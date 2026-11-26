"""Huan luyen PhoBERT multitask ABSA.

    python -m src.train --config configs/multitask.yaml --seed 42 --tag run1
    python -m src.train --config configs/multitask.yaml --tag abl --set pos_weight_power=0 "oversample={STORAGE: 4}"
    python -m src.train --config configs/multitask.yaml --limit 200            # chay thu nhanh
    python -m src.train --config configs/multitask.yaml --overfit 64           # kiem tra code: hoc thuoc 64 mau

Luu y:
  * Checkpoint chon theo cfg["select_metric"] tren Dev (nguong tune tren Dev moi epoch) -> so lieu LAC QUAN.
    Chi dung de chon epoch. So lieu bao cao cuoi cung phai la tren Test voi nguong tu Dev.
  * Cau hinh duoc luu NGUYEN VAN vao checkpoint; predict.py doc lai cau hinh tu do.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.config import load_config
from src.data.dataset import ABSADataset, make_collate
from src.data.schema import ASPECTS, IGNORE_INDEX
from src.evaluate import evaluate
from src.models.multitask import MultiTaskABSA, multitask_loss

GRID = np.arange(0.1, 0.91, 0.05)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    P, S = [], []
    for b in loader:
        pl, sl = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        P.append(torch.sigmoid(pl.float()).cpu())
        S.append(sl.argmax(-1).cpu())
    return torch.cat(P).numpy(), torch.cat(S).numpy()


def tune_thresholds(prob, gold, grid=GRID):
    """Nguong F1 toi uu tung aspect. Aspect khong co mau duong (hoac F1 luon 0) -> 0.5."""
    th = np.full(prob.shape[1], 0.5)
    for a in range(prob.shape[1]):
        if gold[:, a].sum() == 0:
            continue
        scores = [f1_score(gold[:, a], prob[:, a] >= t, zero_division=0) for t in grid]
        if max(scores) > 0:
            th[a] = grid[int(np.argmax(scores))]
    return th


def selection_score(r, mode):
    if mode == "joint":
        return r["joint_f1"]
    if mode == "macro":
        return r["aspect_macro_f1"]
    return 0.5 * (r["joint_f1"] + r["aspect_macro_f1"])     # macro_joint


def make_pos_weight(presence, power, max_w):
    """pos_weight tung aspect = clip(((1-p)/p)**power, 1, max_w). Aspect pho bien khong bi giam trong so."""
    if power <= 0:
        return None
    p = np.clip(presence.mean(0), 1e-4, 1.0)
    return torch.tensor(np.clip(((1 - p) / p) ** power, 1.0, max_w), dtype=torch.float)


def make_sent_weight(sentiment, power, n_pol=3):
    if power <= 0:
        return None
    counts = np.bincount(sentiment[sentiment != IGNORE_INDEX], minlength=n_pol).clip(min=1)
    w = (counts.sum() / (n_pol * counts)) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float)


def make_sampler(presence, oversample, seed):
    """Mau co aspect hiem duoc lay nhieu hon (chi o TRAIN). Do dai epoch giu nguyen."""
    if not oversample:
        return None
    w = np.ones(len(presence))
    for a, f in oversample.items():
        w = np.where(presence[:, ASPECTS.index(a)] == 1, np.maximum(w, f), w)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(w), replacement=True, generator=g)


def param_groups(model, cfg):
    """4 nhom: (encoder|head) x (co|khong weight decay). Bias va LayerNorm khong decay."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "pres_b", "sent_b")
    groups = {}
    for name, p in model.named_parameters():
        enc = name.startswith("encoder.")
        decay = not any(nd in name for nd in no_decay)
        g = groups.setdefault((enc, decay), {
            "params": [], "lr": cfg["lr_encoder"] if enc else cfg["lr_head"],
            "weight_decay": cfg["weight_decay"] if decay else 0.0})
        g["params"].append(p)
    return list(groups.values())


def truncation_stats(tok, ds, max_length, k=2000):
    texts = [ds[i]["text"] for i in range(min(k, len(ds)))]
    lens = np.array([len(x) for x in tok(texts)["input_ids"]])
    return float((lens > max_length).mean()), int(np.percentile(lens, 95))


def main(cfg_path, seed, sets=(), limit=None, overfit=None, tag=""):
    cfg = load_config(cfg_path, sets)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    collate = make_collate(tok, cfg["max_length"])
    d, seg = cfg["data_dir"], cfg["use_segmented"]
    tr = ABSADataset(f"{d}/train.jsonl", cfg["norm"], use_segmented=seg)
    dv = ABSADataset(f"{d}/dev.jsonl", cfg["norm"], use_segmented=seg)
    if limit:                                   # smoke test
        tr.rows = tr.rows[:limit]
        dv.rows = dv.rows[:max(200, limit // 4)]
    if overfit:                                 # kiem tra code: hoc thuoc N mau
        tr.rows = tr.rows[:overfit]
        dv.rows = list(tr.rows)

    trunc, p95 = truncation_stats(tok, tr, cfg["max_length"])
    print(f"device={device} | data={d} (use_segmented={seg}) | train={len(tr)} dev={len(dv)} | "
          f"p95 do dai={p95} token | cat bot >{cfg['max_length']}: {trunc:.1%}", flush=True)
    if trunc > 0.05:
        print(f"  CANH BAO: {trunc:.1%} cau bi cat. Can nhac tang max_length (toi da 256 voi PhoBERT).", flush=True)

    train_p = np.array([r["presence"] for r in tr.rows])
    train_s = np.array([r["sentiment"] for r in tr.rows])
    pos_weight = make_pos_weight(train_p, cfg["pos_weight_power"], cfg["pos_weight_max"])
    sent_weight = make_sent_weight(train_s, cfg["sent_weight_power"])
    sampler = make_sampler(train_p, cfg["oversample"], seed)
    if pos_weight is not None:
        show = {a: round(float(pos_weight[ASPECTS.index(a)]), 2) for a in cfg["monitor_aspects"]}
        print(f"pos_weight (power={cfg['pos_weight_power']}): {show}", flush=True)
        pos_weight = pos_weight.to(device)
    if sent_weight is not None:
        print(f"sent_weight: {sent_weight.numpy().round(2).tolist()}", flush=True)
        sent_weight = sent_weight.to(device)
    if sampler is not None:
        print(f"oversample: {cfg['oversample']}", flush=True)

    if sampler is not None:
        tr_dl = DataLoader(tr, cfg["batch_size"], sampler=sampler, collate_fn=collate)
    else:
        g = torch.Generator().manual_seed(seed)
        tr_dl = DataLoader(tr, cfg["batch_size"], shuffle=True, generator=g, collate_fn=collate)
    dv_dl = DataLoader(dv, 64, collate_fn=collate)

    model = MultiTaskABSA(cfg["encoder"], pooling=cfg["pooling"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg))
    total = len(tr_dl) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    name = f"multitask{('_' + tag) if tag else ''}_seed{seed}"
    ckpt_path, metrics_path = f"outputs/checkpoints/{name}.pt", f"outputs/metrics/{name}.json"
    gold_p = np.array([r["presence"] for r in dv.rows])
    mon = [(a, ASPECTS.index(a)) for a in cfg["monitor_aspects"]]
    best, best_ep, history, t0 = -1.0, -1, [], time.time()

    for ep in range(cfg["epochs"]):
        model.train()
        sum_p = sum_s = torch.zeros((), device=device)
        for b in tr_dl:
            with torch.autocast(device, enabled=use_amp):
                pl, sl = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            loss, lp, ls = multitask_loss(
                pl.float(), sl.float(),
                b["presence"].to(device), b["sentiment"].to(device),
                lam=cfg["lambda"], pos_weight=pos_weight, sent_weight=sent_weight,
            )
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            sum_p, sum_s = sum_p + lp, sum_s + ls
        lp_avg, ls_avg = float(sum_p) / len(tr_dl), float(sum_s) / len(tr_dl)

        prob, sent = predict_probs(model, dv_dl, device)
        th = tune_thresholds(prob, gold_p)
        r = evaluate(dv.rows, (prob >= th).astype(int), sent)
        score = selection_score(r, cfg["select_metric"])

        mon_txt = ""
        for a, j in mon:
            ap = average_precision_score(gold_p[:, j], prob[:, j]) if gold_p[:, j].sum() else float("nan")
            mon_txt += f" | {a} f1={r['aspect_f1_per_aspect'][a]:.3f} ap={ap:.3f}"
        print(f"ep{ep} loss_p={lp_avg:.3f} loss_s={ls_avg:.3f} joint={r['joint_f1']:.4f} "
              f"asp_macro={r['aspect_macro_f1']:.4f} sent|gold={r['sentiment_macro_f1_given_gold_aspect']:.4f} "
              f"sel={score:.4f}{mon_txt} ({time.time() - t0:.0f}s)", flush=True)
        history.append({"epoch": ep, "loss_p": lp_avg, "loss_s": ls_avg, "score": float(score),
                        "joint_f1": r["joint_f1"], "aspect_macro_f1": r["aspect_macro_f1"]})

        if score > best + 1e-6:
            best, best_ep = score, ep
            dev_scores = {k: float(v) for k, v in r.items() if isinstance(v, (int, float))}
            torch.save({"model": model.state_dict(), "thresholds": th.tolist(), "cfg": cfg,
                        "epoch": ep, "dev": dev_scores}, ckpt_path)
            best_metrics = {"dev": dev_scores, "aspect_f1": r["aspect_f1_per_aspect"],
                            "thresholds": dict(zip(ASPECTS, th.tolist()))}
        elif ep - best_ep >= cfg["patience"]:
            print(f"Dung som: {cfg['patience']} epoch khong cai thien (tot nhat o ep{best_ep}).", flush=True)
            break

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"cfg": cfg, "best_epoch": best_ep, "select_metric": cfg["select_metric"],
                   "best_score": best, **best_metrics, "history": history}, f, indent=1, ensure_ascii=False)
    print(f"best {cfg['select_metric']} = {best:.4f} (ep{best_ep})  ->  {ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multitask.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set", nargs="*", default=[], help='ghi de cau hinh, vd lr_encoder=3e-5 "oversample={STORAGE: 4}"')
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overfit", type=int)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.config, a.seed, a.set, a.limit, a.overfit, a.tag)
