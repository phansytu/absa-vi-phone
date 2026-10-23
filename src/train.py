import argparse
import json
import os
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.dataset import ABSADataset, make_collate
from src.evaluate import evaluate
from src.models.multitask import MultiTaskABSA, multitask_loss


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    P, S = [], []
    for b in loader:
        pl, sl = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        P.append(torch.sigmoid(pl.float()).cpu())
        S.append(sl.argmax(-1).cpu())
    return torch.cat(P).numpy(), torch.cat(S).numpy()


def tune_thresholds(prob, gold, grid=np.arange(0.1, 0.91, 0.05)):
    th = np.full(prob.shape[1], 0.5)
    for a in range(prob.shape[1]):
        scores = [f1_score(gold[:, a], prob[:, a] >= t, zero_division=0) for t in grid]
        th[a] = grid[int(np.argmax(scores))]
    return th


def apply_overrides(cfg, sets):
    for s in sets:
        k, v = s.split("=", 1)
        cfg[k] = yaml.safe_load(v)      # "lr_encoder=3e-5" -> float, "pooling=cls" -> str
    return cfg


def main(cfg_path, seed, sets=(), limit=None, overfit=None, tag=""):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = apply_overrides(yaml.safe_load(f), sets)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    collate = make_collate(tok, cfg["max_length"])
    tr = ABSADataset("data/processed/train.jsonl", cfg["norm"])
    dv = ABSADataset("data/processed/dev.jsonl", cfg["norm"])
    if limit:                                   # smoke test
        tr.rows = tr.rows[:limit]
        dv.rows = dv.rows[:max(200, limit // 4)]
    if overfit:                                 # kiểm tra code: học thuộc N mẫu
        tr.rows = tr.rows[:overfit]
        dv.rows = list(tr.rows)

    tr_dl = DataLoader(tr, cfg["batch_size"], shuffle=True, collate_fn=collate)
    dv_dl = DataLoader(dv, 64, collate_fn=collate)

    model = MultiTaskABSA(cfg["encoder"], pooling=cfg["pooling"]).to(device)
    enc_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": cfg["lr_encoder"]},
         {"params": head_params, "lr": cfg["lr_head"]}],
        weight_decay=0.01,
    )
    total = len(tr_dl) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    name = f"multitask{('_' + tag) if tag else ''}_seed{seed}"
    gold_p = np.array([r["presence"] for r in dv.rows])
    best, t0 = -1.0, time.time()

    for ep in range(cfg["epochs"]):
        model.train()
        for b in tr_dl:
            with torch.autocast(device, enabled=use_amp):
                pl, sl = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            loss, lp, ls = multitask_loss(
                pl.float(), sl.float(),
                b["presence"].to(device), b["sentiment"].to(device),
                lam=cfg["lambda"],
            )
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

        prob, sent = predict_probs(model, dv_dl, device)
        th = tune_thresholds(prob, gold_p)
        r = evaluate(dv.rows, (prob >= th).astype(int), sent)
        print(f"ep{ep} loss_p={lp:.3f} loss_s={ls:.3f} joint={r['joint_f1']:.4f} "
              f"asp_macro={r['aspect_macro_f1']:.4f} "
              f"sent|gold={r['sentiment_macro_f1_given_gold_aspect']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

        if r["joint_f1"] > best:
            best = r["joint_f1"]
            torch.save({"model": model.state_dict(), "thresholds": th, "cfg": cfg},
                       f"outputs/checkpoints/{name}.pt")
            dev_scores = {k: float(v) for k, v in r.items() if isinstance(v, (int, float))}
            with open(f"outputs/metrics/{name}.json", "w", encoding="utf-8") as f:
                json.dump({"cfg": cfg, "dev": dev_scores, "epoch": ep}, f, indent=1)

    print(f"best dev joint_f1 = {best:.4f}  ->  outputs/checkpoints/{name}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multitask.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set", nargs="*", default=[], help="ghi đè cấu hình, ví dụ lr_encoder=3e-5")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overfit", type=int)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.config, a.seed, a.set, a.limit, a.overfit, a.tag)