"""Xuat du doan tu checkpoint (dung de cham diem / phan tich / ghep mo hinh).

    python -m src.predict --seed 42 --split dev  --thresholds ckpt
    python -m src.predict --seed 42 --split test --thresholds outputs/metrics/thresholds.json
    python -m src.predict --seed 42 --tag run1 --split dev --threshold 0.5

Cau hinh (encoder, norm, use_segmented, max_length...) duoc doc tu CHECKPOINT, khong doc lai YAML,
de tien xu ly luc suy luan luon khop luc huan luyen.
--thresholds: duong dan file JSON {aspect: nguong} hoac tu khoa "ckpt" (nguong luu trong checkpoint).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import from_checkpoint_cfg, load_config
from src.data.dataset import ABSADataset, make_collate
from src.data.schema import ASPECTS, POLARITES, from_vectors
from src.models.multitask import MultiTaskABSA


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, S = [], []
    for b in loader:
        pl, sl = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        P.append(torch.sigmoid(pl.float()).cpu())
        S.append(sl.argmax(-1).cpu())
    return torch.cat(P).numpy(), torch.cat(S).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/multitask.yaml",
        help="chi dung neu checkpoint khong luu cfg",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--split", default="dev")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="nguong chung (neu khong co --thresholds)",
    )
    ap.add_argument("--thresholds", default=None, help='JSON per-aspect, hoac "ckpt"')
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = f"multitask{('_' + args.tag) if args.tag else ''}_seed{args.seed}"
    ckpt_path = f"outputs/checkpoints/{name}.pt"
    ckpt = torch.load(
        ckpt_path, map_location=device, weights_only=False
    )  # file do chinh ban tao ra
    cfg = (
        from_checkpoint_cfg(ckpt["cfg"])
        if ckpt.get("cfg")
        else load_config(args.config)
    )
    print(
        f"Checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')}) | encoder={cfg['encoder']} "
        f"pooling={cfg['pooling']} use_segmented={cfg['use_segmented']} norm={cfg['norm']}"
    )

    # === nguong ===
    if args.thresholds == "ckpt":
        thr = np.asarray(ckpt["thresholds"], dtype=float)
        print("Dung nguong luu trong checkpoint (tune tren Dev)")
    elif args.thresholds:
        with open(args.thresholds, encoding="utf-8") as f:
            thr_dict = json.load(f)
        thr = np.array([thr_dict.get(a, args.threshold) for a in ASPECTS], dtype=float)
        print(f"Dung nguong per-aspect tu: {args.thresholds}")
    else:
        thr = np.full(len(ASPECTS), args.threshold, dtype=float)
        print(f"Dung nguong chung: {args.threshold}")
    for a, t in zip(ASPECTS, thr):
        print(f"   {a:15s} {t:.2f}")

    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    collate = make_collate(tok, cfg["max_length"])
    model = MultiTaskABSA(
        cfg["encoder"], pooling=cfg["pooling"], dropout=cfg["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["model"])

    ds = ABSADataset(
        f"{cfg['data_dir']}/{args.split}.jsonl",
        cfg["norm"],
        use_segmented=cfg["use_segmented"],
    )
    prob, sent = predict(model, DataLoader(ds, 64, collate_fn=collate), device)

    out_path = Path(f"outputs/predictions/{name}_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r, p, s in zip(ds.rows, prob, sent):
            pres = (p >= thr).astype(int)
            rec = {
                "id": r["id"],
                "text": r["text"],
                "pred": from_vectors(pres.tolist(), s.tolist()),
                "prob": {a: round(float(x), 4) for a, x in zip(ASPECTS, p)},
                # Sentiment du doan cho CA 10 aspect (khong phu thuoc nguong presence)
                "sent_all": {a: POLARITES[int(x)] for a, x in zip(ASPECTS, s)},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
