"""Tìm ngưỡng tối ưu cho PhoBERT trên Dev (per-aspect)."""

import json
from pathlib import Path

import numpy as np

from src.data.load import load_split
from src.data.schema import ASPECTS, POL2ID
from src.evaluate import evaluate
from src.tune_threshold import (
    tune_per_aspect_threshold_floor,
    apply_threshold,
    format_threshold_table,
)


def load_preds(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    dev = load_split(Path("data/raw"), "dev")
    preds = load_preds("outputs/predictions/multitask_seed42_dev.jsonl")

    n = len(preds)
    prob = np.zeros((n, 10))
    sent = np.zeros((n, 10), dtype=int)
    for i, r in enumerate(preds):
        for j, a in enumerate(ASPECTS):
            prob[i, j] = r["prob"].get(a, 0.0)
        for d in r["pred"]:
            j = ASPECTS.index(d["aspect"])
            sent[i, j] = POL2ID[d["sentiment"]]

    gold_p = np.array([r["presence"] for r in dev])

    # Global 0.5
    pres_05 = (prob >= 0.5).astype(int)
    r_05 = evaluate(dev, pres_05, sent)
    print("=== Global 0.5 ===")
    print(
        f"  joint={r_05['joint_f1']:.4f}  macro={r_05['aspect_macro_f1']:.4f}  empty={r_05['gold_empty_predicted_empty_rate']:.4f}"
    )

    # Per-aspect (V2C floor)
    thr_v2c = tune_per_aspect_threshold_floor(prob, gold_p)
    pres_v2c = apply_threshold(prob, thr_v2c)
    r_v2c = evaluate(dev, pres_v2c, sent)
    print("\n=== V2C per-aspect ===")
    print(format_threshold_table(thr_v2c))
    print(
        f"  joint={r_v2c['joint_f1']:.4f}  macro={r_v2c['aspect_macro_f1']:.4f}  empty={r_v2c['gold_empty_predicted_empty_rate']:.4f}"
    )

    # Delta
    print(f"\n=== DELTA (V2C - 0.5) ===")
    print(f"  joint: {r_v2c['joint_f1'] - r_05['joint_f1']:+.4f}")
    print(f"  macro: {r_v2c['aspect_macro_f1'] - r_05['aspect_macro_f1']:+.4f}")
    print(
        f"  empty: {r_v2c['gold_empty_predicted_empty_rate'] - r_05['gold_empty_predicted_empty_rate']:+.4f}"
    )


if __name__ == "__main__":
    main()
