"""Phân tích lỗi cho baseline/multitask — KHÔNG dùng để chấm điểm.

Tách biệt khỏi evaluate.py (gác cổng test của TV2) để tránh làm hỏng metric chuẩn.
Input: gold rows (từ load_split) + prediction rows (từ outputs/predictions/*.jsonl).

Các hàm chính:
- per_label_stats: TP/FP/FN/TN theo từng aspect
- extract_empty_errors: 15 mẫu rỗng dự đoán sai
- analyze_contrasting_sentiment: mẫu có ≥2 aspect trái chiều
- analyze_multi_aspect: recall theo số lượng aspect
- threshold_sweep: quét ngưỡng presence
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix

from .data.schema import ASPECTS, IGNORE_INDEX, POLARITES


# ---------------------------------------------------------------- IO
def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------- helpers
def _gold_presence(gold_rows: list[dict]) -> np.ndarray:
    return np.array([r["presence"] for r in gold_rows], dtype=int)


def _pred_presence(pred_rows: list[dict]) -> np.ndarray:
    """Từ prediction row {id, text, pred:[{aspect,sentiment}], prob:{...}}
    -> ma trận presence [N, 10] in {0,1}."""
    mat = np.zeros((len(pred_rows), len(ASPECTS)), dtype=int)
    for i, r in enumerate(pred_rows):
        for d in r.get("pred", []):
            if d["aspect"] in ASPECTS:
                mat[i, ASPECTS.index(d["aspect"])] = 1
    return mat


def _pred_sentiment(pred_rows: list[dict]) -> np.ndarray:
    """Sentiment dự đoán [N,10]; IGNORE_INDEX nếu aspect vắng."""
    from .data.schema import POL2ID

    mat = np.full((len(pred_rows), len(ASPECTS)), IGNORE_INDEX, dtype=int)
    for i, r in enumerate(pred_rows):
        for d in r.get("pred", []):
            if d["aspect"] in ASPECTS:
                mat[i, ASPECTS.index(d["aspect"])] = POL2ID[d["sentiment"]]
    return mat


# ---------------------------------------------------------------- 1. Confusion per-aspect
def per_label_stats(
    gold_rows: list[dict],
    pred_rows: list[dict],
) -> pd.DataFrame:
    """TP/FP/FN/TN + precision/recall/F1 per aspect.

    Dùng multilabel_confusion_matrix vì baseline dự đoán presence độc lập
    từng aspect (multi-label), KHÔNG phải span-based NER.
    Sắp xếp giảm dần theo FP — nhãn FP cao nhất là nghi phạm over-predict.
    """
    gp = _gold_presence(gold_rows)
    pp = _pred_presence(pred_rows)
    mcm = multilabel_confusion_matrix(gp, pp)  # (10, 2, 2)

    rows = []
    for i, a in enumerate(ASPECTS):
        tn, fp, fn, tp = mcm[i].ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append(
            {
                "aspect": a,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("FP", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- 2. 15 mẫu rỗng sai
def extract_empty_errors(
    gold_rows: list[dict],
    pred_rows: list[dict],
) -> pd.DataFrame:
    """Các mẫu gold KHÔNG có khía cạnh nào nhưng model dự đoán có.

    Trả DataFrame với các cột cần cho review thủ công:
    id, text, pred_text (span bị gán), pred_labels, n_pred, max_prob, category.
    """

    def categorize(text: str, labels: list[str]) -> str:
        t = text.lower().strip()
        if t.endswith("?"):
            return "câu hỏi"
        if re.search(r"\b(không|chưa|chẳng|chả|đâu có)\b", t):
            return "phủ định"
        if re.search(r"\b(nếu|khi|giả sử|miễn là)\b", t):
            return "điều kiện"
        if re.search(r"(hihi|hehe|:\)|:d|=\)|kkk|vl|vãi)", t):
            return "teencode"
        if len(t.split()) < 4:
            return "câu ngắn"
        return "khác"

    rows = []
    for g, p in zip(gold_rows, pred_rows):
        if sum(g["presence"]) != 0:
            continue
        preds = p.get("pred", [])
        if not preds:
            continue
        prob = p.get("prob", {})
        max_p = max((prob.get(d["aspect"], 0.0) for d in preds), default=0.0)
        labels = [d["aspect"] for d in preds]
        rows.append(
            {
                "id": g["id"],
                "text": g["text"],
                "pred_text": " | ".join(
                    f"{d['aspect']}→{d['sentiment']}" for d in preds
                ),
                "pred_labels": ",".join(labels),
                "n_pred": len(preds),
                "max_prob": round(float(max_p), 4),
                "category": categorize(g["text"], labels),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("max_prob", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------- 3. Phân tích sentiment trái chiều
def analyze_contrasting_sentiment(
    gold_rows: list[dict],
    pred_rows: list[dict],
) -> pd.DataFrame:
    """Tìm mẫu có ≥2 aspect với polarity trái chiều (Positive + Negative).

    Trả DataFrame: id, text, gold_pairs, pred_pairs, is_correct.
    """
    rows = []
    for g, p in zip(gold_rows, pred_rows):
        # Gold pairs
        gold_pairs = []
        for i in range(len(ASPECTS)):
            if g["sentiment"][i] != IGNORE_INDEX:
                gold_pairs.append((ASPECTS[i], POLARITES[g["sentiment"][i]]))

        gold_pols = {pol for _, pol in gold_pairs}
        if "Positive" not in gold_pols or "Negative" not in gold_pols:
            continue  # không phải trái chiều

        pred_pairs = [(d["aspect"], d["sentiment"]) for d in p.get("pred", [])]
        pred_pols = {pol for _, pol in pred_pairs}

        # Đúng nếu cả 2 polarity đều được dự đoán đúng aspect
        gold_dict = dict(gold_pairs)
        pred_dict = dict(pred_pairs)
        is_correct = all(pred_dict.get(a) == pol for a, pol in gold_pairs)

        rows.append(
            {
                "id": g["id"],
                "text": g["text"],
                "gold_pairs": " | ".join(f"{a}→{pol}" for a, pol in gold_pairs),
                "pred_pairs": " | ".join(f"{a}→{pol}" for a, pol in pred_pairs),
                "is_correct": is_correct,
                "n_gold": len(gold_pairs),
                "n_pred": len(pred_pairs),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("is_correct").reset_index(drop=True)  # sai trước
    return df


# ---------------------------------------------------------------- 4. Phân tích nhiều khía cạnh
def analyze_multi_aspect(
    gold_rows: list[dict],
    pred_rows: list[dict],
) -> pd.DataFrame:
    """Recall theo số lượng aspect gold (1, 2, 3+).

    Trả DataFrame: n_gold_aspects, n_samples, recall, precision.
    """
    groups = defaultdict(lambda: {"n": 0, "tp": 0, "fp": 0, "fn": 0})

    for g, p in zip(gold_rows, pred_rows):
        n_gold = sum(g["presence"])
        gs = {ASPECTS[i] for i, v in enumerate(g["presence"]) if v == 1}
        ps = {d["aspect"] for d in p.get("pred", [])}

        key = min(n_gold, 3)  # 0, 1, 2, 3+
        groups[key]["n"] += 1
        groups[key]["tp"] += len(gs & ps)
        groups[key]["fp"] += len(ps - gs)
        groups[key]["fn"] += len(gs - ps)

    rows = []
    for k in sorted(groups):
        d = groups[k]
        prec = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) else 0.0
        rec = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append(
            {
                "n_gold_aspects": f"{k}+" if k == 3 else str(k),
                "n_samples": d["n"],
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. Threshold sweep
def threshold_sweep(
    gold_rows: list[dict],
    prob_rows: list[dict],
    sent_rows: list[dict],
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep ngưỡng presence toàn cục, tính joint F1 + empty-accuracy trên Dev.

    Cần prob_rows dạng [{'prob': {aspect: p, ...}}] (khớp format jsonl).
    """
    from .evaluate import evaluate
    from .data.schema import from_vectors

    grid = grid if grid is not None else np.arange(0.2, 0.81, 0.05)
    gp = _gold_presence(gold_rows)
    n = len(gold_rows)

    # Build prob matrix [N, 10] từ dict
    prob_mat = np.zeros((n, len(ASPECTS)))
    for i, r in enumerate(prob_rows):
        for j, a in enumerate(ASPECTS):
            prob_mat[i, j] = r.get("prob", {}).get(a, 0.0)

    sent_mat = _pred_sentiment(sent_rows)
    sent_pred = np.where(sent_mat == IGNORE_INDEX, 0, sent_mat)

    results = []
    for t in grid:
        pp = (prob_mat >= t).astype(int)
        r = evaluate(gold_rows, pp, sent_pred)
        empty_mask = gp.sum(1) == 0
        empty_acc = (
            float((pp[empty_mask].sum(1) == 0).mean()) if empty_mask.any() else None
        )
        results.append(
            {
                "threshold": round(float(t), 3),
                "aspect_micro_f1": r["aspect_micro_f1"],
                "aspect_macro_f1": r["aspect_macro_f1"],
                "joint_f1": r["joint_f1"],
                "empty_accuracy": empty_acc,
            }
        )
    return pd.DataFrame(results)


# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from .data.load import load_split

    ap = argparse.ArgumentParser(description="Phân tích lỗi baseline/PhoBERT.")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--pred", required=True, help="File prediction jsonl")
    ap.add_argument("--out", default="outputs/analysis")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "errors").mkdir(parents=True, exist_ok=True)

    dev = load_split(Path(a.raw), "dev")
    preds = load_jsonl(a.pred)
    assert len(dev) == len(preds), (len(dev), len(preds))

    # 1. Per-label
    stats = per_label_stats(dev, preds)
    stats.to_csv(
        out / "metrics" / "per_label_stats.csv", index=False, encoding="utf-8-sig"
    )
    print("=== PER-LABEL STATS (sort by FP) ===")
    print(stats.to_string(index=False))

    # 2. Empty errors
    empties = extract_empty_errors(dev, preds)
    empties.to_csv(
        out / "errors" / "empty_errors.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n=== EMPTY ERRORS: {len(empties)} mẫu ===")
    if not empties.empty:
        print(
            empties[["id", "text", "pred_text", "max_prob", "category"]].to_string(
                index=False, max_colwidth=50
            )
        )
        print("\nCategory breakdown:")
        print(empties["category"].value_counts().to_string())

    # 3. Contrasting sentiment
    contrasting = analyze_contrasting_sentiment(dev, preds)
    contrasting.to_csv(
        out / "errors" / "contrasting_sentiment.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n=== CONTRASTING SENTIMENT: {len(contrasting)} mẫu ===")
    if not contrasting.empty:
        n_correct = int(contrasting["is_correct"].sum())
        print(
            f"  Đúng: {n_correct}/{len(contrasting)} ({n_correct/len(contrasting):.1%})"
        )

    # 4. Multi-aspect
    multi = analyze_multi_aspect(dev, preds)
    multi.to_csv(
        out / "metrics" / "multi_aspect.csv", index=False, encoding="utf-8-sig"
    )
    print("\n=== MULTI-ASPECT ===")
    print(multi.to_string(index=False))

    # 5. Threshold sweep
    sweep = threshold_sweep(dev, preds, preds)
    sweep.to_csv(
        out / "metrics" / "threshold_sweep.csv", index=False, encoding="utf-8-sig"
    )
    print("\n=== THRESHOLD SWEEP ===")
    print(sweep.to_string(index=False))
