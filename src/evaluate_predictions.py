"""Evaluate a saved prediction JSONL against a labeled data split.

Example:
    python -m src.evaluate_predictions --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.load import load_split
from src.data.schema import ASPECTS, POL2ID
from src.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tính chỉ số từ file dự đoán đã lưu (không huấn luyện lại)."
    )
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--raw", default="data/raw", help="Thư mục chứa nhãn Train/Dev/Test")
    parser.add_argument("--pred", default=None, help="File prediction JSONL; mặc định tự chọn theo split")
    parser.add_argument("--out", default=None, help="File JSON lưu chỉ số; mặc định trong outputs/metrics")
    args = parser.parse_args()

    pred_path = Path(args.pred or f"outputs/predictions/multitask_seed42_{args.split}.jsonl")
    out_path = Path(args.out or f"outputs/metrics/multitask_seed42_{args.split}.json")
    if not pred_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dự đoán: {pred_path}")

    with pred_path.open(encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]
    gold = load_split(Path(args.raw), args.split)
    if len(gold) != len(preds):
        raise ValueError(f"Lệch số mẫu: nhãn thật={len(gold)}, dự đoán={len(preds)}")

    presence = np.zeros((len(preds), len(ASPECTS)), dtype=int)
    sentiment = np.zeros((len(preds), len(ASPECTS)), dtype=int)
    for i, (gold_row, pred_row) in enumerate(zip(gold, preds)):
        if gold_row["id"] != pred_row.get("id"):
            raise ValueError(
                f"Lệch ID ở dòng {i}: {gold_row['id']} != {pred_row.get('id')}"
            )
        for item in pred_row.get("pred", []):
            aspect = item["aspect"]
            if aspect not in ASPECTS or item.get("sentiment") not in POL2ID:
                raise ValueError(f"Nhãn dự đoán không hợp lệ ở dòng {i}: {item}")
            presence[i, ASPECTS.index(aspect)] = 1
        for j, aspect in enumerate(ASPECTS):
            sentiment[i, j] = POL2ID[pred_row["sent_all"][aspect]]

    metrics = evaluate(gold, presence, sentiment)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Split: {args.split} | số mẫu: {len(gold)}")
    for key in (
        "aspect_micro_f1",
        "aspect_macro_f1",
        "sentiment_macro_f1_given_gold_aspect",
        "joint_precision",
        "joint_recall",
        "joint_f1",
    ):
        print(f"{key}: {metrics[key]:.4f}")
    print(f"Đã lưu chỉ số: {out_path}")


if __name__ == "__main__":
    main()
