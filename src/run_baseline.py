import argparse
import json
import os
from pathlib import Path

import numpy as np

from .data.load import load_split
from .data.schema import ASPECTS, POLARITES, from_vectors
from .evaluate import evaluate
from .models.baseline import TfidfBaseline
from .tune_threshold import (
    apply_threshold,
    tune_per_aspect_threshold_floor,
    format_threshold_table,
)


def main(raw_dir: str = "data/raw", out_dir: str = "outputs", score_test: bool = False):
    base_dir = Path(os.getcwd()).resolve()
    raw_path = Path(raw_dir).resolve()
    out_path = Path(out_dir).resolve()

    if not (raw_path.is_relative_to(base_dir) and out_path.is_relative_to(base_dir)):
        raise ValueError("Security Error: Path traversal attempt detected!")

    # 1. Load dữ liệu và Huấn luyện Baseline
    train, dev = load_split(raw_path, "train"), load_split(raw_path, "dev")
    model = TfidfBaseline().fit(train)
    prob, sent = model.predict(dev)
    gold_p = np.array([r["presence"] for r in dev])

    # 2. NGƯỠNG V2C (FLOOR THRESHOLD) - MỐC CHÍNH THỨC
    thr_v2c = tune_per_aspect_threshold_floor(prob, gold_p)
    pres_v2c = apply_threshold(prob, thr_v2c)
    res_v2c = evaluate(dev, pres_v2c, sent)

    print("\n=== V2C FLOOR THRESHOLD (BASELINE CHÍNH THỨC) ===")
    print(format_threshold_table(thr_v2c))
    print(f"  joint_f1      = {res_v2c['joint_f1']:.4f}")
    print(f"  aspect_micro  = {res_v2c['aspect_micro_f1']:.4f}")
    print(f"  aspect_macro  = {res_v2c['aspect_macro_f1']:.4f}")
    print(f"  empty_acc     = {res_v2c['gold_empty_predicted_empty_rate']:.4f}")

    test_result = None
    if score_test:
        # Only score Test after configuration/thresholds are frozen; tune remains Dev-only.
        test = load_split(raw_path, "test")
        test_prob, test_sent = model.predict(test)
        test_presence = apply_threshold(test_prob, thr_v2c)
        test_metrics = evaluate(test, test_presence, test_sent)
        test_result = (test, test_prob, test_sent, test_presence, test_metrics)

        print("\n=== TEST (dùng ngưỡng đã chọn trên Dev) ===")
        print(f"  joint_f1      = {test_metrics['joint_f1']:.4f}")
        print(f"  aspect_micro  = {test_metrics['aspect_micro_f1']:.4f}")
        print(f"  aspect_macro  = {test_metrics['aspect_macro_f1']:.4f}")
        print(f"  sentiment_f1  = {test_metrics['sentiment_macro_f1_given_gold_aspect']:.4f}")

    # 4. Lưu kết quả Dev và Test. Ngưỡng ở cả hai file đều được chọn từ Dev.
    metrics_dir = out_path / "metrics"
    preds_dir = out_path / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "baseline_dev_v2c.json", "w", encoding="utf-8") as f:
        json.dump(
            {**res_v2c, "thresholds": thr_v2c.tolist(), "mode": "v2c"},
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(preds_dir / "baseline_dev_v2c.jsonl", "w", encoding="utf-8") as f:
        for r, p, s, pr in zip(dev, pres_v2c, sent, prob):
            rec = {
                "id": r["id"],
                "text": r["text"],
                "pred": from_vectors(p.tolist(), s.tolist()),
                "prob": {a: round(float(x), 4) for a, x in zip(ASPECTS, pr)},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if test_result is not None:
        test, test_prob, test_sent, test_presence, test_metrics = test_result
        with open(metrics_dir / "baseline_test_v2c.json", "w", encoding="utf-8") as f:
            json.dump(
                {**test_metrics, "thresholds": thr_v2c.tolist(), "threshold_source": "dev_v2c"},
                f,
                indent=2,
                ensure_ascii=False,
            )

        with open(preds_dir / "baseline_test_v2c.jsonl", "w", encoding="utf-8") as f:
            for r, p, s, pr in zip(test, test_presence, test_sent, test_prob):
                rec = {
                    "id": r["id"],
                    "text": r["text"],
                    "pred": from_vectors(p.tolist(), s.tolist()),
                    "prob": {a: round(float(x), 4) for a, x in zip(ASPECTS, pr)},
                    "sent_all": {a: POLARITES[int(x)] for a, x in zip(ASPECTS, s)},
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\nĐã lưu: outputs/metrics/baseline_dev_v2c.json")
    print("Đã lưu: outputs/predictions/baseline_dev_v2c.jsonl")
    if score_test:
        print("Đã lưu: outputs/metrics/baseline_test_v2c.json")
        print("Đã lưu: outputs/predictions/baseline_test_v2c.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá baseline TF-IDF.")
    parser.add_argument("raw_dir", nargs="?", default="data/raw")
    parser.add_argument("out_dir", nargs="?", default="outputs")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Chấm Test sau khi cấu hình đã chốt; ngưỡng vẫn chỉ chọn trên Dev.",
    )
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, score_test=args.test)
