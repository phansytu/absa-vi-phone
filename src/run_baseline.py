import json
import os
import sys
from pathlib import Path

import numpy as np

from .data.load import load_split
from .data.schema import ASPECTS, from_vectors
from .evaluate import evaluate
from .models.baseline import TfidfBaseline
from .tune_threshold import apply_threshold, tune_per_aspect_threshold_floor, format_threshold_table


def main(raw_dir: str = "data/raw", out_dir: str = "outputs"):
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

    # 3. Lưu kết quả V2C
    metrics_dir = out_path / "metrics"
    preds_dir = out_path / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "baseline_dev_v2c.json", "w", encoding="utf-8") as f:
        json.dump({**res_v2c, "thresholds": thr_v2c.tolist(), "mode": "v2c"}, f, indent=2, ensure_ascii=False)

    with open(preds_dir / "baseline_dev_v2c.jsonl", "w", encoding="utf-8") as f:
        for r, p, s, pr in zip(dev, pres_v2c, sent, prob):
            rec = {
                "id": r["id"], "text": r["text"],
                "pred": from_vectors(p.tolist(), s.tolist()),
                "prob": {a: round(float(x), 4) for a, x in zip(ASPECTS, pr)},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nĐã lưu: outputs/metrics/baseline_dev_v2c.json")
    print(f"Đã lưu: outputs/predictions/baseline_dev_v2c.jsonl")


if __name__ == "__main__":
    raw_arg = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    main(raw_arg, out_arg)