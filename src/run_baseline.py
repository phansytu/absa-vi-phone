import json
import os
import sys
from pathlib import Path

import numpy as np

from .data.load import load_split
from .data.schema import ASPECTS, from_vectors
from .evaluate import evaluate
from .models.baseline import TfidfBaseline


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
    # 2. Tìm ngưỡng tối ưu toàn cục (Global Threshold) trên tập Dev theo Joint F1
    grid = np.arange(0.2, 0.81, 0.05)
    scores = [
        evaluate(dev, (prob >= t).astype(int), sent)["joint_f1"] 
        for t in grid
    ]
    best_idx = int(np.argmax(scores))
    thr = float(grid[best_idx])
    
    # 3. Tạo nhãn dự đoán theo ngưỡng tối ưu và Tính toán Metric
    pres = (prob >= thr).astype(int)
    res = evaluate(dev, pres, sent)
    res.update(threshold=thr, split="dev", model="tfidf_baseline")

    # 4. Lưu Kết quả Metric và Dự đoán
    metrics_dir = out_path / "metrics"
    preds_dir = out_path / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Ghi file metrics JSON an toàn với context manager
    with open(metrics_dir / "baseline_dev.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # Ghi file predictions JSONL
    with open(preds_dir / "baseline_dev.jsonl", "w", encoding="utf-8") as f:
        for r, p, s, pr in zip(dev, pres, sent, prob):
            rec = {
                "id": r["id"],
                "text": r["text"],
                "pred": from_vectors(p.tolist(), s.tolist()),
                "prob": {a: round(float(x), 4) for a, x in zip(ASPECTS, pr)},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 5. In tóm tắt chỉ số chính ra màn hình
    summary = {k: v for k, v in res.items() if not isinstance(v, dict)}
    print("\n=== BASELINE DEV EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raw_arg = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    main(raw_arg, out_arg)