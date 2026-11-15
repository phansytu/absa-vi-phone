"""Các hàm tiện ích dùng chung (Utility functions) cho toàn bộ dự án ABSA."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên cho tính tái lập (reproducibility)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: Optional[str] = None) -> str:
    """Xác định thiết bị tính toán (CUDA/CPU)."""
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_yaml_config(cfg_path: str | Path, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """Đọc file cấu hình YAML kèm hỗ trợ ghi đè tham số qua CLI."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if overrides:
        for item in overrides:
            if "=" in item:
                k, v = item.split("=", 1)
                cfg[k.strip()] = yaml.safe_load(v.strip())
    return cfg


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Ghi dữ liệu ra file JSON hỗ trợ tiếng Việt có dấu (ensure_ascii=False)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> Any:
    """Đọc file JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(rows: List[Dict[str, Any]], path: str | Path) -> None:
    """Ghi danh sách bản ghi ra định dạng JSON Lines."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Đọc file JSON Lines."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@contextmanager
def Timer(label: str = "Tác vụ") -> Iterator[None]:
    """Context manager đo thời gian thực thi của một đoạn code."""
    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        print(f"⏱️ [{label}] hoàn thành trong {elapsed:.2f}s ({elapsed / 60:.2f} phút)")


def format_metrics_table(res: Dict[str, Any]) -> str:
    """Định dạng kết quả metric thành chuỗi bảng text đẹp mắt."""
    lines = [
        "=" * 60,
        f"{'CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRIC)':<40} | {'GIÁ TRỊ':<15}",
        "=" * 60,
    ]
    main_keys = [
        ("Số lượng mẫu (Samples)", "n_samples"),
        ("Aspect Micro-F1", "aspect_micro_f1"),
        ("Aspect Macro-F1", "aspect_macro_f1"),
        ("Sentiment Macro-F1 (given gold)", "sentiment_macro_f1_given_gold_aspect"),
        ("Joint Precision", "joint_precision"),
        ("Joint Recall", "joint_recall"),
        (">>> JOINT F1 (END-TO-END) <<<", "joint_f1"),
        ("Tỉ lệ dự đoán đúng mẫu rỗng", "gold_empty_predicted_empty_rate"),
        ("Ngưỡng (Threshold)", "threshold"),
    ]
    for label, k in main_keys:
        if k in res and res[k] is not None:
            val = res[k]
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            lines.append(f"{label:<40} | {val_str:<15}")
    lines.append("=" * 60)
    return "\n".join(lines)
