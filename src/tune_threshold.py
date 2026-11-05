"""Tối ưu ngưỡng presence per-aspect trên tập Dev.

Gồm 4 phương án:
- V1: Tối ưu F1 presence từng aspect (nhanh, nhưng có thể giảm empty_acc).
- V2A: Tối ưu đóng góp vào JOINT F1 toàn cục.
- V2B: Tối ưu joint_f1 + lambda * empty_accuracy (có penalty cho mẫu rỗng).
- V2C: Tune presence F1 rồi áp sàn (floor) cho các aspect over-predict.
       Đây là phương án được chọn làm baseline chính thức.

Cách dùng:
    from src.tune_threshold import (
        tune_per_aspect_threshold,          # V1
        tune_per_aspect_threshold_joint,    # V2A
        tune_per_aspect_threshold_constrained,  # V2B
        tune_per_aspect_threshold_floor,    # V2C
        apply_threshold,
        format_threshold_table,
    )
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from .data.schema import ASPECTS
from .evaluate import evaluate


# ============================================================
# Hằng số cấu hình cho V2C
# ============================================================

# Sàn ngưỡng cho các aspect over-predict, dựa trên phân tích lỗi trên Dev:
# - GENERAL: FP=131 (cao nhất) -> cần sàn cao để giảm over-predict
# - SCREEN: Precision thấp (0.73) -> cần sàn trung bình-cao
# - PERFORMANCE: FP=72 -> cần sàn trung bình
# - SER&ACC: Recall thấp (0.75) -> cần sàn trung bình để cân bằng
FLOOR_V2C: dict[str, float] = {
    "GENERAL": 0.60,
    "SCREEN": 0.55,
    "PERFORMANCE": 0.50,
    "SER&ACC": 0.50,
}

# Lưới ngưỡng mặc định
DEFAULT_GRID = np.arange(0.20, 0.81, 0.05)


# ============================================================
# Helpers
# ============================================================

def _gold_rows(
    gold_presence: np.ndarray,
    gold_sentiment: np.ndarray | None = None,
) -> list[dict]:
    """Chuyển ma trận gold -> list[dict] cho evaluate()."""
    n = len(gold_presence)
    if gold_sentiment is None:
        gold_sentiment = np.full_like(gold_presence, -100)
    return [
        {
            "presence": gold_presence[i].tolist(),
            "sentiment": gold_sentiment[i].tolist(),
        }
        for i in range(n)
    ]

# V1: Tối ưu presence F1 từng aspect (KHÔNG khuyến nghị)

def tune_per_aspect_threshold(
    prob: np.ndarray,
    gold_presence: np.ndarray,
    grid: np.ndarray | None = None,
    default: float = 0.5,
) -> np.ndarray:
    """V1: Tối ưu F1 presence từng aspect.

    CẢNH BÁO: Phương án này có thể làm GIẢM empty_accuracy vì nó
    ưu tiên recall presence, dẫn đến tăng false positive trên mẫu rỗng.
    Chỉ dùng để so sánh, không dùng làm baseline chính thức.

    Args:
        prob: [N, 10] xác suất presence.
        gold_presence: [N, 10] nhãn gold.
        grid: lưới ngưỡng (mặc định DEFAULT_GRID).
        default: ngưỡng mặc định nếu aspect vắng.

    Returns:
        [10] ngưỡng tối ưu.
    """
    grid = grid if grid is not None else DEFAULT_GRID
    thr = np.full(len(ASPECTS), default, dtype=float)

    for i in range(len(ASPECTS)):
        y = gold_presence[:, i]
        if y.sum() == 0:
            thr[i] = default
            continue
        scores = [
            f1_score(y, (prob[:, i] >= t).astype(int), zero_division=0)
            for t in grid
        ]
        thr[i] = float(grid[int(np.argmax(scores))])
    return thr


# V2A: Tối ưu đóng góp vào JOINT F1 toàn cục

def tune_per_aspect_threshold_joint(
    prob: np.ndarray,
    gold_presence: np.ndarray,
    gold_sentiment: np.ndarray,
    sent_pred: np.ndarray,
    grid: np.ndarray | None = None,
    default: float = 0.5,
) -> np.ndarray:
    """V2A: Tối ưu ngưỡng per-aspect theo đóng góp vào JOINT F1 toàn cục.

    Với mỗi aspect i:
      - Giữ ngưỡng các aspect khác cố định (default hoặc đã tune trước đó)
      - Sweep ngưỡng cho riêng aspect i
      - Chọn ngưỡng tối đa hoá joint_f1 toàn cục

    Args:
        prob: [N, 10] xác suất presence.
        gold_presence: [N, 10] nhãn gold presence.
        gold_sentiment: [N, 10] nhãn gold sentiment.
        sent_pred: [N, 10] sentiment dự đoán (từ model).
        grid: lưới ngưỡng.
        default: ngưỡng mặc định ban đầu.

    Returns:
        [10] ngưỡng tối ưu.
    """
    grid = grid if grid is not None else DEFAULT_GRID
    thr = np.full(len(ASPECTS), default, dtype=float)
    gold = _gold_rows(gold_presence, gold_sentiment)

    for i in range(len(ASPECTS)):
        best_f1, best_t = -1.0, default
        for t in grid:
            thr_try = thr.copy()
            thr_try[i] = t
            pres = (prob >= thr_try[None, :]).astype(int)
            r = evaluate(gold, pres, sent_pred)
            if r["joint_f1"] > best_f1:
                best_f1, best_t = r["joint_f1"], t
        thr[i] = best_t
    return thr


# V2B: Tối ưu joint_f1 + lambda * empty_accuracy

def tune_per_aspect_threshold_constrained(
    prob: np.ndarray,
    gold_presence: np.ndarray,
    gold_sentiment: np.ndarray,
    sent_pred: np.ndarray,
    grid: np.ndarray | None = None,
    default: float = 0.5,
    lam_empty: float = 0.15,
) -> np.ndarray:
    """V2B: Tối ưu joint_f1 + lam_empty * empty_accuracy.

    Ưu điểm: Có penalty cho mẫu rỗng -> empty_accuracy cao hơn V2A.
    Nhược điểm: joint_f1 có thể giảm đáng kể (phụ thuộc lam_empty).

    Args:
        prob: [N, 10] xác suất presence.
        gold_presence: [N, 10] nhãn gold presence.
        gold_sentiment: [N, 10] nhãn gold sentiment.
        sent_pred: [N, 10] sentiment dự đoán.
        grid: lưới ngưỡng.
        default: ngưỡng mặc định.
        lam_empty: trọng số của empty_accuracy trong hàm mục tiêu.

    Returns:
        [10] ngưỡng tối ưu.
    """
    grid = grid if grid is not None else DEFAULT_GRID
    thr = np.full(len(ASPECTS), default, dtype=float)
    gold = _gold_rows(gold_presence, gold_sentiment)

    for i in range(len(ASPECTS)):
        best_score, best_t = -1.0, default
        for t in grid:
            thr_try = thr.copy()
            thr_try[i] = t
            pres = (prob >= thr_try[None, :]).astype(int)
            r = evaluate(gold, pres, sent_pred)
            empty_acc = r.get("gold_empty_predicted_empty_rate") or 0.0
            score = r["joint_f1"] + lam_empty * empty_acc
            if score > best_score:
                best_score, best_t = score, t
        thr[i] = best_t
    return thr

# V2C: Tune presence F1 + áp sàn cho aspect over-predict (CHÍNH THỨC)

def tune_per_aspect_threshold_floor(
    prob: np.ndarray,
    gold_presence: np.ndarray,
    grid: np.ndarray | None = None,
    default: float = 0.5,
    floor: dict[str, float] | None = None,
) -> np.ndarray:
    """V2C: Tune presence F1 từng aspect, sau đó áp sàn cho aspect over-predict.

    Đây là phương án được chọn làm BASELINE CHÍNH THỨC vì:
    - Cân bằng tốt nhất giữa joint_f1 và empty_accuracy.
    - Đơn giản, dễ giải thích, dễ tái lập.
    - Sàn ngưỡng được chọn dựa trên phân tích lỗi (per_label_stats).

    Args:
        prob: [N, 10] xác suất presence.
        gold_presence: [N, 10] nhãn gold presence.
        grid: lưới ngưỡng.
        default: ngưỡng mặc định.
        floor: dict {aspect: ngưỡng sàn}. Mặc định dùng FLOOR_V2C.

    Returns:
        [10] ngưỡng tối ưu.
    """
    floor = floor if floor is not None else FLOOR_V2C
    grid = grid if grid is not None else DEFAULT_GRID
    thr = np.full(len(ASPECTS), default, dtype=float)

    for i, a in enumerate(ASPECTS):
        y = gold_presence[:, i]
        if y.sum() == 0:
            thr[i] = default
            continue
        scores = [
            f1_score(y, (prob[:, i] >= t).astype(int), zero_division=0)
            for t in grid
        ]
        thr[i] = float(grid[int(np.argmax(scores))])
        if a in floor:
            thr[i] = max(thr[i], floor[a])
    return thr

# Utility: Áp ngưỡng + format bảng

def apply_threshold(prob: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Áp ngưỡng per-aspect -> ma trận presence [N, 10] in {0,1}."""
    return (prob >= thr[None, :]).astype(int)


def format_threshold_table(thr: np.ndarray) -> str:
    """In bảng ngưỡng theo aspect."""
    lines = [f"{'aspect':15s} {'threshold':>10s}", "-" * 28]
    for a, t in zip(ASPECTS, thr):
        lines.append(f"{a:15s} {t:>10.2f}")
    return "\n".join(lines)