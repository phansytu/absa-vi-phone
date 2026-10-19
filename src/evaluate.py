from typing import Any
import numpy as np
from sklearn.metrics import f1_score

from .data.schema import ASPECTS, IGNORE_INDEX


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f

def evaluate(
    gold_rows: list[dict[str, Any]],
    presence_pred: list[list[int]] | np.ndarray,
    sent_pred: list[list[int]] | np.ndarray,
) -> dict[str, Any]:
    """Tính toán toàn bộ metric đánh giá ABSA theo tiêu chuẩn Mục 13.
    
    Args:
        gold_rows: Danh sách bản ghi chứa 'presence' [10] và 'sentiment' [10].
        presence_pred: Ma trận nhãn khía cạnh dự đoán [N, 10] in {0, 1}.
        sent_pred: Ma trận nhãn cảm xúc dự đoán [N, 10] in {0, 1, 2}.
    """
    gp = np.array([r["presence"] for r in gold_rows])
    gs = np.array([r["sentiment"] for r in gold_rows])
    pp, ps = np.asarray(presence_pred), np.asarray(sent_pred)

    out: dict[str, Any] = {"n_samples": len(gold_rows)}

    # 1. Aspect Detection Metrics
    out["aspect_micro_f1"] = float(f1_score(gp, pp, average="micro", zero_division=0))
    out["aspect_macro_f1"] = float(f1_score(gp, pp, average="macro", zero_division=0))
    out["aspect_f1_per_aspect"] = {
        a: float(f1_score(gp[:, i], pp[:, i], zero_division=0))
        for i, a in enumerate(ASPECTS)
    }

    # 2. Sentiment Macro-F1 (Diagnostic on Gold Aspects)
    m = gs != IGNORE_INDEX
    out["n_gold_present_pairs"] = int(m.sum())
    out["sentiment_macro_f1_given_gold_aspect"] = float(
        f1_score(gs[m], ps[m], average="macro", labels=[0, 1, 2], zero_division=0)
    )

    # 3. Joint Aspect-Polarity Metrics (End-to-End)
    tp = int(((pp == 1) & (gp == 1) & (ps == gs)).sum())
    fp = int((pp == 1).sum()) - tp
    fn = int((gp == 1).sum()) - tp
    p, r, f = _prf(tp, fp, fn)
    out.update(joint_precision=p, joint_recall=r, joint_f1=f)

    out["joint_f1_per_aspect"] = {}
    for i, a in enumerate(ASPECTS):
        t = int(((pp[:, i] == 1) & (gp[:, i] == 1) & (ps[:, i] == gs[:, i])).sum())
        out["joint_f1_per_aspect"][a] = _prf(
            t, int(pp[:, i].sum()) - t, int(gp[:, i].sum()) - t
        )[2]

    # 4. Empty Text Metrics
    empty = gp.sum(1) == 0
    out["n_gold_empty"] = int(empty.sum())
    out["gold_empty_predicted_empty_rate"] = (
        float((pp[empty].sum(1) == 0).mean()) if empty.any() else None
    )

    return out