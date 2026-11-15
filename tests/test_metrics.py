"""Unit test cho bộ tính toán metric đánh giá theo tiêu chuẩn Mục 13 và Mục 6.4."""

import numpy as np
import pytest

from src.data.schema import ASPECTS, IGNORE_INDEX, POL2ID, to_vectors
from src.evaluate import _prf, evaluate


def test_prf_normal_and_zero():
    p, r, f = _prf(tp=10, fp=5, fn=5)
    assert pytest.approx(p, 0.001) == 10 / 15
    assert pytest.approx(r, 0.001) == 10 / 15
    assert pytest.approx(f, 0.001) == 10 / 15

    # Zero division
    p0, r0, f0 = _prf(tp=0, fp=0, fn=0)
    assert p0 == 0.0 and r0 == 0.0 and f0 == 0.0


def test_evaluate_section_6_4_manual_example():
    """Kiểm tra độ chính xác theo đúng bài toán tính tay Mục 6.4 trong HUONG_DAN_NHOM.md:
    Hai câu:
    - Câu 1: gold = {BATTERY: Positive, SCREEN: Negative}
      dự đoán = {BATTERY: Positive, SCREEN: Positive, CAMERA: Positive}
    - Câu 2: gold rỗng (chỉ có OTHERS); dự đoán rỗng.
    """
    # 1. Tạo gold rows
    g1_pres, g1_sent = to_vectors({"BATTERY": "Positive", "SCREEN": "Negative"})
    g2_pres, g2_sent = to_vectors({})

    gold_rows = [
        {"id": "test-1", "presence": g1_pres, "sentiment": g1_sent},
        {"id": "test-2", "presence": g2_pres, "sentiment": g2_sent},
    ]

    # 2. Tạo predictions
    # Câu 1 dự đoán: BATTERY: Positive, SCREEN: Positive, CAMERA: Positive
    p1_pres, p1_sent = to_vectors({
        "BATTERY": "Positive",
        "SCREEN": "Positive",
        "CAMERA": "Positive",
    })
    # Câu 2 dự đoán: Rỗng
    p2_pres, p2_sent = to_vectors({})

    presence_pred = np.array([p1_pres, p2_pres])
    sent_pred = np.array([p1_sent, p2_sent])

    # 3. Đánh giá
    res = evaluate(gold_rows, presence_pred, sent_pred)

    # 4. So sánh số liệu tính tay:
    # Joint: TP = 1 (BATTERY), FP = 2 (SCREEN sai cực tính, CAMERA thừa), FN = 1 (SCREEN gold không được khớp)
    # Precision = 1/3, Recall = 1/2, Joint F1 = 2 * (1/3) * (1/2) / (1/3 + 1/2) = 0.4
    assert pytest.approx(res["joint_precision"], 0.001) == 1 / 3
    assert pytest.approx(res["joint_recall"], 0.001) == 1 / 2
    assert pytest.approx(res["joint_f1"], 0.001) == 0.4

    # Aspect Micro-F1 (chỉ xét có mặt hay không):
    # TP = 2 (BATTERY, SCREEN), FP = 1 (CAMERA), FN = 0 -> 2*2 / (2*2 + 1 + 0) = 4/5 = 0.8
    assert pytest.approx(res["aspect_micro_f1"], 0.001) == 0.8

    # Câu rỗng:
    assert res["n_gold_empty"] == 1
    assert res["gold_empty_predicted_empty_rate"] == 1.0  # Câu 2 gold rỗng được dự đoán đúng rỗng
