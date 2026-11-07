"""Unit test cho End-to-End Pipeline của bài toán ABSA."""

import pytest
from src.data.schema import ASPECTS, POLARITES
from src.pipeline import ABSAPipeline


@pytest.fixture(scope="module")
def pipeline():
    # Sử dụng model_type="auto" để tự động nhận diện mô hình có sẵn trong repo
    try:
        return ABSAPipeline(model_type="auto")
    except Exception as e:
        pytest.skip(f"Bỏ qua test pipeline vì môi trường chưa nạp được mô hình: {e}")


def test_pipeline_empty_input(pipeline):
    res_empty = pipeline.predict("")
    assert res_empty == []

    res_space = pipeline.predict("   \n\t  ")
    assert res_space == []


def test_pipeline_single_prediction_format(pipeline):
    text = "Pin trâu nhưng màn hình xấu"
    res = pipeline.predict(text)
    assert isinstance(res, list)

    for item in res:
        assert "aspect" in item
        assert "sentiment" in item
        assert item["aspect"] in ASPECTS
        assert item["sentiment"] in POLARITES


def test_pipeline_detailed_output(pipeline):
    text = "Máy xài rất mượt, giá rẻ"
    res = pipeline.predict(text, detailed=True)

    assert isinstance(res, dict)
    assert "text" in res
    assert "normalized" in res
    assert "predictions" in res
    assert "aspect_details" in res
    assert "has_aspects" in res

    # Kiểm tra đủ 10 khía cạnh trong aspect_details
    for a in ASPECTS:
        assert a in res["aspect_details"]
        det = res["aspect_details"][a]
        assert "presence_prob" in det
        assert "threshold" in det
        assert "is_present" in det


def test_pipeline_batch_prediction(pipeline):
    texts = [
        "Pin trâu, chụp ảnh nét",
        "Máy hay giật lag",
        "Giao hàng nhanh",
    ]
    res = pipeline.predict(texts)
    assert isinstance(res, list)
    assert len(res) == len(texts)
    for r in res:
        assert isinstance(r, list)


def test_pipeline_threshold_override(pipeline):
    text = "Pin dùng tạm ổn, loa ngoài hơi nhỏ"
    # Với ngưỡng cực cao (0.99), hầu như không khía cạnh nào vượt qua
    pipeline.thresholds = [0.99] * len(ASPECTS)
    res_high = pipeline.predict(text)
    assert len(res_high) == 0

