"""Unit test kiểm tra tính toàn vẹn của schema và quy tắc nhãn."""

import pytest
from src.data.schema import (
    ASPECT2ID,
    ASPECTS,
    IGNORE_INDEX,
    POL2ID,
    POLARITES,
    from_vectors,
    parse_label,
    to_vectors,
)


def test_aspects_and_polarities_constants():
    assert len(ASPECTS) == 10
    assert len(POLARITES) == 3
    assert len(ASPECT2ID) == 10
    assert len(POL2ID) == 3
    assert IGNORE_INDEX == -100


def test_parse_label_normal():
    raw = "{CAMERA#Positive};{BATTERY#Negative};"
    pairs, has_others = parse_label(raw)
    assert pairs == {"CAMERA": "Positive", "BATTERY": "Negative"}
    assert has_others is False


def test_parse_label_with_others():
    raw_only_others = "{OTHERS}"
    pairs, has_others = parse_label(raw_only_others)
    assert pairs == {}
    assert has_others is True

    raw_mixed = "{SCREEN#Neutral};{OTHERS};"
    pairs, has_others = parse_label(raw_mixed)
    assert pairs == {"SCREEN": "Neutral"}
    assert has_others is True


def test_parse_label_invalid_token():
    with pytest.raises(ValueError, match="Token nhãn không hợp lệ"):
        parse_label("{UNKNOWN_ASPECT#Positive}")

    with pytest.raises(ValueError, match="Token nhãn không hợp lệ"):
        parse_label("{SCREEN#Happy}")


def test_parse_label_duplicate_aspect():
    with pytest.raises(ValueError, match="Khía cạnh lặp trong một phản hồi"):
        parse_label("{SCREEN#Positive};{SCREEN#Negative}")


def test_to_vectors_conversion():
    pairs = {"SCREEN": "Positive", "BATTERY": "Negative"}
    pres, sent = to_vectors(pairs)

    assert len(pres) == 10
    assert len(sent) == 10

    # Khía cạnh có mặt
    screen_idx = ASPECT2ID["SCREEN"]
    battery_idx = ASPECT2ID["BATTERY"]
    assert pres[screen_idx] == 1
    assert sent[screen_idx] == POL2ID["Positive"]
    assert pres[battery_idx] == 1
    assert sent[battery_idx] == POL2ID["Negative"]

    # Khía cạnh vắng mặt PHẢI là 0 và -100 (tuyệt đối không phải Neutral)
    camera_idx = ASPECT2ID["CAMERA"]
    assert pres[camera_idx] == 0
    assert sent[camera_idx] == IGNORE_INDEX


def test_from_vectors_roundtrip():
    pairs = {"SCREEN": "Positive", "BATTERY": "Negative"}
    pres, sent = to_vectors(pairs)
    reconstructed = from_vectors(pres, sent)

    # Chuyển về dict để so sánh không phụ thuộc thứ tự
    recon_dict = {item["aspect"]: item["sentiment"] for item in reconstructed}
    assert recon_dict == pairs
