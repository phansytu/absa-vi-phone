from src.data.preprocess import mark_negation, normalize


def test_keeps_negation():
    assert "không" in normalize("Pin KHÔNG tốt")
    assert "chưa" in normalize("Chưa thấy lỗi")


def test_teencode_negation_unified():
    assert normalize("máy ko tốt, k đẹp") == "máy không tốt, không đẹp"


def test_price_unit_k_kept():
    assert "600k" in normalize("giá 600k")
    # Bổ sung: Bảo vệ trường hợp có khoảng trắng giữa số và k
    assert "600 k" in normalize("giá 600 k")


def test_collapse_repeats_and_spaces():
    assert normalize("Đẹppppp   quá\n\nluôn") == "đẹp quá luôn"


def test_number_repeats_not_collapsed():
    # Bổ sung: Đảm bảo chuỗi số không bị rút gọn
    assert normalize("giá 1000000đ") == "giá 1000000đ"


def test_keeps_double_letters():
    assert normalize("loa boong") == "loa boong"


def test_unicode_nfc():
    assert normalize("e\u0301") == "\u00e9"


def test_tone_placement():
    assert normalize("hoà vốn, khoẻ") == "hòa vốn, khỏe"
    assert "quý" in normalize("quý khách")  # qu + ý không bị đổi


def test_negation_scope_stops_at_contrast():
    out = mark_negation("pin không tốt nhưng màn hình đẹp", 3)
    assert "n_tốt" in out and "n_màn" not in out