import re
import unicodedata

# Chỉ rút gọn ký tự không phải chữ số và không phải khoảng trắng
_REPEAT = re.compile(r"([^\d\s])\1{2,}")
_SPACES = re.compile(r"\s+")

# Chuẩn vị trí dấu thanh ở cuối từ (hoà->hòa, thuý->thúy).
# Giữ nguyên nếu sau cặp nguyên âm vẫn còn chữ cái (toàn, thoại, khoảng...)
_TONE = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uý": "úy", "uỳ": "ùy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy"
}
_TONE_RE = re.compile(r"(?<!q)(" + "|".join(map(re.escape, _TONE)) + r")(?!\w)")

# Teencode/viết tắt thông dụng
_TEEN = {
    "không": ["ko", "kg", "hok", "hk", "hong", "hông", "khong", "khôg", "khg"],
    "được": ["dc", "đc"], "điện thoại": ["dt", "đt"], "sản phẩm": ["sp"],
    "nhân viên": ["nv"], "mọi người": ["mn"], "mình": ["mk", "mik", "mih"],
    "với": ["vs"], "wifi": ["wf"], "cũng": ["cx"], "ok": ["oke", "oki", "okê", "okie", "okay"],
    "pin": ["bin"], "gì": ["j"], "rồi": ["r"], "bình thường": ["bt", "bth"],
    "sử dụng": ["sd"], "trước": ["trc"], "luôn": ["lun"], "nhưng": ["nhưg"],
}
_TEEN_MAP = {v: k for k, vs in _TEEN.items() for v in vs}
_TEEN_RE = re.compile(
    r"(?<!\w)(" + "|".join(sorted(map(re.escape, _TEEN_MAP), key=len, reverse=True)) + r")(?!\w)"
)

# "k" đứng riêng = "không"; Giữ nguyên nếu đứng sau chữ số hoặc khoảng trắng sau chữ số (600k, 600 k)
_K_RE = re.compile(r"(?<!\d)(?<!\d\s)(?<!\w)k(?!\w)")

NEGATORS = {"không", "chưa", "chẳng", "chả", "đừng", "chớ"}
_STOP_SCOPE = {"nhưng", "mà", "tuy", "song", "nhưng_mà"}
_TOKEN = re.compile(r"\w+|[^\w\s]")


def normalize(text: str, tone: bool = True, teencode: bool = True, repeat: bool = True) -> str:
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = _SPACES.sub(" ", t).strip()
    if repeat:
        t = _REPEAT.sub(r"\1", t)
    if tone:
        t = _TONE_RE.sub(lambda m: _TONE[m.group(1)], t)
    if teencode:
        t = _K_RE.sub("không", t)
        t = _TEEN_RE.sub(lambda m: _TEEN_MAP[m.group(1)], t)
    return t


def mark_negation(text: str, scope: int = 3) -> str:
    """Thêm tiền tố 'n_' cho tối đa `scope` từ sau từ phủ định, dừng ở dấu câu/từ đối lập.
    Chỉ dùng cho mô hình bag-of-words; transformer tự học phạm vi phủ định."""
    out, left = [], 0
    for tok in _TOKEN.findall(text):
        if not tok[0].isalnum() and tok[0] != "_" or tok in _STOP_SCOPE:
            left = 0
            out.append(tok)
        elif tok in NEGATORS:
            left = scope
            out.append(tok)
        elif left > 0:
            out.append("n_" + tok)
            left -= 1
        else:
            out.append(tok)
    return " ".join(out)