"""Audit văn bản thô của TRAIN: đo tỉ lệ thật của (A) bất thường khoảng trắng/ký tự,
(B) biến thể vị trí dấu thanh, (C) token không phải âm tiết tiếng Việt hợp lệ (dính từ, teencode, ngoại ngữ).

Chạy: python experiments/audit_text.py data/raw/Train.csv
Mọi phép đếm làm SAU khi chuẩn NFC.
"""
import collections
import re
import sys
import unicodedata
import pandas as pd

# 1. Đọc dữ liệu đầu vào
path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/Train.csv"

try:
    df = pd.read_csv(path, encoding="utf-8")
    col_name = "comment" if "comment" in df.columns else df.columns[0]
    raw = df[col_name].astype(str).tolist()
except Exception as e:
    print(f"Lỗi đọc tập tin dữ liệu {path}: {e}")
    sys.exit(1)

N = len(raw)
nfc = [unicodedata.normalize("NFC", t) for t in raw]
pct = lambda k: f"{k:6d} ({k / N:6.2%})"
print(f"== Tổng số văn bản kiểm toán: {N} ==\n")

# ---------- A. Khoảng trắng và ký tự đặc biệt ----------
A = {
    "không ở dạng NFC (chữ bị tách dấu)": lambda r, t: r != t,
    "có xuống dòng \\n": lambda r, t: "\n" in t,
    "có \\r hoặc \\t": lambda r, t: bool(re.search(r"[\r\t]", t)),
    "có khoảng trắng không-ngắt (U+00A0)": lambda r, t: "\u00a0" in t,
    "có ký tự độ rộng 0 (U+200B..U+200D,FEFF)": lambda r, t: bool(
        re.search("[\u200b-\u200d\ufeff]", t)
    ),
    "có 2+ khoảng trắng liên tiếp": lambda r, t: bool(re.search(r" {2,}", t)),
    "khoảng trắng đầu/cuối": lambda r, t: t != t.strip(),
    "khoảng trắng TRƯỚC dấu , . ! ?": lambda r, t: bool(re.search(r"\s[,.!?;:]", t)),
    "THIẾU khoảng trắng SAU dấu , . ! ? (liền chữ)": lambda r, t: bool(
        re.search(r"[,!?;:][^\W\d_]|\w\.[^\W\d_]{2,}", t, re.I)
    ),
}

print("A. KHOẢNG TRẮNG / KÝ TỰ BẤT THƯỜNG")
for k, f in A.items():
    print(f"  {k:48s}", pct(sum(f(r, t) for r, t in zip(raw, nfc))))

# ---------- B. Biến thể vị trí dấu thanh ----------
TONES = "\u0300\u0301\u0303\u0309\u0323"
strip_tone = lambda w: unicodedata.normalize(
    "NFC", "".join(c for c in unicodedata.normalize("NFD", w) if c not in TONES)
)
tone_of = lambda w: tuple(c for c in unicodedata.normalize("NFD", w) if c in TONES)

tok_docs = [re.findall(r"\w+", t.lower()) for t in nfc]
cnt = collections.Counter(w for d in tok_docs for w in d)
tot_tokens = sum(cnt.values())

groups = collections.defaultdict(list)
for w, c in cnt.items():
    groups[(strip_tone(w), tone_of(w))].append((w, c))

multi = {k: sorted(v, key=lambda x: -x[1]) for k, v in groups.items() if len(v) > 1}
var_words = {w for v in multi.values() for w, _ in v}
minor = {w for v in multi.values() for w, _ in v[1:]}  # Dạng không phổ biến nhất

docs_minor = sum(any(w in minor for w in d) for d in tok_docs)

print("\nB. BIẾN THỂ DẤU THANH (cùng gốc, cùng loại dấu, khác vị trí đặt)")
print(
    f"  Số nhóm có >=2 dạng viết: {len(multi)} | Tổng lượt token thuộc các nhóm: "
    f"{sum(c for v in multi.values() for _, c in v)} ({sum(c for v in multi.values() for _, c in v)/tot_tokens:.2%} token)"
)
print(
    f"  Lượt xuất hiện của dạng KHÔNG phổ biến nhất: {sum(cnt[w] for w in minor)} "
    f"({sum(cnt[w] for w in minor)/tot_tokens:.3%} token)"
)
print(f"  Số văn bản chứa >=1 dạng không phổ biến: {pct(docs_minor)}")
print("  Top nhóm (dạng:số lần | tỉ lệ dạng ít gặp trong nhóm):")
for k, v in sorted(multi.items(), key=lambda kv: -sum(c for _, c in kv[1][1:]))[:12]:
    s = sum(c for _, c in v)
    print("   ", "  ".join(f"{w}:{c}" for w, c in v), f"| {1 - v[0][1]/s:.0%}")

# Kiểu cũ (oà, uý) vs Kiểu mới (òa, úy)
old = re.compile(r"(?<!q)(oà|oá|oả|oã|oạ|oè|oé|oẻ|oẽ|oẹ|uý|uỳ|uỷ|uỹ|uỵ)(?!\w)")
new = re.compile(r"(òa|óa|ỏa|õa|ọa|òe|óe|ỏe|õe|ọe|úy|ùy|ủy|ũy|ụy)(?!\w)")
o = sum(cnt[w] for w in cnt if old.search(w))
n = sum(cnt[w] for w in cnt if new.search(w))
print(f"  Kiểu CŨ (hoà, thuý...): {o} lượt | Kiểu MỚI (hòa, thúy...): {n} lượt")

# ---------- C. Token không phải âm tiết tiếng Việt hợp lệ ----------
SYL = re.compile(
    r"^(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdđghklmnpqrstvx])?[aăâeêioôơuưy]{1,3}(ch|ng|nh|[cmnpt])?$"
)
is_syl = lambda w: bool(SYL.match(strip_tone(w)))

alpha = {
    w: c
    for w, c in cnt.items()
    if not any(ch.isdigit() for ch in w) and "_" not in w
}
valid = sum(c for w, c in alpha.items() if is_syl(w))
inval = {w: c for w, c in alpha.items() if not is_syl(w)}

tot_alpha = valid + sum(inval.values())
inval_pct = sum(inval.values()) / tot_alpha if tot_alpha > 0 else 0.0

print("\nC. TOKEN CHỮ (không chứa số) KHÔNG PHẢI ÂM TIẾT TIẾNG VIỆT HỢP LỆ")
print(
    f"  Hợp lệ: {valid} lượt | Không hợp lệ: {sum(inval.values())} lượt ({inval_pct:.2%}) "
    f"| {len(inval)} loại token"
)
print(
    "  Top 30 không hợp lệ:",
    ", ".join(
        f"{w}:{c}" for w, c in sorted(inval.items(), key=lambda x: -x[1])[:30]
    ),
)

# Nghi dính từ: tách được thành 2 âm tiết hợp lệ và cả 2 âm tiết đều phổ biến (>=30 lần)
glued = []
for w, c in inval.items():
    if len(w) >= 4 and c <= 20:
        for i in range(2, len(w) - 1):
            a, b = w[:i], w[i:]
            if is_syl(a) and is_syl(b) and cnt[a] >= 30 and cnt[b] >= 30:
                glued.append((w, a, b, c))
                break

print(
    f"  Nghi DÍNH TỪ (tách được 2 âm tiết phổ biến): {len(glued)} loại, {sum(g[3] for g in glued)} lượt"
)
print(
    "  Ví dụ:",
    ", ".join(
        f"{w}={a}+{b}" for w, a, b, _ in sorted(glued, key=lambda g: -g[3])[:15]
    ),
)