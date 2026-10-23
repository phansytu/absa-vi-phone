import pandas as pd
import re

# Đọc file Train.csv
df = pd.read_csv("data/raw/Train.csv")
text_col = "comment" if "comment" in df.columns else df.columns[0]

# Regex tìm chữ 's' đứng độc lập (không dính chữ cái hoặc chữ số phía trước/sau)
pattern = re.compile(r"(?<![\w\d])s(?!\w)", re.IGNORECASE)

matches = []
for idx, text in enumerate(df[text_col].dropna()):
    text_str = str(text)
    if pattern.search(text_str):
        matches.append(text_str)

print(f"📊 Tổng số câu chứa chữ 's' đứng một mình: {len(matches)}")
print("\n=" * 60)
print("XEM TRƯỚC BỐI CẢNH XUẤT HIỆN CỦA 's':")
print("=" * 60)
for i, sentence in enumerate(matches[:15], 1):
    print(f"[{i}] {sentence}")