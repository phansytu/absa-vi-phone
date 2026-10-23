import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from src.data.preprocess import mark_negation, normalize


def clean_and_export(input_path: str, output_path: str):
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Đang đọc file thô: {input_file}...")
    df = pd.read_csv(input_file)

    text_col = "comment" if "comment" in df.columns else df.columns[0]

    # 1. Xử lý rỗng
    raw_series = df[text_col].astype(str).str.strip()
    is_empty_mask = (
        df[text_col].isna()
        | (raw_series == "")
        | (raw_series.str.lower() == "nan")
    )

    # 2. Áp dụng tiền xử lý
    normalized_list = []
    processed_list = []

    for text, is_empty in zip(df[text_col], is_empty_mask):
        if is_empty:
            normalized_list.append(np.nan)
            processed_list.append(np.nan)
        else:
            norm_text = normalize(str(text))
            final_text = mark_negation(norm_text, scope=2)
            normalized_list.append(norm_text)
            processed_list.append(final_text)

    # 3. Thêm các cột so sánh
    df["raw_text"] = df[text_col]
    df["normalized"] = normalized_list
    df["final_processed"] = processed_list

    df["is_changed"] = (~is_empty_mask) & (
        df["raw_text"].astype(str).str.lower().str.strip() != df["normalized"]
    )

    # 4. Xuất file CSV (mã hóa utf-8-sig cho Excel)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    # 5. Báo cáo kết quả
    n_total = len(df)
    n_nan = is_empty_mask.sum()
    n_changed = df["is_changed"].sum()

    print("\n" + "=" * 50)
    print("📊 BÁO CÁO XỬ LÝ VÀ TẠO FILE SẠCH")
    print("=" * 50)
    print(f"• Tổng số dòng           : {n_total}")
    print(f"• Số dòng rỗng/NaN       : {n_nan} ({n_nan/n_total:.2%})")
    print(f"• Số dòng có biến đổi    : {n_changed} ({n_changed/n_total:.2%})")
    print(f"• Số dòng giữ nguyên     : {n_total - n_nan - n_changed}")
    print(f"📁 File sạch đã được lưu : {output_file}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    in_p = sys.argv[1] if len(sys.argv) > 1 else "data/raw/Train.csv"
    out_p = sys.argv[2] if len(sys.argv) > 2 else "data/processed/Train_clean.csv"
    clean_and_export(in_p, out_p)