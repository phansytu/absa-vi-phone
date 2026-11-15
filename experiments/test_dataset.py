"""Test ABSADataset hoạt động đúng với cả 2 nguồn dữ liệu.

Kiểm tra:
1. Load được file jsonl.
2. Đọc đúng text_segmented (có dấu `_`).
3. Normalize đúng (teencode, lowercase, dấu thanh).
4. Presence/sentiment đúng shape.
5. Collate function hoạt động.
6. So sánh trước/sau normalize.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.data.dataset import ABSADataset, make_collate
from src.data.preprocess import normalize
from src.data.schema import ASPECTS, IGNORE_INDEX


def check_source(name: str, path: str, use_segmented: bool, norm_kwargs: dict | None):
    """Kiểm tra 1 nguồn dữ liệu."""
    print(f"\n{'='*70}")
    print(f"KIỂM TRA: {name}")
    print(f"  Path: {path}")
    print(f"  use_segmented: {use_segmented}")
    print(f"  norm_kwargs: {norm_kwargs}")
    print(f"{'='*70}")

    if not Path(path).exists():
        print(f"❌ File không tồn tại: {path}")
        return False

    # 1. Load dataset
    try:
        ds = ABSADataset(path, norm_kwargs=norm_kwargs, use_segmented=use_segmented)
        print(f"✅ Load được: {len(ds)} mẫu")
    except Exception as e:
        print(f"❌ Lỗi load: {e}")
        return False

    # 2. Kiểm tra 1 mẫu
    try:
        sample = ds[0]
        print(f"\n--- Mẫu 0 ---")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Text: {sample['text'][:80]!r}")
        print(f"  Presence: {sample['presence']}")
        print(f"  Sentiment: {sample['sentiment']}")
    except Exception as e:
        print(f"❌ Lỗi __getitem__: {e}")
        return False

    # 3. Kiểm tra shape
    try:
        assert (
            len(sample["presence"]) == 10
        ), f"presence phải có 10 phần tử, có {len(sample['presence'])}"
        assert (
            len(sample["sentiment"]) == 10
        ), f"sentiment phải có 10 phần tử, có {len(sample['sentiment'])}"
        assert all(v in (0, 1) for v in sample["presence"]), "presence phải là 0/1"
        assert all(
            v == IGNORE_INDEX or 0 <= v < 3 for v in sample["sentiment"]
        ), f"sentiment phải là -100 hoặc 0-2"
        print(f"\n✅ Shape đúng: presence[10], sentiment[10]")
    except AssertionError as e:
        print(f"\n❌ Shape sai: {e}")
        return False

    # 4. So sánh text_segmented (nếu có)
    raw = ds.rows[0]
    if use_segmented and "text_segmented" in raw:
        print(f"\n--- So sánh text ---")
        print(f"  Gốc:          {raw['text'][:70]!r}")
        print(f"  Text_segmented: {raw['text_segmented'][:70]!r}")
        print(f"  Sau dataset:  {sample['text'][:70]!r}")

        # Kiểm tra dấu `_` còn không (nếu text gốc có từ ghép)
        if "_" in raw["text_segmented"]:
            if "_" in sample["text"]:
                print(f"  ✅ Dấu `_` được giữ nguyên")
            else:
                print(f"  ⚠️ Dấu `_` bị mất sau normalize")

    # 5. Kiểm tra normalize (nếu có norm_kwargs)
    if norm_kwargs:
        raw_seg = raw.get("text_segmented", raw["text"])
        expected = normalize(raw_seg, **norm_kwargs)
        if sample["text"] == expected:
            print(f"\n✅ Normalize đúng")
        else:
            print(f"\n⚠️ Normalize khác kỳ vọng:")
            print(f"  Expected: {expected[:70]!r}")
            print(f"  Actual:   {sample['text'][:70]!r}")

    # 6. Kiểm tra 3 mẫu cuối (edge case)
    try:
        for i in [-1, -2, -3]:
            _ = ds[i]
        print(f"✅ 3 mẫu cuối load được")
    except Exception as e:
        print(f"❌ Lỗi load mẫu cuối: {e}")
        return False

    return True


def check_collate(name: str, path: str, encoder: str = "vinai/phobert-base"):
    """Kiểm tra collate function."""
    print(f"\n{'='*70}")
    print(f"KIỂM TRA COLLATE: {name}")
    print(f"{'='*70}")

    try:
        tok = AutoTokenizer.from_pretrained(encoder)
        collate = make_collate(tok, max_length=128)
        ds = ABSADataset(path, norm_kwargs=None, use_segmented=True)

        # Tạo batch 4 mẫu
        batch = [ds[i] for i in range(4)]
        enc = collate(batch)

        print(f"✅ Collate thành công")
        print(f"  input_ids shape: {enc['input_ids'].shape}")
        print(f"  attention_mask shape: {enc['attention_mask'].shape}")
        print(f"  presence shape: {enc['presence'].shape}")
        print(f"  sentiment shape: {enc['sentiment'].shape}")
        print(f"  ids: {enc['ids']}")

        # Kiểm tra shape
        assert enc["input_ids"].shape[0] == 4, "batch size phải là 4"
        assert enc["presence"].shape == (4, 10), "presence phải (4, 10)"
        assert enc["sentiment"].shape == (4, 10), "sentiment phải (4, 10)"
        print(f"✅ Shape collate đúng")

        # Kiểm tra token có `_` không
        tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
        has_underscore = any("_" in t for t in tokens)
        print(f"  Tokens mẫu 0: {tokens[:15]}")
        if has_underscore:
            print(f"  ✅ Token có dấu `_` (tách từ có tác dụng)")
        else:
            print(f"  ⚠️ Không thấy token có `_` — có thể text không có từ ghép")

        return True
    except Exception as e:
        print(f"❌ Lỗi collate: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    results = {}

    # Test 1: processed_segmented (chưa normalize)
    results["segmented (no norm)"] = check_source(
        "processed_segmented (no normalize)",
        "data/processed_segmented/dev.jsonl",
        use_segmented=True,
        norm_kwargs=None,
    )

    # Test 2: processed_segmented + normalize
    results["segmented + norm"] = check_source(
        "processed_segmented + normalize",
        "data/processed_segmented/dev.jsonl",
        use_segmented=True,
        norm_kwargs={"tone": True, "teencode": True, "repeat": True},
    )

    # Test 3: processed_final (đã normalize sẵn) — nếu có
    final_path = Path("data/processed_final/dev.jsonl")
    if final_path.exists():
        results["final (no norm)"] = check_source(
            "processed_final (đã normalize sẵn)",
            "data/processed_final/dev.jsonl",
            use_segmented=False,  # vì text đã sạch
            norm_kwargs=None,
        )
    else:
        print(f"\n⚠️ Bỏ qua processed_final (chưa có file)")
        results["final (no norm)"] = None

    # Test collate
    results["collate"] = check_collate(
        "collate trên processed_segmented",
        "data/processed_segmented/dev.jsonl",
    )

    # Tổng kết
    print(f"\n{'='*70}")
    print("TỔNG KẾT")
    print(f"{'='*70}")
    for name, ok in results.items():
        if ok is None:
            print(f"  ⏭️  {name}: bỏ qua")
        elif ok:
            print(f"  ✅ {name}: PASS")
        else:
            print(f"  ❌ {name}: FAIL")

    if all(v is not False for v in results.values()):
        print(f"\n✅ Tất cả test PASS — dataset sẵn sàng")
        return 0
    else:
        print(f"\n❌ Có test FAIL — cần sửa trước khi train")
        return 1


if __name__ == "__main__":
    sys.exit(main())
