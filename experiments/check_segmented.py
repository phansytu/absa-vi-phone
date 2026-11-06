"""Kiểm tra nhanh chất lượng dữ liệu đã tách từ."""
import json
from pathlib import Path


def check_split(split: str, base_dir: str = "data/processed_segmented") -> dict:
    path = Path(base_dir) / f"{split}.jsonl"
    if not path.exists():
        print(f"❌ {split}: không tồn tại ({path})")
        return {}

    n = 0
    n_with_seg = 0
    n_empty_text = 0
    n_empty_seg = 0
    samples = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            if rec.get("text_segmented"):
                n_with_seg += 1
            else:
                n_empty_seg += 1
            if not rec.get("text", "").strip():
                n_empty_text += 1
            if len(samples) < 2:
                samples.append(rec)

    print(f"✅ {split}: {n} mẫu | có text_segmented: {n_with_seg} | "
          f"thiếu text_segmented: {n_empty_seg} | text rỗng: {n_empty_text}")
    for i, rec in enumerate(samples, 1):
        print(f"   Mẫu {i}:")
        print(f"     Gốc:  {rec.get('text', '')[:60]!r}")
        print(f"     Tách: {rec.get('text_segmented', '')[:60]!r}")
        print(f"     Presence: {rec.get('presence')}")
        print(f"     Sentiment: {rec.get('sentiment')}")
    print()

    return {
        "split": split,
        "n": n,
        "n_with_seg": n_with_seg,
        "n_missing_seg": n_empty_seg,
        "n_empty_text": n_empty_text,
    }


if __name__ == "__main__":
    results = []
    for split in ["train", "dev", "test"]:
        results.append(check_split(split))

    print("=" * 60)
    print("TỔNG HỢP:")
    total = sum(r.get("n", 0) for r in results)
    total_seg = sum(r.get("n_with_seg", 0) for r in results)
    total_missing = sum(r.get("n_missing_seg", 0) for r in results)
    print(f"  Tổng mẫu: {total}")
    print(f"  Có text_segmented: {total_seg} ({total_seg/total:.1%})")
    print(f"  Thiếu text_segmented: {total_missing}")

    if total_missing == 0:
        print("\n✅ Tất cả mẫu đã được tách từ — sẵn sàng cho PhoBERT")
    else:
        print(f"\n⚠️ Có {total_missing} mẫu thiếu text_segmented — cần kiểm tra")