import argparse
import json
from pathlib import Path

import pandas as pd

from .schema import parse_label, to_vectors

SPLITS = {"train": "Train.csv", "dev": "Dev.csv", "test": "Test.csv"}
EXPECTED = {"train": 7786, "dev": 1112, "test": 2224}   # tổng 11.122


def is_safe_path(base_dir: Path, target_dir: Path) -> bool:
    """Kiểm tra bảo mật xem target_dir có nằm trong vùng an toàn base_dir hay không."""
    try:
        target_path = Path(target_dir).resolve()
        base_path = Path(base_dir).resolve()
        return base_path in target_path.parents or target_path == base_path
    except Exception:
        return False


def load_split(raw_dir: Path, split: str) -> list:
    # Đảm bảo raw_dir truyền vào đã là đối tượng Path an toàn
    df = pd.read_csv(raw_dir / SPLITS[split], encoding="utf-8")
    rows = []
    for r in df.itertuples(index=False):
        pairs, has_others = parse_label(r.label)
        presence, sentiment = to_vectors(pairs)
        rows.append({"id": f"{split}-{r.index}", "text": str(r.comment),
                     "n_star": int(r.n_star), "pairs": pairs,
                     "has_others": has_others,
                     "presence": presence, "sentiment": sentiment})
    return rows


def main(raw_dir="data/raw", out_dir="data/processed"):
    # Xác định thư mục gốc hiện tại của dự án làm vùng an toàn
    base_project_dir = Path(".").resolve()
    
    raw_path = Path(raw_dir).resolve()
    out_path = Path(out_dir).resolve()
    
    base_project_dir = Path(".").resolve()
    
    # KIỂM TRA BẢO MẬT: Ngăn chặn triệt để Path Traversal
    if not is_safe_path(base_project_dir, raw_path) or not is_safe_path(base_project_dir, out_path):
        raise ValueError("Security Error: Thư mục đầu vào hoặc đầu ra nằm ngoài phạm vi an toàn của dự án!")

    # Tạo thư mục đầu ra sau khi đã xác thực an toàn
    out_path.mkdir(parents=True, exist_ok=True)
    
    for split in SPLITS:
        rows = load_split(raw_path, split)
        assert len(rows) == EXPECTED[split], (split, len(rows))
        
        target_file = out_path / f"{split}.jsonl"
        with open(target_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} mẫu -> {target_file}")


if __name__ == "__main__":
    # Sử dụng argparse thay cho sys.argv[1:] trực tiếp để đảm bảo an toàn CLI
    parser = argparse.ArgumentParser(description="Xử lý dữ liệu thô sang định dạng jsonl.")
    parser.add_argument("raw_dir", nargs="?", default="data/raw", help="Thư mục chứa dữ liệu thô")
    parser.add_argument("out_dir", nargs="?", default="data/processed", help="Thư mục lưu kết quả xử lý")
    args = parser.parse_args()
    
    main(raw_dir=args.raw_dir, out_dir=args.out_dir)
