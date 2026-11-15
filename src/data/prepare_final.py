from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.preprocess import normalize


def prepare_split(
    input_path: Path,
    output_path: Path,
    norm_kwargs: dict | None = None,
    text_key: str = "text_segmented",
    out_key: str = "text",
) -> int:
    """doc json, normalize text_segmented, ghi ra file moi"""
    norm_kwargs = norm_kwargs or {"tone": True, "teencode": True, "repeat": True}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)
            seg_text = rec.get(text_key) or rec.get("text", "")
            final_text = normalize(seg_text, **norm_kwargs)

            rec[out_key] = final_text
            if "text_segmented" in rec:
                rec["text_segmented"] = seg_text

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"da xu ly {n} mau tu {input_path} -> {output_path}   ")
    return n


def main():
    ap.add_argument("--in-dir", default="data/processed_segmented")
    ap.add_argument("--out-dir", default="data/processed_final")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("tone", action="store_true", default=True)
    ap.add_argument("--no-tone", dest="tone", action="store_false")
    ap.add_argument("--teencode", action="store_true", default=True)
    ap.add_argument("--no-teencode", dest="teencode", action="store_false")
    ap.add_argument("--repeat", action="store_true", default=True)
    ap.add_argument("--no-repeat", dest="repeat", action="store_false")
    args = ap.parse_args()

    norm_kwargs = {"tone": args.tone, "teencode": args.teencode, "repeat": args.repeat}

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    for split in args.splits:
        src = in_dir / f"{split}.jsonl"
        dst = out_dir / f"{split}.jsonl"
        if src.exists():
            prepare_split(src, dst, norm_kwargs)
        else:
            print(f"khong tim thay file {src}, bo qua")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    main()
