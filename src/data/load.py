import json
import sys
from pathlib import Path

import pandas as pd

from .schema import parse_label, to_vectors

SPLITS = {"train": "Train.csv", "dev": "Dev.csv", "test": "Test.csv"}
EXPECTED = {"train": 7786, "dev": 1112, "test": 2224}   # tổng 11.122


def load_split(raw_dir: str, split: str) -> list:
    df = pd.read_csv(Path(raw_dir) / SPLITS[split], encoding="utf-8")
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
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = load_split(raw_dir, split)
        assert len(rows) == EXPECTED[split], (split, len(rows))
        with open(Path(out_dir) / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} mẫu -> {out_dir}/{split}.jsonl")


if __name__ == "__main__":
    main(*sys.argv[1:])