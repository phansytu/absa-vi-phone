# experiments/check_normalize_on_segmented.py
import json
from src.data.preprocess import normalize

with open("data/processed_segmented/dev.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 10: break
        rec = json.loads(line)
        seg = rec["text_segmented"]
        norm = normalize(seg, tone=True, teencode=True, repeat=True)
        print(f"Gốc:  {rec['text'][:100]!r}")
        print(f"Tách: {seg[:100]!r}")
        print(f"Norm: {norm[:100]!r}")
        print()