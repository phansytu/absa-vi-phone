"""Ghep PhoBERT + baseline TF-IDF theo tung aspect va danh gia cong bang.

Moi aspect lay xac suat presence tu mo hinh tot hon. Mac dinh: STORAGE lay tu
baseline (F1 ~0.69), 9 aspect con lai lay tu PhoBERT.

Chay tu thu muc goc project (can file du doan co truong "sent_all", xem predict.py moi):
    python -m src.hybrid \
        --pred-dev  outputs/predictions/multitask_seed42_dev.jsonl \
        --pred-test outputs/predictions/multitask_seed42_test.jsonl

Cach doc ket qua:
  * Nguong duoc tune tren DEV. Cot "dev" la so lieu lac quan (tune va do tren cung tap).
  * Cot "test" dung nguong tu dev -> so lieu trung thuc. Chi nen bao cao cot nay.
  * --blend w: p = w*p_baseline + (1-w)*p_phobert cho aspect trong --swap (w=1: chi baseline).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data.load import load_split
from .data.schema import ASPECTS, POL2ID
from .evaluate import evaluate
from .models.baseline import TfidfBaseline
from .tune_threshold import FLOOR_V2C, apply_threshold, format_threshold_table, tune_per_aspect_threshold_floor

GRID = np.arange(0.10, 0.91, 0.05)


def read_preds(path: str, gold: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Doc file du doan -> (prob [N,10], sent [N,10] cho CA 10 aspect)."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert len(rows) == len(gold), f"{path}: {len(rows)} dong, gold co {len(gold)}"
    prob = np.zeros((len(rows), len(ASPECTS)))
    sent = np.zeros((len(rows), len(ASPECTS)), dtype=int)
    for i, (r, g) in enumerate(zip(rows, gold)):
        assert r["id"] == g["id"], f"{path}: lech id tai dong {i}: {r['id']} != {g['id']}"
        if "sent_all" not in r:
            raise KeyError(f"{path} thieu 'sent_all'. Chay lai predict.py ban moi de tao file du doan.")
        for j, a in enumerate(ASPECTS):
            prob[i, j] = r["prob"].get(a, 0.0)
            sent[i, j] = POL2ID[r["sent_all"][a]]
    return prob, sent


def fuse(pho: np.ndarray, base: np.ndarray, swap: list[str], w: float) -> np.ndarray:
    out = pho.copy()
    for a in swap:
        j = ASPECTS.index(a)
        out[:, j] = w * base[:, j] + (1 - w) * pho[:, j]
    return out


def _f(x, w: int = 8, d: int = 4) -> str:
    """Dinh dang so, chiu duoc None (vd split khong co mau rong)."""
    return f"{x:>{w}.{d}f}" if x is not None else f"{'n/a':>{w}}"


def score(gold, prob, sent, thr) -> dict:
    r = evaluate(gold, apply_threshold(prob, thr), sent)
    return {"joint": r["joint_f1"], "macro": r["aspect_macro_f1"],
            "empty": r["gold_empty_predicted_empty_rate"],
            "per_aspect": r["aspect_f1_per_aspect"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--pred-dev", required=True)
    ap.add_argument("--pred-test", default=None)
    ap.add_argument("--swap", nargs="+", default=["STORAGE"], help="aspect lay xac suat tu baseline")
    ap.add_argument("--blend", type=float, default=1.0)
    ap.add_argument("--floor", choices=["v2c", "none"], default="v2c")
    ap.add_argument("--sent-from", choices=["phobert", "baseline"], default="phobert")
    ap.add_argument("--out", default="outputs/metrics/hybrid.json")
    a = ap.parse_args()
    for s in a.swap:
        assert s in ASPECTS, f"aspect khong hop le: {s}"
    floor = FLOOR_V2C if a.floor == "v2c" else {}

    raw = Path(a.raw)
    train, dev = load_split(raw, "train"), load_split(raw, "dev")
    test = load_split(raw, "test") if a.pred_test else None

    base = TfidfBaseline().fit(train)
    splits = {"dev": (dev, a.pred_dev)}
    if test:
        splits["test"] = (test, a.pred_test)
    P = {}
    for name, (gold, path) in splits.items():
        pho_p, pho_s = read_preds(path, gold)
        b_p, b_s = base.predict(gold)
        P[name] = {"pho": (pho_p, pho_s), "base": (np.asarray(b_p), np.asarray(b_s))}

    def probs(name, kind):
        pho_p, pho_s = P[name]["pho"]
        b_p, b_s = P[name]["base"]
        if kind == "phobert":
            return pho_p, pho_s
        if kind == "baseline":
            return b_p, b_s
        sent = pho_s if a.sent_from == "phobert" else b_s
        return fuse(pho_p, b_p, a.swap, a.blend), sent

    gold_dev_p = np.array([r["presence"] for r in dev])
    results, thresholds = {}, {}
    for kind in ["baseline", "phobert", "hybrid"]:
        pd_, sd_ = probs("dev", kind)
        thr = tune_per_aspect_threshold_floor(pd_, gold_dev_p, grid=GRID, floor=floor)
        thresholds[kind] = thr
        results[kind] = {"dev": score(dev, pd_, sd_, thr)}
        if test:
            pt_, st_ = probs("test", kind)
            results[kind]["test"] = score(test, pt_, st_, thr)

    # ---- in bang
    cols = ["dev"] + (["test"] if test else [])
    print(f"\nSwap={a.swap} blend={a.blend} floor={a.floor} sent_from={a.sent_from}")
    print(f"{'mo hinh':<10}" + "".join(f" | {c:^34}" for c in cols))
    print(f"{'':<10}" + "".join(f" | {'joint':>8} {'macro':>8} {'empty':>8} {'STOR':>6}" for _ in cols))
    for kind in ["baseline", "phobert", "hybrid"]:
        line = f"{kind:<10}"
        for c in cols:
            r = results[kind][c]
            line += f" | {_f(r['joint'])} {_f(r['macro'])} {_f(r['empty'])} {_f(r['per_aspect']['STORAGE'], 6, 3)}"
        print(line)
    print("\nNguong hybrid (tune tren dev):")
    print(format_threshold_table(thresholds["hybrid"]))
    if test:
        print("\n-> Bao cao cot TEST. Cot dev chi de tham khao (tune va do tren cung tap).")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(a), "thresholds": {k: dict(zip(ASPECTS, map(float, v))) for k, v in thresholds.items()},
               "results": {k: {s: {m: v for m, v in r.items()} for s, r in d.items()} for k, d in results.items()}}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDa ghi {out}")


if __name__ == "__main__":
    main()
