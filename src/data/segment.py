"""Tach tu tieng Viet (dung chung cho train va serving) + xuat file ra soat.

Chay tach tu:
    python segmentation.py --in-dir data/processed --out-dir data/processed_segmented

Chay tach tu + xuat file ra soat:
    python segmentation.py --audit-dir reports/segmentation_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None

log = logging.getLogger(__name__)

# Tu phu dinh: neu bi pyvi noi lien voi tu sau (vd "khong_gian") thi can xem lai,
# vi pipeline xu ly phu dinh dua vao viec phu dinh dung rieng.
NEG_WORDS = {"không", "chẳng", "chả", "chưa", "đừng"}
# Teencode con sot lai sau tien xu ly (kiem tra xem buoc teencode da bat het chua).
TEENCODE_LEFT = {"ko", "k", "dc", "đc", "j", "z", "hok", "vs"}
# Co nao bi coi la "nghiem trong" -> dua vao mau ra soat uu tien.
SERIOUS_FLAGS = {
    "seg_failed", "content_changed", "unit_token_broken", "neg_joined",
    "long_compound", "neg_marker_in_input", "underscore_in_input",
}


# ----------------------------------------------------------------------------
# Tach tu
# ----------------------------------------------------------------------------
def segment_with_status(text) -> tuple[str, bool]:
    """Tach tu 1 cau, tra ve (text_da_tach, thanh_cong).

    Raises:
        ImportError: neu chua cai pyvi.
    """
    if ViTokenizer is None:
        raise ImportError("Chua cai pyvi. Chay: pip install pyvi")
    if text is None:
        return "", True
    text = str(text).strip()
    if not text:
        return "", True
    try:
        return ViTokenizer.tokenize(text), True
    except Exception as e:  # fallback an toan nhung KHONG im lang
        log.warning("pyvi loi (%s) voi: %.80r", e, text)
        return text, False


def segment_text(text) -> str:
    """Ban rut gon dung cho serving: chi tra ve chuoi da tach tu."""
    return segment_with_status(text)[0]


# ----------------------------------------------------------------------------
# Phan tich 1 dong de ra soat
# ----------------------------------------------------------------------------
def _is_compound(t: str) -> bool:
    """Tu ghep do tach tu tao ra (bo qua tien to phu dinh 'n_' do pipeline gan)."""
    t = t.strip("_")
    return "_" in t and not t.startswith("n_")


def _norm(s: str) -> str:
    return re.sub(r"[\s_]+", "", s)


def analyze(orig: str, seg: str, ok: bool) -> list[str]:
    """Tra ve danh sach co (flag) can xem lai cho 1 cau."""
    flags: list[str] = []
    toks = seg.split()
    compounds = [t for t in toks if _is_compound(t)]

    if not ok:
        flags.append("seg_failed")
    # Bat bien: bo khoang trang va "_" thi noi dung phai giong het ban goc.
    elif _norm(orig) != _norm(seg):
        flags.append("content_changed")
    if "_" in orig:
        flags.append("underscore_in_input")      # trung ky hieu noi am tiet
    if re.search(r"\bn_\w", orig):
        flags.append("neg_marker_in_input")      # 'n_' chi danh cho TF-IDF, khong dua vao PhoBERT
    if any(t.count("_") >= 3 for t in compounds):
        flags.append("long_compound")            # tu ghep >= 4 am tiet
    if any(t.split("_")[0].lower() in NEG_WORDS for t in compounds):
        flags.append("neg_joined")               # vd 'khong_gian' (co the hop le)
    units = re.findall(r"\b\d+[a-zA-Z]+\b", orig.lower())   # 600k, 4gb, 5000mah
    if units and any(u not in {t.lower() for t in toks} for u in units):
        flags.append("unit_token_broken")
    if any(t.lower() in TEENCODE_LEFT for t in toks):
        flags.append("teencode_left")
    return flags


def _meta_value(v) -> str:
    return v if isinstance(v, (str, int, float)) else json.dumps(v, ensure_ascii=False)


# ----------------------------------------------------------------------------
# Tach tu ca file jsonl
# ----------------------------------------------------------------------------
def segment_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    text_key: str = "text",
    out_key: str = "text_segmented",
    meta_keys: tuple[str, ...] = ("id", "label", "labels"),
    audit_rows: list[dict] | None = None,
) -> int:
    """Tach tu toan bo file jsonl, tra ve so dong da xu ly.

    - Luon tach lai tu text_key (khong dung ket qua cu, tranh du lieu 'cu').
    - Giu nguyen text goc, them truong out_key.
    - Neu audit_rows != None: them 1 dict phan tich cho moi dong vao danh sach do.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(input_path, encoding="utf-8") as f_in, \
            open(output_path, "w", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if text_key not in rec:
                raise KeyError(f"{input_path}:{line_no} thieu truong '{text_key}'")

            orig = "" if rec[text_key] is None else str(rec[text_key])
            seg, ok = segment_with_status(orig)
            rec[out_key] = seg
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

            if audit_rows is not None:
                toks = seg.split()
                row = {"line": line_no}
                for k in meta_keys:
                    if k in rec:
                        row[k] = _meta_value(rec[k])
                row.update({
                    "text": orig,
                    "text_segmented": seg,
                    "n_tokens": len(toks),
                    "n_compounds": sum(1 for t in toks if _is_compound(t)),
                    "flags": ";".join(analyze(orig, seg, ok)),
                })
                audit_rows.append(row)
    print(f"Da tach tu {n} dong: {input_path} -> {output_path}")
    return n


# ----------------------------------------------------------------------------
# Xuat file ra soat
# ----------------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    # utf-8-sig de Excel mo dung tieng Viet
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_audit(
    split: str, rows: list[dict], audit_dir: Path,
    sample_size: int = 200, max_flagged: int = 300, seed: int = 42,
) -> dict:
    """Xuat 3 file cho 1 split: full, mau ra soat, tu dien tu ghep. Tra ve thong ke."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in rows[0].keys()] if rows else ["line", "text", "text_segmented", "flags"]

    # 1) Toan bo dong (loc theo cot flags trong Excel)
    _write_csv(audit_dir / f"{split}_full.csv", rows, cols)

    # 2) Mau ra soat: dong co co nghiem trong + dong ngau nhien
    rng = random.Random(seed)
    flagged = [r for r in rows if SERIOUS_FLAGS & set(filter(None, r["flags"].split(";")))]
    rest = [r for r in rows if r not in flagged]
    rng.shuffle(flagged)
    rng.shuffle(rest)
    picked = [dict(r, sample_type="flagged") for r in flagged[:max_flagged]] + \
             [dict(r, sample_type="random") for r in rest[:sample_size]]
    for r in picked:
        r["reviewer_ok"] = ""      # dien Y / N khi ra soat
        r["reviewer_note"] = ""
    _write_csv(audit_dir / f"{split}_review_sample.csv", picked,
               cols + ["sample_type", "reviewer_ok", "reviewer_note"])

    # 3) Tu dien tu ghep: quet nhanh de bat tu ghep sai
    freq, example = Counter(), {}
    for r in rows:
        for t in r["text_segmented"].split():
            if _is_compound(t):
                key = t.strip("_").lower()
                freq[key] += 1
                example.setdefault(key, r["text_segmented"][:120])
    vocab = []
    for comp, c in freq.most_common():
        parts = comp.split("_")
        notes = []
        if len(parts) >= 4:
            notes.append("long")
        if parts[0] in NEG_WORDS:
            notes.append("neg_joined")
        if any(ch.isdigit() for ch in comp):
            notes.append("has_digit")
        vocab.append({"compound": comp, "syllables": len(parts), "freq": c,
                      "note": ";".join(notes), "example": example[comp],
                      "reviewer_ok": "", "reviewer_note": ""})
    _write_csv(audit_dir / f"{split}_compound_vocab.csv", vocab,
               ["compound", "syllables", "freq", "note", "example",
                "reviewer_ok", "reviewer_note"])

    flag_counts = Counter(f for r in rows for f in r["flags"].split(";") if f)
    return {
        "split": split, "n": len(rows), "n_flagged_serious": len(flagged),
        "n_unique_compounds": len(vocab), "flag_counts": dict(flag_counts),
        "top_compounds": [(v["compound"], v["freq"]) for v in vocab[:15]],
    }


def write_summary(stats: list[dict], audit_dir: Path) -> None:
    lines = ["# Bao cao ra soat tach tu", ""]
    for s in stats:
        lines += [f"## Split: {s['split']}", "",
                  f"- So dong: {s['n']}",
                  f"- So dong co co nghiem trong: {s['n_flagged_serious']}",
                  f"- So tu ghep khac nhau: {s['n_unique_compounds']}", "",
                  "| Co (flag) | So dong |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in sorted(s["flag_counts"].items(),
                                                        key=lambda x: -x[1])] or ["| (khong co) | 0 |"]
        lines += ["", "Top tu ghep: " + ", ".join(f"{c} ({n})" for c, n in s["top_compounds"]), ""]
    lines += ["## Y nghia cac co", "",
              "- seg_failed: pyvi loi, dong nay giu nguyen text chua tach",
              "- content_changed: noi dung bi thay doi sau khi tach (khong chi them/bot khoang trang va '_')",
              "- underscore_in_input: text dau vao da co '_', de lan voi ky hieu noi am tiet",
              "- neg_marker_in_input: text dau vao con tien to 'n_' (chi nen dung cho TF-IDF, khong dua vao PhoBERT)",
              "- long_compound: tu ghep tu 4 am tiet tro len, nen xem co dung khong",
              "- neg_joined: tu phu dinh bi noi voi tu sau (vd khong_gian: co the hop le)",
              "- unit_token_broken: don vi/gia tri nhu 600k, 4gb bi tach sai",
              "- teencode_left: con teencode sot lai sau tien xu ly"]
    (audit_dir / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Tach tu cac file jsonl theo split")
    ap.add_argument("--in-dir", default="data/processed")
    ap.add_argument("--out-dir", default="data/processed_segmented")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--out-key", default="text_segmented")
    ap.add_argument("--meta-keys", nargs="*", default=["id", "label", "labels"],
                    help="Cac truong di kem de hien trong file ra soat")
    ap.add_argument("--audit-dir", default=None,
                    help="Neu dat: xuat file ra soat vao thu muc nay")
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--max-flagged", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    audit_dir = Path(args.audit_dir) if args.audit_dir else None
    all_stats = []
    for split in args.splits:
        src, dst = in_dir / f"{split}.jsonl", out_dir / f"{split}.jsonl"
        if not src.exists():
            print(f"Bo qua: khong thay {src}")
            continue
        rows: list[dict] | None = [] if audit_dir else None
        segment_jsonl(src, dst, text_key=args.text_key, out_key=args.out_key,
                      meta_keys=tuple(args.meta_keys), audit_rows=rows)
        if audit_dir and rows:
            all_stats.append(write_audit(split, rows, audit_dir,
                                         args.sample_size, args.max_flagged, args.seed))
    if audit_dir and all_stats:
        write_summary(all_stats, audit_dir)
        print(f"Da xuat file ra soat vao {audit_dir}/")


if __name__ == "__main__":
    main()