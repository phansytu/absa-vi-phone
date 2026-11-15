"""Doc va kiem tra cau hinh YAML.

Bao loi RO RANG thay vi im lang khi:
  * khoa YAML bi khai bao 2 lan (PyYAML mac dinh lay gia tri cuoi, rat de sai),
  * khoa la / go sai ten (vd "lr_encoer"),
  * gia tri sai kieu (vd "3e-5" bi PyYAML doc thanh chuoi).
"""
from __future__ import annotations

import difflib

import yaml

DEFAULTS: dict = {
    # --- mo hinh / du lieu
    "encoder": "vinai/phobert-base",
    "max_length": 128,
    "pooling": "mean",              # mean | cls | aspect_attn
    "dropout": 0.1,
    "use_segmented": True,          # PhoBERT: true (can tach tu). ViSoBERT: false (nhan van ban tho)
    "norm": {"tone": False, "teencode": True, "repeat": True},   # {} = tat chuan hoa
    "data_dir": "data/processed_final",
    # --- toi uu
    "batch_size": 32,
    "epochs": 8,
    "lr_encoder": 2.0e-5,
    "lr_head": 5.0e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lambda": 1.0,                  # trong so loss sentiment
    # --- mat can bang lop
    "pos_weight_power": 0.5,        # pos_weight = clip(((1-p)/p)**power, 1, pos_weight_max); 0 = tat
    "pos_weight_max": 20.0,
    "sent_weight_power": 0.0,       # trong so lop cho sentiment; 0 = tat
    "oversample": {},               # vd {STORAGE: 4}: mau co STORAGE duoc lay nhieu gap 4 lan
    # --- chon checkpoint
    "select_metric": "macro_joint", # joint | macro | macro_joint
    "patience": 3,
    "monitor_aspects": ["STORAGE"],
}
POOLINGS = ("mean", "cls", "aspect_attn")
SELECT_METRICS = ("joint", "macro", "macro_joint")
NORM_KEYS = ("tone", "teencode", "repeat")
ASPECTS = ["SCREEN", "CAMERA", "FEATURES", "BATTERY", "PERFORMANCE",
           "STORAGE", "DESIGN", "PRICE", "GENERAL", "SER&ACC"]


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader tu choi khoa trung nhau."""


def _no_duplicates(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ValueError(f"Khoa YAML bi khai bao 2 lan: {key!r} (dong {key_node.start_mark.line + 1}). "
                             f"Moi khoa chi duoc xuat hien mot lan.")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def _coerce(key, value):
    """Ep kieu theo DEFAULTS (vd '3e-5' -> 3e-05)."""
    want = type(DEFAULTS[key])
    if want is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{key} phai la true/false, nhan {value!r}")
        return value
    if want in (int, float):
        if isinstance(value, bool):
            raise ValueError(f"{key} phai la so, nhan {value!r}")
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} phai la so, nhan {value!r}") from None
        return int(num) if want is int and num == int(num) else (num if want is float else num)
    return value


def validate(cfg: dict) -> dict:
    unknown = [k for k in cfg if k not in DEFAULTS]
    if unknown:
        hints = {k: difflib.get_close_matches(k, DEFAULTS, n=1) for k in unknown}
        raise ValueError("Khoa cau hinh khong hop le: " +
                         ", ".join(f"{k!r}" + (f" (co phai {h[0]!r}?)" if h else "") for k, h in hints.items()))
    out = {k: _coerce(k, v) for k, v in cfg.items()}
    if out["pooling"] not in POOLINGS:
        raise ValueError(f"pooling phai thuoc {POOLINGS}, nhan {out['pooling']!r}")
    if out["select_metric"] not in SELECT_METRICS:
        raise ValueError(f"select_metric phai thuoc {SELECT_METRICS}, nhan {out['select_metric']!r}")
    norm = out["norm"] or {}
    if not isinstance(norm, dict) or set(norm) - set(NORM_KEYS):
        raise ValueError(f"norm phai la dict voi khoa trong {NORM_KEYS}, nhan {norm!r}")
    out["norm"] = norm
    over = out["oversample"] or {}
    for a, f in over.items():
        if a not in ASPECTS or float(f) < 1:
            raise ValueError(f"oversample: aspect/he so khong hop le {a!r}: {f!r} (he so phai >= 1)")
    out["oversample"] = {a: float(f) for a, f in over.items()}
    bad = [a for a in out["monitor_aspects"] if a not in ASPECTS]
    if bad:
        raise ValueError(f"monitor_aspects chua aspect khong hop le: {bad}")
    if out["max_length"] > 256 and "phobert" in str(out["encoder"]).lower():
        raise ValueError("PhoBERT chi ho tro max_length <= 256")
    return out


def load_config(path, overrides=()) -> dict:
    """Doc YAML (bao loi khi trung khoa), gop voi DEFAULTS, ap --set k=v, kiem tra."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.load(f, Loader=_StrictLoader) or {}
    cfg = {**DEFAULTS, **raw}
    for s in overrides:
        if "=" not in s:
            raise ValueError(f"--set can dang key=value, nhan {s!r}")
        k, v = s.split("=", 1)
        cfg[k.strip()] = yaml.safe_load(v)
    return validate(cfg)


def from_checkpoint_cfg(saved: dict) -> dict:
    """Cau hinh luu trong checkpoint -> day du khoa (checkpoint cu thieu khoa moi).
    Khoa la bi bo qua (khong bao loi) de doc duoc checkpoint cu."""
    cfg = {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}
    return validate(cfg)
