"""So sánh Baseline V2C vs PhoBERT multitask (3 seeds).

Chạy:
    python -m experiments.compare_models

Output:
    - Bảng so sánh trên terminal
    - outputs/metrics/comparison.csv
    - outputs/metrics/comparison.md
    - outputs/metrics/comparison.png (nếu có matplotlib)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import load_split
from src.data.schema import ASPECTS, POL2ID
from src.evaluate import evaluate


def load_preds(path: str | Path) -> list[dict]:
    """Đọc file prediction jsonl."""
    preds = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            preds.append(json.loads(line))
    return preds


def preds_to_matrices(
    preds: list[dict], n_aspects: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Chuyển prediction list -> ma trận presence + sentiment."""
    n = len(preds)
    pres = np.zeros((n, n_aspects), dtype=int)
    sent = np.zeros((n, n_aspects), dtype=int)
    for i, r in enumerate(preds):
        for d in r["pred"]:
            j = ASPECTS.index(d["aspect"])
            pres[i, j] = 1
            sent[i, j] = POL2ID[d["sentiment"]]
    return pres, sent


def evaluate_file(path: str | Path, dev_rows: list[dict]) -> dict:
    """Chấm điểm 1 file prediction."""
    preds = load_preds(path)
    pres, sent = preds_to_matrices(preds)
    return evaluate(dev_rows, pres, sent)


def aggregate_seeds(metrics_list: list[dict]) -> dict:
    """Tính mean ± std qua nhiều seed."""
    keys = [
        "joint_f1",
        "aspect_micro_f1",
        "aspect_macro_f1",
        "sentiment_macro_f1_given_gold_aspect",
        "gold_empty_predicted_empty_rate",
    ]
    agg = {}
    for k in keys:
        vals = [m.get(k) for m in metrics_list if m.get(k) is not None]
        if vals:
            agg[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": vals,
            }
    return agg


def main():
    dev_rows = load_split(Path("data/raw"), "dev")
    pred_dir = Path("outputs/predictions")

    # === Baseline ===
    baseline_path = pred_dir / "baseline_dev_v2c.jsonl"
    if not baseline_path.exists():
        print(f"❌ Không thấy baseline: {baseline_path}")
        print("   Chạy trước: python -m src.run_baseline")
        return

    baseline_metrics = evaluate_file(baseline_path, dev_rows)

    # === PhoBERT 3 seeds ===
    phobert_metrics_list = []
    phobert_seeds_found = []
    for seed in [42, 123, 456]:
        path = pred_dir / f"multitask_seed{seed}_dev.jsonl"
        if path.exists():
            m = evaluate_file(path, dev_rows)
            phobert_metrics_list.append(m)
            phobert_seeds_found.append(seed)
            print(f"✅ seed {seed}: joint={m['joint_f1']:.4f}")
        else:
            print(f"⚠️ Chưa có: {path}")

    if not phobert_metrics_list:
        print("\n❌ Chưa có prediction PhoBERT nào. Chạy trước:")
        print("   python -m src.train --config configs/multitask.yaml --seed 42")
        print(
            "   python -m src.predict --config configs/multitask.yaml --seed 42 --split dev"
        )
        return

    phobert_agg = aggregate_seeds(phobert_metrics_list)

    # === Bảng so sánh ===
    rows = []

    # Baseline
    rows.append(
        {
            "Model": "Baseline V2C",
            "Joint F1": f"{baseline_metrics['joint_f1']:.4f}",
            "Aspect Micro": f"{baseline_metrics['aspect_micro_f1']:.4f}",
            "Aspect Macro": f"{baseline_metrics['aspect_macro_f1']:.4f}",
            "Sentiment (gold)": f"{baseline_metrics['sentiment_macro_f1_given_gold_aspect']:.4f}",
            "Empty Acc": f"{baseline_metrics['gold_empty_predicted_empty_rate']:.4f}",
        }
    )

    # PhoBERT mean ± std
    n_seeds = len(phobert_seeds_found)
    rows.append(
        {
            "Model": f"PhoBERT (n={n_seeds} seeds)",
            "Joint F1": f"{phobert_agg['joint_f1']['mean']:.4f} ± {phobert_agg['joint_f1']['std']:.4f}",
            "Aspect Micro": f"{phobert_agg['aspect_micro_f1']['mean']:.4f} ± {phobert_agg['aspect_micro_f1']['std']:.4f}",
            "Aspect Macro": f"{phobert_agg['aspect_macro_f1']['mean']:.4f} ± {phobert_agg['aspect_macro_f1']['std']:.4f}",
            "Sentiment (gold)": f"{phobert_agg['sentiment_macro_f1_given_gold_aspect']['mean']:.4f} ± {phobert_agg['sentiment_macro_f1_given_gold_aspect']['std']:.4f}",
            "Empty Acc": f"{phobert_agg['gold_empty_predicted_empty_rate']['mean']:.4f} ± {phobert_agg['gold_empty_predicted_empty_rate']['std']:.4f}",
        }
    )

    df = pd.DataFrame(rows)

    print("\n" + "=" * 90)
    print("BẢNG SO SÁNH BASELINE VS PHOBERT")
    print("=" * 90)
    print(df.to_string(index=False))

    # === Delta ===
    print("\n" + "=" * 90)
    print("DELTA (PhoBERT - Baseline)")
    print("=" * 90)
    deltas = {
        "Joint F1": phobert_agg["joint_f1"]["mean"] - baseline_metrics["joint_f1"],
        "Aspect Micro": phobert_agg["aspect_micro_f1"]["mean"]
        - baseline_metrics["aspect_micro_f1"],
        "Aspect Macro": phobert_agg["aspect_macro_f1"]["mean"]
        - baseline_metrics["aspect_macro_f1"],
        "Sentiment (gold)": phobert_agg["sentiment_macro_f1_given_gold_aspect"]["mean"]
        - baseline_metrics["sentiment_macro_f1_given_gold_aspect"],
        "Empty Acc": phobert_agg["gold_empty_predicted_empty_rate"]["mean"]
        - baseline_metrics["gold_empty_predicted_empty_rate"],
    }
    for k, v in deltas.items():
        sign = "+" if v > 0 else ""
        arrow = "🟢" if v > 0.005 else ("🔴" if v < -0.005 else "🟡")
        print(f"  {k:20s} {sign}{v:+.4f}  {arrow}")

    # === Lưu CSV ===
    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ Đã lưu: {out_dir / 'comparison.csv'}")

    # === Lưu Markdown ===
    with open(out_dir / "comparison.md", "w", encoding="utf-8") as f:
        f.write("# So sánh Baseline V2C vs PhoBERT\n\n")
        f.write(f"*Số seed PhoBERT: {n_seeds} (seeds: {phobert_seeds_found})*\n\n")
        try:
            f.write(df.to_markdown(index=False))
        except Exception:
            cols = [str(c) for c in df.columns]
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
            for _, r in df.iterrows():
                f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
        f.write("\n\n## Delta (PhoBERT - Baseline)\n\n")
        for k, v in deltas.items():
            sign = "+" if v > 0 else ""
            f.write(f"- **{k}**: {sign}{v:.4f}\n")
    print(f"✅ Đã lưu: {out_dir / 'comparison.md'}")

    # === Vẽ biểu đồ ===
    try:
        import matplotlib.pyplot as plt

        metrics_to_plot = [
            "joint_f1",
            "aspect_macro_f1",
            "sentiment_macro_f1_given_gold_aspect",
            "gold_empty_predicted_empty_rate",
        ]
        labels = ["Joint F1", "Aspect Macro-F1", "Sentiment (gold)", "Empty Accuracy"]

        baseline_vals = [baseline_metrics[m] for m in metrics_to_plot]
        phobert_vals = [phobert_agg[m]["mean"] for m in metrics_to_plot]
        phobert_stds = [phobert_agg[m]["std"] for m in metrics_to_plot]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(11, 6))
        bars1 = ax.bar(
            x - width / 2, baseline_vals, width, label="Baseline V2C", color="#4C72B0"
        )
        bars2 = ax.bar(
            x + width / 2,
            phobert_vals,
            width,
            yerr=phobert_stds,
            label=f"PhoBERT (n={n_seeds})",
            color="#DD8452",
            capsize=5,
        )

        ax.set_ylabel("Score")
        ax.set_title("So sánh Baseline V2C vs PhoBERT (Dev)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()
        plt.savefig(out_dir / "comparison.png", dpi=150)
        print(f"✅ Đã lưu: {out_dir / 'comparison.png'}")
    except ImportError:
        print("⚠️ Chưa cài matplotlib — bỏ qua biểu đồ")


if __name__ == "__main__":
    main()
