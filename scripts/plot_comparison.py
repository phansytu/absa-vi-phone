"""Create the final Test comparison chart from outputs/metrics/comparison.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "metrics" / "comparison.csv"
OUTPUT = ROOT / "outputs" / "metrics" / "comparison.png"


def main() -> None:
    df = pd.read_csv(SOURCE)
    metrics = [
        ("aspect_micro_f1", "Aspect micro F1"),
        ("aspect_macro_f1", "Aspect macro F1"),
        ("sentiment_macro_f1_given_gold", "Sentiment macro F1"),
        ("joint_f1", "Joint F1"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4), sharey=True)
    colors = ["#94A3B8", "#1D4ED8"]
    for ax, (column, title) in zip(axes, metrics):
        values = df[column].to_numpy()
        bars = ax.bar([0, 1], values, color=colors, width=0.62)
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xticks([0, 1], ["TF-IDF", "PhoBERT"], rotation=15)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    axes[0].set_ylabel("F1 score")
    fig.suptitle("Kết quả trên Test (n = 2.224)", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"Đã lưu biểu đồ: {OUTPUT}")


if __name__ == "__main__":
    main()
