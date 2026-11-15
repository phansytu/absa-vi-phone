import pandas as pd

base = pd.read_csv("outputs/analysis_baseline/metrics/per_label_stats.csv")
pho = pd.read_csv("outputs/analysis_phobert/metrics/per_label_stats.csv")

merged = base[["aspect", "FP", "FN", "f1"]].merge(
    pho[["aspect", "FP", "FN", "f1"]], on="aspect", suffixes=("_base", "_pho")
)
merged["f1_delta"] = merged["f1_pho"] - merged["f1_base"]
merged["FP_delta"] = merged["FP_pho"] - merged["FP_base"]
print(merged.sort_values("f1_delta").to_string(index=False))
