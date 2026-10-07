import pandas as pd
import numpy as np
import glob
from scipy.stats import wilcoxon

print("🔍 loading structured data...")

# =========================
# load all csvs
# =========================
def load_all(files, metric):
    rows = []
    for f in files:
        df = pd.read_csv(f)
        df["source"] = f
        rows.append(df[[metric, "source"]])
    return pd.concat(rows)

base_files = glob.glob("crisp_outputs/mydata_baseline*/**/metrics_summary.csv", recursive=True)
rec_files  = glob.glob("crisp_outputs/mydata_rc_full_seed*/**/metrics_summary.csv", recursive=True)

metrics = ["mse_de", "pearson_de", "pearson_delta_de", "r2score_de"]

results = {}

for m in metrics:

    base_df = load_all(base_files, m)
    rec_df  = load_all(rec_files, m)

    # =========================
    # FIX: match by index alignment
    # =========================
    min_len = min(len(base_df), len(rec_df))

    base_vals = base_df[m].values[:min_len]
    rec_vals  = rec_df[m].values[:min_len]

    diff = rec_vals - base_vals

    results[m] = {
        "baseline_mean": np.mean(base_vals),
        "recalib_mean": np.mean(rec_vals),
        "mean_improve": np.mean(diff),
        "improved_fraction": (diff > 0).mean(),
        "wilcoxon_p": wilcoxon(diff).pvalue if len(diff) > 1 else np.nan
    }

df = pd.DataFrame(results).T
df.to_csv("Table2_final_recovered.csv")

print(df)
print("✅ DONE")