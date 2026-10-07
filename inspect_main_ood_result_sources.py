from pathlib import Path
import pandas as pd
import numpy as np

target = {
    "mse_de": 0.019786,
    "pearson_de": 0.846972,
    "pearson_delta_de": 0.340919,
    "r2score_de": 0.516148,
}

files = sorted(Path("crisp_outputs").glob("mydata_rc_full_seed*/ood/metrics_by_condition.csv"))

rows = []
for p in files:
    df = pd.read_csv(p)

    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.strip()
        rc = df[df["model"].str.contains("DEAwareRC|ReCalib|alpha", case=False, regex=True)].copy()
        if len(rc) == 0:
            rc = df.copy()
    else:
        rc = df.copy()

    item = {"path": str(p), "n_rows": len(df), "n_eval_rows": len(rc)}
    score = 0.0
    for m, v in target.items():
        if m in rc.columns:
            val = pd.to_numeric(rc[m], errors="coerce").mean()
            item[m] = val
            score += abs(val - v)
        else:
            item[m] = np.nan
            score += 999
    item["distance_to_old_main_result"] = score
    rows.append(item)

out = pd.DataFrame(rows).sort_values("distance_to_old_main_result")
out.to_csv("crisp_outputs/main_ood_result_source_inventory.csv", index=False)
print(out.to_string(index=False))
print()
print("saved: crisp_outputs/main_ood_result_source_inventory.csv")
