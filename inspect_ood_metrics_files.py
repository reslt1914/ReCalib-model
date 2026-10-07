from pathlib import Path
import pandas as pd

root = Path("crisp_outputs")
files = sorted(root.glob("**/ood/metrics_by_condition.csv"))

rows = []
for p in files:
    try:
        df = pd.read_csv(p)
    except Exception as e:
        rows.append({
            "path": str(p),
            "n_rows": "ERR",
            "n_conditions": "ERR",
            "models": str(e),
            "mse_de": "",
            "pearson_de": "",
            "r2score_de": ""
        })
        continue

    cond_col = None
    for c in ["condition", "cov_drug_dose_name", "cov_drug_name", "Unnamed: 0", "index"]:
        if c in df.columns:
            cond_col = c
            break

    n_conditions = df[cond_col].astype(str).nunique() if cond_col else len(df)

    if "model" in df.columns:
        models = ";".join(df["model"].astype(str).unique().tolist())
        show = df[df["model"].astype(str).str.contains("ReCalib|DEAwareRC|alpha", case=False, regex=True)]
        if len(show) == 0:
            show = df[df["model"].astype(str).str.lower() == "crisp"]
        if len(show) == 0:
            show = df
    else:
        models = ""
        show = df

    item = {
        "path": str(p),
        "n_rows": len(df),
        "n_conditions": n_conditions,
        "models": models,
        "mse_de": pd.to_numeric(show["mse_de"], errors="coerce").mean() if "mse_de" in show.columns else "",
        "pearson_de": pd.to_numeric(show["pearson_de"], errors="coerce").mean() if "pearson_de" in show.columns else "",
        "r2score_de": pd.to_numeric(show["r2score_de"], errors="coerce").mean() if "r2score_de" in show.columns else ""
    }
    rows.append(item)

out = pd.DataFrame(rows)
out.to_csv("crisp_outputs/ood_metrics_file_inventory.csv", index=False)

print(out.to_string(index=False))
print()
print("saved: crisp_outputs/ood_metrics_file_inventory.csv")


