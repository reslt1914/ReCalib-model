import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def choose_condition_col(df):
    candidates = ["condition", "cov_drug_dose_name", "cov_drug_name", "Unnamed: 0", "index"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def filter_model(df, model_name, prefer_recalib):
    if "model" not in df.columns:
        return df
    models = df["model"].astype(str).unique().tolist()
    if model_name:
        hit = df[df["model"].astype(str) == model_name].copy()
        if len(hit) > 0:
            return hit
        raise RuntimeError(f"model not found: {model_name}; available models: {models}")
    if prefer_recalib:
        mask = df["model"].astype(str).str.contains("ReCalib|DEAwareRC|alpha", case=False, regex=True)
        hit = df[mask].copy()
        if len(hit) > 0:
            return hit
        non_crisp = df[df["model"].astype(str).str.lower() != "crisp"].copy()
        if len(non_crisp) > 0:
            return non_crisp
    else:
        hit = df[df["model"].astype(str).str.lower() == "crisp"].copy()
        if len(hit) > 0:
            return hit
    return df

def read_metrics(path, model_name=None, prefer_recalib=False):
    df = pd.read_csv(path)
    df = filter_model(df, model_name, prefer_recalib)
    cond_col = choose_condition_col(df)
    if cond_col is None:
        df["condition_auto"] = np.arange(len(df)).astype(str)
        cond_col = "condition_auto"
    df[cond_col] = df[cond_col].astype(str)
    df = df.drop_duplicates(subset=[cond_col], keep="last").copy()
    df = df.set_index(cond_col)
    return df

def bootstrap_ci(x, n_boot, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    vals = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        vals[i] = np.mean(x[rng.integers(0, n, n)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def paired_wilcoxon(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x != 0]
    if len(x) == 0:
        return np.nan, np.nan
    p_two = float(wilcoxon(x, alternative="two-sided", zero_method="wilcox").pvalue)
    p_greater = float(wilcoxon(x, alternative="greater", zero_method="wilcox").pvalue)
    return p_two, p_greater

def fmt_p(p):
    if pd.isna(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--recalib", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label", default="main_ood")
    ap.add_argument("--baseline_model", default=None)
    ap.add_argument("--recalib_model", default=None)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = read_metrics(args.baseline, args.baseline_model, False)
    rc = read_metrics(args.recalib, args.recalib_model, True)

    common = base.index.intersection(rc.index)
    base = base.loc[common].copy()
    rc = rc.loc[common].copy()

    print(f"baseline rows after filtering: {len(base)}")
    print(f"recalib rows after filtering: {len(rc)}")
    print(f"paired common conditions: {len(common)}")

    metrics = ["mse_de", "pearson_de", "pearson_delta_de", "r2score_de", "mse", "pearson", "pearson_delta", "r2score"]
    lower_better = {"mse", "mse_de"}

    rows = []
    paired_rows = []

    for m in metrics:
        if m not in base.columns or m not in rc.columns:
            continue

        tmp = pd.DataFrame({
            "baseline": pd.to_numeric(base[m], errors="coerce"),
            "recalib": pd.to_numeric(rc[m], errors="coerce")
        }).dropna()

        b = tmp["baseline"].to_numpy(dtype=float)
        r = tmp["recalib"].to_numpy(dtype=float)

        if m in lower_better:
            improvement = b - r
            improved = r < b
            direction = "lower_is_better"
        else:
            improvement = r - b
            improved = r > b
            direction = "higher_is_better"

        ci_low, ci_high = bootstrap_ci(improvement, args.n_boot, args.seed)
        p_two, p_greater = paired_wilcoxon(improvement)

        rows.append({
            "label": args.label,
            "metric": m,
            "direction": direction,
            "n_conditions": int(len(tmp)),
            "baseline_mean": float(np.mean(b)),
            "recalib_mean": float(np.mean(r)),
            "mean_improvement_positive_is_better": float(np.mean(improvement)),
            "bootstrap95_ci_low": ci_low,
            "bootstrap95_ci_high": ci_high,
            "improved_fraction": float(np.mean(improved)),
            "wilcoxon_p_two_sided": p_two,
            "wilcoxon_p_greater": p_greater
        })

        paired_rows.append(pd.DataFrame({
            "condition": tmp.index.astype(str),
            "metric": m,
            "baseline": b,
            "recalib": r,
            "improvement_positive_is_better": improvement,
            "improved": improved
        }))

    summary = pd.DataFrame(rows)
    paired = pd.concat(paired_rows, ignore_index=True)

    summary.to_csv(out_dir / f"{args.label}_bootstrap_wilcoxon_summary.csv", index=False)
    paired.to_csv(out_dir / f"{args.label}_paired_condition_values.csv", index=False)

    md_rows = []
    md_rows.append("| Metric | Baseline mean | ReCalib mean | Mean improvement | 95% CI | Improved fraction | Wilcoxon p(two-sided) | Wilcoxon p(greater) |")
    md_rows.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        md_rows.append(
            f"| {row['metric']} | "
            f"{row['baseline_mean']:.6f} | "
            f"{row['recalib_mean']:.6f} | "
            f"{row['mean_improvement_positive_is_better']:.6f} | "
            f"[{row['bootstrap95_ci_low']:.6f}, {row['bootstrap95_ci_high']:.6f}] | "
            f"{row['improved_fraction']:.3f} | "
            f"{fmt_p(row['wilcoxon_p_two_sided'])} | "
            f"{fmt_p(row['wilcoxon_p_greater'])} |"
        )

    md = "\n".join(md_rows)
    (out_dir / f"{args.label}_bootstrap_wilcoxon_summary.md").write_text(md)

    print(md)
    print(out_dir / f"{args.label}_bootstrap_wilcoxon_summary.csv")
    print(out_dir / f"{args.label}_paired_condition_values.csv")
    print(out_dir / f"{args.label}_bootstrap_wilcoxon_summary.md")

if __name__ == "__main__":
    main()
