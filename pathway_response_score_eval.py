import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

try:
    import scanpy as sc
except Exception:
    sc = None


def read_matrix(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def parse_gmt(gmt_path):
    pathways = {}
    with open(gmt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = [g.strip() for g in parts[2:] if g.strip()]
            pathways[name] = genes
    return pathways


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(pearsonr(x, y)[0])


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(spearmanr(x, y).correlation)


def mse(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return np.nan
    return float(np.mean((x[ok] - y[ok]) ** 2))


def bootstrap_ci(x, n_boot=10000, seed=2024):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = np.mean(x[rng.integers(0, n, size=n)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def safe_wilcoxon_greater(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x != 0]
    if len(x) == 0:
        return np.nan
    try:
        return float(wilcoxon(x, alternative="greater", zero_method="wilcox").pvalue)
    except Exception:
        return np.nan


def build_gene_alias(matrix_genes, adata_path=None):
    """
    Build mapping from gene symbol / alias to matrix column name.
    If matrix columns are already symbols, this is enough.
    If h5ad var aligns with matrix genes, also map var columns.
    """
    matrix_genes = [str(g) for g in matrix_genes]
    alias = {g: g for g in matrix_genes}

    if adata_path and sc is not None and Path(adata_path).exists():
        try:
            adata = sc.read_h5ad(adata_path, backed="r")
            if adata.n_vars == len(matrix_genes):
                for i, g in enumerate(adata.var_names.astype(str).tolist()):
                    alias[str(g)] = matrix_genes[i]
                for col in adata.var.columns:
                    vals = adata.var[col].astype(str).tolist()
                    for i, g in enumerate(vals):
                        if g and g != "nan":
                            alias[str(g)] = matrix_genes[i]
        except Exception as e:
            print(f"[WARN] Could not use h5ad gene aliases: {e}")

    return alias


def map_pathways_to_columns(pathways, alias, min_genes=5):
    mapped = {}
    rows = []
    for pname, genes in pathways.items():
        cols = []
        for g in genes:
            if g in alias:
                cols.append(alias[g])
        cols = sorted(set(cols))
        rows.append({
            "pathway": pname,
            "n_genes_original": len(set(genes)),
            "n_genes_mapped": len(cols),
        })
        if len(cols) >= min_genes:
            mapped[pname] = cols
    return mapped, pd.DataFrame(rows)


def compute_pathway_scores(delta_df, mapped_pathways):
    """
    pathway_score = mean signed delta of mapped pathway genes.
    Rows: conditions, columns: pathways.
    """
    out = {}
    for pname, cols in mapped_pathways.items():
        cols_exist = [c for c in cols if c in delta_df.columns]
        if len(cols_exist) == 0:
            continue
        out[pname] = delta_df[cols_exist].mean(axis=1)
    score_df = pd.DataFrame(out, index=delta_df.index)
    return score_df


def condition_metrics(true_scores, pred_scores, model_name):
    common_conds = true_scores.index.intersection(pred_scores.index)
    common_paths = true_scores.columns.intersection(pred_scores.columns)

    true_scores = true_scores.loc[common_conds, common_paths]
    pred_scores = pred_scores.loc[common_conds, common_paths]

    rows = []
    for cond in common_conds:
        t = true_scores.loc[cond].to_numpy(dtype=float)
        p = pred_scores.loc[cond].to_numpy(dtype=float)
        rows.append({
            "model": model_name,
            "condition": str(cond),
            "n_pathways": len(common_paths),
            "pathway_score_pearson": safe_pearson(t, p),
            "pathway_score_spearman": safe_spearman(t, p),
            "pathway_score_mse": mse(t, p),
        })
    return pd.DataFrame(rows)


def paired_stats(frozen_rows, recalib_rows, n_boot=10000, seed=2024):
    f = frozen_rows.set_index("condition")
    r = recalib_rows.set_index("condition")
    common = f.index.intersection(r.index)

    metrics = [
        ("pathway_score_pearson", "higher"),
        ("pathway_score_spearman", "higher"),
        ("pathway_score_mse", "lower"),
    ]

    out = []
    for m, direction in metrics:
        tmp = pd.DataFrame({
            "Frozen predictor": f.loc[common, m].astype(float),
            "ReCalib": r.loc[common, m].astype(float),
        }).dropna()

        if len(tmp) == 0:
            continue

        if direction == "higher":
            imp = tmp["ReCalib"].to_numpy() - tmp["Frozen predictor"].to_numpy()
        else:
            imp = tmp["Frozen predictor"].to_numpy() - tmp["ReCalib"].to_numpy()

        ci_low, ci_high = bootstrap_ci(imp, n_boot=n_boot, seed=seed)
        p = safe_wilcoxon_greater(imp)

        out.append({
            "metric": m,
            "direction": "higher is better" if direction == "higher" else "lower is better",
            "Frozen predictor": float(tmp["Frozen predictor"].mean()),
            "ReCalib": float(tmp["ReCalib"].mean()),
            "Mean improvement": float(np.mean(imp)),
            "95% CI low": ci_low,
            "95% CI high": ci_high,
            "Improved fraction": float(np.mean(imp > 0)),
            "Wilcoxon p-value": p,
            "n_conditions": int(len(tmp)),
        })

    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", default="crisp_outputs/mydata_baseline_full/ood")
    ap.add_argument("--recalib_dir", default="crisp_outputs/mydata_rc_full_seed2024_alpha04_w0/ood")
    ap.add_argument("--gmt", required=True)
    ap.add_argument("--adata", default="data/mydata/sciplex_complete_v2.h5ad")
    ap.add_argument("--out_dir", default="outputs/pathway_response_score_eval_seed2024")
    ap.add_argument("--min_genes", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = Path(args.baseline_dir)
    recalib_dir = Path(args.recalib_dir)

    true_df = read_matrix(baseline_dir / "x_true_condition_mean.csv")
    ctrl_df = read_matrix(baseline_dir / "x_ctrl_condition_mean.csv")
    frozen_df = read_matrix(baseline_dir / "x_crisp_condition_mean.csv")
    recalib_df = read_matrix(recalib_dir / "x_calibrated_condition_mean.csv")

    common_conds = true_df.index.intersection(ctrl_df.index).intersection(frozen_df.index).intersection(recalib_df.index)
    common_genes = true_df.columns.intersection(ctrl_df.columns).intersection(frozen_df.columns).intersection(recalib_df.columns)

    true_df = true_df.loc[common_conds, common_genes]
    ctrl_df = ctrl_df.loc[common_conds, common_genes]
    frozen_df = frozen_df.loc[common_conds, common_genes]
    recalib_df = recalib_df.loc[common_conds, common_genes]

    print("[INFO] Matrix shape:", true_df.shape)

    true_delta = true_df - ctrl_df
    frozen_delta = frozen_df - ctrl_df
    recalib_delta = recalib_df - ctrl_df

    pathways = parse_gmt(args.gmt)
    print("[INFO] Pathways loaded:", len(pathways))

    alias = build_gene_alias(common_genes.astype(str).tolist(), adata_path=args.adata)
    mapped_pathways, coverage = map_pathways_to_columns(pathways, alias, min_genes=args.min_genes)

    coverage.to_csv(out_dir / "pathway_mapping_coverage.csv", index=False)

    print("[INFO] Pathways kept after mapping:", len(mapped_pathways))
    print("[INFO] Mapped gene count summary:")
    print(coverage["n_genes_mapped"].describe())

    if len(mapped_pathways) == 0:
        raise RuntimeError(
            "No pathways passed min_genes threshold. "
            "Check whether GMT gene symbols match matrix gene names."
        )

    true_scores = compute_pathway_scores(true_delta, mapped_pathways)
    frozen_scores = compute_pathway_scores(frozen_delta, mapped_pathways)
    recalib_scores = compute_pathway_scores(recalib_delta, mapped_pathways)

    true_scores.to_csv(out_dir / "true_pathway_response_scores.csv")
    frozen_scores.to_csv(out_dir / "frozen_pathway_response_scores.csv")
    recalib_scores.to_csv(out_dir / "recalib_pathway_response_scores.csv")

    frozen_rows = condition_metrics(true_scores, frozen_scores, "Frozen predictor")
    recalib_rows = condition_metrics(true_scores, recalib_scores, "ReCalib")

    all_rows = pd.concat([frozen_rows, recalib_rows], ignore_index=True)
    all_rows.to_csv(out_dir / "pathway_response_score_by_condition.csv", index=False)

    stat = paired_stats(frozen_rows, recalib_rows, n_boot=args.n_boot, seed=args.seed)
    stat.to_csv(out_dir / "pathway_response_score_paired_stats.csv", index=False)

    paper = stat.copy()
    paper["95% CI"] = paper.apply(
        lambda x: f"[{x['95% CI low']:.6f}, {x['95% CI high']:.6f}]",
        axis=1
    )
    paper = paper[
        [
            "metric",
            "direction",
            "Frozen predictor",
            "ReCalib",
            "Mean improvement",
            "95% CI",
            "Improved fraction",
            "Wilcoxon p-value",
            "n_conditions",
        ]
    ]
    paper.to_csv(out_dir / "pathway_response_score_paper_table.csv", index=False)

    print("\n" + "=" * 100)
    print("[Paper-ready pathway response-score table]")
    print(paper.to_string(index=False))
    print("=" * 100)
    print("[Saved]")
    print(out_dir / "pathway_mapping_coverage.csv")
    print(out_dir / "pathway_response_score_by_condition.csv")
    print(out_dir / "pathway_response_score_paired_stats.csv")
    print(out_dir / "pathway_response_score_paper_table.csv")


if __name__ == "__main__":
    main()
