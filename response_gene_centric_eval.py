import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr, wilcoxon


def read_matrix(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


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


def bootstrap_ci(x, n_boot=10000, seed=2024):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = np.empty(n_boot, dtype=float)
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


def build_gene_alias(adata, matrix_genes):
    """
    Build aliases from matrix gene names and adata.var columns.
    This improves DE-gene mapping robustness.
    """
    alias = {}
    matrix_genes = [str(g) for g in matrix_genes]

    for i, g in enumerate(matrix_genes):
        alias[g] = i

    if adata.n_vars == len(matrix_genes):
        for i, g in enumerate(adata.var_names.astype(str).tolist()):
            alias[str(g)] = i

        for col in adata.var.columns:
            vals = adata.var[col].astype(str).tolist()
            for i, g in enumerate(vals):
                if g and g != "nan":
                    alias[str(g)] = i

    return alias


def get_de_dict(adata, preferred_key="lincs_DEGs"):
    """
    Get DE gene dictionary from adata.uns.
    """
    if preferred_key in adata.uns:
        return preferred_key, adata.uns[preferred_key]

    candidates = []
    for k in adata.uns.keys():
        lk = str(k).lower()
        if "deg" in lk or "degs" in lk or "de_" in lk or "rank" in lk:
            candidates.append(k)

    if len(candidates) == 1:
        key = candidates[0]
        print(f"[WARN] preferred DE key not found. Using detected key: {key}")
        return key, adata.uns[key]

    print("[ERROR] Cannot find DE-gene dictionary automatically.")
    print("[INFO] Available adata.uns keys:")
    for k in adata.uns.keys():
        print("  ", k)
    raise KeyError(
        f"DE key '{preferred_key}' not found. "
        f"Please specify --de_key using one of the printed keys."
    )


def condition_key_variants(cond):
    cond = str(cond)
    return [
        cond,
        cond.replace("__", "_"),
        cond.replace("_", "__"),
        cond.replace("|", "_"),
        cond.replace("/", "_"),
        cond.replace(" ", "_"),
    ]


def get_de_indices(cond, de_dict, alias):
    """
    Map DE-gene list of a condition to matrix column indices.
    """
    genes = None
    for key in condition_key_variants(cond):
        if key in de_dict:
            genes = list(de_dict[key])
            break

    if genes is None:
        return np.array([], dtype=int)

    idx = []
    for g in genes:
        g = str(g)
        if g in alias:
            idx.append(alias[g])
    return np.array(sorted(set(idx)), dtype=int)


def direction_agreement(true_delta, pred_delta, idx, eps=1e-8):
    """
    Fraction of DE genes with same perturbation direction.
    Only genes with non-negligible true_delta are counted.
    """
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0:
        return np.nan

    t = true_delta[idx]
    p = pred_delta[idx]

    valid = np.isfinite(t) & np.isfinite(p) & (np.abs(t) > eps)
    if valid.sum() == 0:
        return np.nan

    return float(np.mean(np.sign(t[valid]) == np.sign(p[valid])))


def topk_recovery(true_delta, pred_delta, idx, k):
    """
    Top-k response gene recovery.
    Response strength is absolute perturbation change.
    """
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0:
        return np.nan

    k_eff = min(k, len(idx))
    if k_eff <= 0:
        return np.nan

    true_strength = np.abs(true_delta[idx])
    pred_strength = np.abs(pred_delta[idx])

    true_top = set(idx[np.argsort(true_strength)[::-1][:k_eff]].tolist())
    pred_top = set(idx[np.argsort(pred_strength)[::-1][:k_eff]].tolist())

    return float(len(true_top & pred_top) / k_eff)


def rank_concordance(true_delta, pred_delta, idx):
    """
    Spearman rank concordance of response strength |delta|.
    """
    idx = np.asarray(idx, dtype=int)
    if len(idx) < 3:
        return np.nan

    true_strength = np.abs(true_delta[idx])
    pred_strength = np.abs(pred_delta[idx])
    return safe_spearman(true_strength, pred_strength)


def eval_model(true_df, ctrl_df, pred_df, de_idx_by_cond, model_name, topks):
    """
    Evaluate one model condition by condition.
    """
    common_conds = true_df.index.intersection(ctrl_df.index).intersection(pred_df.index)
    common_genes = true_df.columns.intersection(ctrl_df.columns).intersection(pred_df.columns)

    true_df = true_df.loc[common_conds, common_genes]
    ctrl_df = ctrl_df.loc[common_conds, common_genes]
    pred_df = pred_df.loc[common_conds, common_genes]

    panel_idx = np.arange(len(common_genes), dtype=int)
    rows = []

    for cond in common_conds:
        y_true = true_df.loc[cond].to_numpy(dtype=float)
        y_ctrl = ctrl_df.loc[cond].to_numpy(dtype=float)
        y_pred = pred_df.loc[cond].to_numpy(dtype=float)

        true_delta = y_true - y_ctrl
        pred_delta = y_pred - y_ctrl

        de_idx = de_idx_by_cond.get(str(cond), np.array([], dtype=int))

        row = {
            "model": model_name,
            "condition": str(cond),
            "n_de": int(len(de_idx)),
            "de_direction_agreement": direction_agreement(true_delta, pred_delta, de_idx),
            "response_rank_spearman_panel": rank_concordance(true_delta, pred_delta, panel_idx),
            "response_rank_spearman_de": rank_concordance(true_delta, pred_delta, de_idx),
        }

        for k in topks:
            row[f"top{k}_recovery_panel"] = topk_recovery(true_delta, pred_delta, panel_idx, k)
            row[f"top{k}_recovery_de"] = topk_recovery(true_delta, pred_delta, de_idx, k)

        rows.append(row)

    return pd.DataFrame(rows)


def paired_stats(frozen_rows, recalib_rows, metric_cols, n_boot=10000, seed=2024):
    """
    Higher is better for all response-gene-centric metrics.
    Improvement = ReCalib - Frozen predictor.
    """
    f = frozen_rows.set_index("condition")
    r = recalib_rows.set_index("condition")
    common = f.index.intersection(r.index)

    out = []
    for m in metric_cols:
        tmp = pd.DataFrame({
            "Frozen predictor": f.loc[common, m].astype(float),
            "ReCalib": r.loc[common, m].astype(float),
        }).dropna()

        if len(tmp) == 0:
            continue

        imp = tmp["ReCalib"].to_numpy() - tmp["Frozen predictor"].to_numpy()
        ci_low, ci_high = bootstrap_ci(imp, n_boot=n_boot, seed=seed)
        p = safe_wilcoxon_greater(imp)

        out.append({
            "metric": m,
            "n_conditions": int(len(tmp)),
            "Frozen predictor": float(tmp["Frozen predictor"].mean()),
            "ReCalib": float(tmp["ReCalib"].mean()),
            "Mean improvement": float(np.mean(imp)),
            "95% CI low": ci_low,
            "95% CI high": ci_high,
            "Improved fraction": float(np.mean(imp > 0)),
            "Wilcoxon p-value": p,
        })

    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", default="crisp_outputs/mydata_baseline_full/ood")
    ap.add_argument("--recalib_dir", default="crisp_outputs/mydata_rc_full_seed2024_alpha04_w0/ood")
    ap.add_argument("--adata", default="data/mydata/sciplex_complete_v2.h5ad")
    ap.add_argument("--de_key", default="lincs_DEGs")
    ap.add_argument("--out_dir", default="outputs/response_gene_centric_eval_seed2024")
    ap.add_argument("--topks", nargs="+", type=int, default=[10, 20])
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    recalib_dir = Path(args.recalib_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    true_path = baseline_dir / "x_true_condition_mean.csv"
    ctrl_path = baseline_dir / "x_ctrl_condition_mean.csv"
    frozen_path = baseline_dir / "x_crisp_condition_mean.csv"
    recalib_path = recalib_dir / "x_calibrated_condition_mean.csv"

    true_df = read_matrix(true_path)
    ctrl_df = read_matrix(ctrl_path)
    frozen_df = read_matrix(frozen_path)
    recalib_df = read_matrix(recalib_path)

    print("[INFO] Loaded matrices:")
    print("  true   :", true_path, true_df.shape)
    print("  ctrl   :", ctrl_path, ctrl_df.shape)
    print("  frozen :", frozen_path, frozen_df.shape)
    print("  recalib:", recalib_path, recalib_df.shape)

    print("[INFO] Loading h5ad:", args.adata)
    adata = sc.read_h5ad(args.adata, backed="r")
    used_de_key, de_dict = get_de_dict(adata, args.de_key)
    print("[INFO] Using DE key:", used_de_key)

    alias = build_gene_alias(adata, true_df.columns.astype(str).tolist())

    de_idx_by_cond = {}
    coverage = []
    for cond in true_df.index.astype(str):
        idx = get_de_indices(cond, de_dict, alias)
        de_idx_by_cond[cond] = idx
        coverage.append({"condition": cond, "n_de_mapped": int(len(idx))})

    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(out_dir / "de_mapping_coverage.csv", index=False)

    print("[INFO] DE mapping coverage:")
    print(coverage_df["n_de_mapped"].describe())

    if coverage_df["n_de_mapped"].sum() == 0:
        raise RuntimeError(
            "No DE genes were mapped. Please inspect condition names and adata.uns DE key."
        )

    frozen_rows = eval_model(
        true_df=true_df,
        ctrl_df=ctrl_df,
        pred_df=frozen_df,
        de_idx_by_cond=de_idx_by_cond,
        model_name="Frozen predictor",
        topks=args.topks,
    )

    recalib_rows = eval_model(
        true_df=true_df,
        ctrl_df=ctrl_df,
        pred_df=recalib_df,
        de_idx_by_cond=de_idx_by_cond,
        model_name="ReCalib",
        topks=args.topks,
    )

    all_rows = pd.concat([frozen_rows, recalib_rows], ignore_index=True)
    all_rows.to_csv(out_dir / "response_gene_centric_by_condition.csv", index=False)

    metric_cols = [
        "de_direction_agreement",
        "top10_recovery_panel",
        "top20_recovery_panel",
        "response_rank_spearman_panel",
        "response_rank_spearman_de",
    ]

    # Keep only metrics that exist in case topks were changed.
    metric_cols = [m for m in metric_cols if m in all_rows.columns]

    stat = paired_stats(
        frozen_rows,
        recalib_rows,
        metric_cols=metric_cols,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    stat.to_csv(out_dir / "response_gene_centric_paired_stats.csv", index=False)

    paper = stat.copy()
    paper["95% CI"] = paper.apply(
        lambda x: f"[{x['95% CI low']:.6f}, {x['95% CI high']:.6f}]",
        axis=1
    )
    paper = paper[
        [
            "metric",
            "Frozen predictor",
            "ReCalib",
            "Mean improvement",
            "95% CI",
            "Improved fraction",
            "Wilcoxon p-value",
        ]
    ]
    paper.to_csv(out_dir / "response_gene_centric_paper_table.csv", index=False)

    print("\n" + "=" * 100)
    print("[Paper-ready response-gene-centric table]")
    print(paper.to_string(index=False))
    print("=" * 100)
    print("[Saved]")
    print(out_dir / "de_mapping_coverage.csv")
    print(out_dir / "response_gene_centric_by_condition.csv")
    print(out_dir / "response_gene_centric_paired_stats.csv")
    print(out_dir / "response_gene_centric_paper_table.csv")


if __name__ == "__main__":
    main()
