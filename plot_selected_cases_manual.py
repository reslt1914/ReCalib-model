import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASELINE_DIR = Path("crisp_outputs/mydata_baseline_full")
RC_DIRS = [
    Path("crisp_outputs/mydata_rc_full_seed2024_alpha04_w0"),
    Path("crisp_outputs/mydata_rc_full_seed3407_alpha04_w0"),
    Path("crisp_outputs/mydata_rc_full_seed42_alpha04_w0"),
]

ADATA_PATH = "data/mydata/sciplex_complete_v2.h5ad"
DE_KEY = "lincs_DEGs"
OUT_DIR = Path("crisp_outputs/evidence_mydata/case_studies_manual")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_mat(path):
    df = pd.read_csv(path, index_col=0, engine="python")
    return df.index.astype(str).to_numpy(), df.columns.astype(str).to_numpy(), df.to_numpy(dtype=np.float32)


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot <= 1e-12:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def pearson(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def parse_condition(cond):
    parts = str(cond).split("_")
    if len(parts) >= 3:
        return parts[0], "_".join(parts[1:-1]), parts[-1]
    return str(cond), "", ""


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))[:180]


def build_alias(adata, genes):
    alias = {str(g): i for i, g in enumerate(genes)}
    if adata.n_vars == len(genes):
        for i, g in enumerate(adata.var_names.astype(str).tolist()):
            alias[str(g)] = i
        for col in adata.var.columns:
            if col in ["gene_id", "gene_name", "symbol", "gene_symbols", "features", "id"]:
                for i, g in enumerate(adata.var[col].astype(str).tolist()):
                    alias[str(g)] = i
    return alias


def get_de_idx(cond, degs, alias, true_delta, topk=50):
    idx = []
    for g in list(degs.get(str(cond), [])):
        g = str(g)
        if g in alias:
            idx.append(alias[g])
    idx = np.array(sorted(set(idx)), dtype=np.int64)
    if len(idx) == 0:
        idx = np.argsort(np.abs(true_delta))[::-1][:topk]
    return idx


def load_split(split):
    conds, genes, x_true = read_mat(BASELINE_DIR / split / "x_true_condition_mean.csv")
    _, _, x_crisp = read_mat(BASELINE_DIR / split / "x_crisp_condition_mean.csv")
    _, _, x_ctrl = read_mat(BASELINE_DIR / split / "x_ctrl_condition_mean.csv")

    rc_mats = []
    for d in RC_DIRS:
        _, _, x_cal = read_mat(d / split / "x_calibrated_condition_mean.csv")
        rc_mats.append(x_cal)
    x_rc = np.mean(np.stack(rc_mats, axis=0), axis=0)

    return conds, genes, x_true, x_crisp, x_ctrl, x_rc


def load_all_data():
    all_data = {}
    genes_ref = None

    for split in ["iid", "ood"]:
        conds, genes, x_true, x_crisp, x_ctrl, x_rc = load_split(split)
        genes_ref = genes

        for i, cond in enumerate(conds):
            all_data[str(cond)] = {
                "split": split,
                "true": x_true[i],
                "crisp": x_crisp[i],
                "ctrl": x_ctrl[i],
                "rc": x_rc[i],
            }

    return genes_ref, all_data


def plot_compare_scatter(cond, genes, y_true, y_crisp, y_rc, ctrl, de_idx, out_path):
    true_delta = y_true - ctrl
    crisp_delta = y_crisp - ctrl
    rc_delta = y_rc - ctrl

    x = true_delta[de_idx]
    y1 = crisp_delta[de_idx]
    y2 = rc_delta[de_idx]

    crisp_r = pearson(x, y1)
    rc_r = pearson(x, y2)
    crisp_r2 = safe_r2(x, y1)
    rc_r2 = safe_r2(x, y2)

    lo = float(min(np.min(x), np.min(y1), np.min(y2)))
    hi = float(max(np.max(x), np.max(y1), np.max(y2)))

    plt.figure(figsize=(5.5, 5.2))
    plt.scatter(x, y1, s=22, alpha=0.60, label=f"CRISP r={crisp_r:.2f}, R²={crisp_r2:.2f}")
    plt.scatter(x, y2, s=22, alpha=0.60, label=f"CRISP-RC r={rc_r:.2f}, R²={rc_r2:.2f}")
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True delta expression, official DE genes")
    plt.ylabel("Predicted delta")
    plt.title(cond)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_error_reduction(cond, genes, y_true, y_crisp, y_rc, de_idx, out_path, topn=20):
    crisp_err = np.abs(y_true[de_idx] - y_crisp[de_idx])
    rc_err = np.abs(y_true[de_idx] - y_rc[de_idx])
    improvement = crisp_err - rc_err

    order = np.argsort(improvement)[::-1][:topn]
    vals = improvement[order]
    gene_names = [str(genes[de_idx[i]]) for i in order]

    plt.figure(figsize=(9.0, 4.5))
    plt.bar(range(len(vals)), vals)
    plt.axhline(0, linestyle="--")
    plt.xticks(range(len(vals)), gene_names, rotation=60, ha="right")
    plt.ylabel("|error CRISP| - |error CRISP-RC|")
    plt.title(f"Top DE-gene error reductions\n{cond}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_dose_response(cond, genes, all_data, de_idx, out_path):
    cell_type, drug, _ = parse_condition(cond)

    rows = []
    for c, item in all_data.items():
        ct, dg, dose_str = parse_condition(c)
        if ct != cell_type or dg != drug:
            continue
        try:
            dose = float(dose_str)
        except Exception:
            continue

        true_delta = item["true"] - item["ctrl"]
        crisp_delta = item["crisp"] - item["ctrl"]
        rc_delta = item["rc"] - item["ctrl"]

        rows.append({
            "condition": c,
            "split": item["split"],
            "dose": dose,
            "true_de_magnitude": float(np.mean(np.abs(true_delta[de_idx]))),
            "crisp_de_magnitude": float(np.mean(np.abs(crisp_delta[de_idx]))),
            "rc_de_magnitude": float(np.mean(np.abs(rc_delta[de_idx]))),
        })

    df = pd.DataFrame(rows)
    if df.empty or df["dose"].nunique() < 2:
        print("[WARN] not enough dose points:", cond)
        return

    df = df.sort_values("dose")
    df.to_csv(out_path.with_suffix(".csv"), index=False)

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(df["dose"], df["true_de_magnitude"], marker="o", label="True")
    plt.plot(df["dose"], df["crisp_de_magnitude"], marker="o", label="CRISP")
    plt.plot(df["dose"], df["rc_de_magnitude"], marker="o", label="CRISP-RC")
    plt.xscale("log")
    plt.xlabel("Dose")
    plt.ylabel("Mean |delta| on official DE genes")
    plt.title(f"Dose response: {cell_type} / {drug}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", required=True)
    args = parser.parse_args()

    adata = sc.read_h5ad(ADATA_PATH, backed="r")
    degs = adata.uns[DE_KEY]

    conds, genes, x_true, x_crisp, x_ctrl, x_rc = load_split("ood")
    alias = build_alias(adata, genes)
    _, all_data = load_all_data()

    cond_to_i = {str(c): i for i, c in enumerate(conds)}

    rows = []

    for cond in args.conditions:
        if cond not in cond_to_i:
            print("[MISS] condition not found in OOD:", cond)
            continue

        i = cond_to_i[cond]
        true_delta = x_true[i] - x_ctrl[i]
        crisp_delta = x_crisp[i] - x_ctrl[i]
        rc_delta = x_rc[i] - x_ctrl[i]

        de_idx = get_de_idx(cond, degs, alias, true_delta)

        prefix = OUT_DIR / sanitize(cond)

        plot_compare_scatter(
            cond, genes, x_true[i], x_crisp[i], x_rc[i], x_ctrl[i], de_idx,
            prefix.with_name(prefix.name + "_scatter_compare.png"),
        )

        plot_error_reduction(
            cond, genes, x_true[i], x_crisp[i], x_rc[i], de_idx,
            prefix.with_name(prefix.name + "_top_de_error_reduction.png"),
        )

        plot_dose_response(
            cond, genes, all_data, de_idx,
            prefix.with_name(prefix.name + "_dose_response.png"),
        )

        rows.append({
            "condition": cond,
            "n_de": len(de_idx),
            "base_delta_de_pearson": pearson(true_delta[de_idx], crisp_delta[de_idx]),
            "rc_delta_de_pearson": pearson(true_delta[de_idx], rc_delta[de_idx]),
            "base_delta_de_r2": safe_r2(true_delta[de_idx], crisp_delta[de_idx]),
            "rc_delta_de_r2": safe_r2(true_delta[de_idx], rc_delta[de_idx]),
            "base_de_mse": float(np.mean((x_true[i][de_idx] - x_crisp[i][de_idx]) ** 2)),
            "rc_de_mse": float(np.mean((x_true[i][de_idx] - x_rc[i][de_idx]) ** 2)),
        })

        print("[DONE]", cond)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "manual_selected_case_metrics.csv", index=False)

    print("\nsaved:", OUT_DIR)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
