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
COND_KEY = "cov_drug_dose_name"

OUT_DIR = Path("crisp_outputs/evidence_mydata/case_studies")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_mat(path):
    df = pd.read_csv(path, index_col=0, engine="python")
    return (
        df.index.astype(str).to_numpy(),
        df.columns.astype(str).to_numpy(),
        df.to_numpy(dtype=np.float32),
    )


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


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def sanitize_name(s):
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:180]


def parse_condition(cond):
    """
    适配类似：
    A549_(+)-JQ1_0.001
    A549_Dacinostat_0.01
    K562_xxx_1.0
    """
    cond = str(cond)
    parts = cond.split("_")

    if len(parts) >= 3:
        cell_type = parts[0]
        dose_str = parts[-1]
        drug = "_".join(parts[1:-1])
    elif len(parts) == 2:
        cell_type = parts[0]
        drug = parts[1]
        dose_str = ""
    else:
        cell_type = cond
        drug = ""
        dose_str = ""

    try:
        dose_float = float(dose_str)
    except Exception:
        dose_float = np.nan

    return cell_type, drug, dose_str, dose_float


def build_gene_alias_index(adata, genes):
    """
    建 gene name / gene_id / var_names 到当前矩阵列号的映射。
    """
    alias = {str(g): i for i, g in enumerate(genes)}

    if adata.n_vars == len(genes):
        for i, g in enumerate(adata.var_names.astype(str).tolist()):
            alias[str(g)] = i

        for col in adata.var.columns:
            if col in ["gene_id", "gene_name", "symbol", "gene_symbols", "features", "id"]:
                vals = adata.var[col].astype(str).tolist()
                for i, g in enumerate(vals):
                    alias[str(g)] = i

    return alias


def get_de_idx(cond, degs, alias, true_delta, topk=50):
    """
    优先用官方 lincs_DEGs；如果没有匹配到，就 fallback 到真实 delta 最大的 top50。
    """
    de_genes = list(degs.get(str(cond), []))
    idx = []

    for g in de_genes:
        g = str(g)
        if g in alias:
            idx.append(alias[g])

    idx = np.array(sorted(set(idx)), dtype=np.int64)

    if len(idx) == 0:
        idx = np.argsort(np.abs(true_delta))[::-1][:topk]

    return idx, de_genes


def load_ood_mats():
    conds, genes, x_true = read_mat(BASELINE_DIR / "ood" / "x_true_condition_mean.csv")
    _, _, x_crisp = read_mat(BASELINE_DIR / "ood" / "x_crisp_condition_mean.csv")
    _, _, x_ctrl = read_mat(BASELINE_DIR / "ood" / "x_ctrl_condition_mean.csv")

    rc_mats = []
    for d in RC_DIRS:
        p = d / "ood" / "x_calibrated_condition_mean.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing calibrated matrix: {p}")
        _, _, x_cal = read_mat(p)
        rc_mats.append(x_cal)

    x_rc = np.mean(np.stack(rc_mats, axis=0), axis=0)
    return conds, genes, x_true, x_crisp, x_ctrl, x_rc


def load_all_split_mats():
    """
    用于画 dose-response。把 iid + ood 都读进来。
    """
    all_data = {}

    for split in ["iid", "ood"]:
        conds, genes, x_true = read_mat(BASELINE_DIR / split / "x_true_condition_mean.csv")
        _, _, x_crisp = read_mat(BASELINE_DIR / split / "x_crisp_condition_mean.csv")
        _, _, x_ctrl = read_mat(BASELINE_DIR / split / "x_ctrl_condition_mean.csv")

        rc_mats = []
        for d in RC_DIRS:
            p = d / split / "x_calibrated_condition_mean.csv"
            if not p.exists():
                raise FileNotFoundError(f"Missing calibrated matrix: {p}")
            _, _, x_cal = read_mat(p)
            rc_mats.append(x_cal)

        x_rc = np.mean(np.stack(rc_mats, axis=0), axis=0)

        for i, cond in enumerate(conds):
            all_data[str(cond)] = {
                "split": split,
                "true": x_true[i],
                "crisp": x_crisp[i],
                "ctrl": x_ctrl[i],
                "rc": x_rc[i],
            }

    return genes, all_data


def collect_condition_metadata(adata):
    candidate_cols = [
        "cell_type",
        "condition",
        "cov_drug",
        "cov_drug_name",
        "cov_drug_dose_name",
        "product_name",
        "pathway",
        "pathway_level_1",
        "pathway_level_2",
        "dose_val",
        "dose",
        "SMILES",
    ]
    cols = [c for c in candidate_cols if c in adata.obs.columns]

    if COND_KEY not in cols:
        raise KeyError(f"{COND_KEY} not found in adata.obs")

    meta = adata.obs[cols].copy()
    for c in meta.columns:
        meta[c] = meta[c].astype(str)

    rows = []
    for cond, sub in meta.groupby(COND_KEY, observed=True):
        row = {"condition": str(cond)}
        for c in cols:
            if c == COND_KEY:
                continue
            vals = sub[c].dropna().astype(str)
            row[c] = vals.iloc[0] if len(vals) else ""
        rows.append(row)

    return pd.DataFrame(rows).set_index("condition")


def select_cases(adata, conds, genes, x_true, x_crisp, x_ctrl, x_rc):
    alias = build_gene_alias_index(adata, genes)
    degs = adata.uns[DE_KEY]
    meta = collect_condition_metadata(adata)

    rows = []

    for i, cond in enumerate(conds):
        true_delta = x_true[i] - x_ctrl[i]
        crisp_delta = x_crisp[i] - x_ctrl[i]
        rc_delta = x_rc[i] - x_ctrl[i]

        de_idx, de_genes = get_de_idx(cond, degs, alias, true_delta)

        base_delta_r2 = safe_r2(true_delta[de_idx], crisp_delta[de_idx])
        rc_delta_r2 = safe_r2(true_delta[de_idx], rc_delta[de_idx])

        base_delta_pearson = pearson(true_delta[de_idx], crisp_delta[de_idx])
        rc_delta_pearson = pearson(true_delta[de_idx], rc_delta[de_idx])

        base_de_mse = mse(x_true[i][de_idx], x_crisp[i][de_idx])
        rc_de_mse = mse(x_true[i][de_idx], x_rc[i][de_idx])

        cell_type, drug, dose_str, dose_float = parse_condition(cond)

        row = {
            "condition": str(cond),
            "cell_type_parsed": cell_type,
            "drug_parsed": drug,
            "dose_parsed": dose_str,
            "dose_float": dose_float,
            "n_de": int(len(de_idx)),
            "base_delta_de_r2": base_delta_r2,
            "rc_delta_de_r2": rc_delta_r2,
            "delta_de_r2_improvement": rc_delta_r2 - base_delta_r2,
            "base_delta_de_pearson": base_delta_pearson,
            "rc_delta_de_pearson": rc_delta_pearson,
            "delta_de_pearson_improvement": rc_delta_pearson - base_delta_pearson,
            "base_de_mse": base_de_mse,
            "rc_de_mse": rc_de_mse,
            "de_mse_improvement": base_de_mse - rc_de_mse,
        }

        if str(cond) in meta.index:
            for c in meta.columns:
                row[c] = meta.loc[str(cond), c]

        rows.append(row)

    df = pd.DataFrame(rows)

    # 排序：优先选 delta R2 提升明显、MSE 也下降的 case
    df = df.sort_values(
        ["delta_de_r2_improvement", "de_mse_improvement", "delta_de_pearson_improvement"],
        ascending=False,
    )

    # 选 3 个不同 drug 的 case，避免全是同一个药
    selected = []
    used_drugs = set()

    for _, row in df.iterrows():
        drug = str(row["drug_parsed"])
        if drug in used_drugs:
            continue
        if row["de_mse_improvement"] <= 0:
            continue
        selected.append(row)
        used_drugs.add(drug)
        if len(selected) >= 3:
            break

    # 如果不足 3 个，就不管 drug 是否重复，补满
    if len(selected) < 3:
        for _, row in df.iterrows():
            cond = str(row["condition"])
            if any(str(x["condition"]) == cond for x in selected):
                continue
            selected.append(row)
            if len(selected) >= 3:
                break

    selected = pd.DataFrame(selected)

    df.to_csv(OUT_DIR / "all_ood_case_ranking.csv", index=False)
    selected.to_csv(OUT_DIR / "selected_case_conditions.csv", index=False)

    print("=" * 100)
    print("Selected case conditions:")
    show_cols = [
        "condition",
        "cell_type_parsed",
        "drug_parsed",
        "dose_parsed",
        "base_delta_de_r2",
        "rc_delta_de_r2",
        "delta_de_r2_improvement",
        "base_de_mse",
        "rc_de_mse",
        "de_mse_improvement",
    ]
    show_cols = [c for c in show_cols if c in selected.columns]
    print(selected[show_cols].to_string(index=False))
    print("=" * 100)

    return selected, alias, degs


def plot_scatter(cond, y_true, y_pred, ctrl, de_idx, model_name, out_path):
    true_delta = y_true - ctrl
    pred_delta = y_pred - ctrl

    x = true_delta[de_idx]
    y = pred_delta[de_idx]

    r = pearson(x, y)
    r2 = safe_r2(x, y)

    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(x, y, s=18, alpha=0.75)
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True delta expression, official DE genes")
    plt.ylabel(f"{model_name} predicted delta")
    plt.title(f"{cond}\n{model_name}: Pearson={r:.3f}, R²={r2:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_error_reduction(cond, genes, y_true, y_crisp, y_rc, de_idx, out_path, topn=15):
    crisp_err = np.abs(y_true[de_idx] - y_crisp[de_idx])
    rc_err = np.abs(y_true[de_idx] - y_rc[de_idx])
    improvement = crisp_err - rc_err

    order = np.argsort(improvement)[::-1][:topn]
    vals = improvement[order]
    gene_names = [str(genes[de_idx[i]]) for i in order]

    plt.figure(figsize=(8.2, 4.2))
    plt.bar(range(len(vals)), vals)
    plt.axhline(0, linestyle="--")
    plt.xticks(range(len(vals)), gene_names, rotation=60, ha="right")
    plt.ylabel("|error CRISP| - |error CRISP-RC|")
    plt.title(f"Top DE-gene error reductions\n{cond}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_dose_response(selected_cond, genes, all_data, de_idx, out_path):
    cell_type, drug, _, _ = parse_condition(selected_cond)

    rows = []

    for cond, item in all_data.items():
        ct, dg, dose_str, dose_float = parse_condition(cond)
        if ct != cell_type or dg != drug:
            continue
        if not np.isfinite(dose_float):
            continue

        true_delta = item["true"] - item["ctrl"]
        crisp_delta = item["crisp"] - item["ctrl"]
        rc_delta = item["rc"] - item["ctrl"]

        rows.append({
            "condition": cond,
            "split": item["split"],
            "dose": dose_float,
            "true_de_magnitude": float(np.mean(np.abs(true_delta[de_idx]))),
            "crisp_de_magnitude": float(np.mean(np.abs(crisp_delta[de_idx]))),
            "rc_de_magnitude": float(np.mean(np.abs(rc_delta[de_idx]))),
        })

    df = pd.DataFrame(rows)

    if df.empty or df["dose"].nunique() < 2:
        print(f"[WARN] Not enough dose points for dose-response plot: {selected_cond}")
        return

    df = df.sort_values("dose")
    df.to_csv(out_path.with_suffix(".csv"), index=False)

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(df["dose"], df["true_de_magnitude"], marker="o", label="True")
    plt.plot(df["dose"], df["crisp_de_magnitude"], marker="o", label="CRISP")
    plt.plot(df["dose"], df["rc_de_magnitude"], marker="o", label="CRISP-RC")
    plt.xscale("log")
    plt.xlabel("Dose")
    plt.ylabel("Mean |delta| on selected DE genes")
    plt.title(f"Dose response\n{cell_type} / {drug}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    print("Loading AnnData:", ADATA_PATH)
    adata = sc.read_h5ad(ADATA_PATH, backed="r")

    print("Loading OOD matrices...")
    conds, genes, x_true, x_crisp, x_ctrl, x_rc = load_ood_mats()

    print("Selecting cases...")
    selected, alias, degs = select_cases(adata, conds, genes, x_true, x_crisp, x_ctrl, x_rc)

    print("Loading all split matrices for dose-response...")
    _, all_data = load_all_split_mats()

    cond_to_i = {str(c): i for i, c in enumerate(conds)}

    for _, row in selected.iterrows():
        cond = str(row["condition"])
        i = cond_to_i[cond]

        true_delta = x_true[i] - x_ctrl[i]
        de_idx, de_genes = get_de_idx(cond, degs, alias, true_delta)

        prefix = OUT_DIR / sanitize_name(cond)

        print("Plotting:", cond)

        plot_scatter(
            cond,
            x_true[i],
            x_crisp[i],
            x_ctrl[i],
            de_idx,
            "CRISP",
            prefix.with_name(prefix.name + "_scatter_crisp.png"),
        )

        plot_scatter(
            cond,
            x_true[i],
            x_rc[i],
            x_ctrl[i],
            de_idx,
            "CRISP-RC",
            prefix.with_name(prefix.name + "_scatter_rc.png"),
        )

        plot_error_reduction(
            cond,
            genes,
            x_true[i],
            x_crisp[i],
            x_rc[i],
            de_idx,
            prefix.with_name(prefix.name + "_top_de_error_reduction.png"),
        )

        plot_dose_response(
            cond,
            genes,
            all_data,
            de_idx,
            prefix.with_name(prefix.name + "_dose_response.png"),
        )

    print("=" * 100)
    print("Saved case-study files to:", OUT_DIR)
    print("Selected cases CSV:", OUT_DIR / "selected_case_conditions.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()
