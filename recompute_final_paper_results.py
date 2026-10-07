from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

try:
    import anndata as ad
except Exception:
    ad = None


PROJECT_ROOT = Path(__file__).resolve().parent

BASELINE_OOD = (
    PROJECT_ROOT
    / "crisp_outputs"
    / "mydata_baseline_full"
    / "ood"
)

FINAL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ridge_anchored_recalib_final"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "paper_recalculation"
)

H5AD = (
    PROJECT_ROOT
    / "data"
    / "mydata"
    / "sciplex_complete_v2_evalall.h5ad"
)

SEEDS = [2024, 3407, 42]

REPRESENTATIVE_CASES = [
    ("K562", "Givinostat", 0.01),
    ("A549", "Dacinostat", 1.0),
    ("MCF7", "Tanespimycin", 0.01),
]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.astype(float)


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(pearsonr(x, y).statistic)


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(spearmanr(x, y).statistic)


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return np.nan

    denominator = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if denominator <= 0:
        return np.nan

    numerator = np.sum(
        (y_true - y_pred) ** 2
    )

    return float(
        1.0 - numerator / denominator
    )


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() == 0:
        return np.nan

    return float(
        np.mean(
            (y_true[mask] - y_pred[mask]) ** 2
        )
    )


def align_matrix(
    df: pd.DataFrame,
    true: pd.DataFrame,
    label: str,
):
    missing_rows = (
        set(true.index) - set(df.index)
    )
    missing_cols = (
        set(true.columns) - set(df.columns)
    )

    if missing_rows:
        raise RuntimeError(
            f"{label}: missing conditions: "
            f"{sorted(missing_rows)[:5]}"
        )

    if missing_cols:
        raise RuntimeError(
            f"{label}: missing genes: "
            f"{sorted(missing_cols)[:5]}"
        )

    return df.loc[
        true.index,
        true.columns,
    ].copy()


def discover_final_prediction(
    seed: int,
    true: pd.DataFrame,
):
    path = (
        PROJECT_ROOT
        / "results"
        / "ridge_anchored_recalib_final"
        / f"ridge_anchored_recalib_seed_{seed}"
        / "x_calibrated_condition_mean.csv"
    )

    require_file(path)

    df = align_matrix(
        read_matrix(path),
        true,
        f"Final ReCalib seed={seed}",
    )

    print(
        f"[FINAL PREDICTION] seed={seed}"
    )
    print(
        f"  path : {path.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"  shape: {df.shape}"
    )

    return path, df


def parse_gene_list(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        obj = ast.literal_eval(text)

        if isinstance(
            obj,
            (list, tuple, set, np.ndarray),
        ):
            return [
                str(x).strip()
                for x in obj
                if str(x).strip()
            ]
    except Exception:
        pass

    for sep in [";", ",", "|"]:
        if sep in text:
            return [
                x.strip().strip("'\"")
                for x in text.split(sep)
                if x.strip()
            ]

    return [text.strip("'\"")]


def load_de_mapping(
    conditions,
    genes,
):
    """
    Map official LINCS DE gene symbols to the 977-gene
    Ensembl-ID expression axis.

    Mapping chain:
        lincs_DEGs gene symbol
        -> adata.var_names
        -> adata.var["gene_id"]
        -> expression-matrix column
    """

    if ad is None:
        raise RuntimeError(
            "anndata is required for DE mapping."
        )

    require_file(H5AD)

    print(
        f"[LOAD] DE annotations from "
        f"{H5AD.relative_to(PROJECT_ROOT)}"
    )

    adata = ad.read_h5ad(
        H5AD,
        backed="r",
    )

    if "lincs_DEGs" not in adata.uns:
        adata.file.close()
        raise RuntimeError(
            "Missing adata.uns['lincs_DEGs']"
        )

    if "gene_id" not in adata.var.columns:
        adata.file.close()
        raise RuntimeError(
            "Missing adata.var['gene_id']"
        )

    # --------------------------------------------------------
    # Verify the 977-gene matrix axis
    # --------------------------------------------------------
    matrix_genes = [
        str(g)
        for g in genes
    ]

    var_symbols = [
        str(g)
        for g in adata.var_names
    ]

    var_gene_ids = (
        adata.var["gene_id"]
        .astype(str)
        .tolist()
    )

    if len(var_gene_ids) != len(matrix_genes):
        adata.file.close()
        raise RuntimeError(
            "H5AD gene count does not match "
            "expression matrix gene count: "
            f"{len(var_gene_ids)} vs "
            f"{len(matrix_genes)}"
        )

    if var_gene_ids != matrix_genes:
        adata.file.close()

        same_set = (
            set(var_gene_ids)
            == set(matrix_genes)
        )

        raise RuntimeError(
            "H5AD gene_id order does not exactly "
            "match the 977-gene expression matrix. "
            f"Same set={same_set}. "
            "Paper recalculation stopped."
        )

    print(
        "[GENE AXIS] "
        "adata.var['gene_id'] exactly matches "
        "the 977 matrix columns."
    )

    # Symbol -> Ensembl ID
    symbol_to_gene_id = {
        symbol.upper(): gene_id
        for symbol, gene_id in zip(
            var_symbols,
            var_gene_ids,
        )
    }

    raw_mapping = {
        str(key): value
        for key, value in dict(
            adata.uns["lincs_DEGs"]
        ).items()
    }

    mapping = {}
    coverage_rows = []
    missing_conditions = []

    for condition in conditions:
        condition = str(condition)

        if condition not in raw_mapping:
            missing_conditions.append(
                condition
            )
            continue

        listed_symbols = [
            str(gene).strip()
            for gene in np.asarray(
                raw_mapping[condition],
                dtype=object,
            ).ravel().tolist()
            if str(gene).strip()
        ]

        mapped_gene_ids = []

        for symbol in listed_symbols:
            key = symbol.upper()

            if key in symbol_to_gene_id:
                mapped_gene_ids.append(
                    symbol_to_gene_id[key]
                )

        # Preserve official DE ordering,
        # remove duplicates only if present.
        mapped_gene_ids = list(
            dict.fromkeys(
                mapped_gene_ids
            )
        )

        mapping[condition] = (
            mapped_gene_ids
        )

        coverage_rows.append(
            {
                "condition": condition,
                "n_de_listed":
                    len(listed_symbols),
                "n_de_mapped":
                    len(mapped_gene_ids),
            }
        )

    adata.file.close()

    if missing_conditions:
        raise RuntimeError(
            "Conditions missing from "
            "adata.uns['lincs_DEGs']: "
            + ", ".join(
                missing_conditions[:20]
            )
        )

    coverage = pd.DataFrame(
        coverage_rows
    )

    print(
        "[DE] exact condition matches:",
        len(coverage),
        "/",
        len(conditions),
    )

    print(
        "[DE] conditions with mapped DE genes:",
        int(
            (
                coverage["n_de_mapped"] > 0
            ).sum()
        ),
        "/",
        len(conditions),
    )

    print(
        "[DE] mapping summary:"
    )

    print(
        coverage[
            [
                "n_de_listed",
                "n_de_mapped",
            ]
        ]
        .describe()
        .to_string()
    )

    # --------------------------------------------------------
    # Cross-check against previously exported metadata
    # --------------------------------------------------------
    metadata_path = (
        BASELINE_OOD
        / "condition_metadata.csv"
    )

    require_file(metadata_path)

    metadata = pd.read_csv(
        metadata_path
    )

    metadata["condition"] = (
        metadata["condition"]
        .astype(str)
    )

    expected_cols = {
        "condition",
        "n_de_listed",
        "n_de_mapped",
    }

    if expected_cols.issubset(
        metadata.columns
    ):
        check = coverage.merge(
            metadata[
                [
                    "condition",
                    "n_de_listed",
                    "n_de_mapped",
                ]
            ],
            on="condition",
            how="left",
            suffixes=(
                "_recomputed",
                "_metadata",
            ),
        )

        listed_ok = (
            check[
                "n_de_listed_recomputed"
            ]
            ==
            check[
                "n_de_listed_metadata"
            ]
        ).all()

        mapped_ok = (
            check[
                "n_de_mapped_recomputed"
            ]
            ==
            check[
                "n_de_mapped_metadata"
            ]
        ).all()

        print(
            "[DE CHECK] n_de_listed "
            f"agreement: {listed_ok}"
        )

        print(
            "[DE CHECK] n_de_mapped "
            f"agreement: {mapped_ok}"
        )

        if not listed_ok or not mapped_ok:
            raise RuntimeError(
                "Recomputed DE mapping does not "
                "match existing condition metadata."
            )

    if len(coverage) != len(conditions):
        raise RuntimeError(
            "Incomplete DE mapping coverage."
        )

    if (
        coverage["n_de_mapped"] <= 0
    ).any():
        raise RuntimeError(
            "At least one condition has "
            "zero mapped DE genes."
        )

    coverage.to_csv(
        OUTPUT_ROOT
        / "main_ood_de_mapping_check.csv",
        index=False,
    )

    return mapping


def condition_metrics(
    pred,
    true,
    ctrl,
    de_mapping,
):
    rows = []

    for condition in true.index:
        genes = de_mapping[condition]

        yt = true.loc[
            condition,
            genes,
        ].to_numpy(float)

        yp = pred.loc[
            condition,
            genes,
        ].to_numpy(float)

        xc = ctrl.loc[
            condition,
            genes,
        ].to_numpy(float)

        delta_true = yt - xc
        delta_pred = yp - xc

        rows.append(
            {
                "condition": condition,
                "n_de": len(genes),
                "mse_de": mse(yt, yp),
                "pearson_de": safe_pearson(
                    yt,
                    yp,
                ),
                "pearson_delta_de":
                    safe_pearson(
                        delta_true,
                        delta_pred,
                    ),
                "r2score_de": safe_r2(
                    yt,
                    yp,
                ),
            }
        )

    return pd.DataFrame(rows)


def response_gene_metrics(
    pred,
    true,
    ctrl,
    de_mapping,
):
    rows = []

    for condition in true.index:
        observed_delta = (
            true.loc[condition]
            - ctrl.loc[condition]
        )

        predicted_delta = (
            pred.loc[condition]
            - ctrl.loc[condition]
        )

        obs_abs = observed_delta.abs()
        pred_abs = predicted_delta.abs()

        row = {
            "condition": condition,
        }

        for k in [10, 20]:
            obs_top = set(
                obs_abs.nlargest(k).index
            )
            pred_top = set(
                pred_abs.nlargest(k).index
            )

            row[f"top{k}_recovery"] = (
                len(obs_top & pred_top)
                / float(k)
            )

        de_genes = de_mapping[condition]

        row["de_response_rank_spearman"] = (
            safe_spearman(
                obs_abs.loc[
                    de_genes
                ].to_numpy(float),
                pred_abs.loc[
                    de_genes
                ].to_numpy(float),
            )
        )

        rows.append(row)

    return pd.DataFrame(rows)


def find_hallmark_gmt():
    path = (
        PROJECT_ROOT
        / "data"
        / "pathways"
        / "h.all.v2025.1.Hs.symbols.gmt"
    )
    return path if path.is_file() else None

def read_gmt(
    path: Path,
    expression_genes,
):
    """
    Map Hallmark gene symbols to the same 977-gene
    Ensembl-ID expression axis used by the model outputs.
    """

    if ad is None:
        raise RuntimeError(
            "anndata is required for Hallmark mapping."
        )

    adata = ad.read_h5ad(
        H5AD,
        backed="r",
    )

    if "gene_id" not in adata.var.columns:
        adata.file.close()
        raise RuntimeError(
            "Missing adata.var['gene_id']"
        )

    var_symbols = [
        str(g)
        for g in adata.var_names
    ]

    var_gene_ids = (
        adata.var["gene_id"]
        .astype(str)
        .tolist()
    )

    matrix_genes = [
        str(g)
        for g in expression_genes
    ]

    if var_gene_ids != matrix_genes:
        adata.file.close()
        raise RuntimeError(
            "Hallmark mapping gene axis does not "
            "match expression matrix exactly."
        )

    symbol_to_gene_id = {
        symbol.upper(): gene_id
        for symbol, gene_id in zip(
            var_symbols,
            var_gene_ids,
        )
    }

    adata.file.close()

    gene_sets = {}

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        for line in f:
            parts = (
                line.rstrip("\n")
                .split("\t")
            )

            if len(parts) < 3:
                continue

            name = parts[0]

            mapped = []

            for symbol in parts[2:]:
                key = str(
                    symbol
                ).upper()

                if key in symbol_to_gene_id:
                    mapped.append(
                        symbol_to_gene_id[key]
                    )

            mapped = list(
                dict.fromkeys(mapped)
            )

            if len(mapped) >= 5:
                gene_sets[name] = mapped

    print(
        "[HALLMARK] gene sets with >=5 "
        f"mapped genes: {len(gene_sets)}"
    )

    return gene_sets


def pathway_metrics(
    pred,
    true,
    ctrl,
    gene_sets,
):
    rows = []

    for condition in true.index:
        observed_scores = []
        predicted_scores = []

        for _, genes in gene_sets.items():
            obs = (
                true.loc[
                    condition,
                    genes,
                ]
                -
                ctrl.loc[
                    condition,
                    genes,
                ]
            ).mean()

            prd = (
                pred.loc[
                    condition,
                    genes,
                ]
                -
                ctrl.loc[
                    condition,
                    genes,
                ]
            ).mean()

            observed_scores.append(
                float(obs)
            )
            predicted_scores.append(
                float(prd)
            )

        rows.append(
            {
                "condition": condition,
                "pathway_score_pearson":
                    safe_pearson(
                        observed_scores,
                        predicted_scores,
                    ),
                "pathway_score_spearman":
                    safe_spearman(
                        observed_scores,
                        predicted_scores,
                    ),
                "pathway_score_mse":
                    mse(
                        observed_scores,
                        predicted_scores,
                    ),
                "n_pathways":
                    len(gene_sets),
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(
    df,
    metrics,
):
    rows = []

    for metric in metrics:
        values = pd.to_numeric(
            df[metric],
            errors="coerce",
        ).dropna()

        rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "sample_sd":
                    values.std(ddof=1),
                "median": values.median(),
                "n_valid_conditions":
                    len(values),
            }
        )

    return pd.DataFrame(rows)


def residual_consistency(
    pred,
    crisp,
    true,
    de_mapping,
):
    correction = pred - crisp
    true_residual = true - crisp

    all_x = true_residual.to_numpy().ravel()
    all_y = correction.to_numpy().ravel()

    rows = [
        {
            "scope": "all_genes",
            "corr": safe_pearson(
                all_x,
                all_y,
            ),
            "r2": safe_r2(
                all_x,
                all_y,
            ),
            "mse": mse(
                all_x,
                all_y,
            ),
            "n_values": len(all_x),
        }
    ]

    de_true = []
    de_pred = []

    for condition in true.index:
        genes = de_mapping[condition]

        de_true.extend(
            true_residual.loc[
                condition,
                genes,
            ].to_numpy(float)
        )

        de_pred.extend(
            correction.loc[
                condition,
                genes,
            ].to_numpy(float)
        )

    rows.append(
        {
            "scope": "de_genes",
            "corr": safe_pearson(
                de_true,
                de_pred,
            ),
            "r2": safe_r2(
                de_true,
                de_pred,
            ),
            "mse": mse(
                de_true,
                de_pred,
            ),
            "n_values": len(de_true),
        }
    )

    return pd.DataFrame(rows)


def load_metadata():
    path = require_file(
        BASELINE_OOD
        / "condition_metadata.csv"
    )

    meta = pd.read_csv(path)

    if "condition" not in meta.columns:
        raise RuntimeError(
            "condition column missing "
            "from metadata"
        )

    meta["condition"] = (
        meta["condition"].astype(str)
    )

    return meta


def representative_case_metrics(
    models,
    true,
    ctrl,
    de_mapping,
    metadata,
):
    rows = []

    for cell, drug, dose in REPRESENTATIVE_CASES:
        subset = metadata[
            (
                metadata["cell_type"]
                .astype(str)
                .str.lower()
                == cell.lower()
            )
            &
            (
                metadata["drug"]
                .astype(str)
                .str.lower()
                == drug.lower()
            )
            &
            (
                np.isclose(
                    pd.to_numeric(
                        metadata["dose"],
                        errors="coerce",
                    ),
                    float(dose),
                    rtol=0,
                    atol=1e-12,
                )
            )
        ]

        if subset.empty:
            print(
                "[WARN] representative case "
                f"not found: {cell}, {drug}, {dose}"
            )
            continue

        condition = str(
            subset.iloc[0]["condition"]
        )

        genes = de_mapping[condition]

        yt = true.loc[
            condition,
            genes,
        ].to_numpy(float)

        xc = ctrl.loc[
            condition,
            genes,
        ].to_numpy(float)

        for model_name, pred in models.items():
            yp = pred.loc[
                condition,
                genes,
            ].to_numpy(float)

            rows.append(
                {
                    "condition": condition,
                    "cell_type": cell,
                    "drug": drug,
                    "dose": dose,
                    "model": model_name,
                    "n_de": len(genes),
                    "mse_de":
                        mse(yt, yp),
                    "pearson_delta_de":
                        safe_pearson(
                            yt - xc,
                            yp - xc,
                        ),
                }
            )

    return pd.DataFrame(rows)


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "Final ReCalib paper-result recalculation"
    )
    print(
        "Project:",
        PROJECT_ROOT,
    )
    print("=" * 80)

    true_path = require_file(
        BASELINE_OOD
        / "x_true_condition_mean.csv"
    )
    crisp_path = require_file(
        BASELINE_OOD
        / "x_crisp_condition_mean.csv"
    )
    ctrl_path = require_file(
        BASELINE_OOD
        / "x_ctrl_condition_mean.csv"
    )

    true = read_matrix(true_path)
    crisp = align_matrix(
        read_matrix(crisp_path),
        true,
        "Frozen CRISP",
    )
    ctrl = align_matrix(
        read_matrix(ctrl_path),
        true,
        "control",
    )

    print(
        f"[OK] Main OOD: "
        f"{true.shape[0]} conditions x "
        f"{true.shape[1]} genes"
    )

    final_predictions = {}
    prediction_paths = {}

    for seed in SEEDS:
        path, pred = discover_final_prediction(
            seed,
            true,
        )

        final_predictions[
            seed
        ] = pred

        prediction_paths[
            seed
        ] = path

    de_mapping = load_de_mapping(
        list(true.index),
        list(true.columns),
    )

    models = {
        "frozen_crisp": crisp,
    }

    for seed, pred in final_predictions.items():
        models[
            f"recalib_seed_{seed}"
        ] = pred

    # -------------------------------------------------
    # Main OOD condition metrics
    # -------------------------------------------------
    all_condition_metrics = []

    for model_name, pred in models.items():
        df = condition_metrics(
            pred,
            true,
            ctrl,
            de_mapping,
        )
        df.insert(
            0,
            "model",
            model_name,
        )
        all_condition_metrics.append(df)

    main_condition = pd.concat(
        all_condition_metrics,
        ignore_index=True,
    )

    main_condition.to_csv(
        OUTPUT_ROOT
        / "main_ood_condition_metrics.csv",
        index=False,
    )

    metric_columns = [
        "mse_de",
        "pearson_de",
        "pearson_delta_de",
        "r2score_de",
    ]

    summary_rows = []

    for model_name in main_condition[
        "model"
    ].unique():
        subset = main_condition[
            main_condition["model"]
            == model_name
        ]

        s = summarize_metrics(
            subset,
            metric_columns,
        )

        s.insert(
            0,
            "model",
            model_name,
        )

        summary_rows.append(s)

    main_summary = pd.concat(
        summary_rows,
        ignore_index=True,
    )

    main_summary.to_csv(
        OUTPUT_ROOT
        / "main_ood_model_summary.csv",
        index=False,
    )

    # Seed mean +/- sample SD from per-seed means.
    per_seed = (
        main_summary[
            main_summary["model"]
            .str.startswith(
                "recalib_seed_"
            )
        ][
            ["model", "metric", "mean"]
        ]
        .copy()
    )

    per_seed["seed"] = (
        per_seed["model"]
        .str.extract(
            r"(\d+)$"
        )[0]
        .astype(int)
    )

    seed_summary = (
        per_seed
        .groupby(
            "metric",
            sort=False,
        )["mean"]
        .agg(
            mean="mean",
            sample_sd=lambda x:
                x.std(ddof=1),
        )
        .reset_index()
    )

    seed_summary[
        "n_seeds"
    ] = len(SEEDS)

    seed_summary.to_csv(
        OUTPUT_ROOT
        / "main_ood_recalib_seed_mean_sd.csv",
        index=False,
    )

    # Mean condition metrics across seeds.
    seed_condition = main_condition[
        main_condition["model"]
        .str.startswith(
            "recalib_seed_"
        )
    ].copy()

    seed_condition_mean = (
        seed_condition
        .groupby(
            "condition",
            sort=False,
        )[metric_columns]
        .mean()
        .reset_index()
    )

    seed_condition_mean.insert(
        0,
        "model",
        "recalib_seed_mean",
    )

    seed_condition_mean.to_csv(
        OUTPUT_ROOT
        / "main_ood_recalib_seed_mean_condition_metrics.csv",
        index=False,
    )

    # -------------------------------------------------
    # Response-gene evaluation
    # -------------------------------------------------
    response_rows = []

    for model_name, pred in models.items():
        df = response_gene_metrics(
            pred,
            true,
            ctrl,
            de_mapping,
        )

        df.insert(
            0,
            "model",
            model_name,
        )

        response_rows.append(df)

    response_condition = pd.concat(
        response_rows,
        ignore_index=True,
    )

    response_condition.to_csv(
        OUTPUT_ROOT
        / "response_gene_condition_metrics.csv",
        index=False,
    )

    response_summary_rows = []

    response_metrics = [
        "top10_recovery",
        "top20_recovery",
        "de_response_rank_spearman",
    ]

    for model_name in response_condition[
        "model"
    ].unique():
        subset = response_condition[
            response_condition["model"]
            == model_name
        ]

        s = summarize_metrics(
            subset,
            response_metrics,
        )

        s.insert(
            0,
            "model",
            model_name,
        )

        response_summary_rows.append(s)

    pd.concat(
        response_summary_rows,
        ignore_index=True,
    ).to_csv(
        OUTPUT_ROOT
        / "response_gene_summary.csv",
        index=False,
    )

    # -------------------------------------------------
    # Hallmark evaluation
    # -------------------------------------------------
    hallmark_path = find_hallmark_gmt()

    if hallmark_path is None:
        print(
            "[WARN] No Hallmark GMT found. "
            "Hallmark recalculation skipped."
        )
    else:
        print(
            "[HALLMARK]",
            hallmark_path.relative_to(
                PROJECT_ROOT
            ),
        )

        gene_sets = read_gmt(
            hallmark_path,
            true.columns,
        )

        print(
            "[HALLMARK] retained sets:",
            len(gene_sets),
        )

        pathway_rows = []

        for model_name, pred in models.items():
            df = pathway_metrics(
                pred,
                true,
                ctrl,
                gene_sets,
            )

            df.insert(
                0,
                "model",
                model_name,
            )

            pathway_rows.append(df)

        pathway_condition = pd.concat(
            pathway_rows,
            ignore_index=True,
        )

        pathway_condition.to_csv(
            OUTPUT_ROOT
            / "hallmark_condition_metrics.csv",
            index=False,
        )

        pathway_summary_rows = []

        pathway_metric_names = [
            "pathway_score_pearson",
            "pathway_score_spearman",
            "pathway_score_mse",
        ]

        for model_name in pathway_condition[
            "model"
        ].unique():
            subset = pathway_condition[
                pathway_condition["model"]
                == model_name
            ]

            s = summarize_metrics(
                subset,
                pathway_metric_names,
            )

            s.insert(
                0,
                "model",
                model_name,
            )

            pathway_summary_rows.append(s)

        pd.concat(
            pathway_summary_rows,
            ignore_index=True,
        ).to_csv(
            OUTPUT_ROOT
            / "hallmark_summary.csv",
            index=False,
        )

    # -------------------------------------------------
    # Residual consistency
    # -------------------------------------------------
    residual_rows = []

    for seed, pred in final_predictions.items():
        df = residual_consistency(
            pred,
            crisp,
            true,
            de_mapping,
        )

        df.insert(
            0,
            "seed",
            seed,
        )

        residual_rows.append(df)

    residual_df = pd.concat(
        residual_rows,
        ignore_index=True,
    )

    residual_df.to_csv(
        OUTPUT_ROOT
        / "residual_consistency.csv",
        index=False,
    )

    # -------------------------------------------------
    # Representative cases
    # -------------------------------------------------
    metadata = load_metadata()

    case_df = representative_case_metrics(
        models,
        true,
        ctrl,
        de_mapping,
        metadata,
    )

    case_df.to_csv(
        OUTPUT_ROOT
        / "representative_case_metrics.csv",
        index=False,
    )

    # -------------------------------------------------
    # Provenance
    # -------------------------------------------------
    provenance = {
        "baseline_true": {
            "path": str(
                true_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": sha256(true_path),
        },
        "baseline_crisp": {
            "path": str(
                crisp_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": sha256(crisp_path),
        },
        "baseline_control": {
            "path": str(
                ctrl_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": sha256(ctrl_path),
        },
        "final_predictions": {},
        "main_ood_shape": list(
            true.shape
        ),
        "seeds": SEEDS,
    }

    for seed, path in prediction_paths.items():
        provenance[
            "final_predictions"
        ][str(seed)] = {
            "path": str(
                path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": sha256(path),
        }

    if hallmark_path is not None:
        provenance["hallmark_gmt"] = {
            "path": str(
                hallmark_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": sha256(
                hallmark_path
            ),
        }

    with (
        OUTPUT_ROOT
        / "recalculation_provenance.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            provenance,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("[OK] Main-OOD final recalculation completed")
    print("[OUTPUT]", OUTPUT_ROOT)
    print("=" * 80)

    print()
    print("Final ReCalib seed mean +/- sample SD")
    print(
        seed_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
