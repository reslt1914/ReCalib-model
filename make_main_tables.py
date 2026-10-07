from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path("results/paper_recalculation")

MAIN_FILE = ROOT / "main_ood_condition_metrics.csv"
RESPONSE_FILE = ROOT / "response_gene_condition_metrics.csv"
HALLMARK_FILE = ROOT / "hallmark_condition_metrics.csv"

OUT_DIR = ROOT / "main_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [
    "recalib_seed_2024",
    "recalib_seed_3407",
    "recalib_seed_42",
]

BASE_MODEL = "frozen_crisp"

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 2024


def holm_adjust(pvalues):
    """
    Holm step-down multiple-testing correction.
    Returns adjusted p-values in original order.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)

    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.empty(m, dtype=float)

    running_max = 0.0

    for rank, value in enumerate(sorted_p):
        adjusted = (m - rank) * value
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted


def bootstrap_mean_ci(
    values,
    n_boot=BOOTSTRAP_REPS,
    seed=BOOTSTRAP_SEED,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        len(values),
        size=(n_boot, len(values)),
    )

    boot_means = values[indices].mean(axis=1)

    low, high = np.percentile(
        boot_means,
        [2.5, 97.5],
    )

    return float(low), float(high)


def safe_wilcoxon(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    if np.allclose(values, 0):
        return 1.0

    try:
        result = wilcoxon(
            values,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        return float(result.pvalue)
    except ValueError:
        return 1.0


def prepare_condition_level(
    path,
    metrics,
):
    df = pd.read_csv(path)

    required = {
        "model",
        "condition",
        *metrics,
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    # ---------------------------------------------------------
    # Frozen CRISP: one row per condition
    # ---------------------------------------------------------
    frozen = (
        df[df["model"].astype(str) == BASE_MODEL]
        [["condition"] + metrics]
        .copy()
    )

    if len(frozen) != 108:
        raise RuntimeError(
            f"{path}: expected 108 Frozen CRISP rows, "
            f"found {len(frozen)}"
        )

    if frozen["condition"].duplicated().any():
        raise RuntimeError(
            f"{path}: duplicated Frozen CRISP conditions"
        )

    frozen = frozen.rename(
        columns={
            metric: f"frozen_{metric}"
            for metric in metrics
        }
    )

    # ---------------------------------------------------------
    # Final ReCalib: exactly 3 seeds
    # ---------------------------------------------------------
    recalib = df[
        df["model"].astype(str).isin(SEEDS)
    ].copy()

    found_models = set(
        recalib["model"].astype(str).unique()
    )

    if found_models != set(SEEDS):
        raise RuntimeError(
            f"{path}: ReCalib seed mismatch.\n"
            f"Expected: {SEEDS}\n"
            f"Found: {sorted(found_models)}"
        )

    counts = (
        recalib.groupby("condition")["model"]
        .nunique()
    )

    if not (counts == 3).all():
        bad = counts[counts != 3]
        raise RuntimeError(
            f"{path}: not every condition has 3 seeds:\n"
            f"{bad.to_string()}"
        )

    # Mean across the three independently trained seeds
    agg_dict = {}

    for metric in metrics:
        agg_dict[f"recalib_{metric}"] = (
            metric,
            "mean",
        )

        agg_dict[f"recalib_{metric}_seed_sd"] = (
            metric,
            "std",
        )

    recalib_mean = (
        recalib.groupby(
            "condition",
            as_index=False,
        )
        .agg(**agg_dict)
    )

    merged = frozen.merge(
        recalib_mean,
        on="condition",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 108:
        raise RuntimeError(
            f"{path}: expected 108 merged conditions, "
            f"found {len(merged)}"
        )

    return df, recalib, merged


def make_table(
    path,
    metric_specs,
    output_name,
):
    metrics = list(metric_specs.keys())

    df, recalib_seed_rows, merged = (
        prepare_condition_level(
            path,
            metrics,
        )
    )

    rows = []

    for metric, spec in metric_specs.items():

        higher_is_better = spec["higher_is_better"]
        display_name = spec["display_name"]

        frozen = merged[
            f"frozen_{metric}"
        ].to_numpy(dtype=float)

        recalib = merged[
            f"recalib_{metric}"
        ].to_numpy(dtype=float)

        valid = (
            np.isfinite(frozen)
            &
            np.isfinite(recalib)
        )

        frozen = frozen[valid]
        recalib = recalib[valid]

        if higher_is_better:
            advantage = recalib - frozen
        else:
            advantage = frozen - recalib

        ci_low, ci_high = bootstrap_mean_ci(
            advantage
        )

        raw_p = safe_wilcoxon(
            advantage
        )

        # Overall Frozen CRISP mean across 108 conditions
        frozen_mean = float(
            np.mean(frozen)
        )

        # Overall ReCalib mean across:
        # first mean each seed over 108 conditions,
        # then calculate mean ± sample SD across 3 seeds.
        seed_means = (
            recalib_seed_rows.groupby("model")[metric]
            .mean()
            .reindex(SEEDS)
        )

        recalib_mean = float(
            seed_means.mean()
        )

        recalib_seed_sd = float(
            seed_means.std(ddof=1)
        )

        rows.append(
            {
                "metric": display_name,
                "metric_key": metric,
                "frozen_crisp_mean": frozen_mean,
                "recalib_mean": recalib_mean,
                "recalib_sample_sd": recalib_seed_sd,
                "mean_paired_improvement": float(
                    np.mean(advantage)
                ),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "improved_fraction": float(
                    np.mean(advantage > 0)
                ),
                "wins": int(
                    np.sum(advantage > 0)
                ),
                "losses": int(
                    np.sum(advantage < 0)
                ),
                "ties": int(
                    np.sum(advantage == 0)
                ),
                "n_valid_conditions": int(
                    len(advantage)
                ),
                "wilcoxon_pvalue": raw_p,
            }
        )

    result = pd.DataFrame(rows)

    result["wilcoxon_pvalue_holm"] = holm_adjust(
        result["wilcoxon_pvalue"].to_numpy()
    )

    result["significant_holm_0.05"] = (
        result["wilcoxon_pvalue_holm"] < 0.05
    )

    out = OUT_DIR / output_name
    result.to_csv(out, index=False)

    return result, out


# =============================================================
# TABLE 1
# =============================================================

table1_specs = {
    "mse_de": {
        "display_name": "mse_de",
        "higher_is_better": False,
    },
    "pearson_de": {
        "display_name": "pearson_de",
        "higher_is_better": True,
    },
    "pearson_delta_de": {
        "display_name": "pearson_delta_de",
        "higher_is_better": True,
    },
    "r2score_de": {
        "display_name": "r2score_de",
        "higher_is_better": True,
    },
}

table1, table1_path = make_table(
    MAIN_FILE,
    table1_specs,
    "table1_main_ood.csv",
)


# =============================================================
# TABLE 2
# =============================================================

table2_specs = {
    "top10_recovery": {
        "display_name":
            "Top-10 response-gene recovery",
        "higher_is_better": True,
    },
    "top20_recovery": {
        "display_name":
            "Top-20 response-gene recovery",
        "higher_is_better": True,
    },
    "de_response_rank_spearman": {
        "display_name":
            "DE-gene response-ranking concordance",
        "higher_is_better": True,
    },
}

table2, table2_path = make_table(
    RESPONSE_FILE,
    table2_specs,
    "table2_response_gene.csv",
)


# =============================================================
# TABLE 3
# =============================================================

table3_specs = {
    "pathway_score_pearson": {
        "display_name":
            "Pathway-score Pearson",
        "higher_is_better": True,
    },
    "pathway_score_spearman": {
        "display_name":
            "Pathway-score Spearman",
        "higher_is_better": True,
    },
    "pathway_score_mse": {
        "display_name":
            "Pathway-score MSE",
        "higher_is_better": False,
    },
}

table3, table3_path = make_table(
    HALLMARK_FILE,
    table3_specs,
    "table3_hallmark.csv",
)


# =============================================================
# PRINT PAPER-READY RESULTS
# =============================================================

def print_table(title, df):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)

    cols = [
        "metric",
        "frozen_crisp_mean",
        "recalib_mean",
        "recalib_sample_sd",
        "mean_paired_improvement",
        "ci95_low",
        "ci95_high",
        "improved_fraction",
        "wins",
        "losses",
        "ties",
        "n_valid_conditions",
        "wilcoxon_pvalue",
        "wilcoxon_pvalue_holm",
        "significant_holm_0.05",
    ]

    print(
        df[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.9g}",
        )
    )


print_table(
    "MAIN TABLE 1 — DE-GENE METRICS",
    table1,
)

print_table(
    "MAIN TABLE 2 — RESPONSE-GENE METRICS",
    table2,
)

print_table(
    "MAIN TABLE 3 — HALLMARK METRICS",
    table3,
)


print()
print("=" * 120)
print("OUTPUT FILES")
print("=" * 120)
print(table1_path)
print(table2_path)
print(table3_path)

print()
print("[OK] Main Tables 1–3 statistics generated.")
