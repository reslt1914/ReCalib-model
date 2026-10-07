from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import recompute_final_paper_results as r


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
    / "paper_recalculation"
)

SHUFFLED_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ridge_anchored_recalib_shuffled"
)

OUT_DIR = (
    FINAL_ROOT
    / "supplementary_tables"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = [2024, 3407, 42]

FINAL_MODEL_NAMES = {
    2024: "recalib_seed_2024",
    3407: "recalib_seed_3407",
    42: "recalib_seed_42",
}

RESPONSE_METRICS = [
    "top10_recovery",
    "top20_recovery",
    "de_response_rank_spearman",
]

HALLMARK_METRICS = [
    "pathway_score_pearson",
    "pathway_score_spearman",
    "pathway_score_mse",
]

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 2024


# ============================================================
# Statistics
# ============================================================

def bootstrap_mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    indices = rng.integers(
        0,
        len(values),
        size=(
            BOOTSTRAP_REPS,
            len(values),
        ),
    )

    means = values[indices].mean(axis=1)

    lo, hi = np.percentile(
        means,
        [2.5, 97.5],
    )

    return float(lo), float(hi)


def paired_wilcoxon(values):
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


def holm_adjust(pvalues):
    pvalues = np.asarray(
        pvalues,
        dtype=float,
    )

    n = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]

    adjusted_sorted = np.empty(
        n,
        dtype=float,
    )

    running_max = 0.0

    for i, value in enumerate(sorted_p):
        candidate = (
            (n - i) * value
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted_sorted[i] = min(
            running_max,
            1.0,
        )

    adjusted = np.empty(
        n,
        dtype=float,
    )

    adjusted[order] = adjusted_sorted

    return adjusted


# ============================================================
# Load Main OOD truth / control
# ============================================================

TRUE_FILE = (
    BASELINE_OOD
    / "x_true_condition_mean.csv"
)

CTRL_FILE = (
    BASELINE_OOD
    / "x_ctrl_condition_mean.csv"
)

for path in [TRUE_FILE, CTRL_FILE]:
    if not path.is_file():
        raise RuntimeError(
            f"Missing required file: {path}"
        )

true = pd.read_csv(
    TRUE_FILE,
    index_col=0,
)

ctrl = pd.read_csv(
    CTRL_FILE,
    index_col=0,
)

true.index = true.index.astype(str)
ctrl.index = ctrl.index.astype(str)

true.columns = true.columns.astype(str)
ctrl.columns = ctrl.columns.astype(str)

if true.shape != (108, 977):
    raise RuntimeError(
        f"Unexpected truth shape: {true.shape}"
    )

if ctrl.shape != true.shape:
    raise RuntimeError(
        "Truth/control shape mismatch"
    )

if list(true.index) != list(ctrl.index):
    raise RuntimeError(
        "Truth/control condition order mismatch"
    )

if list(true.columns) != list(ctrl.columns):
    raise RuntimeError(
        "Truth/control gene order mismatch"
    )

print("=" * 100)
print("SUPPLEMENTARY TABLES 15-16 RECALCULATION")
print("=" * 100)
print(
    f"[OK] Main OOD: "
    f"{true.shape[0]} conditions x "
    f"{true.shape[1]} genes"
)


# ============================================================
# Official DE mapping
# ============================================================

de_mapping = r.load_de_mapping(
    true.index.tolist(),
    true.columns.tolist(),
)

if len(de_mapping) != 108:
    raise RuntimeError(
        f"Expected DE mapping for 108 conditions, "
        f"found {len(de_mapping)}"
    )

mapped_counts = [
    len(de_mapping[c])
    for c in true.index
]

print(
    "[OK] DE mapping:",
    len(de_mapping),
    "conditions"
)

print(
    "[OK] DE mapped genes per condition:",
    min(mapped_counts),
    "to",
    max(mapped_counts),
)


# ============================================================
# Hallmark mapping
# ============================================================

hallmark_path = r.find_hallmark_gmt()

if hallmark_path is None:
    raise RuntimeError(
        "Hallmark GMT file not found."
    )

gene_sets = r.read_gmt(
    hallmark_path,
    true.columns.tolist(),
)

if len(gene_sets) != 49:
    raise RuntimeError(
        "Expected 49 retained Hallmark sets, "
        f"found {len(gene_sets)}"
    )

print(
    "[OK] Hallmark sets:",
    len(gene_sets)
)


# ============================================================
# Existing FINAL condition-level metrics
# ============================================================

FINAL_RESPONSE_FILE = (
    FINAL_ROOT
    / "response_gene_condition_metrics.csv"
)

FINAL_HALLMARK_FILE = (
    FINAL_ROOT
    / "hallmark_condition_metrics.csv"
)

for path in [
    FINAL_RESPONSE_FILE,
    FINAL_HALLMARK_FILE,
]:
    if not path.is_file():
        raise RuntimeError(
            f"Missing final metric file: {path}"
        )

final_response_all = pd.read_csv(
    FINAL_RESPONSE_FILE
)

final_hallmark_all = pd.read_csv(
    FINAL_HALLMARK_FILE
)


# ============================================================
# Extract final Frozen CRISP + 3 seed results
# ============================================================

def extract_existing_metrics(
    df,
    metrics,
):
    required = {
        "model",
        "condition",
        *metrics,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    frozen = (
        df[
            df["model"].astype(str)
            == "frozen_crisp"
        ]
        [
            ["condition"] + metrics
        ]
        .copy()
    )

    if len(frozen) != 108:
        raise RuntimeError(
            "Frozen CRISP does not contain "
            "exactly 108 conditions."
        )

    frozen["condition"] = (
        frozen["condition"]
        .astype(str)
    )

    final_seed_frames = []

    for seed in SEEDS:
        model_name = (
            FINAL_MODEL_NAMES[seed]
        )

        part = (
            df[
                df["model"].astype(str)
                == model_name
            ]
            [
                ["condition"] + metrics
            ]
            .copy()
        )

        if len(part) != 108:
            raise RuntimeError(
                f"{model_name}: expected 108 rows, "
                f"found {len(part)}"
            )

        part["condition"] = (
            part["condition"].astype(str)
        )

        part["seed"] = seed

        final_seed_frames.append(part)

    final_seed = pd.concat(
        final_seed_frames,
        ignore_index=True,
    )

    return frozen, final_seed


(
    frozen_response,
    final_response_seed,
) = extract_existing_metrics(
    final_response_all,
    RESPONSE_METRICS,
)

(
    frozen_hallmark,
    final_hallmark_seed,
) = extract_existing_metrics(
    final_hallmark_all,
    HALLMARK_METRICS,
)

print(
    "[OK] Loaded existing final "
    "response-gene metrics"
)

print(
    "[OK] Loaded existing final "
    "Hallmark metrics"
)


# ============================================================
# Recalculate SHUFFLED metrics with EXACT final-paper functions
# ============================================================

shuffled_response_frames = []
shuffled_hallmark_frames = []

for seed in SEEDS:

    pred_file = (
        SHUFFLED_ROOT
        / f"ridge_anchored_recalib_seed_{seed}"
        / "x_calibrated_condition_mean.csv"
    )

    if not pred_file.is_file():
        raise RuntimeError(
            f"Missing shuffled prediction: {pred_file}"
        )

    pred = pd.read_csv(
        pred_file,
        index_col=0,
    )

    pred.index = pred.index.astype(str)
    pred.columns = pred.columns.astype(str)

    if set(pred.index) != set(true.index):
        raise RuntimeError(
            f"Shuffled seed {seed}: "
            "condition set mismatch"
        )

    if set(pred.columns) != set(true.columns):
        raise RuntimeError(
            f"Shuffled seed {seed}: "
            "gene set mismatch"
        )

    pred = pred.reindex(
        index=true.index,
        columns=true.columns,
    )

    if pred.isna().any().any():
        raise RuntimeError(
            f"Shuffled seed {seed}: "
            "NaN introduced during alignment"
        )

    # --------------------------------------------------------
    # EXACT response-gene implementation
    # --------------------------------------------------------

    response = r.response_gene_metrics(
        pred,
        true,
        ctrl,
        de_mapping,
    )

    response.insert(
        0,
        "seed",
        seed,
    )

    shuffled_response_frames.append(
        response
    )

    # --------------------------------------------------------
    # EXACT Hallmark implementation
    # --------------------------------------------------------

    hallmark = r.pathway_metrics(
        pred,
        true,
        ctrl,
        gene_sets,
    )

    hallmark.insert(
        0,
        "seed",
        seed,
    )

    shuffled_hallmark_frames.append(
        hallmark
    )

    print(
        f"[OK] shuffled seed {seed}: "
        "response-gene + Hallmark recalculated"
    )


shuffled_response_seed = pd.concat(
    shuffled_response_frames,
    ignore_index=True,
)

shuffled_hallmark_seed = pd.concat(
    shuffled_hallmark_frames,
    ignore_index=True,
)


# Save raw recalculated shuffled metrics

shuffled_response_seed.to_csv(
    OUT_DIR
    / "table15_shuffled_response_gene_by_seed_condition.csv",
    index=False,
)

shuffled_hallmark_seed.to_csv(
    OUT_DIR
    / "table16_shuffled_hallmark_by_seed_condition.csv",
    index=False,
)


# ============================================================
# Generic final-vs-shuffled table builder
# ============================================================

def build_table(
    *,
    frozen_df,
    final_seed_df,
    shuffled_seed_df,
    metrics,
    lower_is_better,
    output_prefix,
):

    # --------------------------------------------------------
    # Seed-level global means
    # --------------------------------------------------------

    final_seed_means = (
        final_seed_df
        .groupby("seed")[metrics]
        .mean()
        .reindex(SEEDS)
    )

    shuffled_seed_means = (
        shuffled_seed_df
        .groupby("seed")[metrics]
        .mean()
        .reindex(SEEDS)
    )

    print()
    print("=" * 105)
    print(
        output_prefix.upper(),
        "- SEED LEVEL MEANS"
    )
    print("=" * 105)

    print("\nFinal ReCalib:")
    print(
        final_seed_means.to_string(
            float_format=lambda x: f"{x:.9f}"
        )
    )

    print("\nShuffled control:")
    print(
        shuffled_seed_means.to_string(
            float_format=lambda x: f"{x:.9f}"
        )
    )

    # --------------------------------------------------------
    # Average 3 seeds within EACH CONDITION
    # --------------------------------------------------------

    final_condition = (
        final_seed_df
        .groupby(
            "condition",
            as_index=False,
        )[metrics]
        .mean()
    )

    shuffled_condition = (
        shuffled_seed_df
        .groupby(
            "condition",
            as_index=False,
        )[metrics]
        .mean()
    )

    final_condition = (
        final_condition.rename(
            columns={
                metric:
                    f"final_{metric}"
                for metric in metrics
            }
        )
    )

    shuffled_condition = (
        shuffled_condition.rename(
            columns={
                metric:
                    f"shuffled_{metric}"
                for metric in metrics
            }
        )
    )

    paired = final_condition.merge(
        shuffled_condition,
        on="condition",
        how="inner",
        validate="one_to_one",
    )

    if len(paired) != 108:
        raise RuntimeError(
            f"{output_prefix}: "
            "expected 108 paired conditions, "
            f"found {len(paired)}"
        )

    frozen_mean = {
        metric: float(
            frozen_df[metric].mean()
        )
        for metric in metrics
    }

    rows = []

    for metric in metrics:

        final_values = paired[
            f"final_{metric}"
        ].to_numpy(dtype=float)

        shuffled_values = paired[
            f"shuffled_{metric}"
        ].to_numpy(dtype=float)

        valid = (
            np.isfinite(final_values)
            &
            np.isfinite(shuffled_values)
        )

        final_values = final_values[valid]
        shuffled_values = (
            shuffled_values[valid]
        )

        if metric in lower_is_better:
            advantage = (
                shuffled_values
                - final_values
            )
        else:
            advantage = (
                final_values
                - shuffled_values
            )

        ci_low, ci_high = (
            bootstrap_mean_ci(
                advantage
            )
        )

        raw_p = paired_wilcoxon(
            advantage
        )

        final_seed_metric = (
            final_seed_means[metric]
            .to_numpy(dtype=float)
        )

        shuffled_seed_metric = (
            shuffled_seed_means[metric]
            .to_numpy(dtype=float)
        )

        rows.append(
            {
                "metric":
                    metric,

                "frozen_crisp_mean":
                    frozen_mean[metric],

                "final_recalib_mean":
                    float(
                        np.mean(
                            final_seed_metric
                        )
                    ),

                "final_recalib_sample_sd":
                    float(
                        np.std(
                            final_seed_metric,
                            ddof=1,
                        )
                    ),

                "shuffled_mean":
                    float(
                        np.mean(
                            shuffled_seed_metric
                        )
                    ),

                "shuffled_sample_sd":
                    float(
                        np.std(
                            shuffled_seed_metric,
                            ddof=1,
                        )
                    ),

                "mean_paired_advantage":
                    float(
                        np.mean(
                            advantage
                        )
                    ),

                "ci95_low":
                    ci_low,

                "ci95_high":
                    ci_high,

                "final_win_fraction":
                    float(
                        np.mean(
                            advantage > 0
                        )
                    ),

                "final_wins":
                    int(
                        np.sum(
                            advantage > 0
                        )
                    ),

                "final_losses":
                    int(
                        np.sum(
                            advantage < 0
                        )
                    ),

                "ties":
                    int(
                        np.sum(
                            advantage == 0
                        )
                    ),

                "n_valid_conditions":
                    int(
                        len(advantage)
                    ),

                "wilcoxon_pvalue":
                    raw_p,
            }
        )

    result = pd.DataFrame(rows)

    result[
        "wilcoxon_pvalue_holm"
    ] = holm_adjust(
        result[
            "wilcoxon_pvalue"
        ].to_numpy(dtype=float)
    )

    result[
        "significant_holm_0.05"
    ] = (
        result[
            "wilcoxon_pvalue_holm"
        ]
        < 0.05
    )

    result.to_csv(
        OUT_DIR
        / f"{output_prefix}.csv",
        index=False,
    )

    paired.to_csv(
        OUT_DIR
        / f"{output_prefix}_condition_pairs.csv",
        index=False,
    )

    return (
        result,
        final_seed_means,
        shuffled_seed_means,
    )


# ============================================================
# Supplementary Table 15
# ============================================================

(
    table15,
    final_response_seed_means,
    shuffled_response_seed_means,
) = build_table(
    frozen_df=frozen_response,
    final_seed_df=final_response_seed,
    shuffled_seed_df=shuffled_response_seed,
    metrics=RESPONSE_METRICS,
    lower_is_better=set(),
    output_prefix="table15_response_gene_shuffled",
)


# ============================================================
# Supplementary Table 16
# ============================================================

(
    table16,
    final_hallmark_seed_means,
    shuffled_hallmark_seed_means,
) = build_table(
    frozen_df=frozen_hallmark,
    final_seed_df=final_hallmark_seed,
    shuffled_seed_df=shuffled_hallmark_seed,
    metrics=HALLMARK_METRICS,
    lower_is_better={
        "pathway_score_mse",
    },
    output_prefix="table16_hallmark_shuffled",
)


# ============================================================
# Critical consistency checks against locked FINAL results
# ============================================================

expected_response = {
    "top10_recovery": (
        0.098148,
        0.234259,
    ),
    "top20_recovery": (
        0.118981,
        0.286883,
    ),
    "de_response_rank_spearman": (
        0.516592,
        0.527816,
    ),
}

expected_hallmark = {
    "pathway_score_pearson": (
        0.224451,
        0.441727,
    ),
    "pathway_score_spearman": (
        0.253714,
        0.426759,
    ),
    "pathway_score_mse": (
        0.002544,
        0.000363,
    ),
}

print()
print("=" * 105)
print("LOCKED FINAL-RESULT CONSISTENCY CHECK")
print("=" * 105)

for result, expected in [
    (table15, expected_response),
    (table16, expected_hallmark),
]:

    for _, row in result.iterrows():

        metric = row["metric"]

        exp_frozen, exp_final = (
            expected[metric]
        )

        diff_frozen = abs(
            row["frozen_crisp_mean"]
            - exp_frozen
        )

        diff_final = abs(
            row["final_recalib_mean"]
            - exp_final
        )

        print(
            f"{metric:32s} "
            f"frozen_diff={diff_frozen:.8f} "
            f"final_diff={diff_final:.8f}"
        )

        if (
            diff_frozen > 5e-6
            or
            diff_final > 5e-6
        ):
            raise RuntimeError(
                f"{metric}: locked final "
                "result was not reproduced."
            )


# ============================================================
# Paper-ready print
# ============================================================

DISPLAY_COLUMNS = [
    "metric",
    "frozen_crisp_mean",
    "final_recalib_mean",
    "final_recalib_sample_sd",
    "shuffled_mean",
    "shuffled_sample_sd",
    "mean_paired_advantage",
    "ci95_low",
    "ci95_high",
    "final_win_fraction",
    "final_wins",
    "final_losses",
    "ties",
    "n_valid_conditions",
    "wilcoxon_pvalue",
    "wilcoxon_pvalue_holm",
    "significant_holm_0.05",
]

print()
print("=" * 125)
print(
    "SUPPLEMENTARY TABLE 15 "
    "— RESPONSE-GENE SHUFFLED CONTROL"
)
print("=" * 125)

print(
    table15[
        DISPLAY_COLUMNS
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.9g}",
    )
)

print()
print("=" * 125)
print(
    "SUPPLEMENTARY TABLE 16 "
    "— HALLMARK SHUFFLED CONTROL"
)
print("=" * 125)

print(
    table16[
        DISPLAY_COLUMNS
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.9g}",
    )
)

print()
print("[OK] Supplementary Tables 15-16 generated.")
print("[OUTPUT]", OUT_DIR)
