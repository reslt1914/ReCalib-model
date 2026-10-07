from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# Paths
# ============================================================

ROOT = Path("results")

FINAL_FILE = (
    ROOT
    / "paper_recalculation"
    / "main_ood_condition_metrics.csv"
)

SHUFFLED_ROOT = (
    ROOT
    / "ridge_anchored_recalib_shuffled"
)

OUT_DIR = (
    ROOT
    / "paper_recalculation"
    / "main_tables"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL_MODELS = {
    2024: "recalib_seed_2024",
    3407: "recalib_seed_3407",
    42: "recalib_seed_42",
}

SEEDS = [2024, 3407, 42]

METRICS = [
    "mse_de",
    "pearson_de",
    "pearson_delta_de",
    "r2score_de",
]

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 2024


# ============================================================
# Utility functions
# ============================================================

def holm_adjust(pvalues):
    """
    Holm step-down adjusted P values.
    """
    p = np.asarray(
        pvalues,
        dtype=float,
    )

    m = len(p)

    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for rank, value in enumerate(sorted_p):

        adjusted = (
            (m - rank)
            * value
        )

        running_max = max(
            running_max,
            adjusted,
        )

        adjusted_sorted[rank] = min(
            running_max,
            1.0,
        )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    adjusted[order] = adjusted_sorted

    return adjusted


def bootstrap_mean_ci(
    values,
    n_boot=BOOTSTRAP_REPS,
    seed=BOOTSTRAP_SEED,
):
    """
    Percentile bootstrap CI of mean paired advantage.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(
        seed
    )

    indices = rng.integers(
        0,
        len(values),
        size=(
            n_boot,
            len(values),
        ),
    )

    boot_means = (
        values[indices]
        .mean(axis=1)
    )

    low, high = np.percentile(
        boot_means,
        [2.5, 97.5],
    )

    return (
        float(low),
        float(high),
    )


def safe_wilcoxon(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    if np.allclose(
        values,
        0,
    ):
        return 1.0

    try:

        result = wilcoxon(
            values,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        return float(
            result.pvalue
        )

    except ValueError:

        return 1.0


# ============================================================
# Load final ReCalib
# ============================================================

print("=" * 110)
print("TABLE 4 — FINAL RECALIB VS SHUFFLED CONTROL")
print("=" * 110)

if not FINAL_FILE.is_file():
    raise RuntimeError(
        f"Missing final file: {FINAL_FILE}"
    )

final_all = pd.read_csv(
    FINAL_FILE
)

required = {
    "model",
    "condition",
    *METRICS,
}

missing = (
    required
    - set(final_all.columns)
)

if missing:
    raise RuntimeError(
        "Final file missing columns: "
        f"{sorted(missing)}"
    )

final_seed_frames = []

for seed in SEEDS:

    model_name = (
        FINAL_MODELS[seed]
    )

    d = (
        final_all[
            final_all["model"]
            .astype(str)
            == model_name
        ]
        [
            ["condition"]
            + METRICS
        ]
        .copy()
    )

    if len(d) != 108:
        raise RuntimeError(
            f"Final seed {seed}: "
            f"expected 108 conditions, "
            f"found {len(d)}"
        )

    if d["condition"].duplicated().any():
        raise RuntimeError(
            f"Final seed {seed}: "
            "duplicated conditions"
        )

    d["seed"] = seed

    final_seed_frames.append(d)


final_seed = pd.concat(
    final_seed_frames,
    ignore_index=True,
)

print(
    "[OK] Final ReCalib:",
    len(final_seed),
    "rows"
)


# ============================================================
# Load shuffled control
# ============================================================

shuffled_seed_frames = []

for seed in SEEDS:

    path = (
        SHUFFLED_ROOT
        / f"ridge_anchored_recalib_seed_{seed}"
        / "main_ood_condition_metrics.csv"
    )

    if not path.is_file():
        raise RuntimeError(
            f"Missing shuffled file: {path}"
        )

    d = pd.read_csv(path)

    missing = (
        {
            "condition",
            *METRICS,
        }
        - set(d.columns)
    )

    if missing:
        raise RuntimeError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    # Some result files may contain only one method;
    # some may include a model column.
    # We only require exactly one row per Main OOD condition.
    d = d[
        ["condition"] + METRICS
    ].copy()

    if len(d) != 108:
        raise RuntimeError(
            f"Shuffled seed {seed}: "
            f"expected 108 rows, "
            f"found {len(d)}"
        )

    if d["condition"].duplicated().any():
        raise RuntimeError(
            f"Shuffled seed {seed}: "
            "duplicated conditions"
        )

    d["seed"] = seed

    shuffled_seed_frames.append(d)

    print(
        f"[OK] Shuffled seed {seed}: "
        f"108 conditions"
    )


shuffled_seed = pd.concat(
    shuffled_seed_frames,
    ignore_index=True,
)


# ============================================================
# Exact condition-set checks
# ============================================================

for seed in SEEDS:

    f = set(
        final_seed[
            final_seed["seed"] == seed
        ]["condition"].astype(str)
    )

    s = set(
        shuffled_seed[
            shuffled_seed["seed"] == seed
        ]["condition"].astype(str)
    )

    if f != s:
        raise RuntimeError(
            f"Condition mismatch for seed {seed}.\n"
            f"Final only: {sorted(f-s)[:10]}\n"
            f"Shuffled only: {sorted(s-f)[:10]}"
        )

print(
    "[OK] Exact Main OOD condition sets "
    "match for all three seeds."
)


# ============================================================
# Seed-level overall means
# ============================================================

final_seed_means = (
    final_seed
    .groupby("seed")[METRICS]
    .mean()
    .reindex(SEEDS)
)

shuffled_seed_means = (
    shuffled_seed
    .groupby("seed")[METRICS]
    .mean()
    .reindex(SEEDS)
)

print()
print("=" * 110)
print("SEED-LEVEL OVERALL MEANS")
print("=" * 110)

print()
print("Final ReCalib:")
print(
    final_seed_means.to_string(
        float_format=lambda x: f"{x:.9f}"
    )
)

print()
print("Shuffled control:")
print(
    shuffled_seed_means.to_string(
        float_format=lambda x: f"{x:.9f}"
    )
)


# ============================================================
# Three-seed condition means
# ============================================================

final_condition = (
    final_seed
    .groupby(
        "condition",
        as_index=False,
    )[METRICS]
    .mean()
)

shuffled_condition = (
    shuffled_seed
    .groupby(
        "condition",
        as_index=False,
    )[METRICS]
    .mean()
)

final_condition = (
    final_condition.rename(
        columns={
            m: f"final_{m}"
            for m in METRICS
        }
    )
)

shuffled_condition = (
    shuffled_condition.rename(
        columns={
            m: f"shuffled_{m}"
            for m in METRICS
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
        "Expected 108 paired Main OOD "
        f"conditions, found {len(paired)}"
    )

paired.to_csv(
    OUT_DIR
    / "table4_condition_level_pairs.csv",
    index=False,
)


# ============================================================
# Table 4 statistics
# ============================================================

rows = []

for metric in METRICS:

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

    final_values = (
        final_values[valid]
    )

    shuffled_values = (
        shuffled_values[valid]
    )

    # Positive advantage ALWAYS means Final ReCalib is better.
    if metric == "mse_de":

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

    raw_p = safe_wilcoxon(
        advantage
    )

    final_means = (
        final_seed_means[metric]
        .to_numpy(dtype=float)
    )

    shuffled_means = (
        shuffled_seed_means[metric]
        .to_numpy(dtype=float)
    )

    row = {
        "metric": metric,

        "final_recalib_mean":
            float(
                np.mean(final_means)
            ),

        "final_recalib_sample_sd":
            float(
                np.std(
                    final_means,
                    ddof=1,
                )
            ),

        "shuffled_mean":
            float(
                np.mean(
                    shuffled_means
                )
            ),

        "shuffled_sample_sd":
            float(
                np.std(
                    shuffled_means,
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

    rows.append(row)


result = pd.DataFrame(
    rows
)

result[
    "wilcoxon_pvalue_holm"
] = holm_adjust(
    result[
        "wilcoxon_pvalue"
    ].to_numpy()
)

result[
    "significant_holm_0.05"
] = (
    result[
        "wilcoxon_pvalue_holm"
    ]
    < 0.05
)


# ============================================================
# Save
# ============================================================

output_path = (
    OUT_DIR
    / "table4_final_vs_shuffled.csv"
)

result.to_csv(
    output_path,
    index=False,
)


# ============================================================
# Print paper-ready output
# ============================================================

print()
print("=" * 125)
print("MAIN TABLE 4 — FINAL RECALIB VS SHUFFLED CONTROL")
print("=" * 125)

columns = [
    "metric",

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

print(
    result[
        columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.9g}",
    )
)


print()
print("=" * 125)
print("CONSISTENCY CHECK AGAINST PREVIOUS SUMMARY")
print("=" * 125)

expected = {
    "mse_de": {
        "final": 0.016035,
        "shuffled": 0.017730,
    },

    "pearson_de": {
        "final": 0.885316,
        "shuffled": 0.876951,
    },

    "pearson_delta_de": {
        "final": 0.474411,
        "shuffled": 0.441918,
    },

    "r2score_de": {
        "final": 0.672964,
        "shuffled": 0.644320,
    },
}

for _, row in result.iterrows():

    metric = row["metric"]

    exp = expected[metric]

    d_final = abs(
        row["final_recalib_mean"]
        - exp["final"]
    )

    d_shuffle = abs(
        row["shuffled_mean"]
        - exp["shuffled"]
    )

    print(
        f"{metric:18s} "
        f"final_diff={d_final:.8f} "
        f"shuffled_diff={d_shuffle:.8f}"
    )

    if (
        d_final > 5e-6
        or d_shuffle > 5e-6
    ):
        raise RuntimeError(
            f"{metric}: overall mean does not "
            "reproduce the previous final/shuffled "
            "summary."
        )


print()
print("[OK] Table 4 statistics generated.")
print("[OUTPUT]", output_path)
print(
    "[OUTPUT]",
    OUT_DIR
    / "table4_condition_level_pairs.csv"
)
