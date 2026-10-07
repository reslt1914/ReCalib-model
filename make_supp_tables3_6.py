from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# Configuration
# ============================================================

ROOT = Path(".")
OUT = Path("results/paper_recalculation/supplementary_tables")
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [2024, 3407, 42]
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 2024

METRICS = [
    "mse_de",
    "pearson_de",
    "pearson_delta_de",
    "r2score_de",
]

DISPLAY = {
    "mse_de": "MSE-DE",
    "pearson_de": "Pearson-DE",
    "pearson_delta_de": "Pearson-Δ-DE",
    "r2score_de": "R²-DE",
}

DIRECTION = {
    "mse_de": "lower",
    "pearson_de": "higher",
    "pearson_delta_de": "higher",
    "r2score_de": "higher",
}

# ------------------------------------------------------------
# Main OOD
# ------------------------------------------------------------

MAIN_FINAL_CONDITION_METRICS = Path(
    "results/paper_recalculation/"
    "main_ood_condition_metrics.csv"
)

MAIN_SHUFFLED_ROOT = Path(
    "results/ridge_anchored_recalib_shuffled"
)

# ------------------------------------------------------------
# Alternative / pathway-held-out OOD
# ------------------------------------------------------------

ALT_BASELINE_METRICS = Path(
    "crisp_outputs/"
    "mydata_ho_pathway_baseline_full/"
    "ood/"
    "metrics_by_condition.csv"
)

ALT_FINAL_ROOT = Path(
    "results/ridge_anchored_recalib_ho_pathway"
)

ALT_SHUFFLED_ROOT = Path(
    "results/ridge_anchored_recalib_ho_pathway_shuffled"
)


# ============================================================
# Utilities
# ============================================================

def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_dir(path: Path):
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def standardize_condition_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "condition" not in df.columns:
        for candidate in [
            "Condition",
            "condition_name",
            "perturbation",
        ]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "condition"})
                break

    if "condition" not in df.columns:
        raise RuntimeError(
            "Cannot identify condition column. "
            f"Columns: {df.columns.tolist()}"
        )

    df["condition"] = df["condition"].astype(str)

    return df


def validate_metric_columns(df: pd.DataFrame, source: str):
    missing = set(METRICS) - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{source}: missing metric columns "
            f"{sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )


def detect_seed_from_path(path: Path) -> int:
    text = str(path)

    patterns = [
        r"seed[_-]?(\d+)",
        r"seed=(\d+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text)

        if m:
            seed = int(m.group(1))

            if seed in SEEDS:
                return seed

    raise RuntimeError(
        f"Cannot determine seed from path: {path}"
    )


def find_seed_condition_metrics(root: Path):
    """
    Locate one main_ood_condition_metrics.csv for each required seed.
    """

    require_dir(root)

    candidates = sorted(
        root.glob("**/main_ood_condition_metrics.csv")
    )

    if not candidates:
        # Fallback for slightly different historical naming.
        candidates = sorted(
            root.glob("**/metrics_by_condition.csv")
        )

    by_seed = {}

    for path in candidates:
        try:
            seed = detect_seed_from_path(path)
        except RuntimeError:
            continue

        if seed in by_seed:
            raise RuntimeError(
                f"Multiple condition-metric files detected "
                f"for seed {seed} under {root}:\n"
                f"  {by_seed[seed]}\n"
                f"  {path}"
            )

        by_seed[seed] = path

    missing = [
        seed for seed in SEEDS
        if seed not in by_seed
    ]

    if missing:
        print()
        print("[DEBUG] Candidate files under", root)

        for path in candidates:
            print(" ", path)

        raise RuntimeError(
            f"Missing seed condition metrics under {root}: "
            f"{missing}"
        )

    return by_seed


def read_one_seed_metrics(path: Path, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = standardize_condition_column(df)
    validate_metric_columns(df, str(path))

    # Some files may contain more than one model.
    if "model" in df.columns:
        models = df["model"].astype(str).unique().tolist()

        seed_matches = [
            x for x in models
            if str(seed) in x
        ]

        if len(seed_matches) == 1:
            df = df[
                df["model"].astype(str)
                == seed_matches[0]
            ].copy()

    if df["condition"].duplicated().any():
        duplicates = (
            df.loc[
                df["condition"].duplicated(),
                "condition"
            ]
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            f"{path}: duplicated conditions found: "
            f"{duplicates}"
        )

    return (
        df[
            ["condition"] + METRICS
        ]
        .set_index("condition")
        .sort_index()
    )


def load_seed_metrics(root: Path):
    files = find_seed_condition_metrics(root)

    result = {}

    for seed in SEEDS:
        result[seed] = read_one_seed_metrics(
            files[seed],
            seed,
        )

        print(
            f"[LOAD] seed={seed}: "
            f"{files[seed]} "
            f"({len(result[seed])} conditions)"
        )

    # Exact condition-set check
    reference = set(result[SEEDS[0]].index)

    for seed in SEEDS[1:]:
        current = set(result[seed].index)

        if current != reference:
            raise RuntimeError(
                f"Condition mismatch between seeds "
                f"{SEEDS[0]} and {seed}: "
                f"reference={len(reference)}, "
                f"current={len(current)}, "
                f"only_reference={len(reference-current)}, "
                f"only_current={len(current-reference)}"
            )

    return result


def aggregate_seed_metrics(seed_dfs):
    """
    Returns:
      condition_mean:
          mean prediction metric across the 3 nonlinear seeds
          for every condition.

      seed_overall:
          one overall condition-mean value per seed.

      mean_over_seeds:
          mean of the three seed-level overall values.

      sample_sd:
          sample SD across the three seed-level overall values.
    """

    conditions = seed_dfs[SEEDS[0]].index

    for seed in SEEDS[1:]:
        if not seed_dfs[seed].index.equals(conditions):
            seed_dfs[seed] = (
                seed_dfs[seed]
                .reindex(conditions)
            )

    condition_mean = pd.DataFrame(
        index=conditions
    )

    seed_overall_rows = []

    for metric in METRICS:
        matrix = pd.concat(
            [
                seed_dfs[seed][metric].rename(str(seed))
                for seed in SEEDS
            ],
            axis=1,
        )

        condition_mean[metric] = matrix.mean(axis=1)

        for seed in SEEDS:
            seed_overall_rows.append({
                "seed": seed,
                "metric": metric,
                "mean": float(
                    seed_dfs[seed][metric].mean()
                ),
            })

    seed_overall = pd.DataFrame(seed_overall_rows)

    summary_rows = []

    for metric in METRICS:
        vals = (
            seed_overall.loc[
                seed_overall["metric"] == metric,
                "mean",
            ]
            .to_numpy(dtype=float)
        )

        summary_rows.append({
            "metric": metric,
            "mean": float(vals.mean()),
            "sample_sd": float(
                vals.std(ddof=1)
            ),
            "min_seed": float(vals.min()),
            "max_seed": float(vals.max()),
            "n_seeds": len(vals),
        })

    seed_summary = (
        pd.DataFrame(summary_rows)
        .set_index("metric")
    )

    return (
        condition_mean,
        seed_overall,
        seed_summary,
    )


def load_alt_frozen_metrics(path: Path):
    require_file(path)

    df = pd.read_csv(path)
    df = standardize_condition_column(df)
    validate_metric_columns(df, str(path))

    # If method/model is present, select CRISP/Frozen CRISP.
    method_col = None

    for candidate in [
        "model",
        "method",
    ]:
        if candidate in df.columns:
            method_col = candidate
            break

    if method_col is not None:
        labels = (
            df[method_col]
            .astype(str)
            .unique()
            .tolist()
        )

        frozen_candidates = [
            x for x in labels
            if (
                "crisp" in x.lower()
                and "recalib" not in x.lower()
            )
        ]

        if len(frozen_candidates) == 1:
            df = df[
                df[method_col].astype(str)
                == frozen_candidates[0]
            ].copy()

        elif len(labels) > 1:
            raise RuntimeError(
                f"Cannot uniquely identify Frozen CRISP "
                f"in {path}. Labels: {labels}"
            )

    if df["condition"].duplicated().any():
        raise RuntimeError(
            f"{path}: duplicated condition rows."
        )

    result = (
        df[
            ["condition"] + METRICS
        ]
        .set_index("condition")
        .sort_index()
    )

    return result


def load_main_frozen_and_final(path: Path):
    """
    Uses the already locked paper-recalculation table generated
    for Main OOD.
    """

    require_file(path)

    df = pd.read_csv(path)
    df = standardize_condition_column(df)
    validate_metric_columns(df, str(path))

    if "model" not in df.columns:
        raise RuntimeError(
            f"{path}: expected model column."
        )

    models = df["model"].astype(str).unique()

    print("[MAIN] models found:")
    for model in models:
        print(" ", model)

    frozen = df[
        df["model"].astype(str)
        == "frozen_crisp"
    ].copy()

    if len(frozen) == 0:
        candidates = [
            x for x in models
            if (
                "frozen" in x.lower()
                and "crisp" in x.lower()
            )
        ]

        if len(candidates) != 1:
            raise RuntimeError(
                "Cannot uniquely locate Main frozen_crisp."
            )

        frozen = df[
            df["model"].astype(str)
            == candidates[0]
        ].copy()

    frozen = (
        frozen[
            ["condition"] + METRICS
        ]
        .set_index("condition")
        .sort_index()
    )

    final_seed_dfs = {}

    for seed in SEEDS:
        expected = f"recalib_seed_{seed}"

        subset = df[
            df["model"].astype(str)
            == expected
        ].copy()

        if len(subset) == 0:
            candidates = [
                x for x in models
                if (
                    str(seed) in x
                    and "recalib" in x.lower()
                )
            ]

            if len(candidates) != 1:
                raise RuntimeError(
                    f"Cannot uniquely locate Main ReCalib "
                    f"seed {seed}. Candidates: {candidates}"
                )

            subset = df[
                df["model"].astype(str)
                == candidates[0]
            ].copy()

        final_seed_dfs[seed] = (
            subset[
                ["condition"] + METRICS
            ]
            .set_index("condition")
            .sort_index()
        )

    return frozen, final_seed_dfs


def exact_condition_check(
    name_a,
    a,
    name_b,
    b,
    expected_n=None,
):
    A = set(a.index)
    B = set(b.index)

    print(
        f"[CHECK] {name_a} vs {name_b}: "
        f"{len(A)} vs {len(B)}"
    )

    if A != B:
        raise RuntimeError(
            f"Condition mismatch: {name_a} vs {name_b}. "
            f"Only {name_a}: {len(A-B)}; "
            f"only {name_b}: {len(B-A)}."
        )

    if expected_n is not None:
        if len(A) != expected_n:
            raise RuntimeError(
                f"{name_a}: expected {expected_n} "
                f"conditions, found {len(A)}."
            )

    print(
        f"[OK] Exact condition match: {len(A)}"
    )


def paired_advantage(
    reference,
    final,
    metric,
):
    """
    Positive = advantage for final ReCalib.

    MSE:
        reference - final

    Correlation/R2:
        final - reference
    """

    if DIRECTION[metric] == "lower":
        return (
            reference[metric]
            - final[metric]
        )

    return (
        final[metric]
        - reference[metric]
    )


def bootstrap_mean_ci(
    values,
    n_bootstrap=N_BOOTSTRAP,
    seed=BOOTSTRAP_SEED,
):
    x = np.asarray(values, dtype=float)

    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    n = len(x)

    draws = rng.integers(
        0,
        n,
        size=(n_bootstrap, n),
    )

    boot_means = x[draws].mean(axis=1)

    return (
        float(np.quantile(boot_means, 0.025)),
        float(np.quantile(boot_means, 0.975)),
    )


def safe_wilcoxon(values):
    x = np.asarray(values, dtype=float)

    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    if np.allclose(x, 0.0):
        return 0.0, 1.0

    result = wilcoxon(
        x,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)

    adjusted = np.full(
        p.shape,
        np.nan,
        dtype=float,
    )

    valid_idx = np.where(
        np.isfinite(p)
    )[0]

    if len(valid_idx) == 0:
        return adjusted

    pv = p[valid_idx]

    order = np.argsort(pv)
    ordered_p = pv[order]

    m = len(ordered_p)

    ordered_adj = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for i, value in enumerate(ordered_p):
        raw = (m - i) * value
        running_max = max(
            running_max,
            raw,
        )

        ordered_adj[i] = min(
            running_max,
            1.0,
        )

    inverse = np.empty_like(order)
    inverse[order] = np.arange(m)

    adjusted_valid = (
        ordered_adj[inverse]
    )

    adjusted[valid_idx] = adjusted_valid

    return adjusted


def paired_stats(
    reference,
    final,
    comparison_name,
    expected_n,
):
    exact_condition_check(
        f"{comparison_name}: reference",
        reference,
        f"{comparison_name}: final",
        final,
        expected_n,
    )

    final = final.reindex(reference.index)

    rows = []
    pair_frames = []

    for metric_i, metric in enumerate(METRICS):
        advantage = paired_advantage(
            reference,
            final,
            metric,
        )

        advantage = advantage.astype(float)

        ci_low, ci_high = bootstrap_mean_ci(
            advantage.values,
            seed=BOOTSTRAP_SEED + metric_i,
        )

        statistic, pvalue = safe_wilcoxon(
            advantage.values
        )

        eps = 1e-12

        wins = int(
            (advantage > eps).sum()
        )

        losses = int(
            (advantage < -eps).sum()
        )

        ties = int(
            len(advantage) - wins - losses
        )

        row = {
            "comparison": comparison_name,
            "metric": metric,
            "metric_display": DISPLAY[metric],
            "n_valid_conditions": len(advantage),
            "reference_mean": float(
                reference[metric].mean()
            ),
            "final_condition_mean": float(
                final[metric].mean()
            ),
            "mean_paired_advantage": float(
                advantage.mean()
            ),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_fraction": (
                wins / len(advantage)
            ),
            "wilcoxon_stat": statistic,
            "wilcoxon_pvalue": pvalue,
        }

        rows.append(row)

        pair_df = pd.DataFrame({
            "condition": reference.index,
            "comparison": comparison_name,
            "metric": metric,
            "reference_value": (
                reference[metric].values
            ),
            "final_value": (
                final[metric].values
            ),
            "paired_advantage": (
                advantage.values
            ),
        })

        pair_frames.append(pair_df)

    stats = pd.DataFrame(rows)

    stats["wilcoxon_pvalue_holm"] = (
        holm_adjust(
            stats["wilcoxon_pvalue"].values
        )
    )

    stats["significant_holm_0.05"] = (
        stats["wilcoxon_pvalue_holm"]
        < 0.05
    )

    pairs = pd.concat(
        pair_frames,
        ignore_index=True,
    )

    return stats, pairs


def format_mean_sd(mean, sd):
    return f"{mean:.6f} ± {sd:.6f}"


# ============================================================
# 1. Load Alternative OOD
# ============================================================

print("=" * 100)
print("ALTERNATIVE OOD")
print("=" * 100)

alt_frozen = load_alt_frozen_metrics(
    ALT_BASELINE_METRICS
)

print(
    "[LOAD] Alternative Frozen CRISP:",
    len(alt_frozen),
    "conditions",
)

alt_final_seeds = load_seed_metrics(
    ALT_FINAL_ROOT
)

alt_shuffled_seeds = load_seed_metrics(
    ALT_SHUFFLED_ROOT
)

(
    alt_final_condition_mean,
    alt_final_seed_overall,
    alt_final_summary,
) = aggregate_seed_metrics(
    alt_final_seeds
)

(
    alt_shuffled_condition_mean,
    alt_shuffled_seed_overall,
    alt_shuffled_summary,
) = aggregate_seed_metrics(
    alt_shuffled_seeds
)

exact_condition_check(
    "Alternative Frozen CRISP",
    alt_frozen,
    "Alternative Final ReCalib",
    alt_final_condition_mean,
    expected_n=57,
)

exact_condition_check(
    "Alternative Final ReCalib",
    alt_final_condition_mean,
    "Alternative shuffled",
    alt_shuffled_condition_mean,
    expected_n=57,
)


# ============================================================
# 2. Alternative S3: Final vs Frozen
# ============================================================

alt_final_vs_frozen, alt_frozen_pairs = paired_stats(
    alt_frozen,
    alt_final_condition_mean,
    "Alternative OOD: Final ReCalib vs Frozen CRISP",
    expected_n=57,
)

s3_rows = []

for _, row in alt_final_vs_frozen.iterrows():
    metric = row["metric"]

    final_mean = float(
        alt_final_summary.loc[
            metric,
            "mean",
        ]
    )

    final_sd = float(
        alt_final_summary.loc[
            metric,
            "sample_sd",
        ]
    )

    s3_rows.append({
        "metric": metric,
        "metric_display": DISPLAY[metric],
        "frozen_crisp_mean": float(
            alt_frozen[metric].mean()
        ),
        "final_recalib_mean": final_mean,
        "final_recalib_sample_sd": final_sd,
        "final_recalib_mean_pm_sd": (
            format_mean_sd(
                final_mean,
                final_sd,
            )
        ),
        "mean_paired_improvement": (
            row["mean_paired_advantage"]
        ),
        "ci95_low": row["ci95_low"],
        "ci95_high": row["ci95_high"],
        "improved_fraction": row["win_fraction"],
        "wins": row["wins"],
        "losses": row["losses"],
        "ties": row["ties"],
        "n_valid_conditions": (
            row["n_valid_conditions"]
        ),
        "wilcoxon_pvalue": (
            row["wilcoxon_pvalue"]
        ),
        "wilcoxon_pvalue_holm": (
            row["wilcoxon_pvalue_holm"]
        ),
        "significant_holm_0.05": (
            row["significant_holm_0.05"]
        ),
    })

table_s3 = pd.DataFrame(s3_rows)


# ============================================================
# 3. Alternative Final vs shuffled
# ============================================================

alt_final_vs_shuffled, alt_shuffle_pairs = paired_stats(
    alt_shuffled_condition_mean,
    alt_final_condition_mean,
    "Alternative OOD: Final ReCalib vs shuffled control",
    expected_n=57,
)


# ============================================================
# 4. Load Main OOD
# ============================================================

print()
print("=" * 100)
print("MAIN OOD")
print("=" * 100)

main_frozen, main_final_seeds = (
    load_main_frozen_and_final(
        MAIN_FINAL_CONDITION_METRICS
    )
)

(
    main_final_condition_mean,
    main_final_seed_overall,
    main_final_summary,
) = aggregate_seed_metrics(
    main_final_seeds
)

main_shuffled_seeds = load_seed_metrics(
    MAIN_SHUFFLED_ROOT
)

(
    main_shuffled_condition_mean,
    main_shuffled_seed_overall,
    main_shuffled_summary,
) = aggregate_seed_metrics(
    main_shuffled_seeds
)

exact_condition_check(
    "Main Frozen CRISP",
    main_frozen,
    "Main Final ReCalib",
    main_final_condition_mean,
    expected_n=108,
)

exact_condition_check(
    "Main Final ReCalib",
    main_final_condition_mean,
    "Main shuffled",
    main_shuffled_condition_mean,
    expected_n=108,
)


# ============================================================
# 5. Main paired statistics
# ============================================================

main_final_vs_frozen, main_frozen_pairs = paired_stats(
    main_frozen,
    main_final_condition_mean,
    "Main OOD: Final ReCalib vs Frozen CRISP",
    expected_n=108,
)

main_final_vs_shuffled, main_shuffle_pairs = paired_stats(
    main_shuffled_condition_mean,
    main_final_condition_mean,
    "Main OOD: Final ReCalib vs shuffled control",
    expected_n=108,
)


# ============================================================
# 6. Supplementary Table 6 cross-protocol replication
# ============================================================

protocol_rows = []

for (
    protocol,
    n_conditions,
    frozen,
    final_summary,
    shuffled_summary,
) in [
    (
        "Main OOD",
        108,
        main_frozen,
        main_final_summary,
        main_shuffled_summary,
    ),
    (
        "Alternative OOD",
        57,
        alt_frozen,
        alt_final_summary,
        alt_shuffled_summary,
    ),
]:
    for metric in METRICS:
        final_mean = float(
            final_summary.loc[
                metric,
                "mean",
            ]
        )

        final_sd = float(
            final_summary.loc[
                metric,
                "sample_sd",
            ]
        )

        shuffled_mean = float(
            shuffled_summary.loc[
                metric,
                "mean",
            ]
        )

        shuffled_sd = float(
            shuffled_summary.loc[
                metric,
                "sample_sd",
            ]
        )

        protocol_rows.append({
            "ood_protocol": protocol,
            "n_conditions": n_conditions,
            "metric": metric,
            "metric_display": DISPLAY[metric],
            "frozen_crisp_mean": float(
                frozen[metric].mean()
            ),
            "final_recalib_mean": final_mean,
            "final_recalib_sample_sd": final_sd,
            "final_recalib_mean_pm_sd": (
                format_mean_sd(
                    final_mean,
                    final_sd,
                )
            ),
            "shuffled_control_mean": (
                shuffled_mean
            ),
            "shuffled_control_sample_sd": (
                shuffled_sd
            ),
            "shuffled_control_mean_pm_sd": (
                format_mean_sd(
                    shuffled_mean,
                    shuffled_sd,
                )
            ),
        })

table_s6_replication = pd.DataFrame(
    protocol_rows
)


# ============================================================
# 7. Supplementary Table 6 Final vs shuffled statistics
# ============================================================

s6_stats = pd.concat(
    [
        main_final_vs_shuffled.assign(
            ood_protocol="Main OOD"
        ),
        alt_final_vs_shuffled.assign(
            ood_protocol="Alternative OOD"
        ),
    ],
    ignore_index=True,
)

s6_stats = s6_stats[
    [
        "ood_protocol",
        "metric",
        "metric_display",
        "reference_mean",
        "final_condition_mean",
        "mean_paired_advantage",
        "ci95_low",
        "ci95_high",
        "wins",
        "losses",
        "ties",
        "win_fraction",
        "n_valid_conditions",
        "wilcoxon_stat",
        "wilcoxon_pvalue",
        "wilcoxon_pvalue_holm",
        "significant_holm_0.05",
    ]
]


# ============================================================
# 8. Save seed-level summaries
# ============================================================

seed_summary = pd.concat(
    [
        main_final_seed_overall.assign(
            ood_protocol="Main OOD",
            method="Final ReCalib",
        ),
        main_shuffled_seed_overall.assign(
            ood_protocol="Main OOD",
            method="Shuffled control",
        ),
        alt_final_seed_overall.assign(
            ood_protocol="Alternative OOD",
            method="Final ReCalib",
        ),
        alt_shuffled_seed_overall.assign(
            ood_protocol="Alternative OOD",
            method="Shuffled control",
        ),
    ],
    ignore_index=True,
)


# ============================================================
# 9. Save everything
# ============================================================

table_s3.to_csv(
    OUT / "tableS3_alternative_ood.csv",
    index=False,
)

alt_frozen_pairs.to_csv(
    OUT / "tableS3_condition_level_pairs.csv",
    index=False,
)

table_s6_replication.to_csv(
    OUT / "tableS6_cross_protocol_replication.csv",
    index=False,
)

s6_stats.to_csv(
    OUT / "tableS6_final_vs_shuffled.csv",
    index=False,
)

pd.concat(
    [
        main_shuffle_pairs.assign(
            ood_protocol="Main OOD"
        ),
        alt_shuffle_pairs.assign(
            ood_protocol="Alternative OOD"
        ),
    ],
    ignore_index=True,
).to_csv(
    OUT / "tableS6_condition_level_pairs.csv",
    index=False,
)

seed_summary.to_csv(
    OUT / "tableS3_S6_seed_level_means.csv",
    index=False,
)


# ============================================================
# 10. Console report
# ============================================================

print()
print("=" * 110)
print("SUPPLEMENTARY TABLE 3")
print("Alternative pathway-held-out OOD")
print("=" * 110)

print(
    table_s3[
        [
            "metric_display",
            "frozen_crisp_mean",
            "final_recalib_mean_pm_sd",
            "mean_paired_improvement",
            "ci95_low",
            "ci95_high",
            "improved_fraction",
            "wins",
            "losses",
            "ties",
            "wilcoxon_pvalue_holm",
            "significant_holm_0.05",
        ]
    ].to_string(index=False)
)

print()
print("=" * 110)
print("SUPPLEMENTARY TABLE 6A")
print("Cross-protocol replication")
print("=" * 110)

print(
    table_s6_replication[
        [
            "ood_protocol",
            "n_conditions",
            "metric_display",
            "frozen_crisp_mean",
            "final_recalib_mean_pm_sd",
            "shuffled_control_mean_pm_sd",
        ]
    ].to_string(index=False)
)

print()
print("=" * 110)
print("SUPPLEMENTARY TABLE 6B")
print("Final ReCalib vs shuffled remaining-residual control")
print("=" * 110)

print(
    s6_stats[
        [
            "ood_protocol",
            "metric_display",
            "mean_paired_advantage",
            "ci95_low",
            "ci95_high",
            "win_fraction",
            "wins",
            "losses",
            "ties",
            "wilcoxon_pvalue_holm",
            "significant_holm_0.05",
        ]
    ].to_string(index=False)
)

print()
print("=" * 110)
print("OUTPUT FILES")
print("=" * 110)

for path in sorted(OUT.glob("tableS*.csv")):
    print(path)

print()
print("[OK] Supplementary Tables 3 and 6 statistics generated.")
