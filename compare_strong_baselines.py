from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from innovation_data import (
    PROJECT_ROOT,
    load_experiment_data,
)
from run_innovation_baselines import (
    METRIC_DIRECTIONS,
    PRIMARY_METRICS,
    build_official_de_masks,
    evaluate_by_condition,
    summarize_metrics,
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "innovation_experiments.yaml"
)

BASELINE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "innovation_baselines"
)

STRICT_RECALIB_ROOT = (
    PROJECT_ROOT
    / "results"
    / "recalib_strict_main_ood"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "strong_baseline_comparison"
)

SEEDS = [2024, 3407, 42]


def read_prediction(
    path: Path,
    *,
    conditions: list[str],
    genes: list[str],
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, index_col=0)

    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)

    if frame.index.duplicated().any():
        raise ValueError(
            f"Duplicated conditions in {path}"
        )

    missing_conditions = (
        set(conditions) - set(frame.index)
    )
    extra_conditions = (
        set(frame.index) - set(conditions)
    )

    missing_genes = (
        set(genes) - set(frame.columns)
    )
    extra_genes = (
        set(frame.columns) - set(genes)
    )

    if missing_conditions or extra_conditions:
        raise ValueError(
            f"Condition mismatch in {path}\n"
            f"missing={sorted(missing_conditions)[:10]}\n"
            f"extra={sorted(extra_conditions)[:10]}"
        )

    if missing_genes or extra_genes:
        raise ValueError(
            f"Gene mismatch in {path}\n"
            f"missing={sorted(missing_genes)[:10]}\n"
            f"extra={sorted(extra_genes)[:10]}"
        )

    frame = frame.loc[
        conditions,
        genes,
    ]

    values = frame.to_numpy(
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite values in {path}"
        )

    return values


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iterations: int = 20000,
    confidence: float = 0.95,
    seed: int = 2024,
) -> tuple[float, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        len(values),
        size=(iterations, len(values)),
    )

    means = values[indices].mean(axis=1)

    alpha = 1.0 - confidence

    return (
        float(
            np.quantile(
                means,
                alpha / 2.0,
            )
        ),
        float(
            np.quantile(
                means,
                1.0 - alpha / 2.0,
            )
        ),
    )


def holm_adjust(
    pvalues: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(
        pvalues,
        errors="coerce",
    ).to_numpy(dtype=float)

    adjusted = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    valid_positions = np.flatnonzero(
        np.isfinite(values)
    )

    if len(valid_positions) == 0:
        return pd.Series(
            adjusted,
            index=pvalues.index,
        )

    valid_values = values[
        valid_positions
    ]

    order = np.argsort(
        valid_values
    )

    sorted_positions = (
        valid_positions[order]
    )
    sorted_values = (
        valid_values[order]
    )

    m = len(sorted_values)
    previous = 0.0

    for rank, (
        position,
        pvalue,
    ) in enumerate(
        zip(
            sorted_positions,
            sorted_values,
        )
    ):
        candidate = min(
            1.0,
            (m - rank)
            * float(pvalue),
        )

        candidate = max(
            previous,
            candidate,
        )

        adjusted[position] = (
            candidate
        )

        previous = candidate

    return pd.Series(
        adjusted,
        index=pvalues.index,
    )


def average_condition_metrics(
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    if not frames:
        raise ValueError(
            "No seed metric frames"
        )

    reference = (
        frames[0]
        .set_index("condition")
        .copy()
    )

    reference_conditions = (
        reference.index.tolist()
    )

    aligned = []

    for frame in frames:
        current = (
            frame
            .set_index("condition")
        )

        if set(current.index) != set(
            reference_conditions
        ):
            raise ValueError(
                "Condition mismatch "
                "between ReCalib seeds"
            )

        current = current.loc[
            reference_conditions
        ]

        aligned.append(current)

    result = aligned[0].copy()

    result["method"] = (
        "strict_recalib_seed_mean"
    )

    for metric in PRIMARY_METRICS:
        stacked = np.vstack(
            [
                pd.to_numeric(
                    frame[metric],
                    errors="coerce",
                ).to_numpy(dtype=float)
                for frame in aligned
            ]
        )

        result[metric] = (
            np.nanmean(
                stacked,
                axis=0,
            )
        )

    return (
        result
        .reset_index()
    )


def paired_comparison(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    bootstrap_seed: int,
) -> pd.DataFrame:
    merged = (
        frame_a[
            ["condition", *PRIMARY_METRICS]
        ]
        .merge(
            frame_b[
                ["condition", *PRIMARY_METRICS]
            ],
            on="condition",
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
    )

    rows = []

    for metric_index, metric in enumerate(
        PRIMARY_METRICS
    ):
        a = pd.to_numeric(
            merged[f"{metric}_a"],
            errors="coerce",
        )

        b = pd.to_numeric(
            merged[f"{metric}_b"],
            errors="coerce",
        )

        valid = (
            a.notna()
            & b.notna()
        )

        a = a[valid].to_numpy(
            dtype=float
        )
        b = b[valid].to_numpy(
            dtype=float
        )

        if (
            METRIC_DIRECTIONS[metric]
            == "lower"
        ):
            # Positive = A better
            advantage = b - a
        else:
            advantage = a - b

        tolerance = 1e-12

        wins = int(
            np.sum(
                advantage > tolerance
            )
        )

        losses = int(
            np.sum(
                advantage < -tolerance
            )
        )

        ties = int(
            len(advantage)
            - wins
            - losses
        )

        if len(advantage) == 0:
            statistic = np.nan
            pvalue = np.nan

        elif np.all(
            np.abs(advantage)
            <= tolerance
        ):
            statistic = 0.0
            pvalue = 1.0

        else:
            test = wilcoxon(
                advantage,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            statistic = float(
                test.statistic
            )

            pvalue = float(
                test.pvalue
            )

        ci_low, ci_high = (
            bootstrap_mean_ci(
                advantage,
                seed=(
                    bootstrap_seed
                    + metric_index
                ),
            )
        )

        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "metric": metric,
                "n_paired_conditions": (
                    len(advantage)
                ),
                "method_a_mean": (
                    float(np.mean(a))
                ),
                "method_b_mean": (
                    float(np.mean(b))
                ),
                "method_a_advantage_mean": (
                    float(
                        np.mean(
                            advantage
                        )
                    )
                ),
                "method_a_advantage_median": (
                    float(
                        np.median(
                            advantage
                        )
                    )
                ),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "method_a_wins": wins,
                "method_a_losses": losses,
                "ties": ties,
                "method_a_win_fraction": (
                    wins
                    / len(advantage)
                ),
                "wilcoxon_statistic": (
                    statistic
                ),
                "wilcoxon_pvalue": (
                    pvalue
                ),
            }
        )

    return pd.DataFrame(rows)


def build_drug_level_summary(
    all_metrics: pd.DataFrame,
) -> pd.DataFrame:
    return (
        all_metrics
        .groupby(
            [
                "method",
                "drug",
            ],
            as_index=False,
        )[
            PRIMARY_METRICS
        ]
        .mean()
    )


def build_drug_level_wins(
    drug_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    strict = (
        drug_summary[
            drug_summary["method"]
            == "strict_recalib_seed_mean"
        ]
    )

    for baseline in [
        "frozen_crisp",
        "gene_wise_affine",
        "ridge_residual",
    ]:
        other = (
            drug_summary[
                drug_summary["method"]
                == baseline
            ]
        )

        merged = strict.merge(
            other,
            on="drug",
            suffixes=(
                "_strict",
                "_baseline",
            ),
            validate="one_to_one",
        )

        for metric in PRIMARY_METRICS:
            strict_values = (
                merged[
                    f"{metric}_strict"
                ].to_numpy(
                    dtype=float
                )
            )

            baseline_values = (
                merged[
                    f"{metric}_baseline"
                ].to_numpy(
                    dtype=float
                )
            )

            if (
                METRIC_DIRECTIONS[metric]
                == "lower"
            ):
                advantage = (
                    baseline_values
                    - strict_values
                )
            else:
                advantage = (
                    strict_values
                    - baseline_values
                )

            rows.append(
                {
                    "comparison": (
                        "strict_recalib_seed_mean"
                        f"_vs_{baseline}"
                    ),
                    "metric": metric,
                    "n_drugs": (
                        len(advantage)
                    ),
                    "strict_wins": int(
                        np.sum(
                            advantage > 0
                        )
                    ),
                    "strict_losses": int(
                        np.sum(
                            advantage < 0
                        )
                    ),
                    "strict_win_fraction": (
                        float(
                            np.mean(
                                advantage > 0
                            )
                        )
                    ),
                    "mean_drug_level_advantage": (
                        float(
                            np.mean(
                                advantage
                            )
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print(
        "Strict ReCalib strong-baseline comparison"
    )
    print("=" * 88)

    data = load_experiment_data(
        CONFIG_PATH
    )

    _, ood_mask, _, _ = (
        build_official_de_masks(
            data
        )
    )

    split = data.main_ood

    print(
        f"[INFO] Main OOD: "
        f"{split.n_conditions} conditions × "
        f"{split.n_genes} genes"
    )

    metrics_by_method = {}

    # Frozen CRISP
    metrics_by_method[
        "frozen_crisp"
    ] = evaluate_by_condition(
        method="frozen_crisp",
        split=split,
        prediction=(
            split.frozen_crisp
        ),
        de_mask=ood_mask,
    )

    # Gene-wise affine
    affine_prediction = (
        read_prediction(
            BASELINE_ROOT
            / "gene_wise_affine"
            / "x_calibrated_condition_mean.csv",
            conditions=split.conditions,
            genes=split.genes,
        )
    )

    metrics_by_method[
        "gene_wise_affine"
    ] = evaluate_by_condition(
        method="gene_wise_affine",
        split=split,
        prediction=affine_prediction,
        de_mask=ood_mask,
    )

    # Ridge
    ridge_prediction = (
        read_prediction(
            BASELINE_ROOT
            / "ridge_residual"
            / "x_calibrated_condition_mean.csv",
            conditions=split.conditions,
            genes=split.genes,
        )
    )

    metrics_by_method[
        "ridge_residual"
    ] = evaluate_by_condition(
        method="ridge_residual",
        split=split,
        prediction=ridge_prediction,
        de_mask=ood_mask,
    )

    # Strict ReCalib, each seed
    strict_seed_frames = []

    for seed in SEEDS:
        prediction = (
            read_prediction(
                STRICT_RECALIB_ROOT
                / f"seed_{seed}"
                / "x_calibrated_condition_mean.csv",
                conditions=split.conditions,
                genes=split.genes,
            )
        )

        method = (
            f"strict_recalib_seed_{seed}"
        )

        frame = evaluate_by_condition(
            method=method,
            split=split,
            prediction=prediction,
            de_mask=ood_mask,
        )

        metrics_by_method[
            method
        ] = frame

        strict_seed_frames.append(
            frame
        )

    metrics_by_method[
        "strict_recalib_seed_mean"
    ] = average_condition_metrics(
        strict_seed_frames
    )

    # All condition-level metrics
    all_condition_metrics = (
        pd.concat(
            metrics_by_method.values(),
            ignore_index=True,
        )
    )

    all_condition_metrics.to_csv(
        OUTPUT_ROOT
        / "condition_metrics_all.csv",
        index=False,
    )

    # Global method summaries
    summary_frames = []

    for method, frame in (
        metrics_by_method.items()
    ):
        summary = summarize_metrics(
            frame
        )

        summary["method"] = method

        summary_frames.append(
            summary
        )

    summaries = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    summaries.to_csv(
        OUTPUT_ROOT
        / "method_summaries.csv",
        index=False,
    )

    # Strict ReCalib true mean ± SD across random seeds
    strict_seed_summary = (
        summaries[
            summaries["method"].isin(
                [
                    f"strict_recalib_seed_{seed}"
                    for seed in SEEDS
                ]
            )
        ]
        .copy()
    )

    strict_seed_summary[
        "seed"
    ] = (
        strict_seed_summary[
            "method"
        ]
        .str.extract(
            r"seed_(\d+)"
        )[0]
        .astype(int)
    )

    strict_mean_sd = (
        strict_seed_summary
        .groupby(
            "metric",
            as_index=False,
        )
        .agg(
            mean=("mean", "mean"),
            sample_sd=(
                "mean",
                lambda x:
                x.std(ddof=1),
            ),
        )
    )

    strict_mean_sd.to_csv(
        OUTPUT_ROOT
        / "strict_recalib_mean_sd.csv",
        index=False,
    )

    # Paper-friendly comparison table
    paper_methods = [
        "frozen_crisp",
        "gene_wise_affine",
        "ridge_residual",
        "strict_recalib_seed_mean",
    ]

    paper_table = (
        summaries[
            summaries["method"]
            .isin(paper_methods)
        ]
        .pivot(
            index="method",
            columns="metric",
            values="mean",
        )
        .reset_index()
    )

    paper_table.to_csv(
        OUTPUT_ROOT
        / "paper_comparison_table.csv",
        index=False,
    )

    # Primary paired comparisons
    primary_pairs = [
        (
            "strict_recalib_seed_mean",
            "frozen_crisp",
        ),
        (
            "strict_recalib_seed_mean",
            "gene_wise_affine",
        ),
        (
            "strict_recalib_seed_mean",
            "ridge_residual",
        ),
    ]

    primary_frames = []

    for pair_index, (
        method_a,
        method_b,
    ) in enumerate(
        primary_pairs
    ):
        print(
            f"[COMPARE] "
            f"{method_a} vs {method_b}"
        )

        primary_frames.append(
            paired_comparison(
                metrics_by_method[
                    method_a
                ],
                metrics_by_method[
                    method_b
                ],
                method_a=method_a,
                method_b=method_b,
                bootstrap_seed=(
                    2024
                    + pair_index * 100
                ),
            )
        )

    primary_statistics = (
        pd.concat(
            primary_frames,
            ignore_index=True,
        )
    )

    # Holm correction across the complete
    # primary family: 3 baselines × 4 metrics = 12 tests.
    primary_statistics[
        "wilcoxon_pvalue_holm_12"
    ] = holm_adjust(
        primary_statistics[
            "wilcoxon_pvalue"
        ]
    )

    primary_statistics[
        "significant_holm_0.05"
    ] = (
        primary_statistics[
            "wilcoxon_pvalue_holm_12"
        ]
        < 0.05
    )

    primary_statistics.to_csv(
        OUTPUT_ROOT
        / "primary_pairwise_statistics.csv",
        index=False,
    )

    # Seed-specific diagnostics
    seed_specific_frames = []

    for seed in SEEDS:
        strict_method = (
            f"strict_recalib_seed_{seed}"
        )

        for baseline in [
            "gene_wise_affine",
            "ridge_residual",
        ]:
            seed_specific_frames.append(
                paired_comparison(
                    metrics_by_method[
                        strict_method
                    ],
                    metrics_by_method[
                        baseline
                    ],
                    method_a=(
                        strict_method
                    ),
                    method_b=baseline,
                    bootstrap_seed=(
                        seed
                        + len(
                            seed_specific_frames
                        )
                        * 100
                    ),
                )
            )

    seed_specific_statistics = (
        pd.concat(
            seed_specific_frames,
            ignore_index=True,
        )
    )

    seed_specific_statistics.to_csv(
        OUTPUT_ROOT
        / "seed_specific_pairwise_statistics.csv",
        index=False,
    )

    # Drug-level descriptive analysis
    drug_summary = (
        build_drug_level_summary(
            all_condition_metrics
        )
    )

    drug_summary.to_csv(
        OUTPUT_ROOT
        / "per_drug_method_metrics.csv",
        index=False,
    )

    drug_wins = (
        build_drug_level_wins(
            drug_summary
        )
    )

    drug_wins.to_csv(
        OUTPUT_ROOT
        / "per_drug_comparison.csv",
        index=False,
    )

    metadata = {
        "protocol": (
            "strict 1003 fitting / "
            "251 IID validation / "
            "108 unseen-drug Main OOD"
        ),
        "strict_recalib_alpha": 0.50,
        "strict_recalib_de_weight": 0,
        "strict_recalib_seeds": (
            SEEDS
        ),
        "main_ood_conditions": (
            split.n_conditions
        ),
        "main_ood_genes": (
            split.n_genes
        ),
        "main_ood_drugs": 9,
        "paired_statistics": (
            "condition-level"
        ),
        "bootstrap_iterations": (
            20000
        ),
        "multiple_testing": (
            "Holm correction across "
            "12 primary tests"
        ),
        "positive_advantage": (
            "positive values mean "
            "Strict ReCalib is better"
        ),
    }

    (
        OUTPUT_ROOT
        / "run_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("METHOD SUMMARY")
    print("=" * 88)

    print(
        paper_table.to_string(
            index=False
        )
    )

    print("\n" + "=" * 88)
    print("STRICT ReCalib PAIRED STATISTICS")
    print("=" * 88)

    print(
        primary_statistics[
            [
                "method_a",
                "method_b",
                "metric",
                "method_a_mean",
                "method_b_mean",
                "method_a_advantage_mean",
                "ci95_low",
                "ci95_high",
                "method_a_wins",
                "method_a_losses",
                "method_a_win_fraction",
                "wilcoxon_pvalue",
                "wilcoxon_pvalue_holm_12",
                "significant_holm_0.05",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 88)
    print("DRUG-LEVEL DESCRIPTIVE WINS")
    print("=" * 88)

    print(
        drug_wins.to_string(
            index=False
        )
    )

    print("\n[OK] Comparison completed")
    print(f"[OUTPUT] {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
