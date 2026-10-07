from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from innovation_data import (
    PROJECT_ROOT,
    load_experiment_data,
    resolve_path,
)
from run_innovation_baselines import (
    METRIC_DIRECTIONS,
    PRIMARY_METRICS,
    build_official_de_masks,
    evaluate_by_condition,
    summarize_metrics,
)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iterations: int = 10000,
    seed: int = 2024,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)

    for index in range(iterations):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )
        means[index] = np.mean(sample)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def read_prediction(
    path: Path,
    *,
    conditions: list[str],
    genes: list[str],
) -> np.ndarray:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)

    missing_conditions = set(conditions) - set(frame.index)
    missing_genes = set(genes) - set(frame.columns)

    if missing_conditions:
        raise ValueError(
            f"Missing conditions in {path}: "
            f"{sorted(missing_conditions)[:5]}"
        )

    if missing_genes:
        raise ValueError(
            f"Missing genes in {path}: "
            f"{sorted(missing_genes)[:5]}"
        )

    frame = frame.loc[conditions, genes]
    return frame.to_numpy(dtype=np.float64)


def paired_comparison(
    affine: pd.DataFrame,
    recalib: pd.DataFrame,
    *,
    seed_label: str,
) -> pd.DataFrame:
    merged = affine[
        ["condition", *PRIMARY_METRICS]
    ].merge(
        recalib[
            ["condition", *PRIMARY_METRICS]
        ],
        on="condition",
        suffixes=("_affine", "_recalib"),
        validate="one_to_one",
    )

    rows = []

    for metric in PRIMARY_METRICS:
        affine_values = pd.to_numeric(
            merged[f"{metric}_affine"],
            errors="coerce",
        )
        recalib_values = pd.to_numeric(
            merged[f"{metric}_recalib"],
            errors="coerce",
        )

        valid = affine_values.notna() & recalib_values.notna()
        affine_valid = affine_values[valid].to_numpy(dtype=float)
        recalib_valid = recalib_values[valid].to_numpy(dtype=float)

        # Positive advantage means ReCalib is better.
        if METRIC_DIRECTIONS[metric] == "lower":
            advantage = affine_valid - recalib_valid
        else:
            advantage = recalib_valid - affine_valid

        wins = int(np.sum(advantage > 0))
        losses = int(np.sum(advantage < 0))
        ties = int(np.sum(advantage == 0))

        if np.allclose(advantage, 0):
            statistic = 0.0
            p_value = 1.0
        else:
            result = wilcoxon(
                advantage,
                alternative="two-sided",
                zero_method="wilcox",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        ci_low, ci_high = bootstrap_mean_ci(advantage)

        rows.append(
            {
                "recalib_seed": seed_label,
                "metric": metric,
                "n_paired_conditions": int(len(advantage)),
                "gene_wise_affine_mean": float(
                    np.mean(affine_valid)
                ),
                "recalib_mean": float(
                    np.mean(recalib_valid)
                ),
                "recalib_advantage_mean": float(
                    np.mean(advantage)
                ),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "recalib_wins": wins,
                "recalib_losses": losses,
                "ties": ties,
                "recalib_win_fraction": float(
                    wins / len(advantage)
                ),
                "wilcoxon_statistic": statistic,
                "wilcoxon_pvalue": p_value,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    config_path = (
        PROJECT_ROOT
        / "configs"
        / "innovation_experiments.yaml"
    )

    data = load_experiment_data(config_path)
    _, ood_mask, _, _ = build_official_de_masks(data)

    affine_metrics_path = (
        PROJECT_ROOT
        / "results"
        / "innovation_baselines"
        / "gene_wise_affine"
        / "main_ood_condition_metrics.csv"
    )

    if not affine_metrics_path.is_file():
        raise FileNotFoundError(affine_metrics_path)

    affine_metrics = pd.read_csv(affine_metrics_path)

    reference_paths = data.config["paths"]["reference_results"]

    output_root = (
        PROJECT_ROOT
        / "results"
        / "innovation_baselines"
        / "affine_vs_recalib"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_statistics = []

    for seed in [2024, 3407, 42]:
        key = f"recalib_seed_{seed}"
        prediction_path = resolve_path(reference_paths[key])

        prediction = read_prediction(
            prediction_path,
            conditions=data.main_ood.conditions,
            genes=data.main_ood.genes,
        )

        condition_metrics = evaluate_by_condition(
            method=f"recalib_seed_{seed}",
            split=data.main_ood,
            prediction=prediction,
            de_mask=ood_mask,
        )

        condition_metrics.to_csv(
            output_root
            / f"recalib_seed_{seed}_condition_metrics.csv",
            index=False,
        )

        summary = summarize_metrics(condition_metrics)
        summary["seed"] = seed
        all_summaries.append(summary)

        statistics = paired_comparison(
            affine_metrics,
            condition_metrics,
            seed_label=str(seed),
        )
        all_statistics.append(statistics)

    summaries = pd.concat(all_summaries, ignore_index=True)
    statistics = pd.concat(all_statistics, ignore_index=True)

    summaries.to_csv(
        output_root / "recalib_seed_summaries.csv",
        index=False,
    )
    statistics.to_csv(
        output_root
        / "gene_wise_affine_vs_recalib_statistics.csv",
        index=False,
    )

    print("\nReCalib seed summaries")
    print(
        summaries.pivot(
            index="seed",
            columns="metric",
            values="mean",
        ).to_string()
    )

    print("\nGene-wise affine versus ReCalib")
    print(
        statistics[
            [
                "recalib_seed",
                "metric",
                "gene_wise_affine_mean",
                "recalib_mean",
                "recalib_advantage_mean",
                "recalib_wins",
                "recalib_losses",
                "recalib_win_fraction",
                "wilcoxon_pvalue",
            ]
        ].to_string(index=False)
    )

    print(f"\n[OUTPUT] {output_root}")


if __name__ == "__main__":
    main()
