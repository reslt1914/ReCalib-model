from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from innovation_data import (
    PROJECT_ROOT,
    load_experiment_data,
)
from run_innovation_baselines import (
    PRIMARY_METRICS,
    build_official_de_masks,
    evaluate_by_condition,
    summarize_metrics,
)
from compare_strong_baselines import (
    paired_comparison,
    holm_adjust,
)


CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "innovation_experiments.yaml"
)

FINAL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ridge_anchored_recalib_final"
)

BASELINE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "innovation_baselines"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ridge_anchored_final_comparison"
)

SEEDS = [2024, 3407, 42]
ALPHA = 0.50


def read_matrix(
    path: Path,
    conditions,
    genes,
):
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        index_col=0,
    )

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    if set(df.index) != set(conditions):
        raise ValueError(
            f"Condition mismatch: {path}"
        )

    if set(df.columns) != set(genes):
        raise ValueError(
            f"Gene mismatch: {path}"
        )

    df = df.loc[
        conditions,
        genes,
    ]

    x = df.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(x).all():
        raise ValueError(
            f"Non-finite values: {path}"
        )

    return x


def mean_condition_metrics(
    frames,
    method_name,
):
    conditions = (
        frames[0]["condition"]
        .astype(str)
        .tolist()
    )

    aligned = []

    for frame in frames:
        current = frame.copy()
        current["condition"] = (
            current["condition"].astype(str)
        )

        current = (
            current
            .set_index("condition")
            .loc[conditions]
        )

        aligned.append(current)

    result = aligned[0].copy()

    for metric in PRIMARY_METRICS:
        values = np.vstack(
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
                values,
                axis=0,
            )
        )

    result["method"] = method_name

    return result.reset_index()


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_experiment_data(
        CONFIG
    )

    _, ood_mask, _, _ = (
        build_official_de_masks(
            data
        )
    )

    split = data.main_ood

    metrics = {}

    # --------------------------------------------------
    # Frozen CRISP
    # --------------------------------------------------
    metrics["frozen_crisp"] = (
        evaluate_by_condition(
            method="frozen_crisp",
            split=split,
            prediction=split.frozen_crisp,
            de_mask=ood_mask,
        )
    )

    # --------------------------------------------------
    # Gene-wise affine
    # --------------------------------------------------
    affine = read_matrix(
        BASELINE_ROOT
        / "gene_wise_affine"
        / "x_calibrated_condition_mean.csv",
        split.conditions,
        split.genes,
    )

    metrics["gene_wise_affine"] = (
        evaluate_by_condition(
            method="gene_wise_affine",
            split=split,
            prediction=affine,
            de_mask=ood_mask,
        )
    )

    # --------------------------------------------------
    # Ridge
    # --------------------------------------------------
    ridge = read_matrix(
        BASELINE_ROOT
        / "ridge_residual"
        / "x_calibrated_condition_mean.csv",
        split.conditions,
        split.genes,
    )

    metrics["ridge_residual"] = (
        evaluate_by_condition(
            method="ridge_residual",
            split=split,
            prediction=ridge,
            de_mask=ood_mask,
        )
    )

    # --------------------------------------------------
    # Ridge-anchored ReCalib
    # --------------------------------------------------
    final_seed_frames = []

    for seed in SEEDS:
        seed_dir = (
            FINAL_ROOT
            / f"seed_{seed}"
        )

        anchor = read_matrix(
            seed_dir
            / "ridge_anchor_main_ood.csv",
            split.conditions,
            split.genes,
        )

        remaining = read_matrix(
            seed_dir
            / "predicted_remaining_residual_main_ood.csv",
            split.conditions,
            split.genes,
        )

        prediction = (
            anchor
            + ALPHA * remaining
        )

        frame = evaluate_by_condition(
            method=(
                f"ridge_anchored_recalib_seed_{seed}"
            ),
            split=split,
            prediction=prediction,
            de_mask=ood_mask,
        )

        metrics[
            f"ridge_anchored_recalib_seed_{seed}"
        ] = frame

        final_seed_frames.append(frame)

    metrics[
        "ridge_anchored_recalib_seed_mean"
    ] = mean_condition_metrics(
        final_seed_frames,
        "ridge_anchored_recalib_seed_mean",
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    summary_rows = []

    for method, frame in metrics.items():
        summary = summarize_metrics(
            frame
        )

        summary["method"] = method

        summary_rows.append(
            summary
        )

    summaries = pd.concat(
        summary_rows,
        ignore_index=True,
    )

    summaries.to_csv(
        OUTPUT_ROOT
        / "method_summaries.csv",
        index=False,
    )

    paper_methods = [
        "frozen_crisp",
        "gene_wise_affine",
        "ridge_residual",
        "ridge_anchored_recalib_seed_mean",
    ]

    paper_table = (
        summaries[
            summaries["method"].isin(
                paper_methods
            )
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

    # --------------------------------------------------
    # Paired statistics
    # --------------------------------------------------
    final_method = (
        "ridge_anchored_recalib_seed_mean"
    )

    comparisons = []

    for i, baseline in enumerate(
        [
            "frozen_crisp",
            "gene_wise_affine",
            "ridge_residual",
        ]
    ):
        comparison = paired_comparison(
            metrics[final_method],
            metrics[baseline],
            method_a=final_method,
            method_b=baseline,
            bootstrap_seed=2024 + i * 100,
        )

        comparisons.append(
            comparison
        )

    stats = pd.concat(
        comparisons,
        ignore_index=True,
    )

    stats[
        "wilcoxon_pvalue_holm_12"
    ] = holm_adjust(
        stats["wilcoxon_pvalue"]
    )

    stats[
        "significant_holm_0.05"
    ] = (
        stats[
            "wilcoxon_pvalue_holm_12"
        ]
        < 0.05
    )

    stats.to_csv(
        OUTPUT_ROOT
        / "primary_pairwise_statistics.csv",
        index=False,
    )

    # --------------------------------------------------
    # Drug-level
    # --------------------------------------------------
    combined = pd.concat(
        [
            metrics[
                "ridge_anchored_recalib_seed_mean"
            ],
            metrics[
                "gene_wise_affine"
            ],
            metrics[
                "ridge_residual"
            ],
            metrics[
                "frozen_crisp"
            ],
        ],
        ignore_index=True,
    )

    drug_summary = (
        combined
        .groupby(
            [
                "method",
                "drug",
            ],
            as_index=False,
        )[PRIMARY_METRICS]
        .mean()
    )

    drug_summary.to_csv(
        OUTPUT_ROOT
        / "per_drug_metrics.csv",
        index=False,
    )

    drug_rows = []

    final_drug = (
        drug_summary[
            drug_summary["method"]
            == final_method
        ]
    )

    directions = {
        "mse_de": "lower",
        "pearson_de": "higher",
        "pearson_delta_de": "higher",
        "r2score_de": "higher",
    }

    for baseline in [
        "gene_wise_affine",
        "ridge_residual",
        "frozen_crisp",
    ]:
        base_drug = (
            drug_summary[
                drug_summary["method"]
                == baseline
            ]
        )

        merged = final_drug.merge(
            base_drug,
            on="drug",
            suffixes=(
                "_final",
                "_baseline",
            ),
        )

        for metric in PRIMARY_METRICS:
            final_values = merged[
                f"{metric}_final"
            ].to_numpy(dtype=float)

            base_values = merged[
                f"{metric}_baseline"
            ].to_numpy(dtype=float)

            if directions[metric] == "lower":
                advantage = (
                    base_values
                    - final_values
                )
            else:
                advantage = (
                    final_values
                    - base_values
                )

            drug_rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "n_drugs": len(advantage),
                    "final_wins": int(
                        np.sum(
                            advantage > 0
                        )
                    ),
                    "final_losses": int(
                        np.sum(
                            advantage < 0
                        )
                    ),
                    "final_win_fraction": float(
                        np.mean(
                            advantage > 0
                        )
                    ),
                    "mean_advantage": float(
                        np.mean(
                            advantage
                        )
                    ),
                }
            )

    pd.DataFrame(
        drug_rows
    ).to_csv(
        OUTPUT_ROOT
        / "per_drug_comparison.csv",
        index=False,
    )

    print(
        "\n============================================"
    )
    print("FINAL METHOD SUMMARY")
    print(
        "============================================"
    )

    print(
        paper_table.to_string(
            index=False
        )
    )

    print(
        "\n============================================"
    )
    print("PAIRED STATISTICS")
    print(
        "============================================"
    )

    cols = [
        "method_b",
        "metric",
        "method_a_advantage_mean",
        "ci95_low",
        "ci95_high",
        "method_a_wins",
        "method_a_losses",
        "method_a_win_fraction",
        "wilcoxon_pvalue_holm_12",
        "significant_holm_0.05",
    ]

    print(
        stats[cols].to_string(
            index=False
        )
    )

    print(
        "\n============================================"
    )
    print("DRUG LEVEL")
    print(
        "============================================"
    )

    print(
        pd.DataFrame(
            drug_rows
        ).to_string(
            index=False
        )
    )

    print(
        "\n[OK] Final comparison completed"
    )
    print(
        f"[OUTPUT] {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
