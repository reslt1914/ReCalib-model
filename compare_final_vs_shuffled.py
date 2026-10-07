from pathlib import Path

import numpy as np
import pandas as pd

from innovation_data import PROJECT_ROOT, load_experiment_data
from run_innovation_baselines import (
    PRIMARY_METRICS,
    build_official_de_masks,
    evaluate_by_condition,
)
from compare_strong_baselines import (
    paired_comparison,
    holm_adjust,
)

CONFIG = PROJECT_ROOT / "configs" / "innovation_experiments.yaml"

FINAL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ridge_anchored_recalib_final"
)

SHUFFLED_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ridge_anchored_recalib_shuffled"
)

OUT = (
    PROJECT_ROOT
    / "results"
    / "final_vs_shuffled_comparison"
)

SEEDS = [2024, 3407, 42]
ALPHA = 0.50


def read_matrix(path, conditions, genes):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    if set(df.index) != set(conditions):
        raise ValueError(f"Condition mismatch: {path}")

    if set(df.columns) != set(genes):
        raise ValueError(f"Gene mismatch: {path}")

    return (
        df.loc[conditions, genes]
        .to_numpy(dtype=np.float64)
    )


def condition_mean(frames, name):
    conditions = frames[0]["condition"].astype(str).tolist()
    aligned = []

    for frame in frames:
        x = frame.copy()
        x["condition"] = x["condition"].astype(str)
        aligned.append(
            x.set_index("condition").loc[conditions]
        )

    out = aligned[0].copy()

    for metric in PRIMARY_METRICS:
        values = np.vstack([
            pd.to_numeric(
                x[metric],
                errors="coerce",
            ).to_numpy(dtype=float)
            for x in aligned
        ])
        out[metric] = np.nanmean(values, axis=0)

    out["method"] = name
    return out.reset_index()


def evaluate_root(root, prefix, split, de_mask):
    frames = []

    for seed in SEEDS:
        seed_dir = root / f"seed_{seed}"

        anchor = read_matrix(
            seed_dir / "ridge_anchor_main_ood.csv",
            split.conditions,
            split.genes,
        )

        remaining = read_matrix(
            seed_dir / "predicted_remaining_residual_main_ood.csv",
            split.conditions,
            split.genes,
        )

        prediction = anchor + ALPHA * remaining

        frame = evaluate_by_condition(
            method=f"{prefix}_seed_{seed}",
            split=split,
            prediction=prediction,
            de_mask=de_mask,
        )

        frames.append(frame)

    return condition_mean(
        frames,
        f"{prefix}_seed_mean",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data = load_experiment_data(CONFIG)
    _, ood_mask, _, _ = build_official_de_masks(data)
    split = data.main_ood

    final = evaluate_root(
        FINAL_ROOT,
        "final_recalib",
        split,
        ood_mask,
    )

    shuffled = evaluate_root(
        SHUFFLED_ROOT,
        "shuffled_control",
        split,
        ood_mask,
    )

    stats = paired_comparison(
        final,
        shuffled,
        method_a="final_recalib",
        method_b="shuffled_control",
        bootstrap_seed=2024,
    )

    stats["wilcoxon_pvalue_holm_4"] = holm_adjust(
        stats["wilcoxon_pvalue"]
    )

    stats["significant_holm_0.05"] = (
        stats["wilcoxon_pvalue_holm_4"] < 0.05
    )

    stats.to_csv(
        OUT / "paired_statistics.csv",
        index=False,
    )

    cols = [
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
        "wilcoxon_pvalue_holm_4",
        "significant_holm_0.05",
    ]

    print(stats[cols].to_string(index=False))


if __name__ == "__main__":
    main()
