import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRIMARY_METRICS = [
    "mse_de",
    "pearson_de",
    "pearson_delta_de",
    "r2score_de",
]

LOWER_IS_BETTER = {"mse_de"}

INPUT_MAIN = Path("crisp_outputs/evidence_mydata/final_alpha04_w0_paper_main_table.csv")
INPUT_PATHWAY = Path("crisp_outputs/evidence_mydata/ho_pathway/ho_pathway_paper_table.csv")

OUT_DIR = Path("crisp_outputs/evidence_mydata/primary4")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def check_file(p):
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def load_and_filter(path, name):
    check_file(path)
    df = pd.read_csv(path)

    if "metric" not in df.columns:
        raise ValueError(f"{path} does not contain column: metric")

    df = df[df["metric"].isin(PRIMARY_METRICS)].copy()

    df["metric"] = pd.Categorical(df["metric"], categories=PRIMARY_METRICS, ordered=True)
    df = df.sort_values("metric").reset_index(drop=True)

    needed = [
        "metric",
        "CRISP_baseline",
        "normal_RC_mean",
        "normal_RC_delta_positive_is_better",
        "normal_RC_improved_fraction",
        "normal_RC_wilcoxon_p_median",
        "shuffle_RC_mean",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")


    df["raw_change_ReCalib_minus_baseline"] = (
        df["normal_RC_mean"] - df["CRISP_baseline"]
    )

    df["direction"] = df["metric"].map(
        lambda m: "lower is better" if m in LOWER_IS_BETTER else "higher is better"
    )

    df["split_name"] = name

    return df


def format_table(df, out_csv):
    table = df[
        [
            "metric",
            "direction",
            "CRISP_baseline",
            "normal_RC_mean",
            "raw_change_ReCalib_minus_baseline",
            "normal_RC_improved_fraction",
            "normal_RC_wilcoxon_p_median",
            "shuffle_RC_mean",
        ]
    ].copy()

    table = table.rename(
        columns={
            "CRISP_baseline": "Frozen_predictor",
            "normal_RC_mean": "ReCalib",
            "raw_change_ReCalib_minus_baseline": "ReCalib_minus_Frozen",
            "normal_RC_improved_fraction": "Improved_OOD_conditions_fraction",
            "normal_RC_wilcoxon_p_median": "Wilcoxon_p",
            "shuffle_RC_mean": "Shuffled_ReCalib",
        }
    )

    table.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")
    print(table.to_string(index=False))
    print()
    return table


def metric_label(metric):
    return {
        "mse_de": "DE-gene MSE",
        "pearson_de": "DE-gene Pearson",
        "pearson_delta_de": "Delta-DE Pearson",
        "r2score_de": "DE-gene R²",
    }[metric]


def plot_metric_dots(df, title, out_prefix):
    """
    No bar plot. One small dot-comparison panel per metric.
    """
    n = len(PRIMARY_METRICS)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    for ax, metric in zip(axes, PRIMARY_METRICS):
        row = df[df["metric"] == metric].iloc[0]

        vals = {
            "Frozen": row["CRISP_baseline"],
            "ReCalib": row["normal_RC_mean"],
            "Shuffled": row["shuffle_RC_mean"],
        }

        labels = list(vals.keys())
        x = np.array(list(vals.values()), dtype=float)
        y = np.arange(len(labels))[::-1]

        ax.plot(
            [vals["Frozen"], vals["ReCalib"]],
            [y[0], y[1]],
            linewidth=2.5,
            color="#A0AEC0",
            zorder=1,
        )

        ax.scatter(vals["Frozen"], y[0], s=90, color="#718096", zorder=3, label="Frozen")
        ax.scatter(vals["ReCalib"], y[1], s=110, color="#2563EB", zorder=3, label="ReCalib")
        ax.scatter(
            vals["Shuffled"],
            y[2],
            s=95,
            facecolor="white",
            edgecolor="#E11D48",
            linewidth=2,
            zorder=3,
            label="Shuffled",
        )

        for label, val, yy in zip(labels, x, y):
            ax.text(val, yy + 0.12, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_title(metric_label(metric), fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

        if metric in LOWER_IS_BETTER:
            note = "lower is better"
        else:
            note = "higher is better"
        ax.text(
            0.02,
            0.03,
            note,
            transform=ax.transAxes,
            fontsize=9,
            color="#4A5568",
        )

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(f"{out_prefix}.png", dpi=300)
    fig.savefig(f"{out_prefix}.pdf")
    plt.close(fig)

    print(f"[saved] {out_prefix}.png")
    print(f"[saved] {out_prefix}.pdf")


def plot_improved_fraction(df, title, out_prefix):

    vals = []
    labels = []

    for metric in PRIMARY_METRICS:
        row = df[df["metric"] == metric].iloc[0]
        vals.append(float(row["normal_RC_improved_fraction"]))
        labels.append(metric_label(metric))

    vals = np.array(vals)
    y = np.arange(len(vals))[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.2))

    ax.hlines(y, 0, vals, linewidth=2.5, color="#CBD5E0")
    ax.scatter(vals, y, s=120, color="#2563EB", zorder=3)

    for v, yy in zip(vals, y):
        ax.text(v + 0.015, yy, f"{v * 100:.1f}%", va="center", fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Fraction of improved OOD conditions")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=300)
    fig.savefig(f"{out_prefix}.pdf")
    plt.close(fig)

    print(f"[saved] {out_prefix}.png")
    print(f"[saved] {out_prefix}.pdf")


def main():
    main_df = load_and_filter(INPUT_MAIN, "main_ood")
    pathway_df = load_and_filter(INPUT_PATHWAY, "pathway_ood")

    main_table = format_table(
        main_df,
        OUT_DIR / "main_ood_primary4_table.csv",
    )

    pathway_table = format_table(
        pathway_df,
        OUT_DIR / "pathway_ood_primary4_table.csv",
    )

    plot_metric_dots(
        main_df,
        "Main OOD performance on primary DE-gene metrics",
        OUT_DIR / "main_ood_primary4_metric_dots",
    )

    plot_improved_fraction(
        main_df,
        "Main OOD condition-level improvement",
        OUT_DIR / "main_ood_primary4_improved_fraction",
    )

    plot_metric_dots(
        pathway_df,
        "Pathway-level OOD performance on primary DE-gene metrics",
        OUT_DIR / "pathway_ood_primary4_metric_dots",
    )

    plot_improved_fraction(
        pathway_df,
        "Pathway-level OOD condition-level improvement",
        OUT_DIR / "pathway_ood_primary4_improved_fraction",
    )

    print("\nDone. Use files in:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
