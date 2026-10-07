#!/usr/bin/env python3

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

import recompute_final_paper_results as final

# ------------------------------------------------------------------
# Frozen CMonge seed
# ------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="Evaluate Frozen CMonge on the official Main OOD split."
)
_parser.add_argument(
    "--seed",
    type=int,
    choices=[2024, 3407, 42],
    default=42,
    help="Frozen CMonge training seed.",
)
_args = _parser.parse_args()
CMONGE_SEED = int(_args.seed)



PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================================
# Frozen CMonge export
# ============================================================================

CMONGE_ROOT = (
    PROJECT_ROOT
    / "cmonge_outputs"
    / f"mydata_frozen_cmonge_seed{CMONGE_SEED}"
    / "main_ood"
)

CMONGE_PRED = (
    CMONGE_ROOT
    / "x_cmonge_condition_mean.csv"
)

TRUE_PATH = (
    CMONGE_ROOT
    / "x_true_condition_mean.csv"
)

CTRL_PATH = (
    CMONGE_ROOT
    / "x_ctrl_condition_mean.csv"
)

METADATA_PATH = (
    CMONGE_ROOT
    / "condition_metadata.csv"
)


# ============================================================================
# Existing Frozen CRISP Main-OOD predictor
# ============================================================================

CRISP_ROOT = (
    PROJECT_ROOT
    / "crisp_outputs"
    / "mydata_baseline_full"
    / "ood"
)

CRISP_PRED = (
    CRISP_ROOT
    / "x_crisp_condition_mean.csv"
)


# ============================================================================
# Output
# ============================================================================

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / f"frozen_cmonge_seed{CMONGE_SEED}"
    / "main_ood"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CMONGE_CONDITION_OUT = (
    OUTPUT_ROOT
    / "frozen_cmonge_condition_metrics.csv"
)

CRISP_CONDITION_OUT = (
    OUTPUT_ROOT
    / "frozen_crisp_condition_metrics_recomputed.csv"
)

SUMMARY_OUT = (
    OUTPUT_ROOT
    / "frozen_predictor_comparison.csv"
)

JSON_OUT = (
    OUTPUT_ROOT
    / "frozen_predictor_summary.json"
)


METRICS = [
    "mse_de",
    "pearson_de",
    "pearson_delta_de",
    "r2score_de",
]


# ============================================================================
# Helpers
# ============================================================================

def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    return path


def align_to_reference(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    name: str,
) -> pd.DataFrame:

    if set(frame.index) != set(reference.index):
        missing = sorted(
            set(reference.index)
            - set(frame.index)
        )

        extra = sorted(
            set(frame.index)
            - set(reference.index)
        )

        raise RuntimeError(
            f"{name}: condition mismatch\n"
            f"missing={missing[:10]}\n"
            f"extra={extra[:10]}"
        )

    if set(frame.columns) != set(reference.columns):
        missing = sorted(
            set(reference.columns)
            - set(frame.columns)
        )

        extra = sorted(
            set(frame.columns)
            - set(reference.columns)
        )

        raise RuntimeError(
            f"{name}: gene mismatch\n"
            f"missing={missing[:10]}\n"
            f"extra={extra[:10]}"
        )

    return frame.loc[
        reference.index,
        reference.columns,
    ].copy()


def summarize(
    metrics: pd.DataFrame,
    method: str,
):
    row = {
        "method": method,
        "n_conditions": int(len(metrics)),
    }

    for metric in METRICS:

        values = pd.to_numeric(
            metrics[metric],
            errors="coerce",
        )

        row[metric] = float(
            values.mean()
        )

        row[
            f"{metric}_valid_n"
        ] = int(
            values.notna().sum()
        )

        row[
            f"{metric}_std"
        ] = float(
            values.std(
                ddof=1
            )
        )

    return row


# ============================================================================
# Main
# ============================================================================

def main():

    print("=" * 100)
    print("FROZEN CMONGE MAIN-OOD EVALUATION")
    print("=" * 100)

    print("\n[1] INPUT FILES")

    for p in [
        CMONGE_PRED,
        TRUE_PATH,
        CTRL_PATH,
        METADATA_PATH,
        CRISP_PRED,
    ]:
        require(p)
        print(p)

    # ------------------------------------------------------------------------
    # Use EXACTLY the final-paper matrix reader.
    # ------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("[2] LOAD MATRICES USING FINAL-PAPER READER")
    print("=" * 100)

    true = final.read_matrix(
        TRUE_PATH
    )

    ctrl = final.read_matrix(
        CTRL_PATH
    )

    cmonge = final.read_matrix(
        CMONGE_PRED
    )

    crisp = final.read_matrix(
        CRISP_PRED
    )

    print("true   =", true.shape)
    print("ctrl   =", ctrl.shape)
    print("CMonge =", cmonge.shape)
    print("CRISP  =", crisp.shape)

    if true.shape != (108, 977):
        raise RuntimeError(
            f"Expected true=(108,977), got {true.shape}"
        )

    ctrl = align_to_reference(
        ctrl,
        true,
        "control",
    )

    cmonge = align_to_reference(
        cmonge,
        true,
        "CMonge",
    )

    crisp = align_to_reference(
        crisp,
        true,
        "CRISP",
    )

    print(
        "\nCondition order aligned =",
        list(cmonge.index)
        == list(true.index),
    )

    print(
        "Gene order aligned      =",
        list(cmonge.columns)
        == list(true.columns),
    )

    # ------------------------------------------------------------------------
    # DE mapping:
    #
    # IMPORTANT:
    # This is the exact final-paper function in
    # recompute_final_paper_results.py.
    # ------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("[3] LOAD OFFICIAL CONDITION-SPECIFIC DE MAPPING")
    print("=" * 100)

    conditions = [
        str(x)
        for x in true.index
    ]

    genes = [
        str(x)
        for x in true.columns
    ]

    # load_de_mapping() writes its own audit under final.OUTPUT_ROOT.
    # Redirect that audit into this Frozen-CMonge result directory.
    original_output_root = final.OUTPUT_ROOT

    try:
        final.OUTPUT_ROOT = OUTPUT_ROOT

        de_mapping = (
            final.load_de_mapping(
                conditions,
                genes,
            )
        )

    finally:
        final.OUTPUT_ROOT = (
            original_output_root
        )

    if set(de_mapping) != set(conditions):
        missing = (
            set(conditions)
            - set(de_mapping)
        )

        extra = (
            set(de_mapping)
            - set(conditions)
        )

        raise RuntimeError(
            "DE mapping condition mismatch:\n"
            f"missing={sorted(missing)[:10]}\n"
            f"extra={sorted(extra)[:10]}"
        )

    de_counts = np.asarray(
        [
            len(
                de_mapping[
                    condition
                ]
            )
            for condition in conditions
        ],
        dtype=int,
    )

    print(
        "DE mapped conditions =",
        len(de_mapping),
    )

    print(
        "DE genes / condition:"
    )

    print(
        "  min    =",
        int(de_counts.min()),
    )

    print(
        "  median =",
        float(
            np.median(
                de_counts
            )
        ),
    )

    print(
        "  max    =",
        int(de_counts.max()),
    )

    print(
        "  total  =",
        int(
            de_counts.sum()
        ),
    )

    if np.any(
        de_counts <= 0
    ):
        raise RuntimeError(
            "At least one condition has zero mapped DE genes."
        )

    # ------------------------------------------------------------------------
    # Frozen CMonge metrics
    # ------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("[4] FROZEN CMONGE CONDITION METRICS")
    print("=" * 100)

    cmonge_metrics = (
        final.condition_metrics(
            pred=cmonge,
            true=true,
            ctrl=ctrl,
            de_mapping=de_mapping,
        )
    )

    if len(cmonge_metrics) != 108:
        raise RuntimeError(
            "Frozen CMonge metrics did not return 108 rows."
        )

    cmonge_metrics.insert(
        0,
        "method",
        f"frozen_cmonge_seed{CMONGE_SEED}",
    )

    cmonge_metrics.to_csv(
        CMONGE_CONDITION_OUT,
        index=False,
    )

    print(
        "saved:",
        CMONGE_CONDITION_OUT,
    )

    # ------------------------------------------------------------------------
    # Frozen CRISP recomputation:
    #
    # This is our evaluator self-check.
    # ------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("[5] FROZEN CRISP RECOMPUTATION — SELF CHECK")
    print("=" * 100)

    crisp_metrics = (
        final.condition_metrics(
            pred=crisp,
            true=true,
            ctrl=ctrl,
            de_mapping=de_mapping,
        )
    )

    if len(crisp_metrics) != 108:
        raise RuntimeError(
            "Frozen CRISP metrics did not return 108 rows."
        )

    crisp_metrics.insert(
        0,
        "method",
        "frozen_crisp",
    )

    crisp_metrics.to_csv(
        CRISP_CONDITION_OUT,
        index=False,
    )

    print(
        "saved:",
        CRISP_CONDITION_OUT,
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("[6] MAIN-OOD SUMMARY")
    print("=" * 100)

    cmonge_summary = summarize(
        cmonge_metrics,
        f"Frozen CMonge (seed{CMONGE_SEED})",
    )

    crisp_summary = summarize(
        crisp_metrics,
        "Frozen CRISP",
    )

    summary = pd.DataFrame(
        [
            crisp_summary,
            cmonge_summary,
        ]
    )

    # Improvement relative to Frozen CRISP.
    #
    # Positive = CMonge is better.
    improvement = {
        "method":
            "CMonge improvement vs CRISP",

        "n_conditions":
            108,

        "mse_de":
            crisp_summary["mse_de"]
            - cmonge_summary["mse_de"],

        "pearson_de":
            cmonge_summary["pearson_de"]
            - crisp_summary["pearson_de"],

        "pearson_delta_de":
            cmonge_summary["pearson_delta_de"]
            - crisp_summary["pearson_delta_de"],

        "r2score_de":
            cmonge_summary["r2score_de"]
            - crisp_summary["r2score_de"],
    }

    summary.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    print(
        summary[
            [
                "method",
                "n_conditions",
                "mse_de",
                "pearson_de",
                "pearson_delta_de",
                "r2score_de",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.9f}",
        )
    )

    print()
    print("-" * 100)
    print(
        "CMONGE IMPROVEMENT RELATIVE TO FROZEN CRISP"
    )
    print(
        "(positive = CMonge better)"
    )
    print("-" * 100)

    print(
        f"MSE-DE improvement       = "
        f"{improvement['mse_de']:+.9f}"
    )

    print(
        f"Pearson-DE improvement   = "
        f"{improvement['pearson_de']:+.9f}"
    )

    print(
        f"Pearson-Δ-DE improvement = "
        f"{improvement['pearson_delta_de']:+.9f}"
    )

    print(
        f"R²-DE improvement        = "
        f"{improvement['r2score_de']:+.9f}"
    )

    # ------------------------------------------------------------------------
    # Paired descriptive differences — NOT significance testing yet.
    # ------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("[7] CONDITION-LEVEL PAIRED DESCRIPTIVE CHECK")
    print("=" * 100)

    paired = crisp_metrics[
        [
            "condition",
            *METRICS,
        ]
    ].merge(
        cmonge_metrics[
            [
                "condition",
                *METRICS,
            ]
        ],
        on="condition",
        suffixes=(
            "_crisp",
            "_cmonge",
        ),
        validate="one_to_one",
    )

    paired[
        "mse_de_improvement"
    ] = (
        paired["mse_de_crisp"]
        - paired["mse_de_cmonge"]
    )

    for metric in [
        "pearson_de",
        "pearson_delta_de",
        "r2score_de",
    ]:
        paired[
            f"{metric}_improvement"
        ] = (
            paired[
                f"{metric}_cmonge"
            ]
            - paired[
                f"{metric}_crisp"
            ]
        )

    paired_path = (
        OUTPUT_ROOT
        / "frozen_cmonge_vs_crisp_by_condition.csv"
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    for metric in [
        "mse_de",
        "pearson_de",
        "pearson_delta_de",
        "r2score_de",
    ]:
        col = (
            f"{metric}_improvement"
        )

        values = pd.to_numeric(
            paired[col],
            errors="coerce",
        )

        positive = int(
            (values > 0).sum()
        )

        negative = int(
            (values < 0).sum()
        )

        zero = int(
            (values == 0).sum()
        )

        print(
            f"{metric:20s}: "
            f"CMonge better={positive:3d}, "
            f"worse={negative:3d}, "
            f"tie={zero:3d}"
        )

    # ------------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------------

    json_summary = {
        "evaluation": {
            "split":
                "main_ood",

            "conditions":
                108,

            "genes":
                977,

            "de_mapping_function":
                "recompute_final_paper_results.load_de_mapping",

            "metric_function":
                "recompute_final_paper_results.condition_metrics",
        },

        "frozen_crisp":
            crisp_summary,

        f"frozen_cmonge_seed{CMONGE_SEED}":
            cmonge_summary,

        "cmonge_improvement_vs_crisp": {
            k: float(v)
            if isinstance(
                v,
                (
                    float,
                    np.floating,
                ),
            )
            else v

            for k, v
            in improvement.items()
        },
    }

    JSON_OUT.write_text(
        json.dumps(
            json_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nFiles:")
    print(CMONGE_CONDITION_OUT)
    print(CRISP_CONDITION_OUT)
    print(SUMMARY_OUT)
    print(paired_path)
    print(JSON_OUT)

    print("\n" + "=" * 100)
    print("FROZEN CMONGE MAIN-OOD EVALUATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
