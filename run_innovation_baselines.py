from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baseline_models import (
    GeneMeanResidual,
    GeneWiseAffine,
    GlobalMeanResidual,
)
from innovation_data import (
    PROJECT_ROOT,
    ConditionSplit,
    load_experiment_data,
    resolve_path,
)
from train_crisp_rc_deaware import build_de_masks


PRIMARY_METRICS = [
    "mse_de",
    "pearson_de",
    "pearson_delta_de",
    "r2score_de",
]

METRIC_DIRECTIONS = {
    "mse_de": "lower",
    "pearson_de": "higher",
    "pearson_delta_de": "higher",
    "r2score_de": "higher",
}

# 论文中已经确认的 Frozen CRISP Main OOD 结果。
EXPECTED_FROZEN_CRISP = {
    "mse_de": 0.038031,
    "pearson_de": 0.773241,
    "pearson_delta_de": 0.323000,
    "r2score_de": 0.002306,
}

# 允许六位小数显示值带来的微小舍入误差。
FROZEN_CHECK_TOLERANCE = 5e-5


def safe_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) <= 1e-12:
        return np.nan

    if np.std(y_pred) <= 1e-12:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    residual_sum = float(
        np.sum((y_true - y_pred) ** 2)
    )
    total_sum = float(
        np.sum((y_true - np.mean(y_true)) ** 2)
    )

    if total_sum <= 1e-12:
        return np.nan

    return float(1.0 - residual_sum / total_sum)


def mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(
        np.mean(
            (
                np.asarray(y_true, dtype=np.float64)
                - np.asarray(y_pred, dtype=np.float64)
            )
            ** 2
        )
    )


def metadata_value(
    row: pd.Series,
    candidates: list[str],
) -> str:
    for column in candidates:
        if column in row.index:
            value = row[column]
            if pd.notna(value):
                return str(value)
    return ""


def evaluate_by_condition(
    *,
    method: str,
    split: ConditionSplit,
    prediction: np.ndarray,
    de_mask: np.ndarray,
) -> pd.DataFrame:
    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    if prediction.shape != split.true.shape:
        raise ValueError(
            f"{method}: prediction shape mismatch: "
            f"expected={split.true.shape}, "
            f"actual={prediction.shape}"
        )

    if de_mask.shape != split.true.shape:
        raise ValueError(
            f"{method}: DE-mask shape mismatch: "
            f"expected={split.true.shape}, "
            f"actual={de_mask.shape}"
        )

    rows: list[dict[str, Any]] = []

    for condition_idx, condition in enumerate(
        split.conditions
    ):
        official_indices = np.flatnonzero(
            de_mask[condition_idx]
        )

        metadata_row = split.metadata.iloc[condition_idx]

        row: dict[str, Any] = {
            "method": method,
            "split": split.name,
            "condition": condition,
            "cell_type": metadata_value(
                metadata_row,
                ["cell_type", "cell_line", "cell"],
            ),
            "drug": metadata_value(
                metadata_row,
                ["drug", "perturbation"],
            ),
            "dose": metadata_value(
                metadata_row,
                ["dose", "dose_val"],
            ),
            "n_de_official": int(
                len(official_indices)
            ),
        }

        if len(official_indices) == 0:
            for metric in PRIMARY_METRICS:
                row[metric] = np.nan

            row["is_valid_de"] = False
            rows.append(row)
            continue

        observed = split.true[
            condition_idx,
            official_indices,
        ]
        predicted = prediction[
            condition_idx,
            official_indices,
        ]
        control = split.control[
            condition_idx,
            official_indices,
        ]

        observed_delta = observed - control
        predicted_delta = predicted - control

        row.update(
            {
                "mse_de": mse(observed, predicted),
                "pearson_de": safe_pearson(
                    observed,
                    predicted,
                ),
                "pearson_delta_de": safe_pearson(
                    observed_delta,
                    predicted_delta,
                ),
                "r2score_de": safe_r2(
                    observed,
                    predicted,
                ),
                "is_valid_de": True,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_metrics(
    condition_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for metric in PRIMARY_METRICS:
        values = pd.to_numeric(
            condition_metrics[metric],
            errors="coerce",
        )
        valid_values = values.dropna()

        rows.append(
            {
                "method": str(
                    condition_metrics["method"].iloc[0]
                ),
                "split": str(
                    condition_metrics["split"].iloc[0]
                ),
                "metric": metric,
                "mean": (
                    float(valid_values.mean())
                    if len(valid_values)
                    else np.nan
                ),
                "sd_across_conditions": (
                    float(valid_values.std(ddof=1))
                    if len(valid_values) > 1
                    else np.nan
                ),
                "n_valid_conditions": int(
                    len(valid_values)
                ),
            }
        )

    return pd.DataFrame(rows)


def add_improved_fraction(
    summary: pd.DataFrame,
    condition_metrics: pd.DataFrame,
    frozen_metrics: pd.DataFrame,
) -> pd.DataFrame:
    output = summary.copy()
    fractions: dict[str, float] = {}

    merged = condition_metrics.merge(
        frozen_metrics[
            ["condition", *PRIMARY_METRICS]
        ],
        on="condition",
        suffixes=("_method", "_frozen"),
        validate="one_to_one",
    )

    for metric in PRIMARY_METRICS:
        method_values = pd.to_numeric(
            merged[f"{metric}_method"],
            errors="coerce",
        )
        frozen_values = pd.to_numeric(
            merged[f"{metric}_frozen"],
            errors="coerce",
        )

        valid = (
            method_values.notna()
            & frozen_values.notna()
        )

        if not valid.any():
            fractions[metric] = np.nan
            continue

        if METRIC_DIRECTIONS[metric] == "lower":
            improved = (
                method_values[valid]
                < frozen_values[valid]
            )
        else:
            improved = (
                method_values[valid]
                > frozen_values[valid]
            )

        fractions[metric] = float(improved.mean())

    output["improved_fraction_vs_frozen_crisp"] = (
        output["metric"].map(fractions)
    )

    return output


def legacy_split(split: ConditionSplit) -> dict[str, Any]:
    return {
        "conds": np.asarray(
            split.conditions,
            dtype=str,
        ),
        "genes": np.asarray(
            split.genes,
            dtype=str,
        ),
    }


def build_official_de_masks(
    data,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    adata_path = resolve_path(
        data.config["paths"]["adata"]
    )
    de_key = str(
        data.config["paths"].get(
            "de_key",
            "lincs_DEGs",
        )
    )

    train_legacy = legacy_split(data.train)
    validation_legacy = legacy_split(
        data.validation
    )
    ood_legacy = legacy_split(data.main_ood)

    _, validation_mask, _, validation_coverage = (
        build_de_masks(
            str(adata_path),
            train_legacy,
            validation_legacy,
            de_key=de_key,
        )
    )

    _, ood_mask, _, ood_coverage = build_de_masks(
        str(adata_path),
        train_legacy,
        ood_legacy,
        de_key=de_key,
    )

    print(
        "[INFO] Validation conditions with mapped DE genes: "
        f"{int((validation_mask.sum(axis=1) > 0).sum())}"
        f"/{len(validation_mask)}"
    )
    print(
        "[INFO] Main OOD conditions with mapped DE genes: "
        f"{int((ood_mask.sum(axis=1) > 0).sum())}"
        f"/{len(ood_mask)}"
    )

    return (
        validation_mask,
        ood_mask,
        validation_coverage,
        ood_coverage,
    )


def save_prediction(
    *,
    output_root: Path,
    method: str,
    split: ConditionSplit,
    prediction: np.ndarray,
) -> Path:
    method_dir = output_root / method
    method_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        method_dir
        / "x_calibrated_condition_mean.csv"
    )

    frame = pd.DataFrame(
        np.asarray(prediction, dtype=np.float64),
        index=split.conditions,
        columns=split.genes,
    )
    frame.index.name = "condition"
    frame.to_csv(output_path)

    return output_path


def check_frozen_crisp_reproduction(
    frozen_summary: pd.DataFrame,
) -> None:
    print("\nFrozen CRISP reproduction check")

    failures = []

    for metric, expected in EXPECTED_FROZEN_CRISP.items():
        actual_series = frozen_summary.loc[
            frozen_summary["metric"] == metric,
            "mean",
        ]

        if len(actual_series) != 1:
            raise RuntimeError(
                f"Missing Frozen CRISP summary for {metric}"
            )

        actual = float(actual_series.iloc[0])
        difference = abs(actual - expected)

        print(
            f"  {metric:20s} "
            f"actual={actual:.6f} "
            f"expected={expected:.6f} "
            f"abs_diff={difference:.8f}"
        )

        if difference > FROZEN_CHECK_TOLERANCE:
            failures.append(
                {
                    "metric": metric,
                    "actual": actual,
                    "expected": expected,
                    "difference": difference,
                }
            )

    if failures:
        raise RuntimeError(
            "Frozen CRISP metrics do not reproduce the "
            "paper values. Do not continue with innovation "
            f"baselines. Failures: {failures}"
        )

    print("[OK] Frozen CRISP paper metrics reproduced")


def run_and_save_method(
    *,
    method: str,
    validation_prediction: np.ndarray,
    ood_prediction: np.ndarray,
    data,
    validation_mask: np.ndarray,
    ood_mask: np.ndarray,
    output_root: Path,
    frozen_ood_metrics: pd.DataFrame,
    parameters: dict[str, Any],
    fit_seconds: float,
    inference_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_dir = output_root / method
    method_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_metrics = evaluate_by_condition(
        method=method,
        split=data.validation,
        prediction=validation_prediction,
        de_mask=validation_mask,
    )
    ood_metrics = evaluate_by_condition(
        method=method,
        split=data.main_ood,
        prediction=ood_prediction,
        de_mask=ood_mask,
    )

    validation_metrics.to_csv(
        method_dir / "validation_condition_metrics.csv",
        index=False,
    )
    ood_metrics.to_csv(
        method_dir / "main_ood_condition_metrics.csv",
        index=False,
    )

    save_prediction(
        output_root=output_root,
        method=method,
        split=data.main_ood,
        prediction=ood_prediction,
    )

    validation_summary = summarize_metrics(
        validation_metrics
    )
    ood_summary = summarize_metrics(ood_metrics)

    if method == "frozen_crisp":
        ood_summary[
            "improved_fraction_vs_frozen_crisp"
        ] = np.nan
    else:
        ood_summary = add_improved_fraction(
            ood_summary,
            ood_metrics,
            frozen_ood_metrics,
        )

    validation_summary.to_csv(
        method_dir / "validation_summary.csv",
        index=False,
    )
    ood_summary.to_csv(
        method_dir / "main_ood_summary.csv",
        index=False,
    )

    run_metadata = {
        "method": method,
        "parameters": parameters,
        "train_conditions": data.train.n_conditions,
        "validation_conditions": (
            data.validation.n_conditions
        ),
        "main_ood_conditions": (
            data.main_ood.n_conditions
        ),
        "genes": data.main_ood.n_genes,
        "fit_seconds": fit_seconds,
        "main_ood_inference_seconds": inference_seconds,
    }

    (
        method_dir / "run_metadata.json"
    ).write_text(
        json.dumps(
            run_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n[{method}] Main OOD summary")
    print(
        ood_summary[
            [
                "metric",
                "mean",
                "n_valid_conditions",
                "improved_fraction_vs_frozen_crisp",
            ]
        ].to_string(index=False)
    )

    return validation_summary, ood_summary


def parse_methods(value: str) -> list[str]:
    methods = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    allowed = {
        "frozen_crisp",
        "global_mean_residual",
        "gene_mean_residual",
        "gene_wise_affine",
    }

    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown methods: {unknown}. "
            f"Allowed methods: {sorted(allowed)}"
        )

    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic ReCalib innovation baselines."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/innovation_experiments.yaml"
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=(
            "frozen_crisp,"
            "global_mean_residual,"
            "gene_mean_residual,"
            "gene_wise_affine"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (
            PROJECT_ROOT / config_path
        ).resolve()

    methods = parse_methods(args.methods)

    print("=" * 78)
    print("ReCalib deterministic innovation baselines")
    print(f"Config: {config_path}")
    print(f"Methods: {methods}")
    print("=" * 78)

    data = load_experiment_data(config_path)

    (
        validation_mask,
        ood_mask,
        validation_coverage,
        ood_coverage,
    ) = build_official_de_masks(data)

    output_root = resolve_path(
        data.config["output"]["baseline_dir"]
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_coverage.to_csv(
        output_root / "validation_de_coverage.csv",
        index=False,
    )
    ood_coverage.to_csv(
        output_root / "main_ood_de_coverage.csv",
        index=False,
    )

    all_validation_summaries = []
    all_ood_summaries = []

    # Frozen CRISP is always evaluated first because all
    # improved fractions use it as the paired reference.
    frozen_validation_prediction = (
        data.validation.frozen_crisp.astype(
            np.float64,
            copy=True,
        )
    )
    frozen_ood_prediction = (
        data.main_ood.frozen_crisp.astype(
            np.float64,
            copy=True,
        )
    )

    frozen_validation_metrics = evaluate_by_condition(
        method="frozen_crisp",
        split=data.validation,
        prediction=frozen_validation_prediction,
        de_mask=validation_mask,
    )
    frozen_ood_metrics = evaluate_by_condition(
        method="frozen_crisp",
        split=data.main_ood,
        prediction=frozen_ood_prediction,
        de_mask=ood_mask,
    )

    frozen_summary_for_check = summarize_metrics(
        frozen_ood_metrics
    )
    check_frozen_crisp_reproduction(
        frozen_summary_for_check
    )

    frozen_val_summary, frozen_ood_summary = (
        run_and_save_method(
            method="frozen_crisp",
            validation_prediction=(
                frozen_validation_prediction
            ),
            ood_prediction=frozen_ood_prediction,
            data=data,
            validation_mask=validation_mask,
            ood_mask=ood_mask,
            output_root=output_root,
            frozen_ood_metrics=frozen_ood_metrics,
            parameters={
                "calibration": "none",
            },
            fit_seconds=0.0,
            inference_seconds=0.0,
        )
    )

    all_validation_summaries.append(
        frozen_val_summary
    )
    all_ood_summaries.append(frozen_ood_summary)

    if "global_mean_residual" in methods:
        model = GlobalMeanResidual()

        fit_start = time.perf_counter()
        model.fit(
            data.train.true,
            data.train.frozen_crisp,
        )
        fit_seconds = (
            time.perf_counter() - fit_start
        )

        inference_start = time.perf_counter()
        validation_prediction = model.predict(
            data.validation.frozen_crisp
        )
        ood_prediction = model.predict(
            data.main_ood.frozen_crisp
        )
        inference_seconds = (
            time.perf_counter() - inference_start
        )

        val_summary, ood_summary = (
            run_and_save_method(
                method="global_mean_residual",
                validation_prediction=(
                    validation_prediction
                ),
                ood_prediction=ood_prediction,
                data=data,
                validation_mask=validation_mask,
                ood_mask=ood_mask,
                output_root=output_root,
                frozen_ood_metrics=frozen_ood_metrics,
                parameters={
                    "mean_residual": (
                        model.mean_residual
                    ),
                },
                fit_seconds=fit_seconds,
                inference_seconds=inference_seconds,
            )
        )

        all_validation_summaries.append(
            val_summary
        )
        all_ood_summaries.append(ood_summary)

    if "gene_mean_residual" in methods:
        model = GeneMeanResidual()

        fit_start = time.perf_counter()
        model.fit(
            data.train.true,
            data.train.frozen_crisp,
        )
        fit_seconds = (
            time.perf_counter() - fit_start
        )

        inference_start = time.perf_counter()
        validation_prediction = model.predict(
            data.validation.frozen_crisp
        )
        ood_prediction = model.predict(
            data.main_ood.frozen_crisp
        )
        inference_seconds = (
            time.perf_counter() - inference_start
        )

        val_summary, ood_summary = (
            run_and_save_method(
                method="gene_mean_residual",
                validation_prediction=(
                    validation_prediction
                ),
                ood_prediction=ood_prediction,
                data=data,
                validation_mask=validation_mask,
                ood_mask=ood_mask,
                output_root=output_root,
                frozen_ood_metrics=frozen_ood_metrics,
                parameters={
                    "n_gene_offsets": (
                        data.train.n_genes
                    ),
                },
                fit_seconds=fit_seconds,
                inference_seconds=inference_seconds,
            )
        )

        all_validation_summaries.append(
            val_summary
        )
        all_ood_summaries.append(ood_summary)

    if "gene_wise_affine" in methods:
        alphas = [
            float(value)
            for value in data.config[
                "gene_affine"
            ]["ridge_alphas"]
        ]

        sweep_rows = []
        fitted_models = {}

        for alpha in alphas:
            model = GeneWiseAffine(alpha=alpha)

            fit_start = time.perf_counter()
            model.fit(
                data.train.true,
                data.train.frozen_crisp,
            )
            fit_seconds = (
                time.perf_counter() - fit_start
            )

            validation_prediction = model.predict(
                data.validation.frozen_crisp
            )
            validation_metrics = evaluate_by_condition(
                method="gene_wise_affine",
                split=data.validation,
                prediction=validation_prediction,
                de_mask=validation_mask,
            )
            validation_summary = summarize_metrics(
                validation_metrics
            )

            validation_mse = float(
                validation_summary.loc[
                    validation_summary["metric"]
                    == "mse_de",
                    "mean",
                ].iloc[0]
            )

            sweep_rows.append(
                {
                    "alpha": alpha,
                    "validation_mse_de": (
                        validation_mse
                    ),
                    "fit_seconds": fit_seconds,
                }
            )
            fitted_models[alpha] = model

            print(
                "[gene_wise_affine] "
                f"alpha={alpha:g}, "
                f"validation_mse_de="
                f"{validation_mse:.8f}"
            )

        sweep = pd.DataFrame(sweep_rows)
        sweep = sweep.sort_values(
            ["validation_mse_de", "alpha"],
            ascending=[True, True],
        ).reset_index(drop=True)

        selected_alpha = float(
            sweep.iloc[0]["alpha"]
        )
        selected_model = fitted_models[
            selected_alpha
        ]

        print(
            "[gene_wise_affine] "
            f"selected alpha={selected_alpha:g} "
            "using IID validation mse_de"
        )

        validation_prediction = (
            selected_model.predict(
                data.validation.frozen_crisp
            )
        )

        inference_start = time.perf_counter()
        ood_prediction = selected_model.predict(
            data.main_ood.frozen_crisp
        )
        inference_seconds = (
            time.perf_counter() - inference_start
        )

        method_dir = (
            output_root / "gene_wise_affine"
        )
        method_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        sweep.to_csv(
            method_dir
            / "validation_alpha_sweep.csv",
            index=False,
        )

        selected_fit_seconds = float(
            sweep.loc[
                sweep["alpha"] == selected_alpha,
                "fit_seconds",
            ].iloc[0]
        )

        val_summary, ood_summary = (
            run_and_save_method(
                method="gene_wise_affine",
                validation_prediction=(
                    validation_prediction
                ),
                ood_prediction=ood_prediction,
                data=data,
                validation_mask=validation_mask,
                ood_mask=ood_mask,
                output_root=output_root,
                frozen_ood_metrics=frozen_ood_metrics,
                parameters={
                    "selected_alpha": (
                        selected_alpha
                    ),
                    "selection_metric": (
                        "IID validation mse_de"
                    ),
                    "candidate_alphas": alphas,
                },
                fit_seconds=selected_fit_seconds,
                inference_seconds=inference_seconds,
            )
        )

        all_validation_summaries.append(
            val_summary
        )
        all_ood_summaries.append(ood_summary)

    combined_validation = pd.concat(
        all_validation_summaries,
        ignore_index=True,
    )
    combined_ood = pd.concat(
        all_ood_summaries,
        ignore_index=True,
    )

    combined_validation.to_csv(
        output_root / "validation_summary.csv",
        index=False,
    )
    combined_ood.to_csv(
        output_root / "main_ood_summary.csv",
        index=False,
    )

    print("\nCombined Main OOD summary")
    print(
        combined_ood[
            [
                "method",
                "metric",
                "mean",
                "n_valid_conditions",
                "improved_fraction_vs_frozen_crisp",
            ]
        ].to_string(index=False)
    )

    print("=" * 78)
    print(
        "[OK] Deterministic innovation baselines completed"
    )
    print(f"[OUTPUT] {output_root}")
    print("=" * 78)


if __name__ == "__main__":
    main()
