from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from innovation_data import (
    PROJECT_ROOT,
    ConditionSplit,
    load_experiment_data,
)
from run_innovation_baselines import (
    PRIMARY_METRICS,
    build_official_de_masks,
    evaluate_by_condition,
    run_and_save_method,
    summarize_metrics,
)


METHOD_NAME = "ridge_residual"


def require_metadata_column(
    metadata: pd.DataFrame,
    candidates: list[str],
    field_name: str,
) -> str:
    for column in candidates:
        if column in metadata.columns:
            return column

    raise ValueError(
        f"Cannot identify {field_name} column. "
        f"Available columns: {list(metadata.columns)}"
    )


def build_drug_vocabulary(
    train_metadata: pd.DataFrame,
) -> tuple[str, list[str], dict[str, int]]:
    drug_column = require_metadata_column(
        train_metadata,
        ["drug", "perturbation"],
        "drug",
    )

    drugs = sorted(
        train_metadata[drug_column]
        .astype(str)
        .unique()
        .tolist()
    )

    mapping = {
        drug: index
        for index, drug in enumerate(drugs)
    }

    return drug_column, drugs, mapping


def build_scalar_features(
    split: ConditionSplit,
) -> np.ndarray:
    base = split.frozen_crisp.astype(
        np.float64,
        copy=False,
    )
    control = split.control.astype(
        np.float64,
        copy=False,
    )

    delta = base - control
    absolute_delta = np.abs(delta)
    relative_delta = delta / (
        np.abs(control) + 1e-3
    )

    scalar_features = np.column_stack(
        [
            base.reshape(-1),
            control.reshape(-1),
            delta.reshape(-1),
            absolute_delta.reshape(-1),
            relative_delta.reshape(-1),
        ]
    )

    if not np.isfinite(scalar_features).all():
        raise ValueError(
            f"{split.name}: scalar features contain "
            "non-finite values"
        )

    return scalar_features


def fit_scalar_standardizer(
    train_scalar_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(
        train_scalar_features,
        axis=0,
        dtype=np.float64,
    )
    standard_deviations = np.std(
        train_scalar_features,
        axis=0,
        ddof=0,
        dtype=np.float64,
    )

    standard_deviations[
        standard_deviations < 1e-8
    ] = 1.0

    return means, standard_deviations


def build_sparse_design_matrix(
    *,
    split: ConditionSplit,
    drug_column: str,
    drug_mapping: dict[str, int],
    scalar_means: np.ndarray,
    scalar_standard_deviations: np.ndarray,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    n_conditions = split.n_conditions
    n_genes = split.n_genes
    n_rows = n_conditions * n_genes
    n_drugs = len(drug_mapping)

    # Gene one-hot: condition-major ordering.
    row_indices = np.arange(
        n_rows,
        dtype=np.int64,
    )
    gene_indices = np.tile(
        np.arange(n_genes, dtype=np.int64),
        n_conditions,
    )

    gene_matrix = sparse.csr_matrix(
        (
            np.ones(n_rows, dtype=np.float32),
            (row_indices, gene_indices),
        ),
        shape=(n_rows, n_genes),
        dtype=np.float32,
    )

    split_drugs = (
        split.metadata[drug_column]
        .astype(str)
        .tolist()
    )

    condition_drug_indices = np.asarray(
        [
            drug_mapping.get(drug, -1)
            for drug in split_drugs
        ],
        dtype=np.int64,
    )

    unknown_conditions = [
        split.conditions[index]
        for index, value
        in enumerate(condition_drug_indices)
        if value < 0
    ]

    repeated_drug_indices = np.repeat(
        condition_drug_indices,
        n_genes,
    )
    known_rows = np.flatnonzero(
        repeated_drug_indices >= 0
    )

    drug_matrix = sparse.csr_matrix(
        (
            np.ones(
                len(known_rows),
                dtype=np.float32,
            ),
            (
                known_rows,
                repeated_drug_indices[known_rows],
            ),
        ),
        shape=(n_rows, n_drugs),
        dtype=np.float32,
    )

    scalar_features = build_scalar_features(split)
    scalar_features = (
        scalar_features - scalar_means[None, :]
    ) / scalar_standard_deviations[None, :]

    scalar_matrix = sparse.csr_matrix(
        scalar_features.astype(
            np.float32,
            copy=False,
        )
    )

    design = sparse.hstack(
        [
            gene_matrix,
            drug_matrix,
            scalar_matrix,
        ],
        format="csr",
        dtype=np.float32,
    )

    diagnostics = {
        "rows": int(design.shape[0]),
        "columns": int(design.shape[1]),
        "nonzero_entries": int(design.nnz),
        "gene_features": n_genes,
        "drug_features": n_drugs,
        "scalar_features": 5,
        "unknown_drug_conditions": (
            unknown_conditions
        ),
    }

    return design, diagnostics


def get_metric_mean(
    summary: pd.DataFrame,
    metric: str,
) -> float:
    values = summary.loc[
        summary["metric"] == metric,
        "mean",
    ]

    if len(values) != 1:
        raise ValueError(
            f"Expected one summary value for {metric}, "
            f"found {len(values)}"
        )

    return float(values.iloc[0])


def main() -> None:
    config_path = (
        PROJECT_ROOT
        / "configs"
        / "innovation_experiments.yaml"
    )

    print("=" * 78)
    print("Matched-information Ridge residual regression")
    print(f"Config: {config_path}")
    print("=" * 78)

    data = load_experiment_data(config_path)

    (
        validation_mask,
        ood_mask,
        _,
        _,
    ) = build_official_de_masks(data)

    drug_column, drug_vocabulary, drug_mapping = (
        build_drug_vocabulary(
            data.train.metadata
        )
    )

    print(
        f"[INFO] Drug metadata column: {drug_column}"
    )
    print(
        f"[INFO] Training drug vocabulary: "
        f"{len(drug_vocabulary)}"
    )

    train_scalar = build_scalar_features(
        data.train
    )
    scalar_means, scalar_standard_deviations = (
        fit_scalar_standardizer(train_scalar)
    )

    print("[INFO] Building sparse design matrices")

    train_design, train_diagnostics = (
        build_sparse_design_matrix(
            split=data.train,
            drug_column=drug_column,
            drug_mapping=drug_mapping,
            scalar_means=scalar_means,
            scalar_standard_deviations=(
                scalar_standard_deviations
            ),
        )
    )
    validation_design, validation_diagnostics = (
        build_sparse_design_matrix(
            split=data.validation,
            drug_column=drug_column,
            drug_mapping=drug_mapping,
            scalar_means=scalar_means,
            scalar_standard_deviations=(
                scalar_standard_deviations
            ),
        )
    )
    ood_design, ood_diagnostics = (
        build_sparse_design_matrix(
            split=data.main_ood,
            drug_column=drug_column,
            drug_mapping=drug_mapping,
            scalar_means=scalar_means,
            scalar_standard_deviations=(
                scalar_standard_deviations
            ),
        )
    )

    for split_name, diagnostics in [
        ("train", train_diagnostics),
        ("validation", validation_diagnostics),
        ("main_ood", ood_diagnostics),
    ]:
        print(
            f"[OK] {split_name}: "
            f"{diagnostics['rows']} rows × "
            f"{diagnostics['columns']} features, "
            f"nnz={diagnostics['nonzero_entries']}"
        )

        if diagnostics["unknown_drug_conditions"]:
            print(
                f"[WARN] {split_name}: unknown drugs for "
                f"{len(diagnostics['unknown_drug_conditions'])} "
                "conditions"
            )
        else:
            print(
                f"[OK] {split_name}: no unknown drug identity"
            )

    train_target = (
        data.train.residual
        .astype(np.float64, copy=False)
        .reshape(-1)
    )

    ridge_alphas = [
        float(value)
        for value in data.config[
            "ridge_residual"
        ]["ridge_alphas"]
    ]

    output_root = (
        PROJECT_ROOT
        / data.config["output"]["baseline_dir"]
    )
    method_dir = output_root / METHOD_NAME
    method_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen_ood_metrics = evaluate_by_condition(
        method="frozen_crisp",
        split=data.main_ood,
        prediction=data.main_ood.frozen_crisp,
        de_mask=ood_mask,
    )

    sweep_rows: list[dict[str, Any]] = []
    fitted_models: dict[float, Ridge] = {}

    for ridge_lambda in ridge_alphas:
        print(
            "\n"
            + "-" * 78
            + f"\n[FIT] ridge_lambda={ridge_lambda:g}"
        )

        model = Ridge(
            alpha=ridge_lambda,
            fit_intercept=True,
            solver="lsqr",
            tol=1e-5,
            max_iter=5000,
        )

        fit_start = time.perf_counter()
        model.fit(
            train_design,
            train_target,
        )
        fit_seconds = (
            time.perf_counter() - fit_start
        )

        validation_start = time.perf_counter()
        validation_residual = model.predict(
            validation_design
        ).reshape(
            data.validation.true.shape
        )
        validation_prediction = (
            data.validation.frozen_crisp
            + validation_residual
        )
        validation_seconds = (
            time.perf_counter()
            - validation_start
        )

        validation_metrics = evaluate_by_condition(
            method=METHOD_NAME,
            split=data.validation,
            prediction=validation_prediction,
            de_mask=validation_mask,
        )
        validation_summary = summarize_metrics(
            validation_metrics
        )

        row: dict[str, Any] = {
            "ridge_lambda": ridge_lambda,
            "fit_seconds": fit_seconds,
            "validation_inference_seconds": (
                validation_seconds
            ),
        }

        for metric in PRIMARY_METRICS:
            row[f"validation_{metric}"] = (
                get_metric_mean(
                    validation_summary,
                    metric,
                )
            )

        sweep_rows.append(row)
        fitted_models[ridge_lambda] = model

        print(
            f"[VAL] mse_de="
            f"{row['validation_mse_de']:.8f}, "
            f"pearson_de="
            f"{row['validation_pearson_de']:.8f}, "
            f"pearson_delta_de="
            f"{row['validation_pearson_delta_de']:.8f}, "
            f"r2score_de="
            f"{row['validation_r2score_de']:.8f}, "
            f"fit={fit_seconds:.2f}s"
        )

    sweep = pd.DataFrame(sweep_rows)
    sweep = sweep.sort_values(
        [
            "validation_mse_de",
            "ridge_lambda",
        ],
        ascending=[True, True],
    ).reset_index(drop=True)

    selected_lambda = float(
        sweep.iloc[0]["ridge_lambda"]
    )
    selected_model = fitted_models[
        selected_lambda
    ]

    print(
        "\n"
        + "=" * 78
        + f"\n[SELECTED] ridge_lambda="
        f"{selected_lambda:g} "
        "using IID validation mse_de"
    )

    validation_residual = (
        selected_model.predict(
            validation_design
        ).reshape(
            data.validation.true.shape
        )
    )
    validation_prediction = (
        data.validation.frozen_crisp
        + validation_residual
    )

    inference_start = time.perf_counter()
    ood_residual = selected_model.predict(
        ood_design
    ).reshape(
        data.main_ood.true.shape
    )
    ood_prediction = (
        data.main_ood.frozen_crisp
        + ood_residual
    )
    inference_seconds = (
        time.perf_counter()
        - inference_start
    )

    sweep.to_csv(
        method_dir / "validation_lambda_sweep.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": selected_model,
            "drug_vocabulary": drug_vocabulary,
            "drug_column": drug_column,
            "scalar_feature_names": [
                "frozen_crisp_prediction",
                "matched_control_expression",
                "predicted_delta",
                "absolute_predicted_delta",
                "relative_predicted_delta",
            ],
            "scalar_means": scalar_means,
            "scalar_standard_deviations": (
                scalar_standard_deviations
            ),
            "genes": data.train.genes,
        },
        method_dir / "model.joblib",
    )

    selected_fit_seconds = float(
        sweep.loc[
            sweep["ridge_lambda"]
            == selected_lambda,
            "fit_seconds",
        ].iloc[0]
    )

    run_and_save_method(
        method=METHOD_NAME,
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
            "selected_ridge_lambda": (
                selected_lambda
            ),
            "selection_metric": (
                "IID validation mse_de"
            ),
            "candidate_ridge_lambdas": (
                ridge_alphas
            ),
            "gene_identity": (
                "sparse one-hot"
            ),
            "drug_identity": (
                "sparse one-hot from training vocabulary"
            ),
            "scalar_features": [
                "frozen_crisp_prediction",
                "matched_control_expression",
                "predicted_delta",
                "absolute_predicted_delta",
                "relative_predicted_delta",
            ],
            "continuous_feature_standardization": (
                "fitting-subset mean and standard deviation"
            ),
            "design_matrix_diagnostics": {
                "train": train_diagnostics,
                "validation": validation_diagnostics,
                "main_ood": ood_diagnostics,
            },
        },
        fit_seconds=selected_fit_seconds,
        inference_seconds=inference_seconds,
    )

    print("\nValidation sweep")
    print(
        sweep[
            [
                "ridge_lambda",
                "validation_mse_de",
                "validation_pearson_de",
                "validation_pearson_delta_de",
                "validation_r2score_de",
                "fit_seconds",
            ]
        ].to_string(index=False)
    )

    print("=" * 78)
    print("[OK] Ridge residual baseline completed")
    print(f"[OUTPUT] {method_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
