from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

from innovation_data import (
    PROJECT_ROOT,
    load_experiment_data,
    resolve_path,
)
from run_innovation_baselines import (
    PRIMARY_METRICS,
    build_official_de_masks,
    evaluate_by_condition,
    run_and_save_method,
    summarize_metrics,
)
from run_ridge_residual_baseline import (
    build_drug_vocabulary,
    build_scalar_features,
    fit_scalar_standardizer,
    build_sparse_design_matrix,
)
from train_crisp_rc_deaware import (
    ResidualCalibrator,
    make_pairs,
)


DEFAULT_SEEDS = [2024, 3407, 42]

DEFAULT_ALPHA_CANDIDATES = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.75,
    1.00,
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def encode_drug_features(
    metadata: pd.DataFrame,
    *,
    drug_column: str,
    drug_mapping: dict[str, int],
) -> np.ndarray:
    """Encode drug identity using the vocabulary learned from training only."""
    output = np.zeros(
        (len(metadata), len(drug_mapping)),
        dtype=np.float32,
    )

    unknown = []

    for row_index, drug in enumerate(
        metadata[drug_column].astype(str)
    ):
        if drug not in drug_mapping:
            unknown.append(drug)
            continue

        output[
            row_index,
            drug_mapping[drug],
        ] = 1.0

    if unknown:
        unique_unknown = sorted(set(unknown))
        print(
            "[DRUG] unseen-to-fitting drugs encoded as "
            "all-zero one-hot: "
            f"{unique_unknown}"
        )
        print(
            "[DRUG] affected conditions: "
            f"{len(unknown)}"
        )

    return output


def rows_for_conditions(
    condition_indices: np.ndarray,
    n_genes: int,
) -> np.ndarray:
    """Return flattened condition-gene row indices.

    Matrix order:
        condition 0 gene 0..G-1
        condition 1 gene 0..G-1
        ...
    """
    condition_indices = np.asarray(
        condition_indices,
        dtype=np.int64,
    )

    offsets = (
        condition_indices[:, None] * n_genes
    )

    gene_offsets = np.arange(
        n_genes,
        dtype=np.int64,
    )[None, :]

    return (
        offsets + gene_offsets
    ).reshape(-1)


def get_cell_labels(
    metadata: pd.DataFrame,
) -> np.ndarray:
    for column in [
        "cell_type",
        "cell_line",
        "cell",
    ]:
        if column in metadata.columns:
            return metadata[column].astype(str).to_numpy()

    raise ValueError(
        "Cannot identify cell-line column for Ridge cross-fitting. "
        f"Available columns: {list(metadata.columns)}"
    )


def get_selected_ridge_lambda(
    baseline_root: Path,
) -> float:
    sweep_path = (
        baseline_root
        / "ridge_residual"
        / "validation_lambda_sweep.csv"
    )

    if not sweep_path.is_file():
        raise FileNotFoundError(
            "Ridge validation sweep not found: "
            f"{sweep_path}"
        )

    sweep = pd.read_csv(sweep_path)

    required = {
        "ridge_lambda",
        "validation_mse_de",
    }

    missing = required - set(sweep.columns)
    if missing:
        raise ValueError(
            f"Ridge sweep missing columns: {sorted(missing)}"
        )

    sweep = sweep.sort_values(
        [
            "validation_mse_de",
            "ridge_lambda",
        ],
        ascending=[True, True],
    ).reset_index(drop=True)

    selected = float(
        sweep.iloc[0]["ridge_lambda"]
    )

    print(
        "[RIDGE] selected lambda from existing IID validation sweep: "
        f"{selected:g}"
    )

    return selected


def cross_fitted_ridge_anchor(
    *,
    train_design: sparse.csr_matrix,
    train_target: np.ndarray,
    frozen_crisp: np.ndarray,
    metadata: pd.DataFrame,
    n_genes: int,
    ridge_lambda: float,
    n_folds: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate condition-level out-of-fold Ridge predictions.

    The nonlinear residual network must not be trained on residuals from
    a Ridge model that has already fitted the same condition.

    Therefore, Ridge predictions for the ReCalib fitting subset are
    generated through condition-level cross-fitting.
    """
    n_conditions = frozen_crisp.shape[0]

    labels = get_cell_labels(metadata)

    splitter = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    condition_indices = np.arange(
        n_conditions,
        dtype=np.int64,
    )

    cross_fitted_residual = np.full(
        n_conditions * n_genes,
        np.nan,
        dtype=np.float64,
    )

    fold_assignment = np.full(
        n_conditions,
        -1,
        dtype=np.int64,
    )

    print(
        f"[RIDGE] building {n_folds}-fold "
        "condition-level cross-fitted training anchor"
    )

    for fold_index, (
        fit_condition_indices,
        heldout_condition_indices,
    ) in enumerate(
        splitter.split(
            condition_indices,
            labels,
        ),
        start=1,
    ):
        fit_rows = rows_for_conditions(
            fit_condition_indices,
            n_genes,
        )
        heldout_rows = rows_for_conditions(
            heldout_condition_indices,
            n_genes,
        )

        model = Ridge(
            alpha=ridge_lambda,
            fit_intercept=True,
            solver="lsqr",
            tol=1e-5,
            max_iter=5000,
        )

        start = time.perf_counter()

        model.fit(
            train_design[fit_rows],
            train_target[fit_rows],
        )

        cross_fitted_residual[
            heldout_rows
        ] = model.predict(
            train_design[heldout_rows]
        )

        fold_assignment[
            heldout_condition_indices
        ] = fold_index

        elapsed = (
            time.perf_counter() - start
        )

        print(
            f"[RIDGE] fold {fold_index}/{n_folds}: "
            f"fit_conditions={len(fit_condition_indices)}, "
            f"heldout_conditions={len(heldout_condition_indices)}, "
            f"time={elapsed:.2f}s"
        )

    if np.isnan(cross_fitted_residual).any():
        raise RuntimeError(
            "Cross-fitted Ridge residual contains missing predictions"
        )

    if np.any(fold_assignment < 0):
        raise RuntimeError(
            "Some fitting conditions were not assigned a cross-fitting fold"
        )

    cross_fitted_residual = (
        cross_fitted_residual.reshape(
            frozen_crisp.shape
        )
    )

    anchor = (
        frozen_crisp.astype(
            np.float64,
            copy=False,
        )
        + cross_fitted_residual
    )

    return anchor, fold_assignment


def fit_full_ridge_anchor(
    *,
    train_design: sparse.csr_matrix,
    validation_design: sparse.csr_matrix,
    ood_design: sparse.csr_matrix,
    train_target: np.ndarray,
    train_shape: tuple[int, int],
    validation_shape: tuple[int, int],
    ood_shape: tuple[int, int],
    train_frozen: np.ndarray,
    validation_frozen: np.ndarray,
    ood_frozen: np.ndarray,
    ridge_lambda: float,
) -> tuple[
    Ridge,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fit deterministic Ridge on the complete fitting subset."""
    model = Ridge(
        alpha=ridge_lambda,
        fit_intercept=True,
        solver="lsqr",
        tol=1e-5,
        max_iter=5000,
    )

    print(
        "[RIDGE] fitting full Ridge anchor on 100% of fitting conditions"
    )

    start = time.perf_counter()

    model.fit(
        train_design,
        train_target,
    )

    elapsed = time.perf_counter() - start

    print(
        f"[RIDGE] full model fitted in {elapsed:.2f}s"
    )

    train_residual = model.predict(
        train_design
    ).reshape(train_shape)

    validation_residual = model.predict(
        validation_design
    ).reshape(validation_shape)

    ood_residual = model.predict(
        ood_design
    ).reshape(ood_shape)

    train_anchor = (
        train_frozen + train_residual
    )
    validation_anchor = (
        validation_frozen + validation_residual
    )
    ood_anchor = (
        ood_frozen + ood_residual
    )

    return (
        model,
        train_anchor,
        validation_anchor,
        ood_anchor,
    )


def build_anchor_scalar_features(
    anchor: torch.Tensor,
    control: torch.Tensor,
) -> torch.Tensor:
    """Same five-scalar structure as ReCalib, but relative to Ridge anchor."""
    delta = anchor - control

    relative_delta = (
        delta
        / (
            torch.abs(control)
            + 1e-3
        )
    )

    return torch.stack(
        [
            anchor,
            control,
            delta,
            torch.abs(delta),
            relative_delta,
        ],
        dim=1,
    )


def predict_remaining_residual(
    *,
    model: torch.nn.Module,
    anchor: np.ndarray,
    control: np.ndarray,
    drug_features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    n_conditions, n_genes = anchor.shape

    output = np.zeros(
        (n_conditions, n_genes),
        dtype=np.float32,
    )

    anchor_tensor = torch.tensor(
        anchor,
        dtype=torch.float32,
        device=device,
    )

    control_tensor = torch.tensor(
        control,
        dtype=torch.float32,
        device=device,
    )

    drug_tensor = torch.tensor(
        drug_features,
        dtype=torch.float32,
        device=device,
    )

    condition_indices, gene_indices = (
        make_pairs(
            n_conditions,
            n_genes,
        )
    )

    with torch.no_grad():
        for start in range(
            0,
            len(condition_indices),
            batch_size,
        ):
            condition_numpy = (
                condition_indices[
                    start : start + batch_size
                ]
            )

            gene_numpy = (
                gene_indices[
                    start : start + batch_size
                ]
            )

            condition_index = torch.tensor(
                condition_numpy,
                dtype=torch.long,
                device=device,
            )

            gene_index = torch.tensor(
                gene_numpy,
                dtype=torch.long,
                device=device,
            )

            anchor_values = anchor_tensor[
                condition_index,
                gene_index,
            ]

            control_values = control_tensor[
                condition_index,
                gene_index,
            ]

            scalar_features = (
                build_anchor_scalar_features(
                    anchor_values,
                    control_values,
                )
            )

            predicted_residual = model(
                gene_index,
                drug_tensor[condition_index],
                scalar_features,
            )

            output[
                condition_numpy,
                gene_numpy,
            ] = (
                predicted_residual
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    return output


def metric_value(
    summary: pd.DataFrame,
    metric: str,
) -> float:
    values = summary.loc[
        summary["metric"] == metric,
        "mean",
    ]

    if len(values) != 1:
        raise ValueError(
            f"Expected one {metric} value; found {len(values)}"
        )

    return float(values.iloc[0])


def validation_alpha_sweep(
    *,
    seed: int,
    anchor: np.ndarray,
    nonlinear_residual: np.ndarray,
    split,
    de_mask: np.ndarray,
    alpha_candidates: list[float],
) -> pd.DataFrame:
    rows = []

    for alpha in alpha_candidates:
        prediction = (
            anchor
            + alpha * nonlinear_residual
        )

        condition_metrics = evaluate_by_condition(
            method=f"ridge_anchored_recalib_seed_{seed}",
            split=split,
            prediction=prediction,
            de_mask=de_mask,
        )

        summary = summarize_metrics(
            condition_metrics
        )

        rows.append(
            {
                "seed": seed,
                "alpha": alpha,
                "mse_de": metric_value(
                    summary,
                    "mse_de",
                ),
                "pearson_de": metric_value(
                    summary,
                    "pearson_de",
                ),
                "pearson_delta_de": metric_value(
                    summary,
                    "pearson_delta_de",
                ),
                "r2score_de": metric_value(
                    summary,
                    "r2score_de",
                ),
            }
        )

    return pd.DataFrame(rows)


def select_alpha(
    sweep: pd.DataFrame,
) -> float:
    """Preserve original ReCalib protocol: maximize validation r2score_de.

    Smaller alpha wins exact ties, providing the more conservative
    correction.
    """
    ranked = sweep.sort_values(
        [
            "r2score_de",
            "alpha",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    return float(
        ranked.iloc[0]["alpha"]
    )


def save_matrix(
    path: Path,
    matrix: np.ndarray,
    conditions: list[str],
    genes: list[str],
) -> None:
    frame = pd.DataFrame(
        np.asarray(
            matrix,
            dtype=np.float64,
        ),
        index=conditions,
        columns=genes,
    )

    frame.index.name = "condition"

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(path)


def train_nonlinear_remaining_residual(
    *,
    seed: int,
    data,
    train_anchor_crossfit: np.ndarray,
    train_drug_features: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gene_embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    predicted_residual_l2: float,
    patience: int,
    shuffle_remaining_residual: bool = False,
    shuffle_seed: int = 2024,
) -> tuple[
    torch.nn.Module,
    dict[str, Any],
]:
    set_seed(seed)

    remaining_target = (
        data.train.true.astype(
            np.float32,
            copy=False,
        )
        - train_anchor_crossfit.astype(
            np.float32,
            copy=False,
        )
    )

    # ------------------------------------------------------------
    # Negative control:
    # shuffle complete remaining-residual profiles across fitting
    # conditions while preserving the 977-gene structure within
    # each profile.
    #
    # A fixed-point-free permutation is used so that no condition
    # retains its own remaining-residual target.
    # ------------------------------------------------------------
    shuffle_permutation = None

    if shuffle_remaining_residual:
        n_conditions = remaining_target.shape[0]

        shuffle_rng = np.random.default_rng(
            shuffle_seed
        )

        original_indices = np.arange(
            n_conditions,
            dtype=np.int64,
        )

        for _ in range(10000):
            candidate = shuffle_rng.permutation(
                n_conditions
            )

            if np.all(
                candidate != original_indices
            ):
                shuffle_permutation = candidate
                break

        if shuffle_permutation is None:
            raise RuntimeError(
                "Could not generate a fixed-point-free "
                "condition permutation."
            )

        remaining_target = (
            remaining_target[
                shuffle_permutation,
                :
            ]
            .copy()
        )

        print(
            "[SHUFFLE CONTROL] Remaining-residual "
            "profiles shuffled across fitting conditions."
        )
        print(
            "[SHUFFLE CONTROL] shuffle_seed="
            f"{shuffle_seed}"
        )
        print(
            "[SHUFFLE CONTROL] conditions="
            f"{n_conditions}"
        )
        print(
            "[SHUFFLE CONTROL] fixed_points="
            f"{int(np.sum(shuffle_permutation == original_indices))}"
        )
        print(
            "[SHUFFLE CONTROL] gene structure preserved: "
            f"{remaining_target.shape[1]} genes/profile"
        )

    model = ResidualCalibrator(
        n_genes=data.train.n_genes,
        drug_dim=train_drug_features.shape[1],
        gene_emb_dim=gene_embedding_dim,
        hidden=hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    anchor_tensor = torch.tensor(
        train_anchor_crossfit,
        dtype=torch.float32,
        device=device,
    )

    control_tensor = torch.tensor(
        data.train.control,
        dtype=torch.float32,
        device=device,
    )

    target_tensor = torch.tensor(
        remaining_target,
        dtype=torch.float32,
        device=device,
    )

    drug_tensor = torch.tensor(
        train_drug_features,
        dtype=torch.float32,
        device=device,
    )

    condition_indices, gene_indices = (
        make_pairs(
            data.train.n_conditions,
            data.train.n_genes,
        )
    )

    number_of_pairs = len(
        condition_indices
    )

    rng = np.random.default_rng(seed)

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    completed_epochs = 0

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    training_start = time.perf_counter()

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()
        completed_epochs = epoch

        permutation = rng.permutation(
            number_of_pairs
        )

        epoch_losses = []

        for start in range(
            0,
            number_of_pairs,
            batch_size,
        ):
            pair_indices = permutation[
                start : start + batch_size
            ]

            condition_numpy = (
                condition_indices[
                    pair_indices
                ]
            )

            gene_numpy = (
                gene_indices[
                    pair_indices
                ]
            )

            condition_index = torch.tensor(
                condition_numpy,
                dtype=torch.long,
                device=device,
            )

            gene_index = torch.tensor(
                gene_numpy,
                dtype=torch.long,
                device=device,
            )

            anchor_values = anchor_tensor[
                condition_index,
                gene_index,
            ]

            control_values = control_tensor[
                condition_index,
                gene_index,
            ]

            target = target_tensor[
                condition_index,
                gene_index,
            ]

            scalar_features = (
                build_anchor_scalar_features(
                    anchor_values,
                    control_values,
                )
            )

            predicted_remaining = model(
                gene_index,
                drug_tensor[
                    condition_index
                ],
                scalar_features,
            )

            residual_fit_loss = (
                (
                    predicted_remaining
                    - target
                )
                ** 2
            ).mean()

            residual_magnitude_penalty = (
                predicted_remaining.pow(2)
                .mean()
            )

            loss = (
                residual_fit_loss
                + predicted_residual_l2
                * residual_magnitude_penalty
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.detach().cpu()
                )
            )

        epoch_loss = float(
            np.mean(epoch_losses)
        )

        if (
            epoch_loss
            < best_loss - 1e-10
        ):
            best_loss = epoch_loss

            best_state = {
                key: value
                .detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            bad_epochs = 0

        else:
            bad_epochs += 1

        if (
            epoch == 1
            or epoch % 10 == 0
        ):
            print(
                f"[seed={seed}] "
                f"epoch={epoch:03d} "
                f"loss={epoch_loss:.8f} "
                f"best={best_loss:.8f}"
            )

        if bad_epochs >= patience:
            print(
                f"[seed={seed}] "
                f"early stopping at epoch {epoch}"
            )
            break

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    if best_state is None:
        raise RuntimeError(
            "No model checkpoint was produced"
        )

    model.load_state_dict(
        best_state
    )

    if device.type == "cuda":
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated()
            / 1024
            / 1024
        )
    else:
        peak_gpu_memory_mb = 0.0

    metadata = {
        "seed": seed,
        "trainable_parameters": (
            trainable_parameters
        ),
        "completed_epochs": (
            completed_epochs
        ),
        "best_training_loss": (
            best_loss
        ),
        "training_seconds": (
            training_seconds
        ),
        "peak_gpu_memory_mb": (
            peak_gpu_memory_mb
        ),
    }

    metadata["shuffle_remaining_residual"] = bool(
        shuffle_remaining_residual
    )
    metadata["shuffle_seed"] = (
        int(shuffle_seed)
        if shuffle_remaining_residual
        else None
    )
    metadata["shuffle_fixed_points"] = (
        int(
            np.sum(
                shuffle_permutation
                == np.arange(
                    len(shuffle_permutation)
                )
            )
        )
        if shuffle_permutation is not None
        else None
    )

    return model, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Ridge-anchored ReCalib: "
            "Ridge absorbs linear residual structure and "
            "ReCalib predicts only the remaining nonlinear residual."
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
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--ridge-cv-folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--selection-seed",
        type=int,
        default=2024,
    )

    parser.add_argument(
        "--ridge-lambda",
        type=float,
        default=None,
        help=(
            "Use the specified Ridge lambda. "
            "If omitted, lambda is loaded according to "
            "the existing IID validation protocol."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/ridge_anchored_recalib"
        ),
    )

    parser.add_argument(
        "--shuffle-remaining-residual",
        action="store_true",
        help=(
            "Negative control: shuffle complete stage-2 "
            "remaining-residual profiles across fitting "
            "conditions while preserving gene structure."
        ),
    )

    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=2024,
        help=(
            "Random seed for the condition-level "
            "remaining-residual permutation."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Use a pre-specified residual scaling alpha "
            "instead of selecting alpha from validation. "
            "Intended for matched negative controls."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = args.config

    if not config_path.is_absolute():
        config_path = (
            PROJECT_ROOT
            / config_path
        ).resolve()

    output_root = args.output_dir

    if not output_root.is_absolute():
        output_root = (
            PROJECT_ROOT
            / output_root
        ).resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print("Ridge-anchored ReCalib")
    print(f"Config: {config_path}")
    print(f"Seeds: {args.seeds}")
    print(f"Selection seed: {args.selection_seed}")
    print(f"Output: {output_root}")
    print("=" * 88)

    if args.selection_seed not in args.seeds:
        raise ValueError(
            "selection-seed must be included in --seeds"
        )

    # Make sure selection seed is always trained first.
    seeds = [
        args.selection_seed
    ] + [
        seed
        for seed in args.seeds
        if seed != args.selection_seed
    ]

    data = load_experiment_data(
        config_path
    )

    (
        validation_mask,
        ood_mask,
        _,
        _,
    ) = build_official_de_masks(data)

    baseline_root = resolve_path(
        data.config["output"][
            "baseline_dir"
        ]
    )

    if args.ridge_lambda is not None:
        ridge_lambda = float(
            args.ridge_lambda
        )

        print(
            "[RIDGE] Using specified lambda: "
            f"{ridge_lambda:g}"
        )

    else:
        ridge_lambda = (
            get_selected_ridge_lambda(
                baseline_root
            )
        )

    (
        drug_column,
        drug_vocabulary,
        drug_mapping,
    ) = build_drug_vocabulary(
        data.train.metadata
    )

    print(
        f"[DRUG] fitting vocabulary: "
        f"{len(drug_vocabulary)} drugs"
    )

    train_drug = encode_drug_features(
        data.train.metadata,
        drug_column=drug_column,
        drug_mapping=drug_mapping,
    )

    validation_drug = encode_drug_features(
        data.validation.metadata,
        drug_column=drug_column,
        drug_mapping=drug_mapping,
    )

    ood_drug = encode_drug_features(
        data.main_ood.metadata,
        drug_column=drug_column,
        drug_mapping=drug_mapping,
    )

    train_scalar = (
        build_scalar_features(
            data.train
        )
    )

    (
        scalar_means,
        scalar_standard_deviations,
    ) = fit_scalar_standardizer(
        train_scalar
    )

    print("[RIDGE] building design matrices")

    (
        train_design,
        train_design_info,
    ) = build_sparse_design_matrix(
        split=data.train,
        drug_column=drug_column,
        drug_mapping=drug_mapping,
        scalar_means=scalar_means,
        scalar_standard_deviations=(
            scalar_standard_deviations
        ),
    )

    (
        validation_design,
        validation_design_info,
    ) = build_sparse_design_matrix(
        split=data.validation,
        drug_column=drug_column,
        drug_mapping=drug_mapping,
        scalar_means=scalar_means,
        scalar_standard_deviations=(
            scalar_standard_deviations
        ),
    )

    (
        ood_design,
        ood_design_info,
    ) = build_sparse_design_matrix(
        split=data.main_ood,
        drug_column=drug_column,
        drug_mapping=drug_mapping,
        scalar_means=scalar_means,
        scalar_standard_deviations=(
            scalar_standard_deviations
        ),
    )

    train_target = (
        data.train.residual
        .astype(
            np.float64,
            copy=False,
        )
        .reshape(-1)
    )

    # ----------------------------------------------------------
    # IMPORTANT:
    # Stage-2 network uses cross-fitted Ridge predictions on
    # fitting conditions. This prevents in-sample Ridge fitting
    # from artificially simplifying the stage-2 training target.
    # ----------------------------------------------------------
    (
        train_anchor_crossfit,
        ridge_fold_assignment,
    ) = cross_fitted_ridge_anchor(
        train_design=train_design,
        train_target=train_target,
        frozen_crisp=(
            data.train.frozen_crisp
        ),
        metadata=data.train.metadata,
        n_genes=data.train.n_genes,
        ridge_lambda=ridge_lambda,
        n_folds=args.ridge_cv_folds,
        random_state=2024,
    )

    # Full Ridge is used for actual validation/OOD inference.
    (
        full_ridge_model,
        train_anchor_full,
        validation_anchor,
        ood_anchor,
    ) = fit_full_ridge_anchor(
        train_design=train_design,
        validation_design=(
            validation_design
        ),
        ood_design=ood_design,
        train_target=train_target,
        train_shape=data.train.true.shape,
        validation_shape=(
            data.validation.true.shape
        ),
        ood_shape=(
            data.main_ood.true.shape
        ),
        train_frozen=(
            data.train.frozen_crisp
        ),
        validation_frozen=(
            data.validation.frozen_crisp
        ),
        ood_frozen=(
            data.main_ood.frozen_crisp
        ),
        ridge_lambda=ridge_lambda,
    )

    np.save(
        output_root
        / "ridge_crossfit_fold_assignment.npy",
        ridge_fold_assignment,
    )

    save_matrix(
        output_root
        / "train_ridge_anchor_crossfit.csv",
        train_anchor_crossfit,
        data.train.conditions,
        data.train.genes,
    )

    save_matrix(
        output_root
        / "validation_ridge_anchor.csv",
        validation_anchor,
        data.validation.conditions,
        data.validation.genes,
    )

    save_matrix(
        output_root
        / "main_ood_ridge_anchor.csv",
        ood_anchor,
        data.main_ood.conditions,
        data.main_ood.genes,
    )

    # Sanity-check that our deterministic Ridge anchor reproduces
    # the previously established Ridge result.
    ridge_ood_metrics = (
        evaluate_by_condition(
            method="ridge_anchor",
            split=data.main_ood,
            prediction=ood_anchor,
            de_mask=ood_mask,
        )
    )

    ridge_ood_summary = (
        summarize_metrics(
            ridge_ood_metrics
        )
    )

    ridge_ood_summary.to_csv(
        output_root
        / "ridge_anchor_main_ood_summary.csv",
        index=False,
    )

    print("\n[Ridge anchor] Main OOD")
    print(
        ridge_ood_summary[
            [
                "metric",
                "mean",
                "n_valid_conditions",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------
    # Primary Main-OOD Ridge reproducibility guard.
    # The hard-coded reference values below belong specifically to
    # the primary Main-OOD protocol. Alternative OOD protocols use
    # different fitting/validation pools and therefore legitimately
    # produce different fitted Ridge anchors.
    # ------------------------------------------------------------
    if config_path.name == "innovation_experiments.yaml":
        # Existing Ridge result should be reproduced.
        expected_ridge = {
            "mse_de": 0.017511,
            "pearson_de": 0.877750,
            "pearson_delta_de": 0.493635,
            "r2score_de": 0.643481,
        }

        for metric, expected in expected_ridge.items():
            actual = metric_value(
                ridge_ood_summary,
                metric,
            )

            difference = abs(
                actual - expected
            )

            print(
                f"[RIDGE CHECK] "
                f"{metric}: "
                f"actual={actual:.6f}, "
                f"expected≈{expected:.6f}, "
                f"abs_diff={difference:.8f}"
            )

            if difference > 5e-4:
                raise RuntimeError(
                    "Ridge anchor does not reproduce the "
                    "previous Ridge result. Stop before "
                    "training the nonlinear stage."
                )

    else:
        print(
            "[RIDGE CHECK] Hard-coded primary Main-OOD "
            "reference check skipped for protocol config: "
            f"{config_path.name}"
        )

    frozen_ood_metrics = (
        evaluate_by_condition(
            method="frozen_crisp",
            split=data.main_ood,
            prediction=(
                data.main_ood.frozen_crisp
            ),
            de_mask=ood_mask,
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[DEVICE] {device}"
    )

    recalib_config = (
        data.config["recalib"]
    )

    epochs = (
        int(args.epochs)
        if args.epochs is not None
        else int(
            recalib_config["epochs"]
        )
    )

    batch_size = int(
        recalib_config["batch_size"]
    )

    learning_rate = float(
        recalib_config[
            "learning_rate"
        ]
    )

    weight_decay = float(
        recalib_config[
            "weight_decay"
        ]
    )

    hidden_dim = int(
        recalib_config[
            "hidden_dim"
        ]
    )

    gene_embedding_dim = int(
        recalib_config[
            "gene_embedding_dim"
        ]
    )

    dropout = float(
        recalib_config[
            "dropout"
        ]
    )

    patience = int(
        recalib_config[
            "patience"
        ]
    )

    predicted_residual_l2 = float(
        recalib_config.get(
            "predicted_residual_l2",
            1e-4,
        )
    )

    selected_alpha = (
        float(args.alpha)
        if args.alpha is not None
        else None
    )

    if selected_alpha is not None:
        print(
            "[ALPHA] Using specified alpha: "
            f"{selected_alpha:.2f}"
        )
        print(
            "[ALPHA] IID validation sweep will be saved "
            "for diagnostics only and will NOT be used "
            "for parameter selection."
        )

    run_summaries = []
    cost_rows = []

    for seed in seeds:
        print("\n" + "=" * 88)
        print(
            f"Ridge-anchored ReCalib | seed={seed}"
        )
        print("=" * 88)

        (
            model,
            training_metadata,
        ) = (
            train_nonlinear_remaining_residual(
                seed=seed,
                data=data,
                train_anchor_crossfit=(
                    train_anchor_crossfit
                ),
                train_drug_features=(
                    train_drug
                ),
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=(
                    learning_rate
                ),
                weight_decay=(
                    weight_decay
                ),
                gene_embedding_dim=(
                    gene_embedding_dim
                ),
                hidden_dim=hidden_dim,
                dropout=dropout,
                predicted_residual_l2=(
                    predicted_residual_l2
                ),
                patience=patience,
                shuffle_remaining_residual=(
                    args.shuffle_remaining_residual
                ),
                shuffle_seed=args.shuffle_seed,
            )
        )

        print(
            f"[seed={seed}] "
            "predicting remaining validation residual"
        )

        validation_remaining = (
            predict_remaining_residual(
                model=model,
                anchor=validation_anchor,
                control=(
                    data.validation.control
                ),
                drug_features=(
                    validation_drug
                ),
                batch_size=batch_size,
                device=device,
            )
        )

        alpha_sweep = (
            validation_alpha_sweep(
                seed=seed,
                anchor=validation_anchor,
                nonlinear_residual=(
                    validation_remaining
                ),
                split=data.validation,
                de_mask=validation_mask,
                alpha_candidates=(
                    DEFAULT_ALPHA_CANDIDATES
                ),
            )
        )

        seed_directory = (
            output_root
            / f"seed_{seed}"
        )

        seed_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        alpha_sweep.to_csv(
            seed_directory
            / "validation_alpha_sweep.csv",
            index=False,
        )

        if (
            seed == args.selection_seed
            and args.alpha is None
        ):
            selected_alpha = (
                select_alpha(
                    alpha_sweep
                )
            )

            (
                output_root
                / "selected_alpha.json"
            ).write_text(
                json.dumps(
                    {
                        "selection_seed": seed,
                        "selection_metric": (
                            "IID validation r2score_de"
                        ),
                        "selected_alpha": (
                            selected_alpha
                        ),
                        "alpha_candidates": (
                            DEFAULT_ALPHA_CANDIDATES
                        ),
                        "note": (
                            "alpha=0 is included so the "
                            "method can fall back to Ridge "
                            "when nonlinear correction is "
                            "not supported by IID validation."
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                f"[ALPHA] selected on seed {seed}: "
                f"{selected_alpha:.2f}"
            )

        if selected_alpha is None:
            raise RuntimeError(
                "Alpha has not been selected"
            )

        print(
            f"[ALPHA] seed={seed} uses locked "
            f"alpha={selected_alpha:.2f}"
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_start = (
            time.perf_counter()
        )

        ood_remaining = (
            predict_remaining_residual(
                model=model,
                anchor=ood_anchor,
                control=(
                    data.main_ood.control
                ),
                drug_features=ood_drug,
                batch_size=batch_size,
                device=device,
            )
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_seconds = (
            time.perf_counter()
            - inference_start
        )

        validation_final = (
            validation_anchor
            + selected_alpha
            * validation_remaining
        )

        ood_final = (
            ood_anchor
            + selected_alpha
            * ood_remaining
        )

        save_matrix(
            seed_directory
            / "predicted_remaining_residual_main_ood.csv",
            ood_remaining,
            data.main_ood.conditions,
            data.main_ood.genes,
        )

        save_matrix(
            seed_directory
            / "ridge_anchor_main_ood.csv",
            ood_anchor,
            data.main_ood.conditions,
            data.main_ood.genes,
        )

        method_name = (
            f"ridge_anchored_recalib_seed_{seed}"
        )

        _, ood_summary = (
            run_and_save_method(
                method=method_name,
                validation_prediction=(
                    validation_final
                ),
                ood_prediction=(
                    ood_final
                ),
                data=data,
                validation_mask=(
                    validation_mask
                ),
                ood_mask=ood_mask,
                output_root=output_root,
                frozen_ood_metrics=(
                    frozen_ood_metrics
                ),
                parameters={
                    "seed": seed,
                    "ridge_lambda": (
                        ridge_lambda
                    ),
                    "ridge_training_anchor": (
                        f"{args.ridge_cv_folds}-fold "
                        "condition-level cross-fitted"
                    ),
                    "nonlinear_target": (
                        "observed expression minus "
                        "cross-fitted Ridge anchor"
                    ),
                    "selected_alpha": (
                        selected_alpha
                    ),
                    "alpha_selection_seed": (
                        args.selection_seed
                    ),
                    "alpha_selection_metric": (
                        "IID validation r2score_de"
                    ),
                    "gene_embedding_dim": (
                        gene_embedding_dim
                    ),
                    "hidden_dim": (
                        hidden_dim
                    ),
                    "dropout": dropout,
                    "learning_rate": (
                        learning_rate
                    ),
                    "weight_decay": (
                        weight_decay
                    ),
                    "predicted_residual_l2": (
                        predicted_residual_l2
                    ),
                    "batch_size": (
                        batch_size
                    ),
                    "maximum_epochs": (
                        epochs
                    ),
                    **training_metadata,
                },
                fit_seconds=(
                    training_metadata[
                        "training_seconds"
                    ]
                ),
                inference_seconds=(
                    inference_seconds
                ),
            )
        )

        ood_summary["seed"] = seed
        run_summaries.append(
            ood_summary
        )

        cost_rows.append(
            {
                "seed": seed,
                "selected_alpha": (
                    selected_alpha
                ),
                **training_metadata,
                "main_ood_inference_seconds": (
                    inference_seconds
                ),
            }
        )

        torch.save(
            {
                "state_dict": (
                    model.state_dict()
                ),
                "seed": seed,
                "selected_alpha": (
                    selected_alpha
                ),
                "ridge_lambda": (
                    ridge_lambda
                ),
                "training_metadata": (
                    training_metadata
                ),
            },
            seed_directory
            / "model.pt",
        )

    summaries = pd.concat(
        run_summaries,
        ignore_index=True,
    )

    summaries.to_csv(
        output_root
        / "seed_summaries.csv",
        index=False,
    )

    aggregate = (
        summaries
        .groupby(
            "metric",
            as_index=False,
        )
        .agg(
            mean=("mean", "mean"),
            sample_sd=(
                "mean",
                lambda values: values.std(
                    ddof=1
                ),
            ),
            n_valid_conditions=(
                "n_valid_conditions",
                "min",
            ),
            mean_improved_fraction_vs_frozen_crisp=(
                "improved_fraction_vs_frozen_crisp",
                "mean",
            ),
        )
    )

    aggregate.to_csv(
        output_root
        / "mean_sd_summary.csv",
        index=False,
    )

    pd.DataFrame(
        cost_rows
    ).to_csv(
        output_root
        / "cost_summary.csv",
        index=False,
    )

    run_metadata = {
        "method": (
            "Ridge-anchored ReCalib"
        ),
        "ridge_lambda": (
            ridge_lambda
        ),
        "ridge_crossfit_folds": (
            args.ridge_cv_folds
        ),
        "selection_seed": (
            args.selection_seed
        ),
        "selected_alpha": (
            selected_alpha
        ),
        "alpha_candidates": (
            DEFAULT_ALPHA_CANDIDATES
        ),
        "train_conditions": (
            data.train.n_conditions
        ),
        "validation_conditions": (
            data.validation.n_conditions
        ),
        "main_ood_conditions": (
            data.main_ood.n_conditions
        ),
        "genes": (
            data.main_ood.n_genes
        ),
        "drug_vocabulary_size": (
            len(drug_vocabulary)
        ),
        "ridge_design": {
            "train": train_design_info,
            "validation": (
                validation_design_info
            ),
            "main_ood": (
                ood_design_info
            ),
        },
    }

    (
        output_root
        / "run_metadata.json"
    ).write_text(
        json.dumps(
            run_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("Ridge-anchored ReCalib mean ± sample SD")
    print("=" * 88)

    print(
        aggregate.to_string(
            index=False
        )
    )

    print("\n[OK] Ridge-anchored ReCalib completed")
    print(f"[OUTPUT] {output_root}")


if __name__ == "__main__":
    main()
