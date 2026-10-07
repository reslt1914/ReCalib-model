from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from innovation_data import PROJECT_ROOT, load_experiment_data, resolve_path
from run_innovation_baselines import (
    build_official_de_masks,
    evaluate_by_condition,
    run_and_save_method,
)
from train_crisp_rc_deaware import ResidualCalibrator, make_pairs


SEEDS = [2024, 3407, 42]
METHOD_PREFIX = "direct_mlp_strict"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_drug_features(data):
    """Build drug identity vocabulary from fitting conditions only.

    Drugs absent from the 1003-condition fitting subset are represented
    by an all-zero vector during validation/OOD inference.
    """
    train_drugs = sorted(
        set(
            data.train.metadata["drug"]
            .astype(str)
            .tolist()
        )
    )

    drug_to_index = {
        drug: index
        for index, drug in enumerate(train_drugs)
    }

    def encode(
        metadata: pd.DataFrame,
        split_name: str,
    ) -> np.ndarray:
        output = np.zeros(
            (
                len(metadata),
                len(train_drugs),
            ),
            dtype=np.float32,
        )

        unknown = []

        for row_index, drug in enumerate(
            metadata["drug"].astype(str)
        ):
            index = drug_to_index.get(drug)

            if index is None:
                unknown.append(drug)
                continue

            output[
                row_index,
                index,
            ] = 1.0

        if unknown:
            unique_unknown = sorted(
                set(unknown)
            )

            print(
                f"[DRUG] {split_name}: "
                "unseen-to-fitting drugs encoded "
                "as all-zero vectors:"
            )

            for drug in unique_unknown:
                print(f"       {drug}")

            print(
                f"[DRUG] {split_name}: "
                f"affected conditions={len(unknown)}"
            )

        return output

    return (
        encode(
            data.train.metadata,
            "train",
        ),
        encode(
            data.validation.metadata,
            "validation",
        ),
        encode(
            data.main_ood.metadata,
            "main_ood",
        ),
        train_drugs,
    )

def predict_expression(
    *,
    model,
    split,
    drug_features,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    n_conditions = split.n_conditions
    n_genes = split.n_genes

    output = np.zeros(
        (n_conditions, n_genes),
        dtype=np.float32,
    )

    base_tensor = torch.tensor(
        split.frozen_crisp,
        dtype=torch.float32,
        device=device,
    )
    control_tensor = torch.tensor(
        split.control,
        dtype=torch.float32,
        device=device,
    )
    drug_tensor = torch.tensor(
        drug_features,
        dtype=torch.float32,
        device=device,
    )

    all_conditions, all_genes = make_pairs(
        n_conditions,
        n_genes,
    )

    with torch.no_grad():
        for start in range(0, len(all_conditions), batch_size):
            condition_numpy = all_conditions[
                start : start + batch_size
            ]
            gene_numpy = all_genes[
                start : start + batch_size
            ]

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

            base = base_tensor[condition_index, gene_index]
            control = control_tensor[condition_index, gene_index]
            delta = base - control
            relative_delta = delta / (
                torch.abs(control) + 1e-3
            )

            scalar_features = torch.stack(
                [
                    base,
                    control,
                    delta,
                    torch.abs(delta),
                    relative_delta,
                ],
                dim=1,
            )

            prediction = model(
                gene_index,
                drug_tensor[condition_index],
                scalar_features,
            )

            output[
                condition_numpy,
                gene_numpy,
            ] = prediction.detach().cpu().numpy().astype(
                np.float32
            )

    return output


def train_one_seed(
    *,
    seed: int,
    data,
    train_drug: np.ndarray,
    validation_drug: np.ndarray,
    ood_drug: np.ndarray,
    output_root: Path,
    validation_mask: np.ndarray,
    ood_mask: np.ndarray,
    frozen_ood_metrics: pd.DataFrame,
    device: torch.device,
):
    set_seed(seed)

    config = data.config
    direct_config = config["direct_mlp"]
    recalib_config = config["recalib"]

    epochs = int(direct_config["epochs"])
    batch_size = int(direct_config["batch_size"])
    learning_rate = float(direct_config["learning_rate"])
    patience = int(direct_config["patience"])

    hidden_dim = int(recalib_config["hidden_dim"])
    gene_embedding_dim = int(
        recalib_config["gene_embedding_dim"]
    )
    dropout = float(recalib_config["dropout"])
    weight_decay = float(
        recalib_config.get("weight_decay", 1e-4)
    )
    de_gene_weight = float(
        recalib_config.get("de_gene_weight", 0.0)
    )

    method_name = f"{METHOD_PREFIX}_seed_{seed}"

    model = ResidualCalibrator(
        n_genes=data.train.n_genes,
        drug_dim=train_drug.shape[1],
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

    train_base = torch.tensor(
        data.train.frozen_crisp,
        dtype=torch.float32,
        device=device,
    )
    train_control = torch.tensor(
        data.train.control,
        dtype=torch.float32,
        device=device,
    )
    train_true = torch.tensor(
        data.train.true,
        dtype=torch.float32,
        device=device,
    )
    train_residual = torch.tensor(
        data.train.residual,
        dtype=torch.float32,
        device=device,
    )
    train_drug_tensor = torch.tensor(
        train_drug,
        dtype=torch.float32,
        device=device,
    )

    all_conditions, all_genes = make_pairs(
        data.train.n_conditions,
        data.train.n_genes,
    )
    number_of_pairs = len(all_conditions)
    rng = np.random.default_rng(seed)

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    completed_epochs = 0

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        completed_epochs = epoch

        epoch_indices = rng.permutation(number_of_pairs)
        epoch_losses = []

        for start in range(0, number_of_pairs, batch_size):
            pair_indices = epoch_indices[
                start : start + batch_size
            ]

            condition_numpy = all_conditions[pair_indices]
            gene_numpy = all_genes[pair_indices]

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

            base = train_base[
                condition_index,
                gene_index,
            ]
            control = train_control[
                condition_index,
                gene_index,
            ]
            observed = train_true[
                condition_index,
                gene_index,
            ]
            residual = train_residual[
                condition_index,
                gene_index,
            ]

            delta = base - control
            relative_delta = delta / (
                torch.abs(control) + 1e-3
            )

            scalar_features = torch.stack(
                [
                    base,
                    control,
                    delta,
                    torch.abs(delta),
                    relative_delta,
                ],
                dim=1,
            )

            direct_prediction = model(
                gene_index,
                train_drug_tensor[condition_index],
                scalar_features,
            )

            # Use the same residual- and delta-aware sample weighting
            # as the existing ReCalib training procedure.
            weights = torch.ones_like(observed)

            if de_gene_weight != 0:
                # The final ReCalib configuration uses DE weight = 0.
                # This branch is retained only for configuration safety.
                weights = weights + 0.0 * de_gene_weight

            weights = weights + 0.5 * torch.clamp(
                torch.abs(residual)
                / (
                    torch.abs(residual).mean().detach()
                    + 1e-6
                ),
                0.0,
                4.0,
            )
            weights = weights + 0.25 * torch.clamp(
                torch.abs(delta)
                / (
                    torch.abs(delta).mean().detach()
                    + 1e-6
                ),
                0.0,
                4.0,
            )

            loss = (
                weights
                * (direct_prediction - observed) ** 2
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )
            optimizer.step()

            epoch_losses.append(
                float(loss.detach().cpu())
            )

        average_loss = float(np.mean(epoch_losses))

        if average_loss < best_loss:
            best_loss = average_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{method_name}] "
                f"epoch={epoch:04d} "
                f"train_loss={average_loss:.8f} "
                f"best={best_loss:.8f}"
            )

        if bad_epochs >= patience:
            print(
                f"[{method_name}] "
                f"early stopping at epoch {epoch}"
            )
            break

    training_seconds = (
        time.perf_counter() - training_start
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    if device.type == "cuda":
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated()
            / 1024
            / 1024
        )
    else:
        peak_gpu_memory_mb = 0.0

    validation_prediction = predict_expression(
        model=model,
        split=data.validation,
        drug_features=validation_drug,
        batch_size=batch_size,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_start = time.perf_counter()

    ood_prediction = predict_expression(
        model=model,
        split=data.main_ood,
        drug_features=ood_drug,
        batch_size=batch_size,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_seconds = (
        time.perf_counter() - inference_start
    )

    run_and_save_method(
        method=method_name,
        validation_prediction=validation_prediction,
        ood_prediction=ood_prediction,
        data=data,
        validation_mask=validation_mask,
        ood_mask=ood_mask,
        output_root=output_root,
        frozen_ood_metrics=frozen_ood_metrics,
        parameters={
            "seed": seed,
            "target": "observed post-treatment expression",
            "architecture": (
                "identical to the ReCalib residual MLP"
            ),
            "gene_embedding_dim": gene_embedding_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "maximum_epochs": epochs,
            "completed_epochs": completed_epochs,
            "patience": patience,
            "checkpoint_selection": (
                "minimum fitting-subset training loss, "
                "matching the existing ReCalib implementation"
            ),
            "residual_addition": False,
            "residual_scaling_alpha": None,
            "trainable_parameters": trainable_parameters,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
        },
        fit_seconds=training_seconds,
        inference_seconds=inference_seconds,
    )

    method_directory = output_root / method_name

    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "trainable_parameters": trainable_parameters,
            "best_training_loss": best_loss,
            "completed_epochs": completed_epochs,
        },
        method_directory / "model.pt",
    )

    return {
        "method": method_name,
        "seed": seed,
        "trainable_parameters": trainable_parameters,
        "training_seconds": training_seconds,
        "main_ood_inference_seconds": inference_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "best_training_loss": best_loss,
        "completed_epochs": completed_epochs,
    }


def main() -> None:
    config_path = (
        PROJECT_ROOT
        / "configs"
        / "innovation_experiments.yaml"
    )

    data = load_experiment_data(config_path)

    (
        validation_mask,
        ood_mask,
        _,
        _,
    ) = build_official_de_masks(data)

    (
        train_drug,
        validation_drug,
        ood_drug,
        drug_vocabulary,
    ) = build_drug_features(data)

    print(
        f"[INFO] Drug vocabulary size: "
        f"{len(drug_vocabulary)}"
    )

    output_root = resolve_path(
        data.config["output"]["baseline_dir"]
    )
    output_root.mkdir(parents=True, exist_ok=True)

    frozen_ood_metrics = evaluate_by_condition(
        method="frozen_crisp",
        split=data.main_ood,
        prediction=data.main_ood.frozen_crisp,
        de_mask=ood_mask,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"[INFO] Device: {device}")

    run_rows = []

    for seed in SEEDS:
        print("\n" + "=" * 80)
        print(f"DirectMLP seed {seed}")
        print("=" * 80)

        run_rows.append(
            train_one_seed(
                seed=seed,
                data=data,
                train_drug=train_drug,
                validation_drug=validation_drug,
                ood_drug=ood_drug,
                output_root=output_root,
                validation_mask=validation_mask,
                ood_mask=ood_mask,
                frozen_ood_metrics=frozen_ood_metrics,
                device=device,
            )
        )

    cost_table = pd.DataFrame(run_rows)
    cost_table.to_csv(
        output_root / "direct_mlp_strict_cost_summary.csv",
        index=False,
    )

    summary_frames = []

    for seed in SEEDS:
        path = (
            output_root
            / f"{METHOD_PREFIX}_seed_{seed}"
            / "main_ood_summary.csv"
        )
        frame = pd.read_csv(path)
        frame["seed"] = seed
        summary_frames.append(frame)

    all_seed_summaries = pd.concat(
        summary_frames,
        ignore_index=True,
    )
    all_seed_summaries.to_csv(
        output_root / "direct_mlp_strict_seed_summaries.csv",
        index=False,
    )

    aggregate = (
        all_seed_summaries
        .groupby("metric", as_index=False)
        .agg(
            mean=("mean", "mean"),
            sample_sd=("mean", lambda x: x.std(ddof=1)),
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
        output_root / "direct_mlp_strict_mean_sd.csv",
        index=False,
    )

    print("\nDirectMLP mean ± sample SD")
    print(aggregate.to_string(index=False))

    print("\n[OK] DirectMLP experiments completed")


if __name__ == "__main__":
    main()
