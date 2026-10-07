from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from innovation_data import (
    PROJECT_ROOT,
    load_experiment_data,
)
from run_innovation_baselines import (
    build_official_de_masks,
    evaluate_by_condition,
    summarize_metrics,
)
from train_crisp_rc_deaware import (
    ResidualCalibrator,
    make_pairs,
)


ALPHA = 0.50
DE_WEIGHT = 0
SEEDS = [2024, 3407, 42]


def encode_with_checkpoint_vocabulary(
    metadata: pd.DataFrame,
    *,
    drugs: list[str],
) -> np.ndarray:
    drug_to_id = {
        drug: index
        for index, drug in enumerate(drugs)
    }

    features = np.zeros(
        (len(metadata), len(drugs)),
        dtype=np.float32,
    )

    unknown = []

    for row_index, drug in enumerate(
        metadata["drug"].astype(str)
    ):
        index = drug_to_id.get(drug)

        if index is None:
            unknown.append(drug)
            continue

        features[row_index, index] = 1.0

    unique_unknown = sorted(set(unknown))

    if unique_unknown:
        print(
            "[DRUG] Main OOD unseen drugs encoded "
            "as all-zero drug identity:"
        )
        for drug in unique_unknown:
            print(f"       {drug}")

        print(
            "[DRUG] affected Main OOD conditions: "
            f"{len(unknown)}"
        )

    return features


def predict_residual(
    *,
    model: torch.nn.Module,
    split,
    drug_features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    n_conditions = split.n_conditions
    n_genes = split.n_genes

    result = np.zeros(
        (n_conditions, n_genes),
        dtype=np.float32,
    )

    base = torch.tensor(
        split.frozen_crisp,
        dtype=torch.float32,
        device=device,
    )

    control = torch.tensor(
        split.control,
        dtype=torch.float32,
        device=device,
    )

    drug_tensor = torch.tensor(
        drug_features,
        dtype=torch.float32,
        device=device,
    )

    condition_indices, gene_indices = make_pairs(
        n_conditions,
        n_genes,
    )

    with torch.no_grad():
        for start in range(
            0,
            len(condition_indices),
            batch_size,
        ):
            condition_numpy = condition_indices[
                start : start + batch_size
            ]
            gene_numpy = gene_indices[
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

            base_value = base[
                condition_index,
                gene_index,
            ]

            control_value = control[
                condition_index,
                gene_index,
            ]

            delta = (
                base_value - control_value
            )

            relative_delta = (
                delta
                / (
                    torch.abs(control_value)
                    + 1e-3
                )
            )

            scalars = torch.stack(
                [
                    base_value,
                    control_value,
                    delta,
                    torch.abs(delta),
                    relative_delta,
                ],
                dim=1,
            )

            predicted = model(
                gene_index,
                drug_tensor[
                    condition_index
                ],
                scalars,
            )

            result[
                condition_numpy,
                gene_numpy,
            ] = (
                predicted
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    return result


def checkpoint_for_seed(seed: int) -> Path:
    if seed == 2024:
        return (
            PROJECT_ROOT
            / "results"
            / "recalib_hyperparameter_selection"
            / "de_weight_0"
            / "crisp_deaware_rc_model.pt"
        )

    return (
        PROJECT_ROOT
        / "results"
        / "recalib_strict_final"
        / f"seed_{seed}"
        / "crisp_deaware_rc_model.pt"
    )


def main() -> None:
    config_path = (
        PROJECT_ROOT
        / "configs"
        / "innovation_experiments.yaml"
    )

    data = load_experiment_data(
        config_path
    )

    _, ood_mask, _, _ = (
        build_official_de_masks(
            data
        )
    )

    split = data.main_ood

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_root = (
        PROJECT_ROOT
        / "results"
        / "recalib_strict_main_ood"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summaries = []

    for seed in SEEDS:
        print("\n" + "=" * 88)
        print(
            f"STRICT ReCalib Main OOD | seed={seed}"
        )
        print("=" * 88)

        checkpoint_path = (
            checkpoint_for_seed(seed)
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                checkpoint_path
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        genes = [
            str(value)
            for value
            in checkpoint["genes"]
        ]

        drugs = [
            str(value)
            for value
            in checkpoint["drugs"]
        ]

        if genes != split.genes:
            raise ValueError(
                f"seed {seed}: gene order mismatch"
            )

        args = checkpoint["args"]

        print(
            f"[CHECKPOINT] fitting drugs = "
            f"{len(drugs)}"
        )
        print(
            f"[CHECKPOINT] DE weight = "
            f"{args.get('de_weight')}"
        )
        print(
            f"[LOCKED] alpha = {ALPHA}"
        )

        if (
            abs(
                float(args["de_weight"])
                - DE_WEIGHT
            )
            > 1e-12
        ):
            raise ValueError(
                f"seed {seed}: wrong DE weight "
                f"in checkpoint"
            )

        drug_features = (
            encode_with_checkpoint_vocabulary(
                split.metadata,
                drugs=drugs,
            )
        )

        model = ResidualCalibrator(
            n_genes=len(genes),
            drug_dim=len(drugs),
            gene_emb_dim=int(
                args["gene_emb_dim"]
            ),
            hidden=int(
                args["hidden"]
            ),
            dropout=0.05,
        ).to(device)

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        residual = predict_residual(
            model=model,
            split=split,
            drug_features=drug_features,
            batch_size=int(
                args["batch_size"]
            ),
            device=device,
        )

        prediction = (
            split.frozen_crisp
            + ALPHA * residual
        )

        seed_dir = (
            output_root
            / f"seed_{seed}"
        )
        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            prediction,
            index=split.conditions,
            columns=split.genes,
        ).to_csv(
            seed_dir
            / "x_calibrated_condition_mean.csv"
        )

        pd.DataFrame(
            residual,
            index=split.conditions,
            columns=split.genes,
        ).to_csv(
            seed_dir
            / "predicted_residual.csv"
        )

        metrics = (
            evaluate_by_condition(
                method=(
                    f"strict_recalib_seed_{seed}"
                ),
                split=split,
                prediction=prediction,
                de_mask=ood_mask,
            )
        )

        metrics.to_csv(
            seed_dir
            / "condition_metrics.csv",
            index=False,
        )

        summary = (
            summarize_metrics(
                metrics
            )
        )

        summary["seed"] = seed

        summary.to_csv(
            seed_dir
            / "summary.csv",
            index=False,
        )

        all_summaries.append(
            summary
        )

        print(
            summary[
                [
                    "metric",
                    "mean",
                    "n_valid_conditions",
                ]
            ].to_string(
                index=False
            )
        )

    seed_summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    seed_summary.to_csv(
        output_root
        / "seed_summaries.csv",
        index=False,
    )

    aggregate = (
        seed_summary
        .groupby(
            "metric",
            as_index=False,
        )
        .agg(
            mean=("mean", "mean"),
            sample_sd=(
                "mean",
                lambda values:
                values.std(ddof=1),
            ),
            n_valid_conditions=(
                "n_valid_conditions",
                "min",
            ),
        )
    )

    aggregate.to_csv(
        output_root
        / "mean_sd_summary.csv",
        index=False,
    )

    protocol = {
        "fitting_conditions": 1003,
        "iid_validation_conditions": 251,
        "main_ood_conditions": 108,
        "main_ood_drugs": 9,
        "main_ood_type": (
            "drug-held-out / unseen-drug OOD"
        ),
        "de_weight": DE_WEIGHT,
        "alpha": ALPHA,
        "hyperparameters_selected_on": (
            "IID validation only"
        ),
        "main_ood_used_for_hyperparameter_selection": (
            False
        ),
        "unseen_drug_encoding": (
            "all-zero drug identity vector"
        ),
        "seeds": SEEDS,
    }

    (
        output_root
        / "protocol.json"
    ).write_text(
        json.dumps(
            protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("STRICT ReCalib Main OOD mean ± sample SD")
    print("=" * 88)
    print(
        aggregate.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
