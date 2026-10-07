from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class ConditionSplit:
    name: str
    true: np.ndarray
    frozen_crisp: np.ndarray
    control: np.ndarray
    conditions: list[str]
    genes: list[str]
    metadata: pd.DataFrame

    @property
    def residual(self) -> np.ndarray:
        """Observed residual: true expression minus Frozen CRISP."""
        return self.true - self.frozen_crisp

    @property
    def delta_true(self) -> np.ndarray:
        """Observed perturbation change relative to matched control."""
        return self.true - self.control

    @property
    def delta_crisp(self) -> np.ndarray:
        """Frozen CRISP perturbation change relative to matched control."""
        return self.frozen_crisp - self.control

    @property
    def n_conditions(self) -> int:
        return self.true.shape[0]

    @property
    def n_genes(self) -> int:
        return self.true.shape[1]


@dataclass
class ExperimentData:
    train: ConditionSplit
    validation: ConditionSplit
    main_ood: ConditionSplit
    config: dict[str, Any]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")

    return config


def read_expression_matrix(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Matrix not found: {path}")

    frame = pd.read_csv(path, index_col=0)

    if frame.empty:
        raise ValueError(f"Empty expression matrix: {path}")

    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad_count = int(numeric.isna().sum().sum())
        raise ValueError(
            f"Non-numeric or missing values found in {path}: "
            f"{bad_count} cells"
        )

    return numeric


def detect_condition_column(metadata: pd.DataFrame) -> str:
    candidates = [
        "condition",
        "condition_id",
        "cov_drug_dose_name",
    ]

    for candidate in candidates:
        if candidate in metadata.columns:
            return candidate

    raise ValueError(
        "No condition identifier column found in metadata. "
        f"Available columns: {list(metadata.columns)}"
    )


def load_metadata(
    path: Path,
    reference_conditions: list[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata not found: {path}")

    metadata = pd.read_csv(path)
    condition_column = detect_condition_column(metadata)

    metadata = metadata.copy()
    metadata[condition_column] = metadata[condition_column].astype(str)

    if metadata[condition_column].duplicated().any():
        duplicated = (
            metadata.loc[
                metadata[condition_column].duplicated(),
                condition_column,
            ]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"Duplicated metadata conditions in {path}: "
            f"{duplicated[:10]}"
        )

    metadata_conditions = set(metadata[condition_column])
    matrix_conditions = set(reference_conditions)

    if metadata_conditions != matrix_conditions:
        missing_metadata = sorted(
            matrix_conditions - metadata_conditions
        )
        extra_metadata = sorted(
            metadata_conditions - matrix_conditions
        )
        raise ValueError(
            f"Metadata/matrix mismatch in {path}\n"
            f"Missing in metadata: {missing_metadata[:10]}\n"
            f"Extra in metadata: {extra_metadata[:10]}"
        )

    metadata = (
        metadata.set_index(condition_column)
        .loc[reference_conditions]
        .reset_index()
        .rename(columns={condition_column: "condition"})
    )

    return metadata


def load_split(
    split_name: str,
    split_config: dict[str, Any],
) -> ConditionSplit:
    required_keys = [
        "true",
        "frozen_crisp",
        "control",
        "metadata",
    ]

    for key in required_keys:
        if key not in split_config:
            raise KeyError(
                f"Missing paths.{split_name}.{key} in configuration"
            )

    true_path = resolve_path(split_config["true"])
    crisp_path = resolve_path(split_config["frozen_crisp"])
    control_path = resolve_path(split_config["control"])
    metadata_path = resolve_path(split_config["metadata"])

    true_frame = read_expression_matrix(true_path)
    crisp_frame = read_expression_matrix(crisp_path)
    control_frame = read_expression_matrix(control_path)

    if crisp_frame.shape != true_frame.shape:
        raise ValueError(
            f"{split_name}: Frozen CRISP shape mismatch: "
            f"true={true_frame.shape}, crisp={crisp_frame.shape}"
        )

    if control_frame.shape != true_frame.shape:
        raise ValueError(
            f"{split_name}: control shape mismatch: "
            f"true={true_frame.shape}, control={control_frame.shape}"
        )

    if crisp_frame.index.tolist() != true_frame.index.tolist():
        raise ValueError(
            f"{split_name}: Frozen CRISP condition order mismatch"
        )

    if control_frame.index.tolist() != true_frame.index.tolist():
        raise ValueError(
            f"{split_name}: control condition order mismatch"
        )

    if crisp_frame.columns.tolist() != true_frame.columns.tolist():
        raise ValueError(
            f"{split_name}: Frozen CRISP gene order mismatch"
        )

    if control_frame.columns.tolist() != true_frame.columns.tolist():
        raise ValueError(
            f"{split_name}: control gene order mismatch"
        )

    conditions = true_frame.index.tolist()
    genes = true_frame.columns.tolist()
    metadata = load_metadata(metadata_path, conditions)

    split = ConditionSplit(
        name=split_name,
        true=true_frame.to_numpy(dtype=np.float32, copy=True),
        frozen_crisp=crisp_frame.to_numpy(
            dtype=np.float32,
            copy=True,
        ),
        control=control_frame.to_numpy(
            dtype=np.float32,
            copy=True,
        ),
        conditions=conditions,
        genes=genes,
        metadata=metadata,
    )

    arrays = {
        "true": split.true,
        "frozen_crisp": split.frozen_crisp,
        "control": split.control,
        "residual": split.residual,
        "delta_true": split.delta_true,
        "delta_crisp": split.delta_crisp,
    }

    for array_name, array in arrays.items():
        if not np.isfinite(array).all():
            invalid_count = int((~np.isfinite(array)).sum())
            raise ValueError(
                f"{split_name}: {array_name} contains "
                f"{invalid_count} non-finite values"
            )

    print(
        f"[OK] {split_name}: "
        f"{split.n_conditions} conditions × "
        f"{split.n_genes} genes"
    )

    return split


def validate_expected_counts(
    data: ExperimentData,
) -> None:
    expected = data.config["experiment"]

    checks = [
        (
            "train conditions",
            data.train.n_conditions,
            int(expected["expected_train_conditions"]),
        ),
        (
            "validation conditions",
            data.validation.n_conditions,
            int(expected["expected_validation_conditions"]),
        ),
        (
            "Main OOD conditions",
            data.main_ood.n_conditions,
            int(expected["expected_main_ood_conditions"]),
        ),
        (
            "gene count",
            data.main_ood.n_genes,
            int(expected["expected_gene_count"]),
        ),
    ]

    for label, actual, wanted in checks:
        if actual != wanted:
            raise ValueError(
                f"{label} mismatch: expected={wanted}, actual={actual}"
            )

        print(f"[OK] Expected {label}: {actual}")


def validate_gene_order(data: ExperimentData) -> None:
    if data.train.genes != data.validation.genes:
        raise ValueError(
            "Train and validation gene order differs"
        )

    if data.train.genes != data.main_ood.genes:
        raise ValueError(
            "Train and Main OOD gene order differs"
        )

    if len(set(data.train.genes)) != len(data.train.genes):
        raise ValueError("Duplicated gene names detected")

    print(
        f"[OK] Identical gene order across all splits: "
        f"{len(data.train.genes)} genes"
    )


def validate_condition_isolation(data: ExperimentData) -> None:
    train_ids = set(data.train.conditions)
    validation_ids = set(data.validation.conditions)
    ood_ids = set(data.main_ood.conditions)

    overlaps = {
        "train_validation": train_ids & validation_ids,
        "train_main_ood": train_ids & ood_ids,
        "validation_main_ood": validation_ids & ood_ids,
    }

    for label, overlap in overlaps.items():
        if overlap:
            raise ValueError(
                f"Condition overlap detected for {label}: "
                f"{sorted(overlap)[:10]}"
            )

    print("[OK] Train, validation and Main OOD are disjoint")


def validate_metadata_fields(data: ExperimentData) -> None:
    for split in [
        data.train,
        data.validation,
        data.main_ood,
    ]:
        metadata = split.metadata

        possible_drug_columns = [
            "drug",
            "perturbation",
            "condition_name",
        ]
        possible_cell_columns = [
            "cell_type",
            "cell_line",
            "cell",
        ]
        possible_dose_columns = [
            "dose",
            "dose_val",
        ]

        drug_column = next(
            (
                column
                for column in possible_drug_columns
                if column in metadata.columns
            ),
            None,
        )
        cell_column = next(
            (
                column
                for column in possible_cell_columns
                if column in metadata.columns
            ),
            None,
        )
        dose_column = next(
            (
                column
                for column in possible_dose_columns
                if column in metadata.columns
            ),
            None,
        )

        print(
            f"[INFO] {split.name} metadata columns: "
            f"{list(metadata.columns)}"
        )
        print(
            f"[INFO] {split.name} detected fields: "
            f"cell={cell_column}, drug={drug_column}, "
            f"dose={dose_column}"
        )


def load_experiment_data(
    config_path: Path,
) -> ExperimentData:
    config = load_config(config_path)

    paths = config.get("paths", {})
    for split_name in ["train", "validation", "main_ood"]:
        if split_name not in paths:
            raise KeyError(
                f"Missing paths.{split_name} in configuration"
            )

    train = load_split("train", paths["train"])
    validation = load_split(
        "validation",
        paths["validation"],
    )
    main_ood = load_split(
        "main_ood",
        paths["main_ood"],
    )

    data = ExperimentData(
        train=train,
        validation=validation,
        main_ood=main_ood,
        config=config,
    )

    validate_expected_counts(data)
    validate_gene_order(data)
    validate_condition_isolation(data)
    validate_metadata_fields(data)

    return data


def validate_reference_predictions(
    data: ExperimentData,
) -> dict[str, Any]:
    reference_config = (
        data.config
        .get("paths", {})
        .get("reference_results", {})
    )

    results: dict[str, Any] = {}

    for label in [
        "recalib_seed_2024",
        "recalib_seed_3407",
        "recalib_seed_42",
    ]:
        path_value = reference_config.get(label)
        if not path_value:
            results[label] = {
                "status": "not_configured",
            }
            continue

        path = resolve_path(path_value)

        if not path.is_file():
            print(f"[WARN] Missing reference prediction: {path}")
            results[label] = {
                "status": "missing",
                "path": str(path),
            }
            continue

        prediction = read_expression_matrix(path)

        if prediction.index.tolist() != data.main_ood.conditions:
            raise ValueError(
                f"{label}: condition order differs from Main OOD"
            )

        if prediction.columns.tolist() != data.main_ood.genes:
            raise ValueError(
                f"{label}: gene order differs from Main OOD"
            )

        if prediction.shape != data.main_ood.true.shape:
            raise ValueError(
                f"{label}: shape mismatch: {prediction.shape}"
            )

        results[label] = {
            "status": "ok",
            "path": str(path.relative_to(PROJECT_ROOT)),
            "shape": list(prediction.shape),
            "sha256": sha256_file(path),
        }

        print(
            f"[OK] {label}: "
            f"{prediction.shape[0]} × {prediction.shape[1]}"
        )

    return results


def write_validation_report(
    config_path: Path,
    data: ExperimentData,
    references: dict[str, Any],
) -> Path:
    report = {
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "project_root": str(PROJECT_ROOT),
        "cpu_threads": {
            "OMP_NUM_THREADS": os.environ.get(
                "OMP_NUM_THREADS"
            ),
            "MKL_NUM_THREADS": os.environ.get(
                "MKL_NUM_THREADS"
            ),
            "OPENBLAS_NUM_THREADS": os.environ.get(
                "OPENBLAS_NUM_THREADS"
            ),
            "NUMEXPR_NUM_THREADS": os.environ.get(
                "NUMEXPR_NUM_THREADS"
            ),
        },
        "splits": {
            "train": {
                "conditions": data.train.n_conditions,
                "genes": data.train.n_genes,
            },
            "validation": {
                "conditions": data.validation.n_conditions,
                "genes": data.validation.n_genes,
            },
            "main_ood": {
                "conditions": data.main_ood.n_conditions,
                "genes": data.main_ood.n_genes,
            },
        },
        "split_overlap": {
            "train_validation": 0,
            "train_main_ood": 0,
            "validation_main_ood": 0,
        },
        "reference_predictions": references,
    }

    output_path = (
        PROJECT_ROOT
        / "configs"
        / "innovation_data_validation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load and validate unified ReCalib innovation "
            "experiment data."
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
        "--validate-only",
        action="store_true",
        help="Validate all inputs and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    print("=" * 78)
    print("ReCalib unified innovation-data validation")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config: {config_path}")
    print("=" * 78)

    data = load_experiment_data(config_path)
    references = validate_reference_predictions(data)

    report_path = write_validation_report(
        config_path=config_path,
        data=data,
        references=references,
    )

    print(f"[WRITE] {report_path.relative_to(PROJECT_ROOT)}")
    print("=" * 78)
    print("[OK] Unified innovation data validation completed")
    print("=" * 78)


if __name__ == "__main__":
    main()
