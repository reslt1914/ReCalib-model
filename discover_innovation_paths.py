from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

EXPECTED_PATHS = {
    # ReCalib fitting subset: 1003 conditions
    "train_root": Path(
        "crisp_outputs/mydata_baseline_calib_seed2024/iid"
    ),

    # ReCalib IID validation subset: 251 conditions
    # make_mydata_calib_val_baseline.py 将 validation 写在 ood/ 目录，
    # 但它实际上是 IID validation，不是 Main OOD。
    "validation_root": Path(
        "crisp_outputs/mydata_baseline_calib_seed2024/ood"
    ),

    # Main OOD: 108 conditions
    "main_ood_root": Path(
        "crisp_outputs/mydata_baseline_full/ood"
    ),

    "full_iid_metadata": Path(
        "crisp_outputs/mydata_baseline_full/iid/condition_metadata.csv"
    ),

    "train_conditions": Path(
        "crisp_outputs/mydata_baseline_calib_seed2024/"
        "calib_train_conditions.csv"
    ),

    "validation_conditions": Path(
        "crisp_outputs/mydata_baseline_calib_seed2024/"
        "calib_val_conditions.csv"
    ),

    "gene_names": Path(
        "crisp_outputs/mydata_baseline_calib_seed2024/gene_names.csv"
    ),

    # 正式 ReCalib 运行记录使用的 h5ad 与 DE key
    "adata": Path(
        "data/mydata/sciplex_complete_v2.h5ad"
    ),

    # 仅用于核对论文主表，不作为训练输入
    "paper_main_summary": Path(
        "crisp_outputs/evidence_mydata/primary4/"
        "main_ood_primary4_table.csv"
    ),
}


MATRIX_FILES = {
    "true": "x_true_condition_mean.csv",
    "crisp": "x_crisp_condition_mean.csv",
    "control": "x_ctrl_condition_mean.csv",
}

METADATA_FILE = "condition_metadata.csv"


def relative(path: Path) -> str:
    """Return a project-relative path when possible.

    Both absolute and already-relative paths are accepted.
    """
    path = Path(path)

    if not path.is_absolute():
        return str(path)

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def read_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        index_col=0,
        engine="python",
    )


def read_conditions(path: Path) -> list[str]:
    table = pd.read_csv(path)
    if "condition" not in table.columns:
        raise ValueError(
            f"'condition' column missing from {path}; "
            f"columns={list(table.columns)}"
        )
    return table["condition"].astype(str).tolist()


def inspect_split(
    split_name: str,
    root: Path,
) -> dict[str, object]:
    require_dir(root)

    paths = {
        name: root / filename
        for name, filename in MATRIX_FILES.items()
    }
    metadata_path = root / METADATA_FILE

    for path in paths.values():
        require_file(path)
    require_file(metadata_path)

    matrices = {
        name: read_matrix(path)
        for name, path in paths.items()
    }

    reference = matrices["true"]
    reference_conditions = reference.index.astype(str).tolist()
    reference_genes = reference.columns.astype(str).tolist()

    for name, matrix in matrices.items():
        if matrix.shape != reference.shape:
            raise ValueError(
                f"{split_name}: shape mismatch: "
                f"true={reference.shape}, {name}={matrix.shape}"
            )

        if matrix.index.astype(str).tolist() != reference_conditions:
            raise ValueError(
                f"{split_name}: condition order mismatch in {name}"
            )

        if matrix.columns.astype(str).tolist() != reference_genes:
            raise ValueError(
                f"{split_name}: gene order mismatch in {name}"
            )

    metadata = pd.read_csv(metadata_path)
    if "condition" not in metadata.columns:
        raise ValueError(
            f"{split_name}: metadata missing 'condition' column"
        )

    metadata_conditions = metadata["condition"].astype(str).tolist()

    if set(metadata_conditions) != set(reference_conditions):
        missing_from_metadata = sorted(
            set(reference_conditions) - set(metadata_conditions)
        )
        missing_from_matrix = sorted(
            set(metadata_conditions) - set(reference_conditions)
        )
        raise ValueError(
            f"{split_name}: matrix/metadata condition mismatch.\n"
            f"Missing from metadata: {missing_from_metadata[:5]}\n"
            f"Missing from matrix: {missing_from_matrix[:5]}"
        )

    print(
        f"[OK] {split_name}: "
        f"{reference.shape[0]} conditions × "
        f"{reference.shape[1]} genes"
    )

    for name, path in paths.items():
        size_mb = path.stat().st_size / 1024 / 1024
        print(
            f"     {name:12s}: "
            f"{relative(path)} ({size_mb:.2f} MB)"
        )

    return {
        "root": root,
        "paths": paths,
        "metadata": metadata_path,
        "conditions": reference_conditions,
        "genes": reference_genes,
        "shape": reference.shape,
    }


def yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_yaml(
    output_path: Path,
    train: dict[str, object],
    validation: dict[str, object],
    main_ood: dict[str, object],
) -> None:
    if output_path.exists():
        backup = output_path.with_suffix(
            output_path.suffix + ".before_discovery"
        )
        shutil.copy2(output_path, backup)
        print(f"[BACKUP] Existing config copied to: {relative(backup)}")

    p = EXPECTED_PATHS

    reference_runs = {
        "seed_2024": Path(
            "crisp_outputs/"
            "mydata_rc_full_seed2024_alpha04_w0/"
            "ood/x_calibrated_condition_mean.csv"
        ),
        "seed_3407": Path(
            "crisp_outputs/"
            "mydata_rc_full_seed3407_alpha04_w0/"
            "ood/x_calibrated_condition_mean.csv"
        ),
        "seed_42": Path(
            "crisp_outputs/"
            "mydata_rc_full_seed42_alpha04_w0/"
            "ood/x_calibrated_condition_mean.csv"
        ),
    }

    for label, path in reference_runs.items():
        if path.is_file():
            print(f"[OK] Reference ReCalib {label}: {relative(path)}")
        else:
            print(
                f"[WARN] Reference ReCalib {label} not found: "
                f"{relative(path)}"
            )

    lines = [
        "# Auto-generated by discover_innovation_paths.py",
        "# Paths are relative to the ReCalib project root.",
        "# Main OOD is never used for model or hyperparameter selection.",
        "",
        "project:",
        '  name: "ReCalib innovation baselines and ablations"',
        "  config_version: 1",
        "",
        "paths:",
        f"  adata: {yaml_quote(relative(p['adata']))}",
        '  de_key: "lincs_DEGs"',
        f"  gene_names: {yaml_quote(relative(p['gene_names']))}",
        "",
        "  split_manifests:",
        f"    train_conditions: "
        f"{yaml_quote(relative(p['train_conditions']))}",
        f"    validation_conditions: "
        f"{yaml_quote(relative(p['validation_conditions']))}",
        "",
        "  train:",
        f"    root: {yaml_quote(relative(train['root']))}",
        f"    true: "
        f"{yaml_quote(relative(train['paths']['true']))}",
        f"    frozen_crisp: "
        f"{yaml_quote(relative(train['paths']['crisp']))}",
        f"    control: "
        f"{yaml_quote(relative(train['paths']['control']))}",
        "    derive_residual: true",
        "    derive_delta_true: true",
        "    derive_delta_crisp: true",
        f"    metadata: "
        f"{yaml_quote(relative(train['metadata']))}",
        "",
        "  validation:",
        f"    root: {yaml_quote(relative(validation['root']))}",
        f"    true: "
        f"{yaml_quote(relative(validation['paths']['true']))}",
        f"    frozen_crisp: "
        f"{yaml_quote(relative(validation['paths']['crisp']))}",
        f"    control: "
        f"{yaml_quote(relative(validation['paths']['control']))}",
        "    derive_residual: true",
        "    derive_delta_true: true",
        "    derive_delta_crisp: true",
        f"    metadata: "
        f"{yaml_quote(relative(validation['metadata']))}",
        "",
        "  main_ood:",
        f"    root: {yaml_quote(relative(main_ood['root']))}",
        f"    true: "
        f"{yaml_quote(relative(main_ood['paths']['true']))}",
        f"    frozen_crisp: "
        f"{yaml_quote(relative(main_ood['paths']['crisp']))}",
        f"    control: "
        f"{yaml_quote(relative(main_ood['paths']['control']))}",
        "    derive_residual: true",
        "    derive_delta_true: true",
        "    derive_delta_crisp: true",
        f"    metadata: "
        f"{yaml_quote(relative(main_ood['metadata']))}",
        "",
        "  reference_results:",
        "    # Reference only; new experiments must generate and save",
        "    # their own prediction matrices.",
        f"    paper_main_summary: "
        f"{yaml_quote(relative(p['paper_main_summary']))}",
        f"    recalib_seed_2024: "
        f"{yaml_quote(relative(reference_runs['seed_2024']))}",
        f"    recalib_seed_3407: "
        f"{yaml_quote(relative(reference_runs['seed_3407']))}",
        f"    recalib_seed_42: "
        f"{yaml_quote(relative(reference_runs['seed_42']))}",
        "",
        "experiment:",
        "  cpu_threads: 6",
        "  seeds: [2024, 3407, 42]",
        f"  expected_train_conditions: {train['shape'][0]}",
        f"  expected_validation_conditions: "
        f"{validation['shape'][0]}",
        f"  expected_main_ood_conditions: {main_ood['shape'][0]}",
        f"  expected_gene_count: {main_ood['shape'][1]}",
        '  primary_metric_direction:',
        '    mse_de: "lower"',
        '    pearson_de: "higher"',
        '    pearson_delta_de: "higher"',
        '    r2score_de: "higher"',
        "",
        "recalib:",
        "  epochs: 200",
        "  batch_size: 65536",
        "  learning_rate: 0.0008",
        "  weight_decay: 0.0001",
        "  hidden_dim: 256",
        "  gene_embedding_dim: 64",
        "  dropout: 0.05",
        "  patience: 40",
        "  residual_scaling_alpha: 0.40",
        "  de_gene_weight: 0.0",
        "  predicted_residual_l2: 0.0001",
        "",
        "gene_affine:",
        "  ridge_alphas: "
        "[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]",
        "",
        "ridge_residual:",
        "  ridge_alphas: "
        "[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]",
        "  standardize_continuous_features: true",
        "  gene_identity_encoding: \"sparse one-hot\"",
        "  drug_identity_encoding: \"sparse one-hot\"",
        "",
        "direct_mlp:",
        "  match_recalib_architecture: true",
        "  epochs: 200",
        "  batch_size: 65536",
        "  learning_rate: 0.0008",
        "  weight_decay: 0.0001",
        "  patience: 40",
        "",
        "statistics:",
        "  bootstrap_iterations: 10000",
        "  confidence_level: 0.95",
        '  paired_test: "wilcoxon"',
        '  multiple_testing_correction: "holm"',
        "",
        "output:",
        '  root: "results"',
        '  baseline_dir: "results/innovation_baselines"',
        '  ablation_dir: "results/recalib_ablations"',
        '  cost_dir: "results/compute_cost"',
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(f"[WRITE] {relative(output_path)}")


def main() -> None:
    print("=" * 78)
    print("ReCalib innovation-path discovery")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 78)

    for name, path in EXPECTED_PATHS.items():
        absolute = PROJECT_ROOT / path
        if name.endswith("_root"):
            require_dir(absolute)
        else:
            require_file(absolute)
        print(f"[FOUND] {name}: {path}")

    train = inspect_split(
        "ReCalib fitting",
        PROJECT_ROOT / EXPECTED_PATHS["train_root"],
    )
    validation = inspect_split(
        "IID validation",
        PROJECT_ROOT / EXPECTED_PATHS["validation_root"],
    )
    main_ood = inspect_split(
        "Main OOD",
        PROJECT_ROOT / EXPECTED_PATHS["main_ood_root"],
    )

    if train["genes"] != validation["genes"]:
        raise ValueError(
            "Train and validation gene columns are not identical"
        )
    if train["genes"] != main_ood["genes"]:
        raise ValueError(
            "Train and Main OOD gene columns are not identical"
        )

    train_manifest = read_conditions(
        PROJECT_ROOT / EXPECTED_PATHS["train_conditions"]
    )
    validation_manifest = read_conditions(
        PROJECT_ROOT / EXPECTED_PATHS["validation_conditions"]
    )

    if set(train_manifest) != set(train["conditions"]):
        raise ValueError(
            "calib_train_conditions.csv does not match train matrices"
        )

    if set(validation_manifest) != set(validation["conditions"]):
        raise ValueError(
            "calib_val_conditions.csv does not match validation matrices"
        )

    train_ids = set(train["conditions"])
    validation_ids = set(validation["conditions"])
    ood_ids = set(main_ood["conditions"])

    if train_ids & validation_ids:
        raise ValueError("Train/validation condition overlap detected")
    if train_ids & ood_ids:
        raise ValueError("Train/Main OOD condition overlap detected")
    if validation_ids & ood_ids:
        raise ValueError("Validation/Main OOD condition overlap detected")

    full_iid = pd.read_csv(
        PROJECT_ROOT / EXPECTED_PATHS["full_iid_metadata"]
    )
    full_iid_ids = set(
        full_iid["condition"].astype(str)
    )

    if train_ids | validation_ids != full_iid_ids:
        raise ValueError(
            "Train + validation conditions do not reconstruct "
            "the original IID pool"
        )

    print("[OK] Train/validation/Main OOD are mutually disjoint")
    print(
        "[OK] Train + validation reconstruct the full IID pool: "
        f"{len(full_iid_ids)} conditions"
    )
    print(
        "[OK] Unified expression panel: "
        f"{len(train['genes'])} genes"
    )

    output_path = (
        PROJECT_ROOT
        / "configs"
        / "innovation_experiments.yaml"
    )

    write_yaml(
        output_path=output_path,
        train=train,
        validation=validation,
        main_ood=main_ood,
    )

    report = {
        "project_root": str(PROJECT_ROOT),
        "train_conditions": len(train_ids),
        "validation_conditions": len(validation_ids),
        "main_ood_conditions": len(ood_ids),
        "genes": len(train["genes"]),
        "config": relative(output_path),
    }

    report_path = (
        PROJECT_ROOT
        / "configs"
        / "innovation_path_discovery.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"[WRITE] {relative(report_path)}")
    print("=" * 78)
    print("[OK] Discovery and config generation completed")
    print("=" * 78)


if __name__ == "__main__":
    main()
