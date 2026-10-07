#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


EXPECTED_GENES = 977
EXPECTED_FIT = 1003
EXPECTED_VALIDATION = 251
EXPECTED_MAIN_OOD = 108
EXPECTED_MAIN_OOD_DRUGS = 9

VALID_DOSES_NM = {0, 10, 100, 1000, 10000}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the ReCalib sci-Plex dataset for CMonge while "
            "preserving the frozen-predictor and ReCalib partitions."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "mydata"
            / "sciplex_complete_v2_evall.h5ad"
        ),
    )

    parser.add_argument(
        "--partition-dir",
        default=str(
            Path(__file__).resolve().parents[1]
            / "results"
            / "cmonge_partition_audit"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "recalib_sciplex_main.h5ad"
        ),
    )

    parser.add_argument(
        "--manifest-dir",
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "recalib_sciplex_main_metadata"
        ),
    )

    return parser.parse_args()


def stable_drug_token(drug: str) -> str:
    """
    Generate a deterministic CMonge-safe drug identifier.

    CMonge's RDKitEmbedding parses a condition using:
        cond, dose = condition.split("-")

    Therefore the drug token itself must not contain '-'.

    The original drug name is preserved separately.
    """
    normalized = str(drug).strip().lower()

    digest = hashlib.sha1(
        normalized.encode("utf-8")
    ).hexdigest()[:12]

    return f"d{digest}"


def load_partition(path: Path, expected_n: int, role: str) -> set[str]:
    df = pd.read_csv(path)

    if "condition" not in df.columns:
        raise ValueError(
            f"{path} does not contain a 'condition' column"
        )

    ids = set(df["condition"].astype(str))

    if len(ids) != expected_n:
        raise ValueError(
            f"{role}: expected {expected_n} conditions, "
            f"found {len(ids)}"
        )

    return ids


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    partition_dir = Path(args.partition_dir).resolve()
    output_path = Path(args.output).resolve()
    manifest_dir = Path(args.manifest_dir).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("PREPARE RECALIB SCIPLEX FOR CMONGE")
    print("=" * 90)

    print("\nInput:")
    print(input_path)

    print("\nOutput:")
    print(output_path)

    # ------------------------------------------------------------------
    # 1. Load the locked ReCalib condition partitions
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[1] LOAD LOCKED RECALIB PARTITIONS")
    print("=" * 90)

    fit_ids = load_partition(
        partition_dir / "fit_conditions.csv",
        EXPECTED_FIT,
        "fit",
    )

    val_ids = load_partition(
        partition_dir / "validation_conditions.csv",
        EXPECTED_VALIDATION,
        "validation",
    )

    ood_ids = load_partition(
        partition_dir / "main_ood_conditions.csv",
        EXPECTED_MAIN_OOD,
        "main_ood",
    )

    print("fit        =", len(fit_ids))
    print("validation =", len(val_ids))
    print("main_ood   =", len(ood_ids))

    assert not (fit_ids & val_ids)
    assert not (fit_ids & ood_ids)
    assert not (val_ids & ood_ids)

    role_map = {}

    for x in fit_ids:
        role_map[x] = "fit"

    for x in val_ids:
        role_map[x] = "validation"

    for x in ood_ids:
        role_map[x] = "main_ood"

    # ------------------------------------------------------------------
    # 2. Read original ReCalib h5ad
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[2] READ ORIGINAL H5AD")
    print("=" * 90)

    adata = ad.read_h5ad(input_path)

    print("shape =", adata.shape)

    if adata.n_vars != EXPECTED_GENES:
        raise ValueError(
            f"Expected {EXPECTED_GENES} genes, "
            f"found {adata.n_vars}"
        )

    required_obs = {
        "cell_type",
        "dose",
        "product_name",
        "vehicle",
        "cov_drug_dose_name",
        "split_crisp_eval_all",
    }

    missing_columns = required_obs - set(adata.obs.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required obs columns: "
            f"{sorted(missing_columns)}"
        )

    obs = adata.obs

    # ------------------------------------------------------------------
    # 3. Validate controls
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[3] CONTROL VALIDATION")
    print("=" * 90)

    product_name = obs["product_name"].astype(str)

    is_product_control = (
        product_name.str.lower().str.strip() == "control"
    )

    vehicle_numeric = pd.to_numeric(
        obs["vehicle"],
        errors="raise",
    ).astype(int)

    is_vehicle_control = vehicle_numeric.eq(1)

    mismatch = int(
        np.sum(
            is_product_control.to_numpy()
            != is_vehicle_control.to_numpy()
        )
    )

    print("control by product_name =", int(is_product_control.sum()))
    print("control by vehicle      =", int(is_vehicle_control.sum()))
    print("control mismatches       =", mismatch)

    if mismatch != 0:
        raise ValueError(
            "product_name=='control' and vehicle==1 do not agree"
        )

    # ------------------------------------------------------------------
    # 4. Validate real nM dose
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[4] DOSE VALIDATION")
    print("=" * 90)

    dose_numeric = pd.to_numeric(
        obs["dose"],
        errors="raise",
    )

    rounded_dose = np.rint(dose_numeric).astype(np.int64)

    if not np.allclose(
        dose_numeric.to_numpy(dtype=float),
        rounded_dose.astype(float),
    ):
        raise ValueError(
            "obs['dose'] contains non-integer nM values"
        )

    observed_doses = set(
        np.unique(rounded_dose).tolist()
    )

    print("observed nM doses =", sorted(observed_doses))

    invalid_doses = observed_doses - VALID_DOSES_NM

    if invalid_doses:
        raise ValueError(
            f"Unexpected dose values: {sorted(invalid_doses)}"
        )

    if np.any(
        rounded_dose[is_product_control.to_numpy()] != 0
    ):
        raise ValueError(
            "Control cells must have dose=0"
        )

    # ------------------------------------------------------------------
    # 5. Create deterministic CMonge drug identifiers
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[5] CREATE CMONGE DRUG TOKENS")
    print("=" * 90)

    original_drugs = sorted(
        set(
            product_name.loc[
                ~is_product_control
            ].tolist()
        )
    )

    mapping_rows = []

    for drug in original_drugs:
        mapping_rows.append(
            {
                "original_drug": drug,
                "cmonge_drug": stable_drug_token(drug),
            }
        )

    drug_mapping = pd.DataFrame(mapping_rows)

    # Hash collision check
    if drug_mapping["cmonge_drug"].duplicated().any():
        collision = drug_mapping.loc[
            drug_mapping["cmonge_drug"].duplicated(False)
        ]
        raise ValueError(
            "CMonge drug-token collision:\n"
            + collision.to_string(index=False)
        )

    mapping_dict = dict(
        zip(
            drug_mapping["original_drug"],
            drug_mapping["cmonge_drug"],
        )
    )

    cmonge_drug = product_name.map(mapping_dict)

    cmonge_drug = cmonge_drug.astype("object")
    cmonge_drug.loc[is_product_control] = "control"

    if cmonge_drug.isna().any():
        raise ValueError(
            "Some drug names could not be mapped"
        )

    # ------------------------------------------------------------------
    # 6. Build CMonge-compatible drug-dose labels
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[6] BUILD CMONGE CONDITION LABELS")
    print("=" * 90)

    cmonge_dose = rounded_dose.copy()

    cmonge_dose[is_product_control.to_numpy()] = 0

    cmonge_drug_dose = pd.Series(
        [
            f"{drug}-{int(dose)}"
            for drug, dose in zip(
                cmonge_drug.tolist(),
                cmonge_dose.tolist(),
            )
        ],
        index=obs.index,
        dtype="object",
    )

    # Official CMonge uses condition.split("-").
    # Every non-control condition must therefore split into exactly 2 parts.
    bad_labels = [
        x
        for x in pd.unique(cmonge_drug_dose)
        if len(str(x).split("-")) != 2
    ]

    if bad_labels:
        raise ValueError(
            "Unsafe CMonge drug-dose labels detected: "
            f"{bad_labels[:20]}"
        )

    # ------------------------------------------------------------------
    # 7. Attach the locked ReCalib role
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[7] ATTACH RECALIB CONDITION ROLE")
    print("=" * 90)

    recalib_condition = (
        obs["cov_drug_dose_name"]
        .astype(str)
    )

    recalib_role = recalib_condition.map(
        role_map
    ).fillna("other")

    role_counts = (
        pd.DataFrame(
            {
                "condition": recalib_condition,
                "role": recalib_role,
            }
        )
        .drop_duplicates()
        .query("role != 'other'")
        .groupby("role")["condition"]
        .nunique()
        .to_dict()
    )

    print("condition counts by ReCalib role:")
    print(role_counts)

    expected_role_counts = {
        "fit": EXPECTED_FIT,
        "validation": EXPECTED_VALIDATION,
        "main_ood": EXPECTED_MAIN_OOD,
    }

    if role_counts != expected_role_counts:
        raise ValueError(
            "Unexpected condition-role counts:\n"
            f"expected={expected_role_counts}\n"
            f"actual={role_counts}"
        )

    # ------------------------------------------------------------------
    # 8. Preserve both split layers
    # ------------------------------------------------------------------
    base_split = (
        obs["split_crisp_eval_all"]
        .astype(str)
    )

    print("\nFrozen-base split:")
    print(base_split.value_counts())

    unknown_base_split = (
        set(base_split.unique())
        - {"train", "test"}
    )

    if unknown_base_split:
        raise ValueError(
            f"Unexpected frozen-base split labels: "
            f"{sorted(unknown_base_split)}"
        )

    # ------------------------------------------------------------------
    # 9. Critical Main-OOD leakage audit
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[8] MAIN OOD DRUG LEAKAGE AUDIT")
    print("=" * 90)

    ood_mask = recalib_role.eq("main_ood")

    ood_drugs = set(
        product_name.loc[
            ood_mask
            & ~is_product_control
        ]
    )

    print("Main OOD drugs =", len(ood_drugs))
    print(sorted(ood_drugs))

    if len(ood_drugs) != EXPECTED_MAIN_OOD_DRUGS:
        raise ValueError(
            f"Expected {EXPECTED_MAIN_OOD_DRUGS} Main-OOD drugs, "
            f"found {len(ood_drugs)}"
        )

    base_train_target_drugs = set(
        product_name.loc[
            base_split.eq("train")
            & ~is_product_control
        ]
    )

    ood_drug_overlap = (
        ood_drugs & base_train_target_drugs
    )

    print(
        "Main-OOD drugs present in frozen-base train targets =",
        len(ood_drug_overlap),
    )

    if ood_drug_overlap:
        print(
            "WARNING: these Main-OOD drugs occur in "
            "split_crisp_eval_all=train:"
        )
        print(sorted(ood_drug_overlap))

    # Do NOT silently fail here yet:
    # this is reported for protocol auditing and will be resolved
    # before CMonge training.

    # ------------------------------------------------------------------
    # 10. Add CMonge columns without removing original metadata
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[9] WRITE CMONGE METADATA")
    print("=" * 90)

    adata.obs["recalib_condition"] = recalib_condition.values
    adata.obs["recalib_role"] = recalib_role.values

    adata.obs["cmonge_drug"] = cmonge_drug.values
    adata.obs["cmonge_dose"] = cmonge_dose
    adata.obs["cmonge_drug_dose"] = cmonge_drug_dose.values

    adata.obs["frozen_base_split"] = base_split.values

    # Official CMonge-compatible aliases.
    adata.obs["drug"] = cmonge_drug.values
    adata.obs["drug-dose"] = cmonge_drug_dose.values

    # ------------------------------------------------------------------
    # 11. Condition-level manifest
    # ------------------------------------------------------------------
    manifest_frame = pd.DataFrame(
        {
            "recalib_condition": recalib_condition.values,
            "recalib_role": recalib_role.values,
            "cell_type": obs["cell_type"].astype(str).values,
            "original_drug": product_name.values,
            "cmonge_drug": cmonge_drug.values,
            "dose_nm": cmonge_dose,
            "cmonge_drug_dose": cmonge_drug_dose.values,
            "vehicle": vehicle_numeric.values,
            "frozen_base_split": base_split.values,
        },
        index=obs.index,
    )

    manifest_frame["is_base_train"] = (
        manifest_frame["frozen_base_split"] == "train"
    ).astype(int)

    manifest_frame["is_base_test"] = (
        manifest_frame["frozen_base_split"] == "test"
    ).astype(int)

    condition_manifest = (
        manifest_frame
        .groupby(
            [
                "recalib_condition",
                "recalib_role",
                "cell_type",
                "original_drug",
                "cmonge_drug",
                "dose_nm",
                "cmonge_drug_dose",
                "vehicle",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            n_cells=("recalib_condition", "size"),
            n_base_train_cells=("is_base_train", "sum"),
            n_base_test_cells=("is_base_test", "sum"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 12. Export metadata tables
    # ------------------------------------------------------------------
    drug_mapping.to_csv(
        manifest_dir / "drug_mapping.csv",
        index=False,
    )

    condition_manifest.to_csv(
        manifest_dir / "condition_manifest.csv",
        index=False,
    )

    role_manifest = (
        condition_manifest.loc[
            condition_manifest["recalib_role"] != "other"
        ]
        .copy()
    )

    role_manifest.to_csv(
        manifest_dir / "recalib_condition_manifest.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 13. Write summary
    # ------------------------------------------------------------------
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_original_noncontrol_drugs": int(len(original_drugs)),
        "n_fit_conditions": EXPECTED_FIT,
        "n_validation_conditions": EXPECTED_VALIDATION,
        "n_main_ood_conditions": EXPECTED_MAIN_OOD,
        "n_main_ood_drugs": int(len(ood_drugs)),
        "main_ood_drugs": sorted(ood_drugs),
        "main_ood_drugs_in_base_train": sorted(ood_drug_overlap),
        "dose_values_nm": sorted(
            int(x) for x in observed_doses
        ),
        "frozen_base_split_counts": {
            str(k): int(v)
            for k, v in base_split.value_counts().items()
        },
    }

    with open(
        manifest_dir / "adapter_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # 14. Write new h5ad
    # ------------------------------------------------------------------
    print("\nWriting:")
    print(output_path)
    print(
        "The original ReCalib h5ad is NOT modified."
    )

    adata.write_h5ad(
        output_path,
        compression="gzip",
    )

    print("\n" + "=" * 90)
    print("CMONGE SCIPLEX ADAPTER COMPLETE")
    print("=" * 90)

    print("shape =", adata.shape)
    print(
        "drug mapping =",
        manifest_dir / "drug_mapping.csv",
    )
    print(
        "condition manifest =",
        manifest_dir / "condition_manifest.csv",
    )
    print(
        "summary =",
        manifest_dir / "adapter_summary.json",
    )


if __name__ == "__main__":
    main()
