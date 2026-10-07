#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REF_PATH = ROOT / "data/lincs_trapnell.smiles"
TARGET_PATH = (
    ROOT
    / "data/recalib_sciplex_main_metadata"
    / "cmonge_drug_smiles.csv"
)

OUT_DIR = ROOT / "models/embed/recalib"
OUT_PATH = OUT_DIR / "rdkit"

WORK_DIR = (
    ROOT
    / "data/recalib_sciplex_main_metadata"
    / "rdkit_build"
)

CHUNK_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker-batch",
        action="store_true",
    )
    parser.add_argument(
        "--worker-descriptor",
        action="store_true",
    )

    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--smiles-file")
    parser.add_argument("--descriptor")

    return parser.parse_args()


# ============================================================
# CHILD WORKER 1
# Entire batch using official CMonge rdkit_feats()
# ============================================================

def worker_batch(input_path: str, output_path: str):
    from cmonge.models.rdkit import rdkit_feats

    frame = pd.read_csv(input_path)

    rows = []

    for smi in frame["smiles"].astype(str):
        values = rdkit_feats(smi)

        if values is None:
            raise RuntimeError(
                f"Invalid SMILES: {smi}"
            )

        clean = []

        for x in values:
            if x is None:
                clean.append(np.nan)
            else:
                clean.append(float(x))

        rows.append(clean)

    arr = np.asarray(
        rows,
        dtype=np.float64,
    )

    np.save(
        output_path,
        arr,
    )


# ============================================================
# CHILD WORKER 2
# One descriptor for one molecule.
# Native SIGSEGV is contained inside this process.
# ============================================================

def worker_descriptor(
    smiles_file: str,
    descriptor: str,
    output_path: str,
):
    from rdkit import Chem
    from cmonge.models.rdkit import FEAT_FNS

    smi = Path(smiles_file).read_text().strip()

    result = {
        "ok": False,
        "descriptor": descriptor,
        "smiles": smi,
    }

    try:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            result["error"] = "MolFromSmiles returned None"

        else:
            value = FEAT_FNS[descriptor](mol)

            if value is None:
                result["value"] = None
            else:
                result["value"] = float(value)

            result["ok"] = True

    except BaseException as exc:
        result["error"] = repr(exc)

    Path(output_path).write_text(
        json.dumps(result),
        encoding="utf-8",
    )


# ============================================================
# PARENT BUILDER
# ============================================================

def main():
    from cmonge.models.rdkit import RDKIT_PROPS

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunk_dir = WORK_DIR / "chunks"
    input_dir = WORK_DIR / "inputs"
    single_dir = WORK_DIR / "single"

    for p in [
        chunk_dir,
        input_dir,
        single_dir,
    ]:
        p.mkdir(
            parents=True,
            exist_ok=True,
        )

    reference = pd.read_csv(
        REF_PATH
    )

    targets = pd.read_csv(
        TARGET_PATH
    )

    if list(reference.columns) != ["smiles"]:
        raise ValueError(
            f"Unexpected reference columns: "
            f"{reference.columns.tolist()}"
        )

    if len(reference) != 17869:
        raise ValueError(
            f"Expected 17869 reference SMILES, "
            f"found {len(reference)}"
        )

    if len(targets) != 187:
        raise ValueError(
            f"Expected 187 target drugs, "
            f"found {len(targets)}"
        )

    descriptors = list(RDKIT_PROPS)

    n_features_raw = (
        1 + len(descriptors)
    )

    print("=" * 90)
    print("SAFE FULL RDKIT CHECKPOINT BUILD")
    print("=" * 90)

    print(
        "reference molecules =",
        len(reference),
    )
    print(
        "target drugs        =",
        len(targets),
    )
    print(
        "RDKit descriptors   =",
        len(descriptors),
    )
    print(
        "raw feature length  =",
        n_features_raw,
    )

    # --------------------------------------------------------
    # Full raw embedding matrix.
    # --------------------------------------------------------

    embedding = np.full(
        (
            len(reference),
            n_features_raw,
        ),
        np.nan,
        dtype=np.float64,
    )

    failures = []

    script = Path(__file__).resolve()

    # --------------------------------------------------------
    # Descriptor-level recovery for a single bad molecule
    # --------------------------------------------------------

    def recover_single(idx: int):
        smi = str(
            reference.iloc[idx]["smiles"]
        )

        print()
        print(
            f"[RECOVER] molecule {idx}: {smi}"
        )

        row = np.full(
            n_features_raw,
            np.nan,
            dtype=np.float64,
        )

        # Official rdkit_feats starts with True.
        row[0] = 1.0

        smi_file = (
            single_dir
            / f"mol_{idx}.smi"
        )

        smi_file.write_text(
            smi,
            encoding="utf-8",
        )

        for j, descriptor in enumerate(
            descriptors,
            start=1,
        ):
            result_file = (
                single_dir
                / f"mol_{idx}_desc_{j}.json"
            )

            if result_file.exists():
                result_file.unlink()

            command = [
                sys.executable,
                "-X",
                "faulthandler",
                str(script),
                "--worker-descriptor",
                "--smiles-file",
                str(smi_file),
                "--descriptor",
                descriptor,
                "--output",
                str(result_file),
            ]

            proc = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            value = np.nan
            ok = False
            error = ""

            if (
                proc.returncode == 0
                and result_file.exists()
            ):
                try:
                    result = json.loads(
                        result_file.read_text()
                    )

                    if result.get("ok"):
                        value = result.get(
                            "value",
                            np.nan,
                        )

                        if value is None:
                            value = np.nan

                        ok = True

                    else:
                        error = result.get(
                            "error",
                            "unknown Python error",
                        )

                except Exception as exc:
                    error = (
                        "Could not read descriptor "
                        f"result: {exc!r}"
                    )

            else:
                error = (
                    f"process returncode={proc.returncode}; "
                    f"stderr={proc.stderr[-1000:]}"
                )

            if ok:
                row[j] = float(value)

            else:
                row[j] = np.nan

                failures.append(
                    {
                        "reference_index": idx,
                        "smiles": smi,
                        "descriptor_index": j - 1,
                        "descriptor": descriptor,
                        "returncode": proc.returncode,
                        "error": error,
                    }
                )

                print(
                    "  FAILED:",
                    descriptor,
                    "returncode=",
                    proc.returncode,
                )

        embedding[idx] = row

        cache = (
            chunk_dir
            / f"chunk_{idx}_{idx+1}.npy"
        )

        np.save(
            cache,
            row[None, :],
        )

    # --------------------------------------------------------
    # Recursive safe batch computation
    # --------------------------------------------------------

    def compute_range(
        start: int,
        end: int,
    ):
        cache = (
            chunk_dir
            / f"chunk_{start}_{end}.npy"
        )

        expected_shape = (
            end - start,
            n_features_raw,
        )

        # Resume support
        if cache.exists():
            try:
                arr = np.load(cache)

                if arr.shape == expected_shape:
                    embedding[start:end] = arr

                    print(
                        f"[CACHE] {start}:{end}"
                    )
                    return

            except Exception:
                pass

        input_file = (
            input_dir
            / f"chunk_{start}_{end}.csv"
        )

        reference.iloc[
            start:end
        ][["smiles"]].to_csv(
            input_file,
            index=False,
        )

        command = [
            sys.executable,
            "-X",
            "faulthandler",
            str(script),
            "--worker-batch",
            "--input",
            str(input_file),
            "--output",
            str(cache),
        ]

        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if (
            proc.returncode == 0
            and cache.exists()
        ):
            try:
                arr = np.load(cache)

                if arr.shape == expected_shape:
                    embedding[start:end] = arr

                    print(
                        f"[OK] {start}:{end}"
                    )
                    return

            except Exception:
                pass

        # Native or Python failure.
        print(
            f"[FAIL] {start}:{end} "
            f"(returncode={proc.returncode})"
        )

        n = end - start

        if n == 1:
            recover_single(start)
            return

        middle = (
            start + n // 2
        )

        compute_range(
            start,
            middle,
        )

        compute_range(
            middle,
            end,
        )

    # --------------------------------------------------------
    # Process the full reference library
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("[1] COMPUTE RAW REFERENCE DESCRIPTORS")
    print("=" * 90)

    for start in range(
        0,
        len(reference),
        CHUNK_SIZE,
    ):
        end = min(
            start + CHUNK_SIZE,
            len(reference),
        )

        compute_range(
            start,
            end,
        )

    raw_path = (
        WORK_DIR
        / "reference_raw_features.npy"
    )

    np.save(
        raw_path,
        embedding,
    )

    # --------------------------------------------------------
    # Failure audit
    # --------------------------------------------------------

    failure_frame = pd.DataFrame(
        failures
    )

    failure_path = (
        WORK_DIR
        / "descriptor_failures.csv"
    )

    failure_frame.to_csv(
        failure_path,
        index=False,
    )

    print()
    print("=" * 90)
    print("[2] FAILURE AUDIT")
    print("=" * 90)

    print(
        "failed descriptor evaluations =",
        len(failure_frame),
    )

    if len(failure_frame):
        print()
        print(
            failure_frame[
                [
                    "reference_index",
                    "descriptor",
                    "returncode",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Official CMonge post-processing
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("[3] OFFICIAL CMONGE POST-PROCESSING")
    print("=" * 90)

    clean = embedding.copy()

    bad = ~np.isfinite(clean)

    print(
        "non-finite raw values =",
        int(bad.sum()),
    )

    # Official implementation converts nan / inf to zero.
    clean[bad] = 0.0

    smiles_list = (
        reference["smiles"]
        .astype(str)
        .tolist()
    )

    df = pd.DataFrame(
        data=clean,
        index=smiles_list,
        columns=[
            f"latent_{i}"
            for i in range(
                clean.shape[1]
            )
        ],
    )

    # Official code drops latent_0.
    df.drop(
        columns=["latent_0"],
        inplace=True,
    )

    # pandas std() default ddof=1,
    # matching official implementation.
    std_before = df.std()

    drop_columns = (
        std_before[
            std_before <= 0.01
        ]
        .index
        .tolist()
    )

    print(
        "columns with std <= 0.01 =",
        len(drop_columns),
    )

    print(
        "dropped columns =",
        drop_columns,
    )

    df.drop(
        columns=drop_columns,
        inplace=True,
    )

    means = df.mean()
    stds = df.std()

    normalized_df = (
        df - means
    ) / stds

    if not np.isfinite(
        normalized_df.to_numpy()
    ).all():
        raise RuntimeError(
            "Normalized reference embedding "
            "contains NaN/Inf"
        )

    print(
        "final RDKit latent dimensions =",
        normalized_df.shape[1],
    )

    # --------------------------------------------------------
    # Merge 187 ReCalib drugs
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("[4] MERGE 187 RECALIB DRUGS")
    print("=" * 90)

    drug_table = targets[
        [
            "drug",
            "smile",
        ]
    ].copy()

    checkpoint = drug_table.merge(
        normalized_df,
        left_on="smile",
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    latent_columns = [
        c
        for c in checkpoint.columns
        if c.startswith("latent_")
    ]

    missing_rows = (
        checkpoint[
            latent_columns
        ]
        .isna()
        .all(axis=1)
    )

    if missing_rows.any():
        raise RuntimeError(
            "Target drugs missing RDKit vectors:\n"
            + checkpoint.loc[
                missing_rows,
                ["drug", "smile"],
            ].to_string(index=False)
        )

    if len(checkpoint) != 187:
        raise RuntimeError(
            f"Expected 187 checkpoint rows, "
            f"got {len(checkpoint)}"
        )

    if checkpoint["drug"].nunique() != 187:
        raise RuntimeError(
            "Drug tokens are not unique"
        )

    # Match official to_csv behavior:
    # keep index column.
    checkpoint.to_csv(
        OUT_PATH,
    )

    # --------------------------------------------------------
    # Save normalization audit
    # --------------------------------------------------------

    pd.DataFrame(
        {
            "feature": means.index,
            "mean": means.values,
            "std": stds.values,
        }
    ).to_csv(
        WORK_DIR
        / "reference_normalization_stats.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "dropped_feature": drop_columns
        }
    ).to_csv(
        WORK_DIR
        / "dropped_features.csv",
        index=False,
    )

    summary = {
        "reference_smiles": len(reference),
        "reference_unique_smiles": int(
            reference["smiles"].nunique()
        ),
        "target_drugs": len(checkpoint),
        "raw_feature_dim": int(
            clean.shape[1]
        ),
        "descriptor_count": int(
            len(descriptors)
        ),
        "dropped_low_variance_features": int(
            len(drop_columns)
        ),
        "final_latent_dim": int(
            len(latent_columns)
        ),
        "failed_descriptor_evaluations": int(
            len(failure_frame)
        ),
        "output_checkpoint": str(
            OUT_PATH.resolve()
        ),
    }

    (
        WORK_DIR
        / "build_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("FULL RDKIT CHECKPOINT BUILD COMPLETE")
    print("=" * 90)

    print(
        "checkpoint =",
        OUT_PATH,
    )

    print(
        "checkpoint shape =",
        checkpoint.shape,
    )

    print(
        "latent dimensions =",
        len(latent_columns),
    )

    print(
        "failure audit =",
        failure_path,
    )

    print(
        "summary =",
        WORK_DIR
        / "build_summary.json",
    )


if __name__ == "__main__":
    args = parse_args()

    if args.worker_batch:
        worker_batch(
            args.input,
            args.output,
        )

    elif args.worker_descriptor:
        worker_descriptor(
            args.smiles_file,
            args.descriptor,
            args.output,
        )

    else:
        main()
