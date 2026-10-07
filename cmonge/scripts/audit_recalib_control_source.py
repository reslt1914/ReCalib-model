#!/usr/bin/env python3

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


H5AD = Path(
    str(CMONGE_ROOT / "data/recalib_sciplex_main.h5ad")
)

RECALIB = RECALIB_ROOT

FILES = {
    "fit": (
        RECALIB
        / "crisp_outputs/mydata_baseline_calib_seed2024"
        / "iid/x_ctrl_condition_mean.csv"
    ),

    "validation": (
        RECALIB
        / "crisp_outputs/mydata_baseline_calib_seed2024"
        / "ood/x_ctrl_condition_mean.csv"
    ),

    "main_ood": (
        RECALIB
        / "crisp_outputs/mydata_baseline_full"
        / "ood/x_ctrl_condition_mean.csv"
    ),
}


print("=" * 90)
print("RECALIB CONTROL SOURCE AUDIT")
print("=" * 90)

adata = ad.read_h5ad(H5AD, backed="r")
obs = adata.obs

# ReCalib CSV matrices and the adapted h5ad contain the same 977
# genes, but they may use different identifiers (e.g. Ensembl IDs
# versus gene symbols). Find the exact one-to-one identifier namespace.
reference_ctrl = pd.read_csv(
    FILES["fit"],
    index_col=0,
)

csv_genes = set(map(str, reference_ctrl.columns))

gene_id_source = None
genes = None

candidates = [
    (
        "var_names",
        pd.Series(
            adata.var_names.astype(str),
            index=adata.var_names,
        ),
    )
]

for col in adata.var.columns:
    candidates.append(
        (
            str(col),
            adata.var[col].astype(str),
        )
    )

for name, values in candidates:
    values = pd.Series(
        list(map(str, values)),
        index=adata.var_names,
    )

    if (
        values.nunique() == adata.n_vars
        and set(values) == csv_genes
    ):
        gene_id_source = name
        genes = values.tolist()
        break

if gene_id_source is None:
    raise RuntimeError(
        "No exact 1-to-1 gene identifier mapping was found "
        "between h5ad.var and the ReCalib control matrix."
    )

print("\nGene namespace matched through:", gene_id_source)
print("Matched genes =", len(genes))

is_control = (
    obs["product_name"]
    .astype(str)
    .str.lower()
    .eq("control")
)

base_split = (
    obs["frozen_base_split"]
    .astype(str)
)


def mean_expression(mask):
    idx = np.where(
        np.asarray(mask)
    )[0]

    x = adata[idx, :].X

    if sparse.issparse(x):
        result = np.asarray(
            x.mean(axis=0)
        ).ravel()
    else:
        result = np.asarray(
            x
        ).mean(axis=0)

    return result.astype(
        np.float64
    )


# ------------------------------------------------------------
# Candidate control means
# ------------------------------------------------------------

candidates = {
    "all_control": {},
    "base_train_control": {},
    "base_test_control": {},
}

for cell_type in [
    "A549",
    "K562",
    "MCF7",
]:
    same_cell = (
        obs["cell_type"]
        .astype(str)
        .eq(cell_type)
    )

    masks = {
        "all_control":
            is_control
            & same_cell,

        "base_train_control":
            is_control
            & same_cell
            & base_split.eq("train"),

        "base_test_control":
            is_control
            & same_cell
            & base_split.eq("test"),
    }

    print()
    print(
        "=" * 70
    )
    print(
        "CELL TYPE:",
        cell_type
    )
    print(
        "=" * 70
    )

    for name, mask in masks.items():
        n = int(mask.sum())

        print(
            f"{name:22s}: "
            f"{n} cells"
        )

        if n == 0:
            raise RuntimeError(
                f"No cells for "
                f"{cell_type}/{name}"
            )

        candidates[name][cell_type] = (
            mean_expression(mask)
        )


# ------------------------------------------------------------
# Compare against ReCalib x_ctrl_condition_mean.csv
# ------------------------------------------------------------

print()
print("=" * 90)
print("COMPARE AGAINST EXISTING RECALIB CONTROL MATRICES")
print("=" * 90)

summary_rows = []

for role, path in FILES.items():

    print()
    print("-" * 90)
    print(role)
    print(path)
    print("-" * 90)

    df = pd.read_csv(
        path,
        index_col=0,
    )

    df.index = (
        df.index
        .astype(str)
    )

    df.columns = [
        str(x)
        for x in df.columns
    ]

    print(
        "shape =",
        df.shape
    )

    if set(df.columns) != set(genes):
        missing_from_csv = (
            set(genes)
            - set(df.columns)
        )

        extra_in_csv = (
            set(df.columns)
            - set(genes)
        )

        raise RuntimeError(
            "Gene mismatch:\n"
            f"missing={list(missing_from_csv)[:10]}\n"
            f"extra={list(extra_in_csv)[:10]}"
        )

    # exact h5ad gene order
    df = df[genes]

    for candidate_name in candidates:

        per_row_mse = []
        per_row_max_abs = []

        for condition, row in df.iterrows():

            cell_type = (
                condition.split(
                    "_",
                    1,
                )[0]
            )

            if cell_type not in candidates[
                candidate_name
            ]:
                raise RuntimeError(
                    f"Unknown cell type in "
                    f"condition: {condition}"
                )

            expected = candidates[
                candidate_name
            ][cell_type]

            actual = (
                row.to_numpy(
                    dtype=np.float64
                )
            )

            diff = (
                actual
                - expected
            )

            per_row_mse.append(
                np.mean(
                    diff ** 2
                )
            )

            per_row_max_abs.append(
                np.max(
                    np.abs(diff)
                )
            )

        mean_mse = float(
            np.mean(
                per_row_mse
            )
        )

        max_abs = float(
            np.max(
                per_row_max_abs
            )
        )

        print(
            f"{candidate_name:22s} "
            f"mean_MSE={mean_mse:.12g} "
            f"max_abs={max_abs:.12g}"
        )

        summary_rows.append(
            {
                "role":
                    role,

                "candidate":
                    candidate_name,

                "mean_mse":
                    mean_mse,

                "max_abs":
                    max_abs,
            }
        )


summary = pd.DataFrame(
    summary_rows
)

OUT = Path(
    "data/recalib_sciplex_main_metadata/"
    "formal_cmonge/"
    "control_source_audit.csv"
)

summary.to_csv(
    OUT,
    index=False,
)

print()
print("=" * 90)
print("BEST CANDIDATE PER SPLIT")
print("=" * 90)

for role in FILES:
    subset = (
        summary[
            summary["role"]
            == role
        ]
        .sort_values(
            "mean_mse"
        )
    )

    print()
    print(role)
    print(
        subset.to_string(
            index=False
        )
    )

print()
print("saved:", OUT)

print()
print("=" * 90)
print("CONTROL SOURCE AUDIT COMPLETE")
print("=" * 90)
