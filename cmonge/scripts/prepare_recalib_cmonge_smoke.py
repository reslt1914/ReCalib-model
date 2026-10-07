#!/usr/bin/env python3

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import anndata as ad
import pandas as pd
import yaml


INPUT = Path(
    str(ROOT / "data/recalib_sciplex_main.h5ad")
)

OUTPUT = Path(
    str(ROOT / "data/recalib_sciplex_smoke.h5ad")
)

CONFIG = Path(
    str(ROOT / "configs/recalib_sciplex_smoke.yml")
)

N_DRUGS = 6


print("=" * 90)
print("PREPARE REAL SCIPLEX CMONGE SMOKE TEST")
print("=" * 90)


# ------------------------------------------------------------
# 1. Read metadata only first
# ------------------------------------------------------------

adata = ad.read_h5ad(INPUT, backed="r")
obs = adata.obs.copy()

base_train = (
    obs["frozen_base_split"].astype(str) == "train"
)

control = (
    obs["product_name"].astype(str).str.lower() == "control"
)

treated = (
    base_train & ~control
)


# ------------------------------------------------------------
# 2. Select 6 well-populated conditions from different drugs
# ------------------------------------------------------------

tmp = obs.loc[
    treated,
    [
        "cmonge_drug",
        "cmonge_drug_dose",
        "product_name",
        "dose",
    ],
].copy()

counts = (
    tmp.groupby(
        [
            "cmonge_drug",
            "cmonge_drug_dose",
            "product_name",
            "dose",
        ],
        observed=True,
    )
    .size()
    .reset_index(name="n_cells")
    .sort_values(
        "n_cells",
        ascending=False,
    )
)

selected = (
    counts
    .drop_duplicates(
        subset=["cmonge_drug"]
    )
    .head(N_DRUGS)
    .copy()
)

if len(selected) != N_DRUGS:
    raise RuntimeError(
        f"Expected {N_DRUGS} distinct drugs, "
        f"found {len(selected)}"
    )

conditions = (
    selected["cmonge_drug_dose"]
    .astype(str)
    .tolist()
)

print("\nSelected smoke conditions:")
print(
    selected[
        [
            "product_name",
            "dose",
            "cmonge_drug",
            "cmonge_drug_dose",
            "n_cells",
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# 3. Main-OOD safety check
# ------------------------------------------------------------

ood_drugs = {
    "CUDC-101",
    "CUDC-907",
    "Dacinostat",
    "Givinostat",
    "Hesperadin",
    "Pirarubicin",
    "Raltitrexed",
    "Tanespimycin",
    "Trametinib",
}

selected_original = set(
    selected["product_name"].astype(str)
)

overlap = selected_original & ood_drugs

print("\nMain-OOD overlap =", overlap)

if overlap:
    raise RuntimeError(
        "Smoke dataset contains Main-OOD drug!"
    )


# ------------------------------------------------------------
# 4. Keep train controls + selected train targets
# ------------------------------------------------------------

keep = (
    base_train
    & (
        control
        |
        obs["cmonge_drug_dose"]
        .astype(str)
        .isin(conditions)
    )
)

print("\nSelected cells =", int(keep.sum()))
print(
    "control cells =",
    int((keep & control).sum()),
)

for cond in conditions:
    n = int(
        (
            keep
            & (
                obs["cmonge_drug_dose"]
                .astype(str)
                == cond
            )
        ).sum()
    )

    print(cond, "=", n)


# ------------------------------------------------------------
# 5. Load only selected cells into memory
# ------------------------------------------------------------

smoke = adata[keep.to_numpy(), :].to_memory()

if smoke.n_vars != 977:
    raise RuntimeError(
        f"Expected 977 genes, got {smoke.n_vars}"
    )

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

smoke.write_h5ad(
    OUTPUT,
    compression="gzip",
)

print("\nSmoke h5ad:")
print(OUTPUT)
print("shape =", smoke.shape)


# ------------------------------------------------------------
# 6. Create smoke config
# ------------------------------------------------------------

config = {
    "model": {
        "method": "monge",

        "fitting_loss": {
            "name": "sinkhorn",
            "kwargs": {
                "epsilon_fitting": 1,
            },
        },

        "regularizer": {
            "name": "monge",
            "kwargs": {
                "epsilon_regularizer": 1.0e-2,
                "cost": "euclidean",
            },
        },

        "optim": {
            "lr": 1.0e-4,
            "name": "adamw",
            "kwargs": {
                "weight_decay": 1.0e-5,
            },
        },

        "mlp": {
            "dim_hidden": [
                64,
                64,
                64,
                64,
            ],

            "dim_data": 50,

            "context_entity_bonds": [
                [0, 194],
                [0, 195],
            ],

            "dim_cond": 195,
            "dim_cond_map": 50,
        },

        "embedding": {
            "name": "rdkit",

            "smile_path":
                str(ROOT / "data/lincs_trapnell.smiles"),

            "drug_to_smile_path":
                str(
            ROOT
            / "data"
            / "recalib_sciplex_main_metadata"
            / "cmonge_drug_smiles.csv"
        ),

            "model_dir":
                str(ROOT / "models/embed/recalib/"),

            "checkpoint": True,
        },

        # Smoke only
        "num_train_iters": 100,

        "seed": 42,
    },

    "data": {
        "name": "sciplex",

        "file_path": str(
            OUTPUT.resolve()
        ),

        "batch_size": 512,

        "split": [
            0.8,
            0.2,
            0.0,
        ],

        "drug_col": "drug-dose",

        "drug_condition":
            conditions[0],

        "control_condition":
            "control-0",

        "seed": 42,

        "ae": False,

        "reduction": "ae",

        "ae_config_path": None,
    },

    "condition": {
        "mode": "homogeneous",

        "conditions": conditions,

        "split": [
            0.8,
            0.2,
            0.0,
        ],
    },

    "ae": {
        "model": {
            "hidden_dims": [
                512,
                512,
            ],

            "latent_dim": 50,

            # ReCalib uses 977 genes.
            "data_dim": 977,

            "seed": 1,

            "act_fn": "gelu",
        },

        "optim": {
            "lr": 1.0e-4,

            "optimizer": "adamw",

            "kwargs": {
                "weight_decay": 1.0e-5,
            },
        },

        "training": {
            "n_epochs": 50,

            "valid": False,

            "ckpt": False,

            "model_dir":
                str(ROOT / "models/recalib_smoke/"),
        },
    },
}


CONFIG.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    CONFIG,
    "w",
    encoding="utf-8",
) as f:
    yaml.safe_dump(
        config,
        f,
        sort_keys=False,
    )


print("\nConfig:")
print(CONFIG)

print("\nConditions:")
for x in conditions:
    print(" ", x)

print("\n" + "=" * 90)
print("REAL SCIPLEX SMOKE DATA PREPARED")
print("=" * 90)
