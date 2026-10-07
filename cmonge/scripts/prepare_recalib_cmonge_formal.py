#!/usr/bin/env python3

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import json

import anndata as ad
import pandas as pd
import yaml


INPUT = Path(
    str(ROOT / "data/recalib_sciplex_main.h5ad")
)

OUTPUT = Path(
    str(ROOT / "data/recalib_sciplex_frozen_train.h5ad")
)

CONFIG = Path(
    str(ROOT / "configs/recalib_sciplex_frozen.yml")
)

META_DIR = (
    ROOT
    / "data"
    / "recalib_sciplex_main_metadata"
    / "formal_cmonge"
)

META_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_CELLS = 312405
EXPECTED_GENES = 977

MAIN_OOD_DRUGS = {
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


print("=" * 90)
print("PREPARE FORMAL FROZEN CMONGE TRAINING DATA")
print("=" * 90)


# ============================================================
# 1. Read adapted full data
# ============================================================

adata = ad.read_h5ad(INPUT, backed="r")
obs = adata.obs.copy()

base_train = (
    obs["frozen_base_split"].astype(str)
    == "train"
)

print("\n[1] Frozen-base train cells")
print("cells =", int(base_train.sum()))

if int(base_train.sum()) != EXPECTED_CELLS:
    raise RuntimeError(
        f"Expected {EXPECTED_CELLS} train cells, "
        f"found {int(base_train.sum())}"
    )

if adata.n_vars != EXPECTED_GENES:
    raise RuntimeError(
        f"Expected {EXPECTED_GENES} genes, "
        f"found {adata.n_vars}"
    )


# ============================================================
# 2. Audit training targets
# ============================================================

train_obs = obs.loc[base_train].copy()

is_control = (
    train_obs["product_name"]
    .astype(str)
    .str.lower()
    .eq("control")
)

treated = train_obs.loc[~is_control].copy()

conditions = sorted(
    treated["cmonge_drug_dose"]
    .astype(str)
    .unique()
    .tolist()
)

print("\n[2] Training conditions")
print("non-control drug-dose conditions =", len(conditions))

if "control-0" in conditions:
    raise RuntimeError(
        "control-0 unexpectedly appears in target conditions"
    )

condition_counts = (
    treated["cmonge_drug_dose"]
    .astype(str)
    .value_counts()
    .rename_axis("cmonge_drug_dose")
    .reset_index(name="n_cells")
)

condition_info = (
    treated[
        [
            "cmonge_drug_dose",
            "cmonge_drug",
            "product_name",
            "dose",
        ]
    ]
    .drop_duplicates()
    .merge(
        condition_counts,
        on="cmonge_drug_dose",
        how="left",
        validate="one_to_one",
    )
    .sort_values(
        ["product_name", "dose"]
    )
)

print("minimum cells per condition =", int(condition_info["n_cells"].min()))
print("maximum cells per condition =", int(condition_info["n_cells"].max()))

too_small = condition_info.loc[
    condition_info["n_cells"] < 2
]

if len(too_small):
    raise RuntimeError(
        "Conditions with fewer than 2 cells:\n"
        + too_small.to_string(index=False)
    )


# ============================================================
# 3. Main-OOD leakage audit
# ============================================================

training_drugs = set(
    treated["product_name"].astype(str)
)

overlap = training_drugs & MAIN_OOD_DRUGS

print("\n[3] Main-OOD leakage audit")
print("training drugs =", len(training_drugs))
print("Main-OOD overlap =", sorted(overlap))

if overlap:
    raise RuntimeError(
        f"Main-OOD drug leakage detected: {sorted(overlap)}"
    )

print("Main-OOD leakage = 0  [PASS]")


# ============================================================
# 4. Control audit
# ============================================================

control_count = int(is_control.sum())

print("\n[4] Controls")
print("train control cells =", control_count)

if control_count == 0:
    raise RuntimeError(
        "No base-training control cells found"
    )

control_labels = set(
    train_obs.loc[
        is_control,
        "cmonge_drug_dose"
    ].astype(str)
)

print("control labels =", sorted(control_labels))

if control_labels != {"control-0"}:
    raise RuntimeError(
        f"Unexpected control labels: {control_labels}"
    )


# ============================================================
# 5. Materialize formal training h5ad
# ============================================================

print("\n[5] Writing formal training h5ad")

formal = adata[
    base_train.to_numpy(),
    :
].to_memory()

formal.write_h5ad(
    OUTPUT,
    compression="gzip",
)

print("output =", OUTPUT)
print("shape =", formal.shape)

if formal.shape != (
    EXPECTED_CELLS,
    EXPECTED_GENES,
):
    raise RuntimeError(
        f"Unexpected output shape: {formal.shape}"
    )


# ============================================================
# 6. Save training-condition manifest
# ============================================================

condition_info.to_csv(
    META_DIR / "training_conditions.csv",
    index=False,
)

pd.DataFrame(
    {"condition": conditions}
).to_csv(
    META_DIR / "training_condition_list.csv",
    index=False,
)


# ============================================================
# 7. Formal CMonge configuration
# ============================================================

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
            "dim_hidden": [64, 64, 64, 64],

            "dim_data": 50,

            # Current CMonge demo-compatible RDKit+dose layout.
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

        # Formal CMonge training scale.
        "num_train_iters": 10000,

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

            # Must be absolute because of the checkpoint
            # behavior we already observed on this server.
            "model_dir":
            str(ROOT / "models" / "recalib_frozen_ae"),
        },
    },
}


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


summary = {
    "input_cells": int(adata.n_obs),
    "formal_train_cells": int(formal.n_obs),
    "genes": int(formal.n_vars),
    "control_cells": control_count,
    "training_drugs": len(training_drugs),
    "training_drug_dose_conditions": len(conditions),
    "min_cells_per_condition": int(
        condition_info["n_cells"].min()
    ),
    "max_cells_per_condition": int(
        condition_info["n_cells"].max()
    ),
    "main_ood_drug_overlap": sorted(overlap),
    "cmonge_iterations": 10000,
    "batch_size": 512,
    "seed": 42,
}

with open(
    META_DIR / "formal_training_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


print("\nConfig =", CONFIG)
print(
    "Condition manifest =",
    META_DIR / "training_conditions.csv"
)

print("\n" + "=" * 90)
print("FORMAL FROZEN CMONGE DATA PREPARED")
print("=" * 90)
