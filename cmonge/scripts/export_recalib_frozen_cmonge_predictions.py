#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import anndata as ad
import jax
from flax import serialization
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from cmonge.trainers.ae_trainer import AETrainerModule
from cmonge.trainers.conditional_monge_trainer import ConditionalMongeTrainer
from cmonge.utils import load_config


# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
RECALIB = Path(__file__).resolve().parents[2]

H5AD = ROOT / "data/recalib_sciplex_main.h5ad"

CONFIG = ROOT / "configs/recalib_sciplex_frozen.yml"

DEFAULT_SEED = 42

CMONGE_CKPT = (
    ROOT
    / f"models/recalib_frozen_cmonge/seed{DEFAULT_SEED}"
)

MANIFEST = (
    ROOT
    / "data/recalib_sciplex_main_metadata"
    / "recalib_condition_manifest.csv"
)

OUTPUT_ROOT = (
    RECALIB
    / "cmonge_outputs"
    / f"mydata_frozen_cmonge_seed{DEFAULT_SEED}"
)

LOG_PATH = (
    ROOT
    / "logs"
    / f"load_frozen_cmonge_for_export_seed{DEFAULT_SEED}.yml"
)


# Canonical ReCalib files.
#
# We keep the exact existing:
#   - true expression
#   - control expression
#   - metadata
#
# and only replace the frozen predictor with CMonge.

CANONICAL = {
    "fit": {
        "root": (
            RECALIB
            / "crisp_outputs"
            / "mydata_baseline_calib_seed2024"
            / "iid"
        ),
        "expected": 1003,
    },

    "validation": {
        "root": (
            RECALIB
            / "crisp_outputs"
            / "mydata_baseline_calib_seed2024"
            / "ood"
        ),
        "expected": 251,
    },

    "main_ood": {
        "root": (
            RECALIB
            / "crisp_outputs"
            / "mydata_baseline_full"
            / "ood"
        ),
        "expected": 108,
    },
}


BATCH_SIZE = 512

CELL_TYPES = [
    "A549",
    "K562",
    "MCF7",
]

EXPECTED_CONTROL_COUNTS = {
    "A549": 276,
    "K562": 320,
    "MCF7": 536,
}


# ============================================================================
# Minimal datamodule for loading RDKit embedding + CMonge checkpoint
# ============================================================================

class InferenceDataModule:
    """
    CMonge's checkpointed RDKitEmbedding only needs datamodule.batch_size
    during initialization.

    No treated target data are exposed to this object.
    """

    def __init__(self, batch_size: int):
        self.batch_size = batch_size


# ============================================================================
# Helpers
# ============================================================================

def dense_float32(x):
    if sparse.issparse(x):
        x = x.toarray()

    return np.asarray(
        x,
        dtype=np.float32,
    )


def load_source_controls(
    adata,
    obs: pd.DataFrame,
    cell_type: str,
):
    """
    Exact source protocol recovered from the existing ReCalib
    x_ctrl_condition_mean.csv files:

        control cells
        AND same cell_type
        AND frozen_base_split == test
    """

    mask = (
        obs["product_name"]
        .astype(str)
        .str.lower()
        .eq("control")
        &
        obs["cell_type"]
        .astype(str)
        .eq(cell_type)
        &
        obs["frozen_base_split"]
        .astype(str)
        .eq("test")
    )

    idx = np.flatnonzero(
        mask.to_numpy()
    )

    expected = EXPECTED_CONTROL_COUNTS[
        cell_type
    ]

    if len(idx) != expected:
        raise RuntimeError(
            f"{cell_type}: expected "
            f"{expected} base-test controls, "
            f"found {len(idx)}"
        )

    x = dense_float32(
        adata[idx, :].X
    )

    if x.shape != (
        expected,
        977,
    ):
        raise RuntimeError(
            f"{cell_type}: unexpected "
            f"source shape {x.shape}"
        )

    return x


# ============================================================================
# Main
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export Frozen CMonge condition-level predictions "
            "for the ReCalib SciPlex experiment."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=[42, 2024, 3407],
        help="Frozen CMonge training seed.",
    )

    return parser.parse_args()


def main():
    global CMONGE_CKPT
    global OUTPUT_ROOT
    global LOG_PATH

    args = parse_args()
    seed = int(args.seed)

    CMONGE_CKPT = (
        ROOT
        / f"models/recalib_frozen_cmonge/seed{seed}"
    )

    OUTPUT_ROOT = (
        RECALIB
        / "cmonge_outputs"
        / f"mydata_frozen_cmonge_seed{seed}"
    )

    LOG_PATH = (
        ROOT
        / "logs"
        / f"load_frozen_cmonge_for_export_seed{seed}.yml"
    )

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("EXPORT FROZEN CMONGE PREDICTIONS")
    print("=" * 100)
    print("Seed:", seed)
    print("Checkpoint:", CMONGE_CKPT)
    print("Output:", OUTPUT_ROOT)

    print("=" * 100)
    print("EXPORT FORMAL FROZEN CMONGE PREDICTIONS")
    print("=" * 100)

    print("\nCMonge checkpoint:")
    print(CMONGE_CKPT)

    print("\nInput h5ad:")
    print(H5AD)

    print("\nOutput:")
    print(OUTPUT_ROOT)

    if not CMONGE_CKPT.exists():
        raise FileNotFoundError(
            CMONGE_CKPT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================================
    # 1. Read adapted full ReCalib sci-Plex
    # ========================================================================

    print("\n" + "=" * 100)
    print("[1] LOAD FULL ADAPTED SCIPLEX")
    print("=" * 100)

    adata = ad.read_h5ad(
        H5AD,
        backed="r",
    )

    obs = adata.obs.copy()

    print("shape =", adata.shape)

    if adata.shape != (
        354640,
        977,
    ):
        raise RuntimeError(
            f"Unexpected h5ad shape: "
            f"{adata.shape}"
        )

    # CMonge decoded vectors follow h5ad X order.
    if "gene_id" not in adata.var.columns:
        raise RuntimeError(
            "gene_id missing from adata.var"
        )

    gene_ids_x_order = (
        adata.var["gene_id"]
        .astype(str)
        .tolist()
    )

    if (
        len(gene_ids_x_order) != 977
        or len(set(gene_ids_x_order)) != 977
    ):
        raise RuntimeError(
            "gene_id is not a unique 977-gene axis"
        )

    print(
        "Gene namespace = adata.var['gene_id']"
    )
    print(
        "unique genes =",
        len(set(gene_ids_x_order)),
    )

    # ========================================================================
    # 2. Read condition mapping
    # ========================================================================

    print("\n" + "=" * 100)
    print("[2] LOAD 1003 / 251 / 108 CONDITION MANIFEST")
    print("=" * 100)

    manifest = pd.read_csv(
        MANIFEST
    )

    required_manifest = {
        "recalib_condition",
        "recalib_role",
        "cell_type",
        "original_drug",
        "cmonge_drug",
        "dose_nm",
        "cmonge_drug_dose",
    }

    missing = (
        required_manifest
        - set(manifest.columns)
    )

    if missing:
        raise RuntimeError(
            f"Manifest missing: "
            f"{sorted(missing)}"
        )

    if (
        manifest["recalib_condition"]
        .duplicated()
        .any()
    ):
        dup = manifest.loc[
            manifest[
                "recalib_condition"
            ].duplicated(False)
        ]

        raise RuntimeError(
            "Duplicate ReCalib conditions:\n"
            + dup.head(20).to_string(
                index=False
            )
        )

    manifest = (
        manifest
        .set_index(
            "recalib_condition",
            drop=False,
        )
    )

    role_counts = (
        manifest["recalib_role"]
        .value_counts()
        .to_dict()
    )

    print("manifest role counts:")
    print(role_counts)

    for role, info in CANONICAL.items():
        actual = int(
            (
                manifest["recalib_role"]
                == role
            ).sum()
        )

        if actual != info["expected"]:
            raise RuntimeError(
                f"{role}: expected "
                f"{info['expected']} "
                f"conditions, got {actual}"
            )

    # ========================================================================
    # 3. Recover exact base-test control source populations
    # ========================================================================

    print("\n" + "=" * 100)
    print("[3] LOAD EXACT RECALIB CONTROL SOURCE POPULATIONS")
    print("=" * 100)

    source_controls = {}

    for cell_type in CELL_TYPES:

        x = load_source_controls(
            adata,
            obs,
            cell_type,
        )

        source_controls[
            cell_type
        ] = x

        print(
            f"{cell_type}: "
            f"{x.shape[0]} cells × "
            f"{x.shape[1]} genes"
        )

    print(
        "Total base-test controls =",
        sum(
            x.shape[0]
            for x in source_controls.values()
        ),
    )

    # ========================================================================
    # 4. Load frozen AutoEncoder
    # ========================================================================

    print("\n" + "=" * 100)
    print("[4] LOAD FROZEN AUTOENCODER")
    print("=" * 100)

    config = load_config(
        CONFIG
    )

    # Fresh load_config gives the string expected by AETrainerModule.
    config.ae.model.act_fn = "gelu"

    ae_trainer = AETrainerModule(
        config.ae
    )

    ae_trainer.load_model(
        dataset_name="cond-sciplex",
        drug_condition="homogeneous",
    )

    ae_model = ae_trainer.model.bind(
        {
            "params":
                ae_trainer.state.params
        }
    )

    print(
        "Frozen AE loaded."
    )

    # ========================================================================
    # 5. Load frozen CMonge
    # ========================================================================

    print("\n" + "=" * 100)
    print("[5] LOAD FROZEN CMONGE")
    print("=" * 100)

    inference_dm = InferenceDataModule(
        batch_size=BATCH_SIZE
    )

    # Reload config because AETrainerModule mutates act_fn.
    cmonge_config = load_config(
        CONFIG
    )

    # Use the matching training seed when reconstructing
    # the model structure prior to checkpoint restoration.
    cmonge_config.model.seed = seed

    msgpack_path = (
        CMONGE_CKPT
        / "model.msgpack"
    )

    metadata_path = (
        CMONGE_CKPT
        / "checkpoint_metadata.json"
    )

    if msgpack_path.exists():
        # ----------------------------------------------------
        # Flax MessagePack checkpoint.
        # Used by runs saved with the audited MessagePack
        # serialization path.
        # ----------------------------------------------------
        print(
            "Checkpoint format: Flax MessagePack"
        )

        if metadata_path.exists():
            checkpoint_metadata = json.loads(
                metadata_path.read_text()
            )

            metadata_seed = int(
                checkpoint_metadata.get(
                    "seed",
                    seed,
                )
            )

            if metadata_seed != seed:
                raise RuntimeError(
                    "Checkpoint seed mismatch: "
                    f"requested={seed}, "
                    f"metadata={metadata_seed}"
                )

            verification_diff = float(
                checkpoint_metadata.get(
                    "verification_max_abs_diff",
                    0.0,
                )
            )

            if verification_diff != 0.0:
                raise RuntimeError(
                    "Checkpoint metadata reports a "
                    "non-zero restoration difference: "
                    f"{verification_diff}"
                )

            print(
                "Checkpoint metadata verified."
            )

        trainer = ConditionalMongeTrainer(
            jobid=seed,
            logger_path=LOG_PATH,
            config=cmonge_config.model,
            datamodule=inference_dm,
        )

        trainer.model = serialization.from_bytes(
            trainer.model,
            msgpack_path.read_bytes(),
        )

        leaves = jax.tree_util.tree_leaves(
            jax.device_get(
                trainer.model
            )
        )

        for leaf_index, leaf in enumerate(leaves):
            array = np.asarray(leaf)

            if (
                np.issubdtype(
                    array.dtype,
                    np.number,
                )
                and not np.isfinite(array).all()
            ):
                raise RuntimeError(
                    "Non-finite values detected in "
                    f"restored model leaf {leaf_index}."
                )

        print(
            "MessagePack model restored."
        )
        print(
            "PyTree leaves =",
            len(leaves),
        )

    else:
        # ----------------------------------------------------
        # Original Orbax checkpoint.
        # Used by the already-completed seed42 / seed2024 runs.
        # ----------------------------------------------------
        print(
            "Checkpoint format: Orbax PyTree"
        )

        trainer = (
            ConditionalMongeTrainer
            .load_checkpoint(
                jobid=seed,
                logger_path=LOG_PATH,
                config=cmonge_config.model,
                ckpt_path=CMONGE_CKPT,
                datamodule=inference_dm,
            )
        )

    print(
        "Frozen CMonge checkpoint loaded."
    )

    print(
        "RDKit checkpoint drugs =",
        len(
            trainer
            .embedding_module
            .embeddings
        ),
    )

    if (
        len(
            trainer
            .embedding_module
            .embeddings
        )
        != 187
    ):
        raise RuntimeError(
            "Expected 187 RDKit drug embeddings"
        )

    # ========================================================================
    # 6. JIT inference functions
    # ========================================================================

    print("\n" + "=" * 100)
    print("[6] PREPARE JIT INFERENCE")
    print("=" * 100)

    @jax.jit
    def encode_batch(x):
        return ae_model.encoder(x)

    @jax.jit
    def decode_batch(z):
        return ae_model.decoder(z)

    @jax.jit
    def transport_batch(
        z,
        c,
    ):
        return trainer.transport(
            z,
            c,
            2,
        )

    # Encode the same source control populations only once.
    encoded_controls = {}

    for cell_type in CELL_TYPES:

        x = jnp.asarray(
            source_controls[
                cell_type
            ],
            dtype=jnp.float32,
        )

        z = encode_batch(x)

        z = np.asarray(
            jax.device_get(z),
            dtype=np.float32,
        )

        if z.shape != (
            x.shape[0],
            50,
        ):
            raise RuntimeError(
                f"{cell_type}: unexpected "
                f"AE latent shape {z.shape}"
            )

        encoded_controls[
            cell_type
        ] = z

        print(
            f"{cell_type}: "
            f"{source_controls[cell_type].shape} "
            f"-> {z.shape}"
        )

    # ========================================================================
    # 7. Prediction function
    # ========================================================================

    def predict_condition(
        cell_type: str,
        cmonge_condition: str,
    ) -> np.ndarray:

        if (
            cell_type
            not in encoded_controls
        ):
            raise ValueError(
                f"Unknown cell type: "
                f"{cell_type}"
            )

        z_source = encoded_controls[
            cell_type
        ]

        cond_full, n_contexts = (
            trainer
            .embedding_module(
                cmonge_condition
            )
        )

        if int(n_contexts) != 2:
            raise RuntimeError(
                f"{cmonge_condition}: "
                f"expected n_contexts=2, "
                f"got {n_contexts}"
            )

        sum_pred = np.zeros(
            977,
            dtype=np.float64,
        )

        n_total = 0

        for start in range(
            0,
            len(z_source),
            BATCH_SIZE,
        ):

            end = min(
                start + BATCH_SIZE,
                len(z_source),
            )

            z = jnp.asarray(
                z_source[start:end],
                dtype=jnp.float32,
            )

            n = end - start

            # RDKitEmbedding creates a condition batch
            # using datamodule.batch_size (=512).
            c = cond_full[:n]

            z_pred = transport_batch(
                z,
                c,
            )

            x_pred = decode_batch(
                z_pred
            )

            x_pred = np.asarray(
                jax.device_get(
                    x_pred
                ),
                dtype=np.float64,
            )

            if x_pred.shape != (
                n,
                977,
            ):
                raise RuntimeError(
                    f"{cmonge_condition}: "
                    f"unexpected decoded "
                    f"shape {x_pred.shape}"
                )

            if not np.isfinite(
                x_pred
            ).all():
                raise RuntimeError(
                    f"{cmonge_condition}: "
                    "NaN/Inf prediction"
                )

            sum_pred += (
                x_pred.sum(axis=0)
            )

            n_total += n

        if n_total != len(
            z_source
        ):
            raise RuntimeError(
                "Prediction cell count mismatch"
            )

        return (
            sum_pred
            / float(n_total)
        )

    # ========================================================================
    # 8. One-condition inference smoke
    # ========================================================================

    print("\n" + "=" * 100)
    print("[7] ONE-CONDITION CHECK")
    print("=" * 100)

    first_condition = (
        manifest.iloc[0]
    )

    first_pred = predict_condition(
        str(
            first_condition[
                "cell_type"
            ]
        ),
        str(
            first_condition[
                "cmonge_drug_dose"
            ]
        ),
    )

    print(
        "condition =",
        first_condition[
            "recalib_condition"
        ],
    )

    print(
        "CMonge context =",
        first_condition[
            "cmonge_drug_dose"
        ],
    )

    print(
        "prediction shape =",
        first_pred.shape,
    )

    print(
        "finite =",
        bool(
            np.isfinite(
                first_pred
            ).all()
        ),
    )

    print(
        "mean =",
        float(first_pred.mean()),
    )

    print(
        "std =",
        float(first_pred.std()),
    )

    # ========================================================================
    # 9. Export each ReCalib split
    # ========================================================================

    print("\n" + "=" * 100)
    print("[8] EXPORT 1003 / 251 / 108 PREDICTIONS")
    print("=" * 100)

    export_summary = {
        "model": "Frozen CMonge",
        "seed": 42,
        "checkpoint": str(
            CMONGE_CKPT
        ),
        "control_source":
            "split_crisp_eval_all=test, "
            "same cell_type, control cells",
        "control_counts":
            EXPECTED_CONTROL_COUNTS,
        "splits": {},
    }

    for role, info in CANONICAL.items():

        canonical_root = (
            info["root"]
        )

        out_dir = (
            OUTPUT_ROOT / role
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        true_path = (
            canonical_root
            / "x_true_condition_mean.csv"
        )

        ctrl_path = (
            canonical_root
            / "x_ctrl_condition_mean.csv"
        )

        metadata_path = (
            canonical_root
            / "condition_metadata.csv"
        )

        for p in [
            true_path,
            ctrl_path,
            metadata_path,
        ]:
            if not p.exists():
                raise FileNotFoundError(
                    p
                )

        true_df = pd.read_csv(
            true_path,
            index_col=0,
        )

        ctrl_df = pd.read_csv(
            ctrl_path,
            index_col=0,
        )

        metadata_df = pd.read_csv(
            metadata_path,
        )

        true_df.index = (
            true_df.index
            .astype(str)
        )

        ctrl_df.index = (
            ctrl_df.index
            .astype(str)
        )

        true_df.columns = (
            true_df.columns
            .astype(str)
        )

        ctrl_df.columns = (
            ctrl_df.columns
            .astype(str)
        )

        if len(true_df) != info["expected"]:
            raise RuntimeError(
                f"{role}: true matrix "
                f"expected {info['expected']}, "
                f"got {len(true_df)}"
            )

        if true_df.shape[1] != 977:
            raise RuntimeError(
                f"{role}: true matrix "
                f"gene count != 977"
            )

        if list(
            true_df.columns
        ) != list(
            ctrl_df.columns
        ):
            raise RuntimeError(
                f"{role}: true/control "
                "gene order differs"
            )

        # Metadata condition order is the authoritative
        # ReCalib condition ordering.
        if "condition" not in metadata_df.columns:
            raise RuntimeError(
                f"{role}: metadata has no "
                "'condition' column"
            )

        condition_order = (
            metadata_df[
                "condition"
            ]
            .astype(str)
            .tolist()
        )

        if (
            len(condition_order)
            != info["expected"]
        ):
            raise RuntimeError(
                f"{role}: metadata "
                f"condition count mismatch"
            )

        if (
            len(set(condition_order))
            != len(condition_order)
        ):
            raise RuntimeError(
                f"{role}: duplicate "
                "metadata conditions"
            )

        if (
            set(condition_order)
            != set(true_df.index)
        ):
            raise RuntimeError(
                f"{role}: metadata and "
                "true matrix conditions differ"
            )

        # Make sure all required conditions exist
        # in our CMonge mapping.
        absent = [
            c
            for c in condition_order
            if c not in manifest.index
        ]

        if absent:
            raise RuntimeError(
                f"{role}: conditions missing "
                f"from CMonge manifest:\n"
                f"{absent[:20]}"
            )

        print()
        print("-" * 100)
        print(
            role,
            "=",
            len(condition_order),
        )
        print("-" * 100)

        predictions = []

        prediction_metadata = []

        for condition in tqdm(
            condition_order,
            desc=f"CMonge {role}",
        ):

            row = manifest.loc[
                condition
            ]

            actual_role = str(
                row["recalib_role"]
            )

            if actual_role != role:
                raise RuntimeError(
                    f"{condition}: "
                    f"manifest role={actual_role}, "
                    f"expected={role}"
                )

            cell_type = str(
                row["cell_type"]
            )

            cmonge_condition = str(
                row[
                    "cmonge_drug_dose"
                ]
            )

            pred = predict_condition(
                cell_type,
                cmonge_condition,
            )

            predictions.append(
                pred
            )

            prediction_metadata.append(
                {
                    "condition":
                        condition,

                    "recalib_role":
                        role,

                    "cell_type":
                        cell_type,

                    "original_drug":
                        str(
                            row[
                                "original_drug"
                            ]
                        ),

                    "dose_nm":
                        float(
                            row[
                                "dose_nm"
                            ]
                        ),

                    "cmonge_drug":
                        str(
                            row[
                                "cmonge_drug"
                            ]
                        ),

                    "cmonge_drug_dose":
                        cmonge_condition,

                    "source_control_cells":
                        int(
                            len(
                                source_controls[
                                    cell_type
                                ]
                            )
                        ),
                }
            )

        pred_array = np.vstack(
            predictions
        )

        if pred_array.shape != (
            info["expected"],
            977,
        ):
            raise RuntimeError(
                f"{role}: prediction "
                f"shape={pred_array.shape}"
            )

        if not np.isfinite(
            pred_array
        ).all():
            raise RuntimeError(
                f"{role}: predictions "
                "contain NaN/Inf"
            )

        # First create using the actual h5ad X order.
        pred_df = pd.DataFrame(
            pred_array,
            index=condition_order,
            columns=gene_ids_x_order,
        )

        # Reorder to EXACTLY the existing ReCalib
        # true/control gene order.
        if set(
            pred_df.columns
        ) != set(
            true_df.columns
        ):
            raise RuntimeError(
                f"{role}: prediction "
                "gene set differs from "
                "ReCalib true matrix"
            )

        pred_df = pred_df[
            true_df.columns
        ]

        if list(
            pred_df.index
        ) != condition_order:
            raise RuntimeError(
                "Condition order changed"
            )

        # ----------------------------------------------------
        # Write CMonge predictor
        # ----------------------------------------------------

        pred_path = (
            out_dir
            / "x_cmonge_condition_mean.csv"
        )

        pred_df.to_csv(
            pred_path
        )

        # ----------------------------------------------------
        # Copy the ORIGINAL ReCalib ground truth,
        # control matrix, and metadata unchanged.
        # ----------------------------------------------------

        shutil.copy2(
            true_path,
            out_dir
            / "x_true_condition_mean.csv",
        )

        shutil.copy2(
            ctrl_path,
            out_dir
            / "x_ctrl_condition_mean.csv",
        )

        shutil.copy2(
            metadata_path,
            out_dir
            / "condition_metadata.csv",
        )

        pd.DataFrame(
            prediction_metadata
        ).to_csv(
            out_dir
            / "cmonge_prediction_metadata.csv",
            index=False,
        )

        # ----------------------------------------------------
        # Final reload audit
        # ----------------------------------------------------

        check = pd.read_csv(
            pred_path,
            index_col=0,
        )

        check.index = (
            check.index
            .astype(str)
        )

        check.columns = (
            check.columns
            .astype(str)
        )

        if check.shape != (
            info["expected"],
            977,
        ):
            raise RuntimeError(
                f"{role}: saved shape "
                f"incorrect: {check.shape}"
            )

        if list(
            check.index
        ) != condition_order:
            raise RuntimeError(
                f"{role}: saved "
                "condition order differs"
            )

        if list(
            check.columns
        ) != list(
            true_df.columns
        ):
            raise RuntimeError(
                f"{role}: saved "
                "gene order differs"
            )

        if not np.isfinite(
            check.to_numpy(
                dtype=float
            )
        ).all():
            raise RuntimeError(
                f"{role}: saved "
                "predictions non-finite"
            )

        print(
            "saved:",
            pred_path
        )

        print(
            "shape:",
            check.shape
        )

        print(
            "prediction mean:",
            float(
                check.to_numpy(
                    dtype=float
                ).mean()
            ),
        )

        print(
            "prediction std:",
            float(
                check.to_numpy(
                    dtype=float
                ).std()
            ),
        )

        export_summary[
            "splits"
        ][role] = {
            "conditions":
                int(check.shape[0]),

            "genes":
                int(check.shape[1]),

            "prediction_file":
                str(
                    pred_path.resolve()
                ),

            "all_finite":
                True,
        }

    # ========================================================================
    # 10. Global completeness check
    # ========================================================================

    print("\n" + "=" * 100)
    print("[9] GLOBAL COMPLETENESS CHECK")
    print("=" * 100)

    total_conditions = sum(
        x["expected"]
        for x in CANONICAL.values()
    )

    print(
        "fit        =",
        CANONICAL["fit"]["expected"],
    )

    print(
        "validation =",
        CANONICAL["validation"]["expected"],
    )

    print(
        "main_ood   =",
        CANONICAL["main_ood"]["expected"],
    )

    print(
        "TOTAL      =",
        total_conditions,
    )

    if total_conditions != 1362:
        raise RuntimeError(
            "Expected total 1362 conditions"
        )

    summary_path = (
        OUTPUT_ROOT
        / "export_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            export_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nSummary:",
        summary_path
    )

    print("\n" + "=" * 100)
    print(
        "FROZEN CMONGE PREDICTION EXPORT PASSED"
    )
    print(
        "1003 + 251 + 108 = 1362 CONDITIONS"
    )
    print(
        "977 / 977 GENES"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
