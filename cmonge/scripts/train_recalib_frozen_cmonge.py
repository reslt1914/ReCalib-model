#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from cmonge.datasets.conditional_loader import (
    ConditionalDataModule,
)
from cmonge.datasets.recalib_loader import (
    ReCalibSciPlexModule,
)
from cmonge.trainers.ae_trainer import (
    AETrainerModule,
)
from cmonge.trainers.conditional_monge_trainer import (
    ConditionalMongeTrainer,
)
from cmonge.utils import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    PROJECT_ROOT
    / "configs/recalib_sciplex_frozen.yml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train Frozen CMonge for the ReCalib "
            "SciPlex experiment."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for CMonge and data sampling.",
    )

    parser.add_argument(
        "--train-ae",
        action="store_true",
        help=(
            "Train the autoencoder before CMonge. "
            "Normally omit this option because the "
            "shared ReCalib autoencoder already exists."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()
    seed = int(args.seed)

    # ------------------------------------------------------------------
    # Per-seed CMonge outputs
    # ------------------------------------------------------------------

    log_path = (
        PROJECT_ROOT
        / "logs"
        / f"recalib_frozen_cmonge_seed{seed}.yml"
    )

    cmonge_ckpt = (
        PROJECT_ROOT
        / "models"
        / "recalib_frozen_cmonge"
        / f"seed{seed}"
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmonge_ckpt.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Protect completed checkpoints from accidental overwrite.
    if (
        cmonge_ckpt.exists()
        and any(cmonge_ckpt.iterdir())
    ):
        raise RuntimeError(
            f"CMonge checkpoint already exists: "
            f"{cmonge_ckpt}\n"
            "Refusing to overwrite an existing run."
        )

    # ------------------------------------------------------------------
    # Loader registration
    # ------------------------------------------------------------------

    ConditionalDataModule.datamodule_factory[
        "sciplex"
    ] = ReCalibSciPlexModule

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    config = load_config(CONFIG)

    # These are the two experiment seeds.
    config.model.seed = seed
    config.data.seed = seed

    # IMPORTANT:
    # config.ae.model.seed remains exactly as configured:
    # seed = 1.
    #
    # The same frozen AE is reused across all CMonge seeds.

    print("=" * 90)
    print("FORMAL FROZEN CMONGE TRAINING")
    print("=" * 90)

    print("seed             =", seed)
    print(
        "model seed       =",
        config.model.seed,
    )
    print(
        "data seed        =",
        config.data.seed,
    )
    print(
        "AE seed          =",
        config.ae.model.seed,
    )
    print(
        "conditions       =",
        len(config.condition.conditions),
    )
    print(
        "iterations       =",
        config.model.num_train_iters,
    )
    print(
        "batch size       =",
        config.data.batch_size,
    )
    print(
        "CMonge checkpoint=",
        cmonge_ckpt,
    )
    print(
        "log              =",
        log_path,
    )

    # ------------------------------------------------------------------
    # Stage 1: Autoencoder
    #
    # The formal AE has already been trained for the seed42 experiment.
    # For additional CMonge seeds we reuse that exact AE.
    # ------------------------------------------------------------------

    if args.train_ae:

        print()
        print("=" * 90)
        print("STAGE 1: AUTOENCODER")
        print("=" * 90)

        config.data.ae = True
        config.data.reduction = None

        ae_datamodule = ConditionalDataModule(
            config.data,
            config.condition,
        )

        ae_trainer = AETrainerModule(
            config.ae,
        )

        ae_trainer.train(
            ae_datamodule,
        )

    else:

        print()
        print("=" * 90)
        print("STAGE 1: AUTOENCODER")
        print("=" * 90)

        print(
            "Using existing shared ReCalib "
            "autoencoder checkpoint."
        )

        print(
            "AE model directory =",
            config.ae.training.model_dir,
        )

    # ------------------------------------------------------------------
    # Stage 2: Conditional Monge
    # ------------------------------------------------------------------

    print()
    print("=" * 90)
    print("STAGE 2: CONDITIONAL MONGE")
    print("=" * 90)

    config.data.ae = False

    # Restore setting used by the established workflow.
    config.ae.model.act_fn = "gelu"
    config.data.reduction = "ae"

    datamodule = ConditionalDataModule(
        config.data,
        config.condition,
        ae_config=config.ae,
    )

    trainer = ConditionalMongeTrainer(
        jobid=seed,
        logger_path=log_path,
        config=config.model,
        datamodule=datamodule,
    )

    trainer.train(
        datamodule,
    )

    # ------------------------------------------------------------------
    # Save CMonge checkpoint
    # ------------------------------------------------------------------

    print()
    print("=" * 90)
    print("SAVE FROZEN CMONGE CHECKPOINT")
    print("=" * 90)

    # --------------------------------------------------------------
    # Save the trained CMonge state with Flax MessagePack.
    #
    # Orbax PyTreeCheckpointer is not used here because the current
    # runtime has shown non-deterministic native SIGSEGV failures
    # while writing checkpoint metadata after otherwise successful
    # training. MessagePack serialization has been verified against
    # the completed seed2024 CMonge model with exact reconstruction.
    # --------------------------------------------------------------

    cmonge_ckpt.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        cmonge_ckpt
        / "model.msgpack"
    )

    temp_model_path = (
        cmonge_ckpt
        / "model.msgpack.tmp"
    )

    print(
        "Serializing trained CMonge model to:",
        model_path,
    )

    payload = serialization.to_bytes(
        trainer.model
    )

    with open(temp_model_path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(
        temp_model_path,
        model_path,
    )

    print(
        "MessagePack bytes =",
        len(payload),
    )

    # --------------------------------------------------------------
    # Immediate restoration / equality audit
    # --------------------------------------------------------------

    print()
    print("Verifying saved CMonge model...")

    restored_model = serialization.from_bytes(
        trainer.model,
        model_path.read_bytes(),
    )

    original_host = jax.device_get(
        trainer.model
    )

    restored_host = jax.device_get(
        restored_model
    )

    original_leaves = jax.tree_util.tree_leaves(
        original_host
    )

    restored_leaves = jax.tree_util.tree_leaves(
        restored_host
    )

    if len(original_leaves) != len(restored_leaves):
        raise RuntimeError(
            "Checkpoint verification failed: "
            f"{len(original_leaves)} original leaves versus "
            f"{len(restored_leaves)} restored leaves."
        )

    max_abs_diff = 0.0

    for leaf_index, (original, restored) in enumerate(
        zip(
            original_leaves,
            restored_leaves,
        )
    ):
        original = np.asarray(original)
        restored = np.asarray(restored)

        if original.shape != restored.shape:
            raise RuntimeError(
                "Checkpoint verification failed at leaf "
                f"{leaf_index}: shape "
                f"{original.shape} != {restored.shape}"
            )

        if (
            np.issubdtype(original.dtype, np.number)
            and np.issubdtype(restored.dtype, np.number)
        ):
            difference = (
                float(
                    np.max(
                        np.abs(
                            original - restored
                        )
                    )
                )
                if original.size
                else 0.0
            )

            max_abs_diff = max(
                max_abs_diff,
                difference,
            )

        elif not np.array_equal(
            original,
            restored,
        ):
            raise RuntimeError(
                "Checkpoint verification failed at "
                f"non-numeric leaf {leaf_index}."
            )

    if max_abs_diff != 0.0:
        raise RuntimeError(
            "Checkpoint numerical verification failed: "
            f"max_abs_diff={max_abs_diff}"
        )

    print(
        "Original leaves =",
        len(original_leaves),
    )

    print(
        "Restored leaves =",
        len(restored_leaves),
    )

    print(
        "max_abs_diff =",
        max_abs_diff,
    )

    print(
        "Checkpoint verification PASSED"
    )

    # --------------------------------------------------------------
    # Audit metadata
    # --------------------------------------------------------------

    metadata = {
        "checkpoint_format": "flax_messagepack",
        "seed": seed,
        "model_seed": int(config.model.seed),
        "data_seed": int(config.data.seed),
        "ae_seed": int(config.ae.model.seed),
        "num_train_iters": int(
            config.model.num_train_iters
        ),
        "batch_size": int(
            config.data.batch_size
        ),
        "checkpoint_file": str(
            model_path.resolve()
        ),
        "checkpoint_bytes": int(
            model_path.stat().st_size
        ),
        "pytree_leaves": int(
            len(original_leaves)
        ),
        "verification_max_abs_diff": float(
            max_abs_diff
        ),
    }

    metadata_path = (
        cmonge_ckpt
        / "checkpoint_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n"
    )

    print(
        "Saved Frozen CMonge model:",
        model_path,
    )

    print(
        "Saved checkpoint metadata:",
        metadata_path,
    )

    print()
    print("=" * 90)
    print("FORMAL FROZEN CMONGE TRAINING COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
