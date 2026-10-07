from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


PROJECT = Path.cwd()

CONFIG = PROJECT / "configs" / "innovation_experiments.yaml"

RUNNER = PROJECT / "run_ridge_anchored_recalib.py"

BASELINE_OOD = (
    PROJECT
    / "crisp_outputs"
    / "mydata_baseline_full"
    / "ood"
)

TRUE_FILE = (
    BASELINE_OOD
    / "x_true_condition_mean.csv"
)

RESIDUAL_FILE = (
    BASELINE_OOD
    / "residual_condition_mean.csv"
)

OUTPUT_ROOT = (
    PROJECT
    / "results"
    / "final_poisoning_audit"
)

SEED = 2024


def check_file(path):
    if not path.is_file():
        raise RuntimeError(
            f"Missing required file: {path}"
        )


def poison_matrix(path, random_seed):
    df = pd.read_csv(
        path,
        index_col=0,
    )

    rng = np.random.default_rng(
        random_seed
    )

    poisoned = pd.DataFrame(
        rng.normal(
            loc=0.0,
            scale=100.0,
            size=df.shape,
        ),
        index=df.index,
        columns=df.columns,
    )

    poisoned.to_csv(path)

    print(
        f"[POISONED] {path}"
    )
    print(
        " shape:",
        poisoned.shape,
    )
    print(
        " mean :",
        float(
            poisoned.values.mean()
        ),
    )
    print(
        " std  :",
        float(
            poisoned.values.std()
        ),
    )


def run_recalib(tag):
    out = OUTPUT_ROOT / tag

    if out.exists():
        shutil.rmtree(out)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 90)
    print(
        f"RUNNING FINAL RECALIB: {tag}"
    )
    print("=" * 90)

    env = os.environ.copy()

    env["OMP_NUM_THREADS"] = "6"
    env["MKL_NUM_THREADS"] = "6"
    env["OPENBLAS_NUM_THREADS"] = "6"
    env["NUMEXPR_NUM_THREADS"] = "6"

    cmd = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--config",
        str(CONFIG),
        "--seeds",
        str(SEED),
        "--selection-seed",
        str(SEED),
        "--output-dir",
        str(out),
    ]

    print(
        "Command:",
        " ".join(cmd),
    )

    subprocess.run(
        cmd,
        cwd=PROJECT,
        env=env,
        check=True,
    )

    alpha_json = (
        out
        / "selected_alpha.json"
    )

    if alpha_json.is_file():
        with open(
            alpha_json,
            "r",
            encoding="utf-8",
        ) as f:
            info = json.load(f)

        alpha = float(
            info["selected_alpha"]
        )

        print(
            f"[CHECK] selected alpha = {alpha}"
        )

        if abs(alpha - 0.50) > 1e-12:
            raise RuntimeError(
                "Final audit did not select "
                "alpha=0.50."
            )

    candidates = list(
        out.rglob(
            "x_calibrated_condition_mean.csv"
        )
    )

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one final "
            f"prediction matrix in {out}, "
            f"found {len(candidates)}:\n"
            + "\n".join(
                str(x)
                for x in candidates
            )
        )

    prediction = candidates[0]

    print(
        "[FINAL PREDICTION]",
        prediction,
    )

    return prediction


def compare(
    clean_path,
    poisoned_path,
    label,
):
    clean = pd.read_csv(
        clean_path,
        index_col=0,
    )

    poison = pd.read_csv(
        poisoned_path,
        index_col=0,
    )

    if clean.shape != poison.shape:
        raise RuntimeError(
            f"{label}: shape mismatch: "
            f"{clean.shape} vs "
            f"{poison.shape}"
        )

    if list(clean.index) != list(
        poison.index
    ):
        raise RuntimeError(
            f"{label}: row-order mismatch"
        )

    if list(clean.columns) != list(
        poison.columns
    ):
        raise RuntimeError(
            f"{label}: gene-order mismatch"
        )

    diff = (
        clean.values.astype(float)
        -
        poison.values.astype(float)
    )

    result = {
        "audit": label,
        "mean_abs_diff": float(
            np.mean(
                np.abs(diff)
            )
        ),
        "max_abs_diff": float(
            np.max(
                np.abs(diff)
            )
        ),
        "rmse_diff": float(
            np.sqrt(
                np.mean(
                    diff ** 2
                )
            )
        ),
        "n_conditions": int(
            clean.shape[0]
        ),
        "n_genes": int(
            clean.shape[1]
        ),
    }

    return result


def main():

    check_file(CONFIG)
    check_file(RUNNER)
    check_file(TRUE_FILE)
    check_file(RESIDUAL_FILE)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print(
        "FINAL RIDGE-ANCHORED RECALIB "
        "POISONING AUDIT"
    )
    print("=" * 90)

    print(
        "[SOURCE] true:",
        TRUE_FILE,
    )

    print(
        "[SOURCE] residual:",
        RESIDUAL_FILE,
    )

    with tempfile.TemporaryDirectory(
        prefix="recalib_poison_backup_"
    ) as tmp_name:

        tmp = Path(tmp_name)

        true_backup = (
            tmp / TRUE_FILE.name
        )

        residual_backup = (
            tmp / RESIDUAL_FILE.name
        )

        shutil.copy2(
            TRUE_FILE,
            true_backup,
        )

        shutil.copy2(
            RESIDUAL_FILE,
            residual_backup,
        )

        try:
            # ==================================================
            # CLEAN
            # ==================================================
            clean_prediction = (
                run_recalib(
                    "clean"
                )
            )

            # ==================================================
            # TRUE-EXPRESSION POISON
            # ==================================================
            poison_matrix(
                TRUE_FILE,
                random_seed=998,
            )

            true_poison_prediction = (
                run_recalib(
                    "poison_true"
                )
            )

            # Restore true before next audit
            shutil.copy2(
                true_backup,
                TRUE_FILE,
            )

            print(
                "[RESTORED]",
                TRUE_FILE,
            )

            # ==================================================
            # RESIDUAL POISON
            # ==================================================
            poison_matrix(
                RESIDUAL_FILE,
                random_seed=999,
            )

            residual_poison_prediction = (
                run_recalib(
                    "poison_residual"
                )
            )

        finally:
            # ALWAYS restore original source data.
            shutil.copy2(
                true_backup,
                TRUE_FILE,
            )

            shutil.copy2(
                residual_backup,
                RESIDUAL_FILE,
            )

            print()
            print(
                "[RESTORED ORIGINAL INPUTS]"
            )
            print(
                TRUE_FILE
            )
            print(
                RESIDUAL_FILE
            )

        rows = []

        rows.append(
            compare(
                clean_prediction,
                true_poison_prediction,
                "ood_true_expression_poison",
            )
        )

        rows.append(
            compare(
                clean_prediction,
                residual_poison_prediction,
                "ood_residual_poison",
            )
        )

        result = pd.DataFrame(rows)

        result_path = (
            OUTPUT_ROOT
            / "poisoning_audit_summary.csv"
        )

        result.to_csv(
            result_path,
            index=False,
        )

        print()
        print("=" * 90)
        print(
            "FINAL POISONING AUDIT RESULTS"
        )
        print("=" * 90)

        print(
            result.to_string(
                index=False
            )
        )

        print()
        print(
            "[OUTPUT]",
            result_path,
        )

        threshold = 1e-8

        if (
            result[
                "max_abs_diff"
            ].max()
            <= threshold
        ):
            print()
            print(
                "[PASS] Final predictions "
                "were unchanged within "
                f"tolerance {threshold:g}."
            )
        else:
            print()
            print(
                "[CHECK] Prediction changes "
                "were detected. Inspect "
                "the audit before using "
                "the manuscript statement."
            )


if __name__ == "__main__":
    main()
