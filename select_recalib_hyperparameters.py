from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from innovation_data import (
    PROJECT_ROOT,
    load_experiment_data,
)
from run_innovation_baselines import (
    PRIMARY_METRICS,
    build_official_de_masks,
    evaluate_by_condition,
    summarize_metrics,
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "innovation_experiments.yaml"
)

RUN_ROOT = (
    PROJECT_ROOT
    / "results"
    / "recalib_hyperparameter_selection"
)

DE_WEIGHTS = [0, 10, 30, 50]

ALPHAS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
]


def read_prediction(
    path: Path,
    *,
    conditions: list[str],
    genes: list[str],
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(
        path,
        index_col=0,
    )

    frame.index = (
        frame.index.astype(str)
    )
    frame.columns = (
        frame.columns.astype(str)
    )

    missing_conditions = (
        set(conditions)
        - set(frame.index)
    )
    extra_conditions = (
        set(frame.index)
        - set(conditions)
    )

    missing_genes = (
        set(genes)
        - set(frame.columns)
    )
    extra_genes = (
        set(frame.columns)
        - set(genes)
    )

    if (
        missing_conditions
        or extra_conditions
    ):
        raise ValueError(
            f"Condition mismatch in {path}\n"
            f"Missing={sorted(missing_conditions)[:5]}\n"
            f"Extra={sorted(extra_conditions)[:5]}"
        )

    if missing_genes or extra_genes:
        raise ValueError(
            f"Gene mismatch in {path}\n"
            f"Missing={sorted(missing_genes)[:5]}\n"
            f"Extra={sorted(extra_genes)[:5]}"
        )

    frame = frame.loc[
        conditions,
        genes,
    ]

    values = frame.to_numpy(
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite prediction values: {path}"
        )

    return values


def get_metric(
    summary: pd.DataFrame,
    metric: str,
) -> float:
    values = summary.loc[
        summary["metric"] == metric,
        "mean",
    ]

    if len(values) != 1:
        raise ValueError(
            f"Expected exactly one value for {metric}"
        )

    return float(values.iloc[0])


def check_run_config(
    run_dir: Path,
    expected_de_weight: int,
) -> None:
    path = run_dir / "run_config.json"

    if not path.is_file():
        raise FileNotFoundError(path)

    config = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    actual_weight = float(
        config["de_weight"]
    )
    actual_alpha = float(
        config["alpha"]
    )

    baseline_dir = str(
        config["baseline_dir"]
    )

    if abs(
        actual_weight
        - expected_de_weight
    ) > 1e-12:
        raise ValueError(
            f"{run_dir}: expected "
            f"DE weight={expected_de_weight}, "
            f"found {actual_weight}"
        )

    if abs(
        actual_alpha - 1.0
    ) > 1e-12:
        raise ValueError(
            f"{run_dir}: training alpha "
            f"must equal 1.0; "
            f"found {actual_alpha}"
        )

    if (
        "mydata_baseline_calib_seed2024"
        not in baseline_dir
    ):
        raise ValueError(
            f"Unexpected baseline_dir: "
            f"{baseline_dir}"
        )

    print(
        f"[OK] DE weight={expected_de_weight}: "
        "strict 1003/251 training run verified"
    )


def main() -> None:
    print("=" * 88)
    print(
        "Strict ReCalib IID-validation "
        "hyperparameter selection"
    )
    print("=" * 88)

    data = load_experiment_data(
        CONFIG_PATH
    )

    validation = data.validation

    (
        validation_de_mask,
        _,
        _,
        _,
    ) = build_official_de_masks(
        data
    )

    print(
        "[INFO] Validation matrix: "
        f"{validation.n_conditions} "
        "conditions × "
        f"{validation.n_genes} genes"
    )

    frozen = (
        validation.frozen_crisp
        .astype(
            np.float64,
            copy=False,
        )
    )

    sweep_rows = []
    residual_cache = {}

    for de_weight in DE_WEIGHTS:
        run_dir = (
            RUN_ROOT
            / f"de_weight_{de_weight}"
        )

        check_run_config(
            run_dir,
            de_weight,
        )

        alpha1_path = (
            run_dir
            / "ood"
            / "x_calibrated_condition_mean.csv"
        )

        alpha1_prediction = (
            read_prediction(
                alpha1_path,
                conditions=(
                    validation.conditions
                ),
                genes=validation.genes,
            )
        )

        # Training run used alpha=1.0:
        # x_cal = Frozen CRISP + predicted residual
        predicted_residual = (
            alpha1_prediction
            - frozen
        )

        residual_cache[
            de_weight
        ] = predicted_residual

        print(
            f"[OK] DE weight={de_weight}: "
            "predicted validation residual recovered"
        )

        for alpha in ALPHAS:
            prediction = (
                frozen
                + alpha
                * predicted_residual
            )

            condition_metrics = (
                evaluate_by_condition(
                    method=(
                        "strict_recalib_"
                        f"w{de_weight}_"
                        f"a{alpha}"
                    ),
                    split=validation,
                    prediction=prediction,
                    de_mask=(
                        validation_de_mask
                    ),
                )
            )

            summary = (
                summarize_metrics(
                    condition_metrics
                )
            )

            row = {
                "de_weight": (
                    de_weight
                ),
                "alpha": alpha,
            }

            for metric in PRIMARY_METRICS:
                row[metric] = (
                    get_metric(
                        summary,
                        metric,
                    )
                )

            sweep_rows.append(row)

            print(
                f"[VAL] "
                f"w={de_weight:2d} "
                f"alpha={alpha:.2f} "
                f"mse={row['mse_de']:.6f} "
                f"pearson={row['pearson_de']:.6f} "
                f"delta={row['pearson_delta_de']:.6f} "
                f"r2={row['r2score_de']:.6f}"
            )

    sweep = pd.DataFrame(
        sweep_rows
    )

    sweep.to_csv(
        RUN_ROOT
        / "validation_joint_sweep.csv",
        index=False,
    )

    # Primary selection rule:
    # 1. maximum IID-validation r2score_de
    # 2. lower mse_de
    # 3. smaller alpha
    # 4. smaller DE weight
    #
    # OOD data are never used here.
    ranked = sweep.sort_values(
        [
            "r2score_de",
            "mse_de",
            "alpha",
            "de_weight",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    best = ranked.iloc[0]

    selected_weight = int(
        best["de_weight"]
    )
    selected_alpha = float(
        best["alpha"]
    )

    selection = {
        "selection_dataset": (
            "IID validation only"
        ),
        "validation_conditions": (
            validation.n_conditions
        ),
        "main_ood_used_for_selection": (
            False
        ),
        "primary_selection_metric": (
            "r2score_de"
        ),
        "tie_breaking": [
            "lower mse_de",
            "smaller alpha",
            "smaller DE weight",
        ],
        "selected_de_weight": (
            selected_weight
        ),
        "selected_alpha": (
            selected_alpha
        ),
        "selected_validation_metrics": {
            metric: float(
                best[metric]
            )
            for metric
            in PRIMARY_METRICS
        },
        "candidate_de_weights": (
            DE_WEIGHTS
        ),
        "candidate_alphas": (
            ALPHAS
        ),
    }

    (
        RUN_ROOT
        / "selected_hyperparameters.json"
    ).write_text(
        json.dumps(
            selection,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Supplementary Table 4:
    # alpha sweep with final DE weight fixed.
    table4 = (
        sweep.loc[
            sweep["de_weight"]
            == selected_weight
        ]
        .sort_values("alpha")
        .reset_index(drop=True)
    )

    table4.to_csv(
        RUN_ROOT
        / "supplementary_table4_alpha_sweep.csv",
        index=False,
    )

    # Supplementary Table 5:
    # DE-weight sweep with final alpha fixed.
    table5 = (
        sweep.loc[
            np.isclose(
                sweep["alpha"],
                selected_alpha,
            )
        ]
        .sort_values(
            "de_weight"
        )
        .reset_index(drop=True)
    )

    table5.to_csv(
        RUN_ROOT
        / "supplementary_table5_de_weight_sweep.csv",
        index=False,
    )

    ranked.to_csv(
        RUN_ROOT
        / "validation_joint_sweep_ranked.csv",
        index=False,
    )

    print("\n" + "=" * 88)
    print("SELECTED STRICT ReCalib HYPERPARAMETERS")
    print("=" * 88)

    print(
        f"DE weight = "
        f"{selected_weight}"
    )
    print(
        f"alpha     = "
        f"{selected_alpha:.2f}"
    )

    print("\nValidation metrics:")
    for metric in PRIMARY_METRICS:
        print(
            f"  {metric:20s} "
            f"{float(best[metric]):.6f}"
        )

    print(
        "\n[IMPORTANT] "
        "Main OOD was not loaded for "
        "hyperparameter selection."
    )

    print(
        "\n[WRITE] "
        "validation_joint_sweep.csv"
    )
    print(
        "[WRITE] "
        "selected_hyperparameters.json"
    )
    print(
        "[WRITE] "
        "supplementary_table4_alpha_sweep.csv"
    )
    print(
        "[WRITE] "
        "supplementary_table5_de_weight_sweep.csv"
    )


if __name__ == "__main__":
    main()
