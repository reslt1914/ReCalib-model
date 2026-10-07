from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path(
    "results/paper_recalculation/"
    "main_ood_condition_metrics.csv"
)

OUT_DIR = Path("results/paper_recalculation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

print("=" * 90)
print("FINAL REPRESENTATIVE-CASE SELECTION")
print("=" * 90)
print("Input:", INPUT)
print("Rows :", len(df))
print("Columns:")
print(df.columns.tolist())
print()

required = {
    "condition",
    "model",
    "mse_de",
    "pearson_delta_de",
}

missing = required - set(df.columns)
if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}"
    )

# ------------------------------------------------------------
# 1. Frozen CRISP
# ------------------------------------------------------------
base = (
    df[df["model"].astype(str) == "frozen_crisp"]
    .copy()
)

if len(base) == 0:
    raise RuntimeError("No frozen_crisp rows found.")

if base["condition"].duplicated().any():
    raise RuntimeError(
        "Frozen CRISP contains duplicated conditions."
    )

# Metadata columns if available
metadata_cols = [
    c
    for c in [
        "condition",
        "cell_type",
        "drug",
        "dose",
        "n_de",
        "n_de_official",
    ]
    if c in base.columns
]

base_keep = list(
    dict.fromkeys(
        metadata_cols
        + [
            "condition",
            "mse_de",
            "pearson_delta_de",
        ]
    )
)

base = base[base_keep].rename(
    columns={
        "mse_de": "frozen_mse_de",
        "pearson_delta_de":
            "frozen_pearson_delta_de",
    }
)

# ------------------------------------------------------------
# 2. Three final ReCalib seeds
# ------------------------------------------------------------
expected_models = [
    "recalib_seed_2024",
    "recalib_seed_3407",
    "recalib_seed_42",
]

rec = df[
    df["model"].astype(str).isin(expected_models)
].copy()

models_found = sorted(
    rec["model"].astype(str).unique()
)

print("ReCalib models found:")
for x in models_found:
    print(" ", x)

missing_models = (
    set(expected_models) - set(models_found)
)

if missing_models:
    raise RuntimeError(
        "Missing ReCalib seed models: "
        f"{sorted(missing_models)}"
    )

# ------------------------------------------------------------
# 3. Three-seed mean and sample SD per condition
# ------------------------------------------------------------
agg = (
    rec.groupby("condition", as_index=False)
    .agg(
        recalib_mse_de=(
            "mse_de",
            "mean",
        ),
        recalib_mse_de_sd=(
            "mse_de",
            "std",
        ),
        recalib_pearson_delta_de=(
            "pearson_delta_de",
            "mean",
        ),
        recalib_pearson_delta_de_sd=(
            "pearson_delta_de",
            "std",
        ),
        n_seeds=(
            "model",
            "nunique",
        ),
    )
)

merged = base.merge(
    agg,
    on="condition",
    how="inner",
    validate="one_to_one",
)

# ------------------------------------------------------------
# 4. Require mean improvement in BOTH metrics
# ------------------------------------------------------------
merged["mse_improvement"] = (
    merged["frozen_mse_de"]
    - merged["recalib_mse_de"]
)

merged["pearson_delta_improvement"] = (
    merged["recalib_pearson_delta_de"]
    - merged["frozen_pearson_delta_de"]
)

merged["mean_both_improved"] = (
    (merged["mse_improvement"] > 0)
    &
    (merged["pearson_delta_improvement"] > 0)
)

# ------------------------------------------------------------
# 5. Check whether ALL THREE seeds improve BOTH metrics
# ------------------------------------------------------------
rec_check = rec[
    [
        "condition",
        "model",
        "mse_de",
        "pearson_delta_de",
    ]
].merge(
    base[
        [
            "condition",
            "frozen_mse_de",
            "frozen_pearson_delta_de",
        ]
    ],
    on="condition",
    how="left",
)

rec_check["both_improved"] = (
    (
        rec_check["mse_de"]
        <
        rec_check["frozen_mse_de"]
    )
    &
    (
        rec_check["pearson_delta_de"]
        >
        rec_check[
            "frozen_pearson_delta_de"
        ]
    )
)

robust = (
    rec_check.groupby("condition")
    ["both_improved"]
    .agg(
        n_seed_both_improved="sum",
        all_three_both_improved="all",
    )
    .reset_index()
)

merged = merged.merge(
    robust,
    on="condition",
    how="left",
)

# ------------------------------------------------------------
# 6. Recover cell/drug/dose from condition only if needed
# ------------------------------------------------------------
if "cell_type" not in merged.columns:
    merged["cell_type"] = (
        merged["condition"]
        .astype(str)
        .str.split("_")
        .str[0]
    )

# drug/dose normally already exist in the recalculation file.
# If not, leave condition as the canonical identifier.
if "drug" not in merged.columns:
    merged["drug"] = ""

if "dose" not in merged.columns:
    merged["dose"] = ""

# ------------------------------------------------------------
# 7. Save every eligible candidate
# ------------------------------------------------------------
candidates = merged[
    merged["mean_both_improved"]
].copy()

candidates = candidates.sort_values(
    [
        "cell_type",
        "all_three_both_improved",
        "mse_improvement",
        "pearson_delta_improvement",
    ],
    ascending=[
        True,
        False,
        False,
        False,
    ],
)

candidate_path = (
    OUT_DIR
    / "representative_case_candidates_final.csv"
)

candidates.to_csv(
    candidate_path,
    index=False,
)

print()
print("=" * 90)
print("CANDIDATE COUNTS")
print("=" * 90)

print(
    "Mean improves BOTH metrics:",
    int(merged["mean_both_improved"].sum()),
    "/",
    len(merged),
)

print(
    "All three seeds improve BOTH:",
    int(
        merged[
            "all_three_both_improved"
        ].sum()
    ),
    "/",
    len(merged),
)

print()
print("By cell type:")
print(
    merged.groupby("cell_type")
    .agg(
        total=("condition", "size"),
        mean_both=(
            "mean_both_improved",
            "sum",
        ),
        all3_both=(
            "all_three_both_improved",
            "sum",
        ),
    )
    .to_string()
)

# ------------------------------------------------------------
# 8. Select ONE representative case per cell line.
#
# Priority:
#   all three seeds improve both metrics.
#
# Among these robust cases, choose a central rather than
# extreme example: the point closest to the median percentile
# of BOTH improvement measures within the cell line.
# ------------------------------------------------------------
selected_rows = []

for cell in sorted(
    merged["cell_type"]
    .dropna()
    .astype(str)
    .unique()
):

    g = merged[
        (merged["cell_type"].astype(str) == cell)
        &
        (merged["all_three_both_improved"])
    ].copy()

    selection_level = (
        "all_three_seeds_improve_both"
    )

    if len(g) == 0:
        g = merged[
            (merged["cell_type"].astype(str) == cell)
            &
            (merged["mean_both_improved"])
        ].copy()

        selection_level = (
            "three_seed_mean_improves_both"
        )

    if len(g) == 0:
        print(
            f"[WARN] {cell}: no condition improves "
            "both metrics."
        )
        continue

    g["mse_rank"] = (
        g["mse_improvement"]
        .rank(
            pct=True,
            method="average",
        )
    )

    g["delta_rank"] = (
        g["pearson_delta_improvement"]
        .rank(
            pct=True,
            method="average",
        )
    )

    # Distance from the middle of the improvement distribution.
    # Smaller = more representative / less extreme.
    g["representative_distance"] = (
        (g["mse_rank"] - 0.5) ** 2
        +
        (g["delta_rank"] - 0.5) ** 2
    )

    chosen = (
        g.sort_values(
            [
                "representative_distance",
                "condition",
            ]
        )
        .iloc[0]
        .copy()
    )

    chosen["selection_level"] = (
        selection_level
    )

    selected_rows.append(chosen)

selected = pd.DataFrame(
    selected_rows
)

selected_path = (
    OUT_DIR
    / "representative_cases_final.csv"
)

selected.to_csv(
    selected_path,
    index=False,
)

# ------------------------------------------------------------
# 9. Print exact values for manuscript writing
# ------------------------------------------------------------
print()
print("=" * 90)
print("SELECTED REPRESENTATIVE CASES")
print("=" * 90)

show_cols = [
    c
    for c in [
        "condition",
        "cell_type",
        "drug",
        "dose",
        "selection_level",
        "frozen_mse_de",
        "recalib_mse_de",
        "recalib_mse_de_sd",
        "mse_improvement",
        "frozen_pearson_delta_de",
        "recalib_pearson_delta_de",
        "recalib_pearson_delta_de_sd",
        "pearson_delta_improvement",
        "n_seed_both_improved",
    ]
    if c in selected.columns
]

if len(selected):
    print(
        selected[
            show_cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

print()
print("=" * 90)
print("MANUSCRIPT-READY SENTENCES")
print("=" * 90)

for _, r in selected.iterrows():

    condition = str(r["condition"])

    cell = str(
        r.get(
            "cell_type",
            "",
        )
    )

    drug = str(
        r.get(
            "drug",
            "",
        )
    )

    dose = str(
        r.get(
            "dose",
            "",
        )
    )

    if (
        drug
        and drug != "nan"
        and dose
        and dose != "nan"
    ):
        intro = (
            f"For {cell} cells treated with "
            f"{drug} at dose {dose},"
        )
    else:
        intro = (
            f"For condition {condition},"
        )

    sentence = (
        f"{intro} final Ridge-anchored ReCalib "
        f"reduced mse_de from "
        f"{r['frozen_mse_de']:.6f} to "
        f"{r['recalib_mse_de']:.6f} "
        f"± {r['recalib_mse_de_sd']:.6f} "
        f"and increased pearson_delta_de from "
        f"{r['frozen_pearson_delta_de']:.6f} to "
        f"{r['recalib_pearson_delta_de']:.6f} "
        f"± {r['recalib_pearson_delta_de_sd']:.6f}."
    )

    print(sentence)

print()
print("[OK] Candidate table:")
print(candidate_path)

print("[OK] Selected cases:")
print(selected_path)
