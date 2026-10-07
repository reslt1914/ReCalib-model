#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


OFFICIAL = Path("models/embed/demo/rdkit")
REBUILT = Path("models/embed/recalib/rdkit")


official = pd.read_csv(OFFICIAL)
rebuilt = pd.read_csv(REBUILT)

print("=" * 90)
print("OFFICIAL DEMO vs REBUILT FULL RDKIT CHECKPOINT")
print("=" * 90)

print("\n[1] SHAPES")
print("official =", official.shape)
print("rebuilt  =", rebuilt.shape)

official_latent = [
    c for c in official.columns
    if c.startswith("latent_")
]

rebuilt_latent = [
    c for c in rebuilt.columns
    if c.startswith("latent_")
]

print("\nofficial latent dims =", len(official_latent))
print("rebuilt latent dims  =", len(rebuilt_latent))

print(
    "latent column order identical =",
    official_latent == rebuilt_latent,
)

print(
    "latent column sets identical  =",
    set(official_latent) == set(rebuilt_latent),
)

if set(official_latent) != set(rebuilt_latent):
    print("\nOnly in official:")
    print(
        sorted(
            set(official_latent)
            - set(rebuilt_latent)
        )
    )

    print("\nOnly in rebuilt:")
    print(
        sorted(
            set(rebuilt_latent)
            - set(official_latent)
        )
    )

    raise RuntimeError(
        "Latent feature columns differ."
    )

# Always use official column order.
latent = official_latent

print("\n[2] MATCH OFFICIAL DEMO DRUGS BY EXACT SMILES")

# Match by SMILES rather than drug name because the rebuilt
# checkpoint intentionally uses CMonge-safe hashed drug tokens.
merged = official[
    ["drug", "smile"] + latent
].merge(
    rebuilt[
        ["drug", "smile"] + latent
    ],
    on="smile",
    how="inner",
    suffixes=("_official", "_rebuilt"),
    validate="one_to_one",
)

print("official demo drugs =", len(official))
print("matched by SMILES   =", len(merged))

if len(merged) != len(official):
    official_smiles = set(
        official["smile"].astype(str)
    )
    rebuilt_smiles = set(
        rebuilt["smile"].astype(str)
    )

    missing = (
        official_smiles
        - rebuilt_smiles
    )

    print("missing SMILES:")
    for x in sorted(missing):
        print(" ", x)

    raise RuntimeError(
        "Not all official demo drugs were found "
        "in rebuilt checkpoint."
    )

A = merged[
    [
        f"{c}_official"
        for c in latent
    ]
].to_numpy(dtype=float)

B = merged[
    [
        f"{c}_rebuilt"
        for c in latent
    ]
].to_numpy(dtype=float)

diff = B - A
abs_diff = np.abs(diff)

print("\n[3] GLOBAL NUMERICAL COMPARISON")

print(
    "max_abs_diff  =",
    float(abs_diff.max()),
)

print(
    "mean_abs_diff =",
    float(abs_diff.mean()),
)

print(
    "RMSE          =",
    float(
        np.sqrt(
            np.mean(diff ** 2)
        )
    ),
)

print(
    "allclose(atol=1e-8, rtol=1e-8) =",
    np.allclose(
        A,
        B,
        atol=1e-8,
        rtol=1e-8,
    ),
)

print(
    "allclose(atol=1e-6, rtol=1e-6) =",
    np.allclose(
        A,
        B,
        atol=1e-6,
        rtol=1e-6,
    ),
)

flat_corr = np.corrcoef(
    A.ravel(),
    B.ravel(),
)[0, 1]

print(
    "global Pearson =",
    float(flat_corr),
)

print("\n[4] PER-DRUG COMPARISON")

rows = []

for i in range(len(merged)):
    a = A[i]
    b = B[i]

    d = b - a

    if np.std(a) > 0 and np.std(b) > 0:
        corr = np.corrcoef(
            a,
            b,
        )[0, 1]
    else:
        corr = np.nan

    rows.append(
        {
            "official_drug":
                merged.iloc[i]["drug_official"],
            "rebuilt_drug":
                merged.iloc[i]["drug_rebuilt"],
            "max_abs_diff":
                float(np.max(np.abs(d))),
            "mean_abs_diff":
                float(np.mean(np.abs(d))),
            "rmse":
                float(np.sqrt(np.mean(d ** 2))),
            "pearson":
                float(corr),
        }
    )

report = pd.DataFrame(rows)

print(
    report.to_string(
        index=False,
        float_format=lambda x: f"{x:.10g}",
    )
)

print("\n[5] TOP 20 FEATURE DIFFERENCES")

flat_idx = np.argsort(
    abs_diff.ravel()
)[::-1][:20]

for idx in flat_idx:
    row, col = np.unravel_index(
        idx,
        abs_diff.shape,
    )

    print(
        f"{merged.iloc[row]['drug_official']:15s} "
        f"{latent[col]:12s} "
        f"official={A[row, col]: .10f} "
        f"rebuilt={B[row, col]: .10f} "
        f"abs_diff={abs_diff[row, col]:.10g}"
    )

report.to_csv(
    "data/recalib_sciplex_main_metadata/"
    "rdkit_build/"
    "official_demo_comparison.csv",
    index=False,
)

print("\n" + "=" * 90)
print("RDKIT CHECKPOINT COMPARISON COMPLETE")
print("=" * 90)
