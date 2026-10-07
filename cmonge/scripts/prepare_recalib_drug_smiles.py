#!/usr/bin/env python3

from pathlib import Path
import re
import pandas as pd

MAPPING = Path(
    "data/recalib_sciplex_main_metadata/drug_mapping.csv"
)

OFFICIAL_SMILES = Path(
    "data/trapnell_drugs_smiles.csv"
)

OUT = Path(
    "data/recalib_sciplex_main_metadata/"
    "cmonge_drug_smiles.csv"
)

AUDIT = Path(
    "data/recalib_sciplex_main_metadata/"
    "cmonge_drug_smiles_audit.csv"
)

ALIASES = {
    "(+)-JQ1": "JQ1",
}


def norm(x: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(x).strip().lower(),
    )


mapping = pd.read_csv(MAPPING)
official = pd.read_csv(OFFICIAL_SMILES)

required_mapping_cols = {
    "original_drug",
    "cmonge_drug",
}

if not required_mapping_cols.issubset(mapping.columns):
    raise ValueError(
        f"Missing columns in {MAPPING}: "
        f"{required_mapping_cols - set(mapping.columns)}"
    )

if not {"drug", "smile"}.issubset(official.columns):
    raise ValueError(
        "Official SMILES table must contain drug and smile"
    )

# Explicit alias only. No silent fuzzy matching.
mapping["lookup_drug"] = (
    mapping["original_drug"]
    .map(ALIASES)
    .fillna(mapping["original_drug"])
)

official = official.copy()

# Control is not a drug condition embedding target.
# CMonge represents control separately as "control-0",
# so exclude control from the drug-to-SMILES lookup table.
official = official.loc[
    official["drug"].astype(str).str.strip().str.lower() != "control"
].copy()

official["_norm"] = official["drug"].map(norm)

# Ensure normalized non-control drug names are unique.
dup = official[
    official["_norm"].duplicated(False)
]

if len(dup):
    raise ValueError(
        "Normalized duplicate drug names in official table:\n"
        + dup[["drug", "smile"]].to_string(index=False)
    )

official_by_norm = (
    official
    .set_index("_norm")
)

rows = []

for _, row in mapping.iterrows():
    original_drug = str(row["original_drug"])
    cmonge_drug = str(row["cmonge_drug"])
    lookup_drug = str(row["lookup_drug"])

    key = norm(lookup_drug)

    if key not in official_by_norm.index:
        rows.append({
            "original_drug": original_drug,
            "lookup_drug": lookup_drug,
            "cmonge_drug": cmonge_drug,
            "official_drug": None,
            "smile": None,
            "matched": False,
        })
        continue

    hit = official_by_norm.loc[key]

    rows.append({
        "original_drug": original_drug,
        "lookup_drug": lookup_drug,
        "cmonge_drug": cmonge_drug,
        "official_drug": str(hit["drug"]),
        "smile": str(hit["smile"]),
        "matched": True,
    })

audit = pd.DataFrame(rows)

audit.to_csv(AUDIT, index=False)

missing = audit.loc[
    ~audit["matched"]
]

print("=" * 90)
print("RECALIB → CMONGE SMILES MAPPING")
print("=" * 90)

print("total drugs =", len(audit))
print("matched     =", int(audit["matched"].sum()))
print("missing     =", len(missing))

if len(missing):
    print("\nMissing:")
    print(
        missing[
            ["original_drug", "lookup_drug"]
        ].to_string(index=False)
    )
    raise RuntimeError(
        "SMILES mapping is incomplete"
    )

# Critical integrity checks
assert len(audit) == 187
assert audit["cmonge_drug"].nunique() == 187
assert audit["original_drug"].nunique() == 187
assert audit["smile"].notna().all()

# CMonge RDKitEmbedding expects:
# drug, smile
#
# IMPORTANT:
# use cmonge_drug here, because the actual
# conditions in our adapted h5ad use hashed tokens.
out = (
    audit[
        [
            "cmonge_drug",
            "smile",
            "original_drug",
            "official_drug",
        ]
    ]
    .rename(
        columns={
            "cmonge_drug": "drug",
        }
    )
)

out.to_csv(
    OUT,
    index=False,
)

print("\nAlias mappings used:")
for src, dst in ALIASES.items():
    print(f"  {src} -> {dst}")

print("\nOutput:")
print(OUT)

print("\nFirst 10:")
print(out.head(10).to_string(index=False))

print("\n" + "=" * 90)
print("187 / 187 SMILES MAPPING PASSED")
print("=" * 90)
