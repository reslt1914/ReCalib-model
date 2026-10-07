import pandas as pd
import numpy as np
from pathlib import Path

p = Path("crisp_outputs/evidence_mydata/case_studies/all_ood_case_ranking.csv")
df = pd.read_csv(p)

# 有些列可能名字不同，先确认
print("columns:")
print(list(df.columns))

needed = [
    "condition",
    "cell_type_parsed",
    "drug_parsed",
    "dose_parsed",
    "base_delta_de_r2",
    "rc_delta_de_r2",
    "delta_de_r2_improvement",
    "base_delta_de_pearson",
    "rc_delta_de_pearson",
    "delta_de_pearson_improvement",
    "base_de_mse",
    "rc_de_mse",
    "de_mse_improvement",
]
needed = [c for c in needed if c in df.columns]

# 计算几个更适合展示的指标
df["mse_reduction_frac"] = (df["base_de_mse"] - df["rc_de_mse"]) / (df["base_de_mse"].abs() + 1e-12)
df["rc_delta_de_r2_clipped"] = df["rc_delta_de_r2"].clip(lower=-2, upper=2)

# 候选：不是只看巨大负 R2 的改善，而是看 RC 后是否像样
cand = df.copy()

# 基础过滤：MSE 要下降，Pearson 要提升
cand = cand[
    (cand["de_mse_improvement"] > 0) &
    (cand["delta_de_pearson_improvement"] > 0)
].copy()

# 更适合画图的严格过滤
strict = cand[
    (cand["rc_delta_de_pearson"] >= 0.45) &
    (cand["mse_reduction_frac"] >= 0.15) &
    (cand["rc_delta_de_r2"] > -3)
].copy()

# 如果严格过滤太少，就放宽
if len(strict) < 10:
    strict = cand[
        (cand["rc_delta_de_pearson"] >= 0.35) &
        (cand["mse_reduction_frac"] >= 0.10) &
        (cand["rc_delta_de_r2"] > -5)
    ].copy()

# 综合分数：优先展示“预测质量好 + 有提升 + MSE降得多”
strict["case_score"] = (
    2.0 * strict["rc_delta_de_pearson"].fillna(0) +
    1.0 * strict["delta_de_pearson_improvement"].fillna(0) +
    1.0 * strict["mse_reduction_frac"].fillna(0) +
    0.5 * strict["rc_delta_de_r2_clipped"].fillna(-2)
)

strict = strict.sort_values("case_score", ascending=False)

# 选不同 drug 的 top cases
selected = []
used_drugs = set()

for _, row in strict.iterrows():
    drug = str(row["drug_parsed"])
    if drug in used_drugs:
        continue
    selected.append(row)
    used_drugs.add(drug)
    if len(selected) >= 10:
        break

selected = pd.DataFrame(selected)

out_dir = Path("crisp_outputs/evidence_mydata/case_studies")
strict.to_csv(out_dir / "better_case_candidates_ranked.csv", index=False)
selected.to_csv(out_dir / "better_case_candidates_top10_distinct_drugs.csv", index=False)

print("\nTop better candidates:")
show_cols = [
    "condition",
    "cell_type_parsed",
    "drug_parsed",
    "dose_parsed",
    "base_delta_de_r2",
    "rc_delta_de_r2",
    "delta_de_r2_improvement",
    "base_delta_de_pearson",
    "rc_delta_de_pearson",
    "delta_de_pearson_improvement",
    "base_de_mse",
    "rc_de_mse",
    "mse_reduction_frac",
    "case_score",
]
show_cols = [c for c in show_cols if c in selected.columns]

print(selected[show_cols].head(10).to_string(index=False))

print("\nsaved:")
print(out_dir / "better_case_candidates_ranked.csv")
print(out_dir / "better_case_candidates_top10_distinct_drugs.csv")


