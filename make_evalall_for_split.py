import argparse
from pathlib import Path
import scanpy as sc

parser = argparse.ArgumentParser()
parser.add_argument("--infile", default="data/mydata/sciplex_complete_v2.h5ad")
parser.add_argument("--outfile", required=True)
parser.add_argument("--source_split", required=True)
parser.add_argument("--new_split", required=True)
args = parser.parse_args()

adata = sc.read_h5ad(args.infile)

src = adata.obs[args.source_split].astype(str).copy()
new = src.copy()

# CRISP 默认只预测 test，所以把原 ood 临时并入 test
new[src == "ood"] = "test"

adata.obs[args.new_split] = new

print("original split:")
print(src.value_counts())

print("\nnew split:")
print(adata.obs[args.new_split].value_counts())

# 检查 test 和 ood condition 是否重叠
cond_key = "cov_drug_dose_name"
test_conds = set(adata.obs.loc[src == "test", cond_key].dropna().astype(str))
ood_conds = set(adata.obs.loc[src == "ood", cond_key].dropna().astype(str))
print("\ntest conditions:", len(test_conds))
print("ood conditions:", len(ood_conds))
print("test/ood overlap:", len(test_conds & ood_conds))
print("overlap examples:", list(test_conds & ood_conds)[:10])

adata.write_h5ad(args.outfile)
print("\nsaved:", args.outfile)
