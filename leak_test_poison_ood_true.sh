set -e

BASE="crisp_outputs/mydata_baseline_full"
TMP="crisp_outputs/_leaktest_poison_true_baseline_cpu"
CLEAN_OUT="crisp_outputs/_leaktest_true_clean_seed998_cpu"
POISON_OUT="crisp_outputs/_leaktest_true_poison_seed998_cpu"

rm -rf "$TMP" "$CLEAN_OUT" "$POISON_OUT"
cp -r "$BASE" "$TMP"

python - <<'PY'
import pandas as pd
import numpy as np
from pathlib import Path

p = Path("crisp_outputs/_leaktest_poison_true_baseline_cpu/ood/x_true_condition_mean.csv")
df = pd.read_csv(p, index_col=0, engine="python")

rng = np.random.default_rng(998)
noise = rng.normal(loc=0, scale=100.0, size=df.shape)

df2 = pd.DataFrame(noise, index=df.index, columns=df.columns)
df2.to_csv(p)

print("poisoned:", p)
print("shape:", df2.shape)
print("mean/std:", float(df2.values.mean()), float(df2.values.std()))
PY

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""

echo "Running clean short training on CPU..."
python train_crisp_rc_deaware.py \
  --baseline_dir "$BASE" \
  --adata data/mydata/sciplex_complete_v2.h5ad \
  --de_key lincs_DEGs \
  --out_dir "$CLEAN_OUT" \
  --alpha 0.4 \
  --de_weight 0 \
  --seed 998 \
  --epochs 30 \
  --batch_size 65536 \
  --device cpu

echo "Running poisoned OOD true short training on CPU..."
python train_crisp_rc_deaware.py \
  --baseline_dir "$TMP" \
  --adata data/mydata/sciplex_complete_v2.h5ad \
  --de_key lincs_DEGs \
  --out_dir "$POISON_OUT" \
  --alpha 0.4 \
  --de_weight 0 \
  --seed 998 \
  --epochs 30 \
  --batch_size 65536 \
  --device cpu

python - <<'PY'
import pandas as pd
import numpy as np

clean = pd.read_csv(
    "crisp_outputs/_leaktest_true_clean_seed998_cpu/ood/x_calibrated_condition_mean.csv",
    index_col=0,
    engine="python",
)
pois = pd.read_csv(
    "crisp_outputs/_leaktest_true_poison_seed998_cpu/ood/x_calibrated_condition_mean.csv",
    index_col=0,
    engine="python",
)

assert clean.shape == pois.shape
assert list(clean.index) == list(pois.index)
assert list(clean.columns) == list(pois.columns)

diff = clean.values - pois.values

print("=" * 80)
print("OOD TRUE POISON TEST RESULT")
print("mean abs pred diff:", float(np.mean(np.abs(diff))))
print("max abs pred diff:", float(np.max(np.abs(diff))))
print("rmse pred diff:", float(np.sqrt(np.mean(diff ** 2))))
print("=" * 80)

if np.mean(np.abs(diff)) > 1e-5:
    print("[DANGER] Predictions changed after poisoning OOD x_true. Possible OOD true leakage.")
else:
    print("[PASS] Predictions did not change materially. OOD x_true likely not used in training/prediction.")
PY