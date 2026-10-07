#!/usr/bin/env bash
set -e

cd ~/code/crisp
source ~/miniconda3/etc/profile.d/conda.sh
conda activate crisp

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false

SEEDS=(2024 3407 42)
WEIGHTS=(0 10 30 50)

for W in "${WEIGHTS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    OUT="crisp_outputs/mydata_rc_full_seed${SEED}_alpha02_w${W}"

    if [ -f "${OUT}/metrics_summary.csv" ]; then
      echo "[SKIP] ${OUT}"
      continue
    fi

    echo "============================================================"
    echo "Running de_weight=${W}, seed=${SEED}"
    echo "OUT=${OUT}"
    echo "============================================================"

    python train_crisp_rc_deaware.py \
      --baseline_dir crisp_outputs/mydata_baseline_full \
      --adata data/mydata/sciplex_complete_v2.h5ad \
      --de_key lincs_DEGs \
      --out_dir "${OUT}" \
      --alpha 0.2 \
      --de_weight "${W}" \
      --seed "${SEED}" \
      --epochs 200 \
      --batch_size 65536 \
      --device cuda
  done
done

