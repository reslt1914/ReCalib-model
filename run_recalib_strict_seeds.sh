#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

PYTHON="$(which python)"

BASELINE_DIR="crisp_outputs/mydata_baseline_calib_seed2024"
ADATA="data/mydata/sciplex_complete_v2.h5ad"

OUT_ROOT="results/recalib_strict_final"

EPOCHS=200
BATCH_SIZE=65536
LR=0.0008
HIDDEN=256
GENE_EMB_DIM=64
DE_WEIGHT=0
L2_PRED_RESIDUAL=0.0001

mkdir -p "${OUT_ROOT}"

for SEED in 3407 42
do
    OUT_DIR="${OUT_ROOT}/seed_${SEED}"
    LOG="${OUT_ROOT}/seed_${SEED}.log"

    echo
    echo "============================================================"
    echo "STRICT ReCalib"
    echo "seed       = ${SEED}"
    echo "train      = 1003 fitting conditions"
    echo "validation = 251 conditions"
    echo "DE weight  = ${DE_WEIGHT}"
    echo "============================================================"

    rm -rf "${OUT_DIR}"

    "${PYTHON}" -u train_crisp_rc_deaware.py \
        --baseline_dir "${BASELINE_DIR}" \
        --adata "${ADATA}" \
        --de_key lincs_DEGs \
        --out_dir "${OUT_DIR}" \
        --seed "${SEED}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --hidden "${HIDDEN}" \
        --gene_emb_dim "${GENE_EMB_DIM}" \
        --alpha 1.0 \
        --de_weight "${DE_WEIGHT}" \
        --l2_pred_residual "${L2_PRED_RESIDUAL}" \
        --device cuda \
        --topk 50 \
        --max_train_pairs 0 \
        2>&1 | tee "${LOG}"

    echo "[DONE] seed=${SEED}"
done

echo
echo "[OK] Strict ReCalib additional seeds completed"
