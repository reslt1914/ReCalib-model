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
OUT_ROOT="results/recalib_hyperparameter_selection"

SEED=2024
EPOCHS=200
BATCH_SIZE=65536
LR=0.0008
HIDDEN=256
GENE_EMB_DIM=64
L2_PRED_RESIDUAL=0.0001

mkdir -p "${OUT_ROOT}"

echo "============================================================"
echo "Strict ReCalib hyperparameter selection"
echo "============================================================"
echo "Training conditions   : 1003"
echo "IID validation        : 251"
echo "Main OOD              : NOT USED"
echo "Seed                  : ${SEED}"
echo "Baseline directory    : ${BASELINE_DIR}"
echo "============================================================"

for DE_WEIGHT in 0 10 30 50
do
    OUT_DIR="${OUT_ROOT}/de_weight_${DE_WEIGHT}"
    LOG="${OUT_ROOT}/de_weight_${DE_WEIGHT}.log"

    echo
    echo "============================================================"
    echo "[RUN] DE weight = ${DE_WEIGHT}"
    echo "[OUT] ${OUT_DIR}"
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

    echo "[DONE] DE weight = ${DE_WEIGHT}"
done

echo
echo "============================================================"
echo "[OK] All four strict ReCalib training runs completed"
echo "============================================================"
