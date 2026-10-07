#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================================"
echo "[INFO] Project directory: $(pwd)"
echo "[INFO] Stage-1 cleanup: caches deleted; old files archived"
echo "============================================================"

mkdir -p deprecated
mkdir -p _archive/old_ablations
mkdir -p _archive/test_runs
mkdir -p _archive/audit_runs
mkdir -p _archive/quarantined_code
mkdir -p _archive/diagnostic_reports
mkdir -p experiments/configs/examples
mkdir -p experiments/configs/deprecated

move_to_dir() {
    local src="$1"
    local dst="$2"

    if [[ -e "$src" ]]; then
        echo "[MOVE] $src -> $dst/"
        mv -vn -- "$src" "$dst/"
    else
        echo "[SKIP] Not found: $src"
    fi
}

echo
echo "========== 1. Remove regenerable caches =========="

rm -rf -- CRISP.egg-info
rm -rf -- __pycache__

find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

rm -f -- grep_r2score_delta_de_locations.txt
rm -f -- crisp_outputs/grep_r2score_delta_de_locations.txt

echo
echo "========== 2. Archive obsolete or incorrect scripts =========="

move_to_dir "run_recalib_final.py" "deprecated"
move_to_dir "run_fair_residual_baselines_iid_ood.py" "deprecated"
move_to_dir "simple_residual_baselines_csv.py" "deprecated"
move_to_dir "export_crisp_baseline_residual.py" "deprecated"

# Log produced by the obsolete baseline script
move_to_dir "fair_residual_baselines_iid_ood.log" "_archive/diagnostic_reports"

echo
echo "========== 3. Archive old ablation outputs =========="

move_to_dir "crisp_outputs/ablation_full" "_archive/old_ablations"
move_to_dir "crisp_outputs/ablation_no_drug" "_archive/old_ablations"
move_to_dir "crisp_outputs/ablation_no_raw_ctrl" "_archive/old_ablations"
move_to_dir "crisp_outputs/ablation_no_delta_features" "_archive/old_ablations"
move_to_dir "crisp_outputs/ablation_no_scalar" "_archive/old_ablations"

echo
echo "========== 4. Archive smoke-test and small-data runs =========="

move_to_dir "crisp_outputs/baseline_small50k" "_archive/test_runs"
move_to_dir "crisp_outputs/mydata_baseline_test" "_archive/test_runs"
move_to_dir "crisp_outputs/mydata_rc_test_alpha02_w30" "_archive/test_runs"
move_to_dir "crisp_outputs/rc_deaware_small50k_alpha005_w30" "_archive/test_runs"
move_to_dir "crisp_outputs/rc_deaware_small50k_alpha02_w30" "_archive/test_runs"
move_to_dir "crisp_outputs/rc_small50k_alpha01" "_archive/test_runs"

echo
echo "========== 5. Archive leakage-audit run directories =========="

shopt -s nullglob
audit_dirs=(crisp_outputs/_leaktest_*)
for path in "${audit_dirs[@]}"; do
    move_to_dir "$path" "_archive/audit_runs"
done
shopt -u nullglob

echo
echo "========== 6. Archive previously quarantined source code =========="

move_to_dir \
  "deleted_r2score_delta_de_20260605_150726" \
  "_archive/quarantined_code"

echo
echo "========== 7. Organize CRISP configuration files =========="

move_to_dir \
  "experiments/configs/mydata.evalall.test.yaml" \
  "experiments/configs/examples"

move_to_dir \
  "experiments/configs/mydata.test.yaml" \
  "experiments/configs/examples"

move_to_dir \
  "experiments/configs/nips.local.yaml" \
  "experiments/configs/examples"

move_to_dir \
  "experiments/configs/nips.small50k.yaml" \
  "experiments/configs/examples"

move_to_dir \
  "experiments/configs/nips.yaml" \
  "experiments/configs/examples"

move_to_dir \
  "experiments/configs/sci.yaml" \
  "experiments/configs/examples"

move_to_dir \
  "experiments/configs/mydata.local.yaml" \
  "experiments/configs/deprecated"

echo
echo "========== 8. Archive temporary recovered tables =========="

move_to_dir \
  "Table2_final_recovered.csv" \
  "_archive/diagnostic_reports"

echo
echo "========== 9. Standardize requirements filename =========="

if [[ -f "requirement.txt" && ! -e "requirements.txt" ]]; then
    mv -v -- "requirement.txt" "requirements.txt"
elif [[ -f "requirement.txt" && -e "requirements.txt" ]]; then
    echo "[WARN] Both requirement.txt and requirements.txt exist."
    echo "[WARN] Keeping both; compare them manually."
else
    echo "[SKIP] requirement.txt not found or already renamed."
fi

echo
echo "============================================================"
echo "[OK] Stage-1 cleanup completed."
echo "[INFO] Formal data and candidate paper results were not deleted."
echo "============================================================"
