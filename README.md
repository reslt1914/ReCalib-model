# ReCalib

## Residual Calibration for Single-Cell Drug Perturbation Response Prediction with a Frozen Base Predictor

ReCalib is a post-hoc calibration framework for single-cell perturbation-response prediction.

The key idea is to keep the upstream perturbation-response predictor frozen, model its systematic residual, and use the learned residual to calibrate the original prediction.

The repository contains the ReCalib analysis pipeline, configuration files, supplementary data, evaluation scripts, and the CMonge implementation used as an independent frozen-predictor setting.

---

## 1. Overview

ReCalib does not replace the upstream perturbation-response predictor. Instead, a frozen predictor first produces an initial perturbation-response prediction, and ReCalib learns the systematic residual left by that predictor.

Two frozen predictors are considered in this work:

1. **CRISP** — the primary frozen perturbation-response predictor.
2. **CMonge** — an independently developed second frozen predictor used for an additional validation setting.

The main workflow is:

```text
sci-Plex perturbation condition
            |
            v
     Frozen predictor
            |
            v
      Base prediction
            |
            v
Observed response - prediction
            |
            v
         Residual
            |
            v
         ReCalib
            |
            v
   Calibrated prediction
```

---

## 2. CRISP: Primary Frozen Predictor

CRISP is the **primary frozen perturbation-response predictor** used in the main ReCalib experiments.

CRISP is treated as a frozen upstream predictor during ReCalib calibration.

```text
sci-Plex perturbation condition
            |
            v
       Frozen CRISP
            |
            v
      CRISP prediction
            |
            v
Observed response - CRISP prediction
            |
            v
         Residual
            |
            v
         ReCalib
            |
            v
   Calibrated prediction
```

ReCalib therefore performs post-hoc residual calibration rather than retraining the frozen CRISP predictor.

---

## 3. ReCalib Framework

Let `y` denote the observed perturbation response and `ŷ_base` denote the prediction produced by the frozen predictor. The residual is defined as:

```text
r = y - ŷ_base
```

ReCalib learns systematic structure contained in this residual.

The residual-calibration workflow is:

```text
Base prediction
      |
      v
Residual construction
      |
      v
Ridge residual anchor
      |
      v
Remaining residual
      |
      v
Nonlinear residual calibration
      |
      v
Final ReCalib prediction
```

The Ridge component is used to capture a linearly recoverable part of the residual. The remaining residual can then be modeled separately.

---

## 4. CMonge

CMonge is included as an independently developed second frozen predictor. It is not the ReCalib model itself; it provides an additional frozen-predictor setting for validation.

```text
CRISP
  |
  +-- Primary frozen predictor
  +-- Main ReCalib experiments

CMonge
  |
  +-- Independent frozen predictor
  +-- Additional ReCalib validation
```

The CMonge implementation is included under:

```text
cmong/
```

Its main structure is:

```text
cmong/
├── cmong/
│   ├── datasets/
│   ├── models/
│   ├── trainers/
│   ├── evaluate.py
│   ├── metrics.py
│   └── utils.py
├── configs/
├── scripts/
├── tests/
├── pyproject.toml
├── README.md
└── uv.lock
```

Important preparation scripts include:

```text
cmong/scripts/prepare_cmonge_sciplex.py
cmong/scripts/prepare_recalib_cmonge_smoke.py
cmong/scripts/prepare_recalib_cmonge_formal.py
```

---

## 5. Dataset and Supplementary Data

The experiments are based on the sci-Plex single-cell chemical perturbation setting.

The repository contains the supplementary resources used by the analyses:

```text
supplementary_data/
├── Supplementary_Data_1_977_gene_panel.csv
├── Supplementary_Data_1_977_gene_panel.xlsx
├── Supplementary_Data_2_condition_specific_LINCS_DE_annotations.csv
└── Supplementary_Data_2_condition_specific_LINCS_DE_annotations.xlsx
```

The first pair contains the 977-gene panel used in the analyses. The second pair contains condition-specific LINCS differential-expression annotations.

Large prediction-level and model files are kept outside Git when excluded by the repository configuration.

---

## 6. Repository Structure

```text
ReCalib/
│
├── cmong/
│   ├── cmong/
│   ├── configs/
│   ├── scripts/
│   ├── tests/
│   ├── pyproject.toml
│   ├── README.md
│   └── uv.lock
│
├── configs/
│   ├── cmong_recalib_experiments.yaml
│   ├── cmong_recalib_seed2024.yaml
│   ├── cmong_recalib_seed3407.yaml
│   ├── cmong_recalib_seed42.yaml
│   ├── innovation_data_validation.json
│   ├── innovation_experiments.yaml
│   ├── innovation_experiments_ho_pathway.yaml
│   └── innovation_path_discovery.json
│
├── supplementary_data/
│   ├── Supplementary_Data_1_977_gene_panel.csv
│   ├── Supplementary_Data_1_977_gene_panel.xlsx
│   ├── Supplementary_Data_2_condition_specific_LINCS_DE_annotations.csv
│   └── Supplementary_Data_2_condition_specific_LINCS_DE_annotations.xlsx
│
├── audit_final_poisoning.py
├── baseline_models.py
├── compare_affine_recalib.py
├── compare_final_vs_shuffled.py
├── compare_ridge_anchored_final.py
├── compare_strong_baselines.py
├── discover_innovation_paths.py
├── evaluate_frozen_cmonge_main_ood.py
├── infer_recalib_main_ood.py
├── innovation_data.py
├── inspect_main_ood_result_sources.py
├── inspect_ood_metrics_files.py
├── make_evalall_for_split.py
├── make_main_tables.py
├── make_mydataset_calib_val_baseline.py
├── make_primary4_results.py
├── make_supp_tables3_6.py
├── make_supp_tables15_16.py
├── make_table4.py
├── pathway_response_score_eval.py
├── plot_biological_case_studies.py
├── plot_selected_cases_manual.py
├── response_gene_centric_eval.py
├── recompute_final_paper_results.py
├── recover_table2_from_outputs.py
├── run_deweight_ablation_mydata.sh
├── run_direct_mlp_baseline.py
├── run_innovation_baselines.py
├── run_recalib_hyperparameter_selection.sh
├── run_recalib_strict_seeds.sh
├── run_ridge_anchored_recalib.py
├── run_ridge_residual_baseline.py
├── select_better_case_candidates.py
├── select_recalib_hyperparameters.py
├── select_representative_cases.py
├── statistical_test_recalib.py
├── train_crisp_rc_deaware.py
├── setup.py
├── requirements.txt
├── README.md
└── LICENSE.txt
```

---

## 7. Installation

Create the main ReCalib environment:

```bash
conda create -n ReCalib python=3.9
conda activate ReCalib
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

The CMonge component has its own project configuration in `cmong/pyproject.toml` and `cmong/uv.lock`.

---

## 8. Configuration

Main configuration files are stored under:

```text
configs/
```

Important files include:

```text
configs/cmong_recalib_experiments.yaml
configs/cmong_recalib_seed2024.yaml
configs/cmong_recalib_seed3407.yaml
configs/cmong_recalib_seed42.yaml
configs/innovation_data_validation.json
configs/innovation_experiments.yaml
configs/innovation_experiments_ho_pathway.yaml
configs/innovation_path_discovery.json
```

---

## 9. Hyperparameter Selection

The repository provides a dedicated hyperparameter-selection workflow:

```bash
python select_recalib_hyperparameters.py
```

or:

```bash
bash run_recalib_hyperparameter_selection.sh
```

The selected configuration is then used in the corresponding ReCalib experiments.

---

## 10. Baselines and Comparisons

Important comparison scripts include:

```text
baseline_models.py
compare_affine_recalib.py
compare_final_vs_shuffled.py
compare_ridge_anchored_final.py
compare_strong_baselines.py
run_direct_mlp_baseline.py
run_ridge_residual_baseline.py
```

These scripts provide baseline and residual-correction comparisons for the ReCalib analysis.

---

## 11. OOD Evaluation

The repository contains dedicated scripts for out-of-distribution evaluation:

```text
evaluate_frozen_cmonge_main_ood.py
infer_recalib_main_ood.py
inspect_main_ood_result_sources.py
inspect_ood_metrics_files.py
response_gene_centric_eval.py
```

The conceptual workflow is:

```text
Frozen predictor
       |
       v
OOD prediction
       |
       v
Residual calibration
       |
       v
ReCalib prediction
       |
       v
OOD evaluation
```

---

## 12. Leakage and Robustness Checks

The repository includes dedicated checks for the reliability of the evaluation pipeline:

```text
audit_final_poisoning.py
leak_test_poison_ood_residual.sh
leak_test_poison_ood_true.sh
statistical_test_recalib.py
```

These scripts are provided to audit the experimental pipeline and test for unintended information leakage.

---

## 13. Biological Evaluation

Additional analyses include response-gene-centric and pathway-level evaluation.

Important scripts include:

```text
response_gene_centric_eval.py
pathway_response_score_eval.py
plot_biological_case_studies.py
plot_selected_cases_manual.py
select_better_case_candidates.py
select_representative_cases.py
```

The analysis flow is:

```text
ReCalib prediction
       |
       v
Response-gene analysis
       |
       v
Pathway-level analysis
       |
       v
Biological case studies
```

---

## 14. Main Result Generation

Scripts for generating the main and supplementary results include:

```text
make_main_tables.py
make_primary4_results.py
make_table4.py
make_supp_tables3_6.py
make_supp_tables15_16.py
recompute_final_paper_results.py
recover_table2_from_outputs.py
```

These scripts operate on the corresponding experiment outputs and prediction resources.

---

## 15. Random Seeds

Multiple random seeds are explicitly configured for the ReCalib/CMonge experiments:

```text
configs/cmong_recalib_seed2024.yaml
configs/cmong_recalib_seed3407.yaml
configs/cmong_recalib_seed42.yaml
```

The corresponding experiments can be used to assess stability across random seeds.

---

## 16. Large Files and Generated Outputs

Large datasets, checkpoints, caches, and generated outputs should not be committed directly to Git unless explicitly required.

Common excluded binary types include:

```text
*.h5
*.h5ad
*.pkl
*.pickle
*.pt
*.pth
*.ckpt
*.npy
*.npz
```

Generated directories may include:

```text
outputs/
runs/
checkpoints/
results/
tmp/
```

The purpose is to keep the GitHub repository focused on source code, configuration, documentation, and required supplementary resources.

---

## 17. Reproduction Workflow

A high-level reproduction workflow is:

```text
Step 1  Install the ReCalib environment
   |
Step 2  Prepare required data and annotations
   |
Step 3  Obtain frozen predictor resources
   |
Step 4  Generate or load frozen CRISP predictions
   |
Step 5  Construct prediction residuals
   |
Step 6  Run residual calibration
   |
Step 7  Generate ReCalib predictions
   |
Step 8  Run baseline comparisons
   |
Step 9  Run OOD evaluation
   |
Step 10 Run biological and pathway analyses
   |
Step 11 Generate tables and figures
```

For the independent CMonge setting:

```text
CMonge
  |
  v
CMonge prediction
  |
  v
Residual construction
  |
  v
ReCalib calibration
  |
  v
Evaluation
```

---

## 18. Reproducibility Notes

Exact numerical reproduction can depend on:

- Python version
- PyTorch version
- CUDA version
- GPU hardware
- Dataset version
- Frozen predictor version
- Preprocessing
- Random seed
- Numerical-library versions

The repository therefore separates source code, configuration, supplementary data, frozen prediction resources, and generated experiment outputs.

---

## 19. Project Organization

The overall project can be understood as:

```text
                         ReCalib Project
                               |
             +-----------------+-----------------+
             |                                   |
             v                                   v
      Frozen predictors                    Calibration
             |                                   |
       +-----+-----+                             |
       |           |                             v
       v           v                          ReCalib
     CRISP       CMonge                         |
       |           |                            |
       +-----------+----------------------------+
                               |
                               v
                       Evaluation pipeline
                               |
              +----------------+----------------+
              |                |                |
              v                v                v
          Baselines           OOD       Biological analysis
```

CRISP and CMonge are frozen predictors. ReCalib is the post-hoc calibration component applied to frozen predictions.

---

## 20. Citation

If you use this implementation, please cite the associated ReCalib manuscript.

```bibtex
@article{ReCalib,
  title   = {ReCalib: Residual Calibration for Single-Cell Drug Perturbation Response Prediction with a Frozen Base Predictor},
  author  = {},
  journal = {},
  year    = {}
}
```

The bibliographic fields should be updated with the final published information when available.

---

## 21. License

This project is distributed under the MIT License.

See `LICENSE.txt` for the complete license text.
