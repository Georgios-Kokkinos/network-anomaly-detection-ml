# Network Anomaly Detection - Version 3 (UNSW-NB15 Tree-Based Checkpoint)

**Date:** 27/3/2026

---

## Project Overview

This checkpoint mirrors Version 1.5 on the **UNSW-NB15 / CIC-UNSW-NB15** dataset. The pipeline automates data ingestion, preprocessing, class balancing, multi-model training, evaluation, and visualization. All outputs (plots, metrics, reports) are saved in the `results` folder at the project root for this version.

---

## What Has Been Done So Far (Version 3)

1. **Data Loading**
   - Supports UNSW file layouts:
     - `Data.csv` + `Label.csv` (preferred)
     - A single labeled CSV (for example `CICFlowMeter_out.csv`) with a label-like column.

2. **Preprocessing & Feature Engineering**
   - Column names are sanitized.
   - Missing values, duplicates, and infinite values are handled.
   - Labels are encoded with `LabelEncoder`.
   - Object-like numeric columns are coerced to numeric when possible.
   - Remaining categorical/non-numeric features are encoded consistently.

3. **Class Balancing**
   - Majority class downsampling + SMOTE upsampling (same strategy as Version 1.5).
   - Downsampling and SMOTE settings are guarded against edge cases in minority/majority counts.

4. **Model Training (Multi-Model Benchmark)**
   - Three classifiers are trained on the same hold-out protocol:
     - RandomForest
     - XGBoost
     - LightGBM

5. **Evaluation & Visualization**
   - Metrics (accuracy, precision, recall, F1-score) are saved per model.
   - Detailed classification reports are saved per model.
   - Confusion matrices (raw and normalized) and feature-importance plots are generated per model.
   - Comparative metrics and comparative feature-importance heatmap are generated.
   - Class distribution and class proportion plots are generated.
   - A short narrative report is written to `short_report.txt`.

6. **Label Name Mapping for Outputs**
   - If labels are numeric IDs (`0-9`), all plots/reports are rendered using canonical UNSW names:
     - `0=Benign`, `1=Analysis`, `2=Backdoor`, `3=DoS`, `4=Exploits`
     - `5=Fuzzers`, `6=Generic`, `7=Reconnaissance`, `8=Shellcode`, `9=Worms`

---

## Expected Dataset Files

Put your downloaded UNSW files in `data/`.

Preferred minimum set:
- `Data.csv`
- `Label.csv`

Optional extras (can coexist):
- `CICFlowMeter_out.csv`
- `Readme.txt`

---

## How to Run

1. Install dependencies from this folder:
   ```
   pip install -r requirements.txt
   ```

2. Put UNSW files in `data/`.

3. Execute the pipeline:
   ```
   python -m src.main
   ```

4. Find outputs in `results/` (per-model subfolders and comparative plots in the root of `results/`).

---

## Outputs (after latest run)

- Per-model folders: `rf/`, `xgboost/`, `lightgbm/` containing metrics, classification reports, confusion matrices (raw/normalized), and feature-importance plots.
- Comparative visuals: `comparative_metrics.png`, `comparative_feature_importance.png`.
- Dataset visuals: `class_distribution.png`, `class_proportion.png`.
- Narrative report: `short_report.txt`.
- Metrics snapshot (test set):
  - RandomForest: Acc 93.4314%, Prec 93.3343%, Rec 93.4314%, F1 93.1880%
  - XGBoost: Acc 93.9087%, Prec 93.9709%, Rec 93.9087%, F1 93.6255%
  - LightGBM: Acc 93.3585%, Prec 93.5140%, Rec 93.3585%, F1 93.3640%

---

## Notes

- This version is intentionally parallel to Version 1.5 (same modeling/evaluation style, new dataset).
- Use `python -m src.main` from the Version 3 folder so package imports resolve correctly.
- Benign warnings may appear during runs (joblib core count, XGBoost parameter notice, LightGBM split saturation) without affecting outputs.

---

## How to Use This Version

- This folder contains a snapshot of the tree-based UNSW pipeline (`src/`) and outputs (`results/`).
- To restore this version, copy the contents of this folder back to the project root.
- Use this README as a summary of Version 3 behavior, outputs, and validated baseline metrics.
