# Network Anomaly Detection - Version 4 (UNSW-NB15 Deep Learning Checkpoint)

**Date:** 27/3/2026

---

## Project Overview

This checkpoint mirrors Version 2 on the **UNSW-NB15 / CIC-UNSW-NB15** dataset. It provides an MLP baseline pipeline for multiclass network traffic anomaly detection with end-to-end preprocessing, training, evaluation, and artifact export. All outputs are saved per run in a timestamped folder under `results/`.

---

## What Has Been Done So Far (Version 4)

1. **Data Loading**
   - Supports UNSW file layouts:
     - `Data.csv` + `Label.csv` (preferred)
     - A single labeled CSV containing a label-like column.
   - Optional row caps and sampling are available.

2. **Preprocessing & Feature Engineering**
   - Column names and labels are sanitized for clean reporting.
   - Features are coerced to numeric and cleaned for inf/NaN.
   - Labels are encoded for multiclass classification.

3. **Train/Validation/Test Split**
   - Stratified split preserves class distribution.
   - StandardScaler is fitted on train and applied to validation/test.

4. **Deep Learning Model (MLP Baseline)**
   - Configurable hidden layers and dropout (same architecture philosophy as Version 2).
   - Adam optimizer with tuned architecture defaults for this checkpoint.
   - Train-set rebalancing (per-class downsampling/upsampling) before fitting to improve minority-class learning.
   - Early stopping and LR reduction callbacks are active for stable training.

5. **Evaluation & Visualization**
   - Metrics (accuracy, macro/weighted precision/recall/F1) are saved to CSV.
   - Classification report is saved in CSV and TXT.
   - Confusion matrix heatmaps are generated in raw and normalized forms.
   - ROC and Precision-Recall curves are generated (one-vs-rest plus micro-average).
   - Training curves and epoch history are exported.

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

Optional extras:
- `CICFlowMeter_out.csv`
- `Readme.txt`

---

## Folder Structure
```
Version 4/
  data/                  # UNSW-NB15 CSVs (not committed)
  results/               # Timestamped run outputs
  src/
    main.py              # Entry point: python -m src.main
    config.py            # Central configuration
    data/                # Data loading
    features/            # Preprocessing
    models/              # MLP definition
    evaluation/          # Metrics + plots
    utils/               # Helpers
```

## How to Run
From the Version 4 folder:
```
python -m src.main
```

---

## Outputs (per run)
Each run creates a timestamped folder under `results/` with:
- `config.json`
- `metrics.csv`
- `classification_report.csv`
- `classification_report.txt`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png` — normalized confusion matrix heatmap (0-1).
- `roc_curves.png`
- `precision_recall_curves.png`
- `training_curves.png`
- `history.csv`
- `model.keras`
- `scaler.joblib`
- `label_encoder.joblib`
- `class_distribution.csv`
- `train_class_balance.csv`
- `feature_names.txt`

Latest validated run snapshot (folder `run_20260327_162554`):
- Accuracy: 0.965188
- Weighted F1: 0.967010
- Macro F1: 0.431079

Compared with the earlier baseline snapshot (`run_20260326_150110`), macro F1 improved from 0.205451 to 0.431079 after the rebalancing and tuning updates.

---

## Notes

- This version is intentionally parallel to Version 2 (same DL workflow, new dataset).
- Current validated defaults in `src/config.py`:
   - `sample_fraction=0.8`, `max_rows_per_file=200000`
   - `batch_size=512`, `epochs=14`, `hidden_units=(512, 256, 128)`, `dropout=0.2`
   - train rebalancing enabled with `min_train_samples_per_class=1500`, `max_train_samples_per_class=40000`
- TensorFlow may print oneDNN/CPU/FutureWarning messages that are informational and do not indicate failure.

---

## How to Use This Version

- This folder contains a snapshot of the deep-learning UNSW pipeline (`src/`) and outputs (`results/`).
- To restore this version, copy the contents of this folder back to the project root.
- Use this README as a summary of Version 4 architecture, outputs, and validated baseline behavior.
