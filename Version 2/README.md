# Network Anomaly Detection - Version 2 (Deep Learning Checkpoint)

**Date:** 19/1/2026

---

## Project Overview

This checkpoint introduces a Deep Learning pipeline (MLP baseline) for **multiclass network traffic anomaly detection** using the CICIDS 2017 dataset. The pipeline automates data ingestion, preprocessing, label encoding, model training, evaluation, and plotting. All outputs are saved per run in a timestamped folder under `results/`.

---

## What Has Been Done So Far (Version 2)

1. **Data Loading**
  - All CICIDS CSVs in `data/` are loaded and merged into a single DataFrame.
  - Optional row limits and sampling are supported for memory‑safe experimentation.

2. **Preprocessing & Feature Engineering**
  - Column names and labels are sanitized for clean plotting/reporting.
  - Features are coerced to numeric, with inf/NaN handled robustly.
  - Labels are encoded for multiclass classification.

3. **Train/Validation/Test Split**
  - Stratified splits preserve class distribution.
  - StandardScaler is fitted on train and applied to val/test.

4. **Deep Learning Model (MLP Baseline)**
  - Multi‑layer perceptron with configurable hidden units and dropout.
  - Trained with Adam and early stopping for stability.
  - Class weights used to mitigate class imbalance.

5. **Evaluation & Visualization**
  - Metrics (accuracy, macro/weighted precision/recall/F1) saved to CSV.
  - Classification report saved in CSV and TXT.
  - Confusion matrix heatmap (plasma) with per‑cell counts.
  - ROC curves and Precision‑Recall curves (one‑vs‑rest + micro‑average).
  - Training curves and history saved for analysis.

6. **Reproducibility**
  - Fixed random seed.
  - Full config snapshot saved per run.

---

## Folder Structure
```
Version 2/
  data/                  # CICIDS2017 CSVs (already provided)
  results/               # Run outputs (auto-created)
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
From the Version 2 folder:
```
python -m src.main
```

---

## Outputs (per run)
Each run creates a timestamped folder under `results/` with:
- `config.json` — exact configuration values used for the run.
- `metrics.csv` — single‑row table of accuracy + macro/weighted precision/recall/F1.
- `classification_report.csv` — per‑class precision/recall/F1/support (CSV).
- `classification_report.txt` — readable text version of the classification report.
- `confusion_matrix.png` — raw confusion matrix heatmap with counts.
- `roc_curves.png` — one‑vs‑rest ROC curves (plus micro‑average).
- `precision_recall_curves.png` — one‑vs‑rest PR curves (plus micro‑average).
- `training_curves.png` — training/validation loss and accuracy over epochs.
- `history.csv` — epoch‑by‑epoch training history.
- `model.keras` — trained Keras model checkpoint.
- `scaler.joblib` — fitted StandardScaler for inference preprocessing.
- `label_encoder.joblib` — fitted label encoder (class name ↔ integer).
- `class_distribution.csv` — counts per class in the full dataset used.
- `feature_names.txt` — ordered list of input feature names.

## Notes
- No absolute paths are used.
- Reproducible via fixed seed in `config.py`.
- By default, the run uses a 20% sample and caps rows per CSV for stability. Adjust `sample_fraction` and `max_rows_per_file` in `config.py` for full-data runs.
- TensorFlow may print oneDNN or CPU feature warnings; these are informational and do not indicate failure.

---

## How to Use This Version

- This folder contains a snapshot of the Deep Learning pipeline (`src/`) and its outputs (`results/`).
- To restore this version, copy the contents of this folder back to the project root.
- Use this README as a summary of the Version 2 architecture, outputs, and current baseline behavior.
