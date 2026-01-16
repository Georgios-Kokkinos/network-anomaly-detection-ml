# Network Anomaly Detection - Version 1.5 Checkpoint

**Date:** 30/9/2025

---

## Project Overview

This checkpoint extends the network anomaly detection pipeline using the CICIDS 2017 dataset. The pipeline automates data ingestion, preprocessing, class balancing, multi-model training, evaluation, and visualization. All outputs (plots, metrics, reports) are saved in the `results` folder at the project root for this version.

---

## What Has Been Done So Far (Version 1.5)

1. **Data Loading**
   - All CSV files from the `data` directory are loaded and merged into a single DataFrame.
   - Initial cleaning of column and label characters to remove stray symbols.

2. **Preprocessing & Feature Engineering**
   - Leading/trailing spaces are removed from column names.
   - Rows with missing values and duplicates are dropped.
   - Target labels are encoded with `LabelEncoder`.
   - Object-like numeric columns are coerced to numeric when possible.
   - Remaining categorical features are encoded per column.
   - Infinite values are replaced and dropped.

3. **Class Balancing**
   - The majority class is downsampled to a configurable maximum (`N_SAMPLES`).
   - Minority classes are upsampled with SMOTE to a configurable target (`TARGET_COUNT`).

4. **Model Training (Multi-Model Benchmark)**
   - Three classifiers are trained on the same balanced split:
     - RandomForest
     - XGBoost
     - LightGBM

5. **Evaluation & Visualization**
   - Metrics (accuracy, precision, recall, F1-score) are calculated and saved as CSVs.
   - Detailed classification reports are saved as CSVs.
   - Confusion matrices (raw and normalized) and feature importance plots are generated per model.
   - Comparative metrics and feature-importance heatmap are generated.
   - Class distribution and class proportion plots are generated.
   - A short narrative report is written to `short_report.txt`.

6. **Code Organization**
   - The codebase is modular: `main.py` orchestrates the pipeline, with helpers for plotting and exporting, and separate modules for data loading, feature engineering, model, and evaluation.

---

## What We Intend To Do Next

- Introduce deep learning models (e.g., feed-forward or sequence models) to compare against the current tree-based baselines.

---

## How to Run

1. Install dependencies from this folder:
   ```
   pip install -r requirements.txt
   ```

2. Place the CICIDS 2017 CSVs into `data/` (same layout as Version 1).

3. Execute the pipeline:
   ```
   python -m src.main
   ```

4. Find outputs in `results/` (per-model subfolders and comparative plots in the root of `results/`).

---

## Outputs (after latest run)

- Per-model folders: metrics CSVs, classification reports, confusion matrices (raw/normalized), feature importance.
- Comparative visuals: `comparative_metrics.png`, `comparative_feature_importance.png`, `class_distribution.png`, `class_proportion.png`.
- Narrative report: `short_report.txt`.
- Legacy artifacts (if present): `extended_report.txt` and the feature description table should be placed in `results_archive/` (a sibling to `results/`) so they are not wiped on new runs.
- Metrics snapshot (test set):
  - RandomForest: Acc 0.9968, Prec 0.9969, Rec 0.9968, F1 0.9968
  - XGBoost: Acc 0.9973, Prec 0.9974, Rec 0.9973, F1 0.9973
  - LightGBM: Acc 0.9973, Prec 0.9974, Rec 0.9973, F1 0.9973

---

## Notes

- LightGBM may log "No further splits with positive gain" when trees exhaust useful splits; metrics remain unaffected in this run.
- All plots use the plasma palette for consistency with confusion matrices and feature-importance heatmaps.
- Feature names in plots are prettified (underscores removed); model inputs remain sanitized internally.

---

## How to Use This Version

- This folder contains a snapshot of the code (`src/`) and results (`results/`) as of this checkpoint.
- To restore this version, copy the contents of this folder back to the project root.
- Refer to this file for a summary of the project state and outputs.
