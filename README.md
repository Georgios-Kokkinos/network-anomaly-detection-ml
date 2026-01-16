# Network Anomaly Detection with ML

A staged project for an undergraduate diploma thesis (University of Patras, Supervisor: Prof. Dimitrios Serpanos) that implements and evaluates machine‑learning pipelines for network traffic anomaly detection using the CICIDS 2017 dataset. This repository contains two checkpointed versions of the pipeline: Version 1 and Version 1.5.

---

## Project structure (root)
- Version 1/ — original checkpoint (baseline RandomForest pipeline, documentation, results snapshot)
- Version 1.5/ — extended checkpoint (multi‑model benchmark: RandomForest, XGBoost, LightGBM; improved reporting and plotting)
- README.md (this file)

Each version folder is self-contained: source code under `src/`, a `data/` placeholder (CSV dataset not included), and `results/` containing run artifacts for that version.

---

## High-level overview

- Data: CICIDS 2017 CSVs (not included due to size). Place CSV files in the `data/` directory of the version you run.
- Pipeline steps (per version):
  - Load and clean CSVs
  - Feature engineering and sanitization
  - Class balancing (configurable)
  - Train / evaluate models on a shared hold-out set
  - Save metrics, reports, confusion matrices, and comparative plots to `results/`

Version differences
- Version 1: single-model baseline (RandomForest) with modular code and initial reports.
- Version 1.5: multi-model benchmark (RF, XGBoost, LightGBM), comparative visuals, improved logging, reproducible run artifacts, and a short narrative report.

---

## Quick start (per-version)

1. Open a terminal in the version folder you want to run (e.g., `Version 1.5`).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Add CICIDS 2017 CSVs to `Version X/data/`.
4. Run the pipeline as a module (recommended):
   ```
   python -m src.main
   ```
Notes:
- Use `python -m src.main` so package imports like `from src...` resolve correctly.
- The `data/` folder should contain the CSVs; each version includes a .gitignore that prevents data being pushed to remote.

---

## Outputs and artifacts

Each run writes a set of artifacts to `Version X/results/`:
- Per-model metrics CSV and classification reports
- Confusion matrices (raw and normalized)
- Feature-importance plots
- Comparative metrics and comparative feature‑importance visuals
- `short_report.txt` summarizing the run

Legacy or archived artifacts (e.g., previous extended reports or feature-descriptions) should be placed in `results_archive/` (sibling to `results/`) to avoid accidental overwrites by new runs.

---

## Notes on warnings and run logs

- Some runs may emit benign warnings:
  - loky/core-count detection (joblib) — set `LOKY_MAX_CPU_COUNT` to silence.
  - XGBoost deprecated `use_label_encoder` notice — parameter ignored.
  - LightGBM "No further splits with positive gain" — indicates tree saturation; metrics unaffected.
- Each version’s `main.py` prints concise step markers and only prints guidance messages when the corresponding warnings actually occur.

---

## Recommended pre-push checklist

Before uploading a version folder to GitHub:
- Ensure `Version X/data/.gitkeep` is present and `data/*` is in `.gitignore`.
- Verify `requirements.txt` lists used packages (xgboost/lightgbm if used).
- Confirm `README.md` inside the version folder accurately reflects actual outputs.
- Run the pipeline locally and confirm `results/short_report.txt` exists.

---

## Next steps (research plan)

Planned next work (not implemented in this checkpoint):
- Add deep learning baselines (feed-forward, sequence models) and compare to tree-based models.
- Run k-fold CV and targeted improvements for rare-class performance.
- Expand reproducibility: run manifest (seed, data hashes), config file for hyperparameters.

---

## License & citation

Refer to the LICENSE file inside each version folder for licensing. When using or citing this work in academic contexts, acknowledge the supervisor and institution as appropriate.

---