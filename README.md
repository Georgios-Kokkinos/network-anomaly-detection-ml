# Network Anomaly Detection with ML

A staged undergraduate diploma thesis project (University of Patras, Supervisor: Prof. Dimitrios Serpanos) implementing and evaluating machine-learning pipelines for network traffic anomaly detection across two dataset families:

- CICIDS 2017 (Versions 1, 1.5, 2)
- UNSW-NB15 / CIC-UNSW-NB15 (Versions 3, 4)

---

## Project Structure (Root)

- `Version 1/` - original checkpoint (single-model RandomForest baseline, CICIDS 2017)
- `Version 1.5/` - extended tree-based checkpoint (RF, XGBoost, LightGBM, CICIDS 2017)
- `Version 2/` - deep-learning checkpoint (MLP baseline, CICIDS 2017)
- `Version 3/` - tree-based UNSW checkpoint (RF, XGBoost, LightGBM)
- `Version 4/` - deep-learning UNSW checkpoint (MLP baseline)
- `presentation/` - presentation assets and guide material
- `README.md` (this file)

Each version folder is self-contained with source code under `src/`, a dataset placeholder in `data/` (raw CSVs not committed), and run artifacts under `results/`.

---

## High-Level Overview

Shared pipeline stages (version-specific implementations):
- Load and clean CSV data.
- Perform feature engineering and sanitization.
- Handle class imbalance (tree-based versions) or class-weighted/rebalanced training (deep-learning versions).
- Train and evaluate models on hold-out splits.
- Save metrics, reports, confusion matrices, and comparative plots to `results/`.

Version differences:
- **Version 1:** CICIDS single-model baseline (RandomForest).
- **Version 1.5:** CICIDS multi-model tree benchmark (RF, XGBoost, LightGBM) with comparative reporting.
- **Version 2:** CICIDS deep-learning MLP baseline with ROC/PR curves and timestamped run artifacts.
- **Version 3:** UNSW multi-model tree benchmark mirroring Version 1.5.
- **Version 4:** UNSW deep-learning MLP benchmark mirroring Version 2.

---

## Prerequisites

- Python 3.10+ (validated runs were performed on Python 3.12)
- `pip` for dependency installation
- Sufficient RAM for medium/large CSV processing (recommended 16 GB+ for full-scale runs)
- Optional: GPU support for faster deep-learning training in Versions 2 and 4

Each version has its own `requirements.txt`. Install dependencies from inside the specific version folder you plan to run.

---

## Dataset Notes

- **CICIDS 2017** is expected for Versions 1, 1.5, and 2.
- **UNSW-NB15 / CIC-UNSW-NB15** is expected for Versions 3 and 4.

Official dataset pages:
- CICIDS 2017: https://www.unb.ca//cic/datasets/ids-2017.html
- CIC-UNSW-NB15: https://www.unb.ca/cic/datasets/cic-unsw-nb15.html

For Versions 3 and 4, the loader supports:
- `Data.csv` + `Label.csv` (preferred)
- Single labeled CSV fallback (for example `CICFlowMeter_out.csv`)

Recommended directory layouts:

- CICIDS versions (1, 1.5, 2):
	- place CICIDS CSV files directly in `Version X/data/`
- UNSW versions (3, 4):
	- preferred: `Data.csv` and `Label.csv` in `Version X/data/`
	- fallback: one labeled CSV that already contains a label-like target column

---

## Quick Start (Per Version)

1. Open a terminal inside the version folder you want to run.
2. Install dependencies:
	```
	pip install -r requirements.txt
	```
3. Place the required dataset CSVs in that version's `data/` folder.
4. Run as a module:
	```
	python -m src.main
	```

Use module execution (`python -m src.main`) so imports of the form `from src...` resolve correctly.

---

## Run Matrix

Use this mapping when choosing a version to execute:

- `Version 1`: CICIDS 2017, RandomForest baseline
- `Version 1.5`: CICIDS 2017, tree-based benchmark (RF/XGBoost/LightGBM)
- `Version 2`: CICIDS 2017, deep-learning MLP baseline
- `Version 3`: UNSW-NB15, tree-based benchmark (RF/XGBoost/LightGBM)
- `Version 4`: UNSW-NB15, deep-learning MLP baseline

Typical workflow:
1. `cd` into one version folder.
2. Install dependencies from that folder.
3. Place the right dataset files into that version's `data/` folder.
4. Run `python -m src.main`.
5. Inspect generated artifacts in `results/`.

---

## Outputs and Artifacts

Tree-based versions (1, 1.5, 3) produce:
- Per-model metrics and classification reports
- Confusion matrices (raw and normalized)
- Feature-importance plots
- Comparative metrics/feature-importance visuals
- `short_report.txt`

Deep-learning versions (2, 4) produce timestamped run folders with:
- `config.json`, `metrics.csv`, `classification_report.csv`, `classification_report.txt`
- `confusion_matrix.png`, `roc_curves.png`, `precision_recall_curves.png`, `training_curves.png`
- `history.csv`, `model.keras`, `scaler.joblib`, `label_encoder.joblib`, `feature_names.txt`, `class_distribution.csv`

Interpretation notes:
- Weighted metrics are often higher than macro metrics on imbalanced datasets.
- Confusion matrices should be read alongside class supports in classification reports.
- ROC/PR curves in deep-learning versions are one-vs-rest per class with a micro-average overlay.

---

## Validation Status

As of the latest verification pass, Versions 3 and 4 were run end-to-end and their generated artifacts were audited for:
- File presence/completeness
- Image readability and non-blank content
- CSV/TXT/JSON schema sanity
- Model/scaler/encoder loadability

Manual visual checks also confirmed that generated plots are semantically consistent and class names render correctly.

Current baseline snapshots from latest validated runs:
- Version 3 (UNSW tree-based): around 93% accuracy across RF/XGBoost/LightGBM.
- Version 4 (UNSW deep-learning): accuracy around 0.94 with macro-F1 around 0.38 under severe class imbalance.

---

## Notes on Warnings

Some runs may emit benign warnings:
- joblib/loky core-count detection
- XGBoost parameter deprecation notices
- LightGBM split-saturation messages
- TensorFlow oneDNN/CPU/FutureWarning messages

These are informational in this project context and do not imply run failure when artifacts are produced normally.

---

## Reproducibility Guidance

- Use fixed seeds defined in each version's config/main flow.
- Keep dataset composition stable between comparisons (same files, same row caps/sampling settings).
- Compare models using the same split protocol inside each version.
- Store and reference the generated `config.json` (deep-learning versions) and `short_report.txt` (tree-based versions).

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'src'`
	- run from inside the version folder using `python -m src.main`, not `python src/main.py`
- Dependency conflicts after package updates
	- reinstall from that version's `requirements.txt`
- Missing dataset columns/labels
	- verify files are placed in the correct version `data/` folder and include required label information
- Slow or memory-heavy runs
	- use sampling/row-cap settings where available (especially Versions 2 and 4)

---

## Recommended Pre-Push Checklist

- Keep dataset CSVs out of git (`data/` contains placeholders only).
- Keep generated outputs out of git where intended (`results/` ignored except placeholders).
- Ensure each version's `README.md` matches actual behavior and output files.
- Re-run the intended version and confirm expected artifacts in `results/`.

---

## Research Continuation Ideas

- Cross-dataset evaluation (train on one family, test on the other) for robustness analysis.
- Rare-class optimization (focal loss, calibrated sampling, threshold tuning).
- Hyperparameter sweeps and controlled run manifests (seeds, data hashes, config snapshots).
- Additional deep-learning baselines and ensemble comparisons.

---

## License and Citation

Refer to the `LICENSE` file inside each version folder. For academic usage, acknowledge the supervisor and institution appropriately.
