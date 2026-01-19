# Version 2 — Deep Learning (MLP Baseline)

This version implements a clean, reproducible Deep Learning pipeline for **multiclass network traffic anomaly detection** on CICIDS2017 CSVs.

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

## Outputs (per run)
Each run creates a timestamped folder under `results/` with:
- `config.json`
- `metrics.csv`
- `classification_report.csv` + `classification_report.txt`
- `confusion_matrix.png`
- `roc_curves.png`
- `precision_recall_curves.png`
- `training_curves.png`
- `history.csv`
- `model.keras`
- `scaler.joblib`
- `label_encoder.joblib`
- `class_distribution.csv`
- `feature_names.txt`

## Notes
- No absolute paths are used.
- Reproducible via fixed seed in `config.py`.
- By default, the run uses a 20% sample and caps rows per CSV for stability. Adjust `sample_fraction` and `max_rows_per_file` in `config.py` for full-data runs.
- TensorFlow may print oneDNN or CPU feature warnings; these are informational and do not indicate failure.
