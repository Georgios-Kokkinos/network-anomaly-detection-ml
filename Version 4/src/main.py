"""
main.py

End-to-end training pipeline entry point for Version 4 (DL baseline on UNSW-NB15).
Run with: python -m src.main
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.config import Config
from src.data.data_loader import load_unsw_data
from src.evaluation.metrics import (
    evaluate_model,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_training_curves,
    save_classification_report,
    save_history_csv,
    save_metrics,
)
from src.features.preprocess import preprocess_dataframe
from src.models.mlp import build_mlp
from src.utils.helpers import (
    ensure_dir,
    map_unsw_class_names,
    project_root,
    save_dataframe,
    save_json,
    save_text,
    set_seeds,
)


def main() -> None:
    """Run the full training + evaluation pipeline and save all artifacts.

    Parameters:
    None

    Returns:
    None
    """
    config = Config()
    set_seeds(config.seed)

    root = project_root()
    data_dir = root / "data"
    results_root = root / config.results_dir_name

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(results_root / run_id)

    # Persist the exact configuration for reproducibility.
    save_json(asdict(config), run_dir / "config.json")

    print("[1/7] Loading data...")
    # Load and merge UNSW-NB15 files.
    df = load_unsw_data(
        data_dir,
        max_rows_per_file=config.max_rows_per_file,
        sample_fraction=config.sample_fraction,
        random_state=config.seed,
    )
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]:,} columns")

    print("[2/7] Preprocessing...")
    # Clean data, encode labels, and extract features.
    X, y, feature_names, label_encoder = preprocess_dataframe(df, config.label_col)
    num_classes = len(label_encoder.classes_)
    print(f"Classes: {num_classes}")

    print("[3/7] Splitting data...")
    # Stratified split to preserve class distribution.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.seed
    )
    val_ratio = config.val_size / (1.0 - config.test_size)
    # Validation split from the training portion.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, stratify=y_train, random_state=config.seed
    )

    print("[4/7] Scaling features...")
    # Standardize features for stable MLP training.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Optional class weights to mitigate imbalance.
    class_weights = None
    if config.use_class_weight:
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weights = {cls: weight for cls, weight in zip(classes, weights)}

    print("[5/7] Building model...")
    # Build the baseline MLP.
    model = build_mlp(input_dim=X_train.shape[1], num_classes=num_classes, config=config)

    # Early stopping and LR scheduling for stable convergence.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]

    print("[6/7] Training...")
    # Train the model.
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=class_weights,
        verbose=1,
    )

    print("[7/7] Evaluating...")
    # Predict class probabilities and labels on the test set.
    y_proba = model.predict(X_test, batch_size=config.batch_size, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = evaluate_model(y_test, y_pred)
    save_metrics(metrics, run_dir / "metrics.csv")

    # Human-readable labels for reports and plots.
    labels = map_unsw_class_names(list(label_encoder.classes_))
    save_classification_report(
        y_test, y_pred, labels, run_dir / "classification_report.csv", run_dir / "classification_report.txt"
    )
    # Core evaluation plots.
    plot_confusion_matrix(y_test, y_pred, labels, run_dir / "confusion_matrix.png")
    plot_roc_curves(y_test, y_proba, labels, run_dir / "roc_curves.png")
    plot_precision_recall_curves(y_test, y_proba, labels, run_dir / "precision_recall_curves.png")
    plot_training_curves(history.history, run_dir / "training_curves.png")
    save_history_csv(history.history, run_dir / "history.csv")

    # Save artifacts
    # Save model and preprocessing artifacts.
    model.save(run_dir / "model.keras")
    joblib.dump(scaler, run_dir / "scaler.joblib")
    joblib.dump(label_encoder, run_dir / "label_encoder.joblib")
    save_text("\n".join(feature_names), run_dir / "feature_names.txt")

    # Save class distribution for transparency.
    class_dist = np.bincount(y).astype(int)
    class_df = pd.DataFrame({"class": labels, "count": class_dist})
    save_dataframe(class_df, run_dir / "class_distribution.csv")

    print(f"Done. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
