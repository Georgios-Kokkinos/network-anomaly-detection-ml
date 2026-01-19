"""
metrics.py

Evaluation utilities:
- classification metrics
- confusion matrix plot (Version 1/1.5 color style)
- training curves
- report saving
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy plus macro and weighted precision/recall/F1.

    Parameters:
    y_true (np.ndarray): True labels.
    y_pred (np.ndarray): Predicted labels.

    Returns:
    dict[str, float]: Metric name -> value.
    """
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f_macro,
        "precision_weighted": p_weight,
        "recall_weighted": r_weight,
        "f1_weighted": f_weight,
    }


def save_metrics(metrics: dict[str, float], path: Path) -> None:
    """Save metrics dictionary as a single-row CSV.

    Parameters:
    metrics (dict[str, float]): Metrics to save.
    path (Path): Output CSV path.

    Returns:
    None
    """
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)


def save_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], csv_path: Path, txt_path: Path
) -> None:
    """Save the sklearn classification report in CSV and plain text formats.

    Parameters:
    y_true (np.ndarray): True labels.
    y_pred (np.ndarray): Predicted labels.
    labels (list[str]): Class label names.
    csv_path (Path): Output CSV path.
    txt_path (Path): Output TXT path.

    Returns:
    None
    """
    report_dict = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
    )
    pd.DataFrame(report_dict).transpose().to_csv(csv_path)

    report_text = classification_report(
        y_true, y_pred, target_names=labels, zero_division=0
    )
    txt_path.write_text(report_text, encoding="utf-8")


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], path: Path
) -> None:
    """Plot and save the confusion matrix with Version 1/1.5 styling.

    Parameters:
    y_true (np.ndarray): True labels.
    y_pred (np.ndarray): Predicted labels.
    labels (list[str]): Class label names.
    path (Path): Output image path.

    Returns:
    None
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="plasma",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 7},
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_training_curves(history: dict, path: Path) -> None:
    """Plot training/validation loss and accuracy curves.

    Parameters:
    history (dict): Keras history dictionary.
    path (Path): Output image path.

    Returns:
    None
    """
    hist_df = pd.DataFrame(history)

    plt.figure(figsize=(10, 5))
    if "loss" in hist_df:
        plt.plot(hist_df["loss"], label="train_loss")
    if "val_loss" in hist_df:
        plt.plot(hist_df["val_loss"], label="val_loss")
    if "accuracy" in hist_df:
        plt.plot(hist_df["accuracy"], label="train_acc")
    if "val_accuracy" in hist_df:
        plt.plot(hist_df["val_accuracy"], label="val_acc")

    plt.title("Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_history_csv(history: dict, path: Path) -> None:
    """Save raw Keras history to CSV for analysis.

    Parameters:
    history (dict): Keras history dictionary.
    path (Path): Output CSV path.

    Returns:
    None
    """
    pd.DataFrame(history).to_csv(path, index=False)


def plot_roc_curves(
    y_true: np.ndarray, y_proba: np.ndarray, labels: list[str], path: Path
) -> None:
    """Plot one-vs-rest ROC curves (plus micro-average) for multiclass outputs.

    Parameters:
    y_true (np.ndarray): True labels.
    y_proba (np.ndarray): Predicted class probabilities.
    labels (list[str]): Class label names.
    path (Path): Output image path.

    Returns:
    None
    """
    n_classes = y_proba.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{labels[i]} (AUC={roc_auc:.3f})")

    # Micro-average
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linestyle="--", color="black", label=f"micro-average (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_precision_recall_curves(
    y_true: np.ndarray, y_proba: np.ndarray, labels: list[str], path: Path
) -> None:
    """Plot one-vs-rest precision-recall curves (plus micro-average).

    Parameters:
    y_true (np.ndarray): True labels.
    y_proba (np.ndarray): Predicted class probabilities.
    labels (list[str]): Class label names.
    path (Path): Output image path.

    Returns:
    None
    """
    n_classes = y_proba.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, label=f"{labels[i]} (AP={ap:.3f})")

    # Micro-average
    precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_proba.ravel())
    ap = average_precision_score(y_true_bin.ravel(), y_proba.ravel())
    plt.plot(recall, precision, linestyle="--", color="black", label=f"micro-average (AP={ap:.3f})")

    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
