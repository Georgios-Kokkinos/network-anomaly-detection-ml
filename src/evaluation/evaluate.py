"""
evaluate.py

This module provides evaluation utilities for the network anomaly detection pipeline.
It includes:
- Calculation of standard classification metrics (accuracy, precision, recall, F1-score)
- Visualization of confusion matrices (raw and normalized)
- Visualization of feature importances for trained models
- Saving evaluation metrics to a file

These tools help assess model performance, interpret results, and generate figures for reporting or further analysis.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of the model using various metrics.

    Parameters:
    y_true (array-like): True labels of the data.
    y_pred (array-like): Predicted labels from the model.

    Returns:
    dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None, normalize=False):
    """
    Plot the confusion matrix (raw or normalized).

    Parameters:
    y_true (array-like): True labels of the data.
    y_pred (array-like): Predicted labels from the model.
    classes (list): List of class names.
    save_path (str, optional): Path to save the plot. If None, displays the plot.
    normalize (bool): Whether to normalize the confusion matrix.
    """
    # Compute confusion matrix (normalized if requested)
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    plt.figure(figsize=(14, 10))  # Increased figure size
    fmt = '.2f' if normalize else 'd'
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='plasma',
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={"size": 9}  # Smaller font for annotations
    )
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plot feature importance from the trained model.

    Parameters:
    model: Trained model object (must have feature_importances_ attribute).
    feature_names (list): List of feature names.
    save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_evaluation_results(metrics, filename='evaluation_results.txt'):
    """
    Save evaluation metrics to a text file.

    Parameters:
    metrics (dict): A dictionary containing evaluation metrics.
    filename (str): The name of the file to save the results.
    """
    with open(filename, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")