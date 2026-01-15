"""
helpers.py

This module provides general-purpose plotting and reporting utilities for the network anomaly detection pipeline.
It includes:
- Visualization of class distributions (bar and pie charts)
- Exporting evaluation metrics and classification reports to CSV files

These functions are designed to be reusable and to keep the main pipeline code clean and modular.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_class_distribution(y, title="Class Distribution", save_path=None, class_names=None):
    """
    Plot the distribution of classes as a bar chart (log scale for clarity).

    Parameters:
    y (pd.Series): Encoded class labels.
    title (str): Title for the plot.
    save_path (str, optional): If provided, saves the plot to this path.
    class_names (list, optional): List of class names to use as x-tick labels.
    """
    plt.figure(figsize=(8, 5))
    counts = y.value_counts()  # Do NOT sort_index(), keep by count
    ax = counts.plot(kind='bar')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count (log scale)')
    plt.yscale('log')  # Use log scale for better visibility of minority classes

    # Set class names as x-tick labels if provided
    if class_names is not None:
        # Map each label in counts.index to its class name
        ax.set_xticklabels([class_names[int(idx)] for idx in counts.index], rotation=45, ha='right')

    # Annotate each bar with its count
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, rotation=90)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_class_pie(y, title="Class Proportion", save_path=None, class_names=None):
    """
    Plot the class proportions as a pie chart.

    Parameters:
    y (pd.Series): Encoded class labels.
    title (str): Title for the plot.
    save_path (str, optional): If provided, saves the plot to this path.
    class_names (list, optional): List of class names to use as labels.
    """
    plt.figure(figsize=(8, 8))
    counts = y.value_counts()
    labels = counts.index.astype(str)
    # If class_names mapping is provided, use it for labels
    if class_names is not None:
        labels = [class_names[int(idx)] for idx in counts.index]

    # Show only percentages > 2% on the pie, rest in legend
    def autopct(pct):
        return ('%.1f%%' % pct) if pct > 2 else ''
    wedges, texts, autotexts = plt.pie(
        counts, labels=None, autopct=autopct, startangle=90,
        pctdistance=0.8, labeldistance=1.2
    )
    plt.legend(wedges, labels, title="Class", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_metrics_and_report(evaluation_results, y_test, y_pred, target_names, results_dir):
    """
    Save overall evaluation metrics and a detailed classification report as CSV files.

    Parameters:
    evaluation_results (dict): Overall metrics (accuracy, precision, recall, f1-score).
    y_test (array-like): True labels.
    y_pred (array-like): Predicted labels.
    target_names (list): List of class names for the report.
    results_dir (str): Directory to save the CSV files.
    """
    # Save overall metrics
    pd.DataFrame([evaluation_results]).to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    # Save detailed per-class metrics
    from sklearn.metrics import classification_report
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(os.path.join(results_dir, "classification_report.csv"))