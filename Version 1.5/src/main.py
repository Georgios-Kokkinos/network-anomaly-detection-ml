"""
main.py

Entry point for Version 1.5 of the network anomaly detection pipeline. It orchestrates:
- Loading and cleaning all CICIDS 2017 CSVs from the local data directory.
- Feature engineering with column/label sanitization (human-friendly names kept for plots).
- Hybrid class balancing (majority downsampling + SMOTE) on a stratified 80/20 split.
- Training and evaluating three models (RandomForest, XGBoost, LightGBM) on the same split.
- Generating per-model plots (confusion matrices, feature importance) and comparative plots
    (metrics, feature-importance heatmap), plus class distribution/proportion visuals.
- Persisting all artifacts (CSVs and PNGs) under results/ without auto-opening windows.
"""

import os
import shutil
import warnings
from datetime import datetime
from textwrap import dedent
from src.data.data_loader import load_csv_files
from src.features.feature_engineering import preprocess_data
from src.models.model import AnomalyDetectionModel
from src.evaluation.evaluate import evaluate_model, plot_confusion_matrix, plot_feature_importance, get_feature_importances
from src.utils.helpers import plot_class_distribution, plot_class_pie, save_metrics_and_report
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==== CONFIGURABLE PARAMETERS ====
NROWS = 100000  # Number of rows to read from the CSV files (default is None, eg. read all rows)
N_SAMPLES = 50000  # Maximum number of samples to keep for the majority class during downsampling (default is 100000)
TARGET_COUNT = 500  # Target number of samples for each minority class during upsampling (SMOTE) (default is 20000)

# Define the data and results directories relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def prepare_results_dir():
    """
    Removes the existing results directory (if any) and creates a fresh one.
    Ensures that each run starts with a clean results folder.
    """
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR)

def write_reports(metrics_df, save_dir, n_total, n_train, n_test, balanced_sizes, class_names, feature_names_pretty):
    """Persist the short narrative report into the results directory."""
    today = datetime.now().strftime("%Y-%m-%d")
    metrics_lines = []
    for _, row in metrics_df.iterrows():
        metrics_lines.append(
            f"{row['model']}: Acc {row['accuracy']:.4f}%, Prec {row['precision']:.4f}%, Rec {row['recall']:.4f}%, F1 {row['f1_score']:.4f}%"
        )
    metrics_block = "\n".join(f"- {line}" for line in metrics_lines)

    best_row = metrics_df.sort_values("f1_score", ascending=False).iloc[0]
    best_model_line = (
        f"{best_row['model']} leads the pack with Acc {best_row['accuracy']:.4f}%, Prec {best_row['precision']:.4f}%, "
        f"Rec {best_row['recall']:.4f}%, and F1 {best_row['f1_score']:.4f}% on the common hold-out."
    )

    short_report = dedent(
        f"""
        Network Anomaly Detection - Short Report
        Date: {today}

        Overview
        Three tree-based classifiers (RandomForest, XGBoost, LightGBM) were trained on a unified CICIDS 2017 pipeline with hybrid class balancing (downsample majority + SMOTE). A single stratified 80/20 hold-out enables fair comparison across models.

        Data Slice
        Total rows after cleaning: {n_total}. Train: {n_train}. Test: {n_test}. Features retained: {len(feature_names_pretty)}. Classes observed: {len(class_names)}.

        Aggregate Metrics (percent scale)
        {metrics_block}
        """
    ).strip() + "\n"
    with open(os.path.join(save_dir, "short_report.txt"), "w", encoding="utf-8") as f:
        f.write(short_report)

def main():
    # Capture warnings so we can print a helpful note only if they occur.
    warnings_triggered = []
    default_showwarning = warnings.showwarning

    def capture_and_show(msg, cat, fname, lineno, file=None, line=None):
        warnings_triggered.append((msg, cat))
        default_showwarning(msg, cat, fname, lineno, file, line)

    warnings.showwarning = capture_and_show

    print("[step] Preparing results directory...")
    prepare_results_dir()
    print("[done] Results directory ready.")

    # Load and preprocess data
    print("[step] Loading data...")
    data = load_csv_files(DATA_DIR, NROWS)
    print(f"[done] Loaded data with shape {data.shape}.")
    data.columns = data.columns.str.replace('�', '', regex=False)
    data[' Label'] = data[' Label'].str.replace('�', '', regex=False)
    # Keep human-readable names for documentation
    feature_names_original = [c.strip() for c in data.columns if c.strip() != 'Label']
    # Sanitize column names for models (remove spaces to avoid LightGBM warnings)
    data.columns = data.columns.str.strip().str.replace(r'\s+', '_', regex=True)
    print("[step] Preprocessing data (cleaning, encoding)...")
    processed_data, label_encoder = preprocess_data(data)
    print(f"[done] Preprocessing complete; rows={len(processed_data)}, cols={processed_data.shape[1]}.")
    # Pretty names for plots (replace underscores with spaces)
    feature_names_pretty = [name.replace('_', ' ') for name in feature_names_original]

    # Feature/label split
    X = processed_data.iloc[:, :-1]
    y = processed_data.iloc[:, -1]
    n_total = len(processed_data)

    # Ensure object-like numeric columns are coerced to numeric (avoid XGBoost dtype errors)
    X = X.copy()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce').fillna(0)

    # Visualize class distribution and proportions
    class_names = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
    print("[step] Plotting class distribution visuals...")
    plot_class_distribution(
        y,
        title="Class Distribution After Preprocessing",
        save_path=os.path.join(RESULTS_DIR, "class_distribution.png"),
        class_names=class_names
    )
    class_names = label_encoder.inverse_transform(sorted(y.unique()))
    plot_class_pie(
        y,
        title="Class Proportion After Preprocessing",
        save_path=os.path.join(RESULTS_DIR, "class_proportion.png"),
        class_names=class_names
    )
    print("[done] Class distribution visuals saved.")

    # Create a single hold-out test set to fairly compare models
    print("[step] Building stratified train/hold-out split (80/20)...")
    X_train_full, X_test_holdout, y_train_full, y_test_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[done] Split complete; train={len(X_train_full)}, test={len(X_test_holdout)}.")
    n_train = len(X_train_full)
    n_test = len(X_test_holdout)

    models = [
        ("rf", "RandomForest"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
    ]
    model_order = [name for _, name in models]

    metrics_list = []
    fi_dict = {}
    balanced_sizes = {}

    for model_key, model_name in models:
        print(f"Training and evaluating: {model_name}")
        model = AnomalyDetectionModel(model_type=model_key)

        # Balance the training set (downsample + SMOTE) then train
        X_resampled, y_resampled = model.handle_class_imbalance(
            X_train_full, y_train_full, n_samples=N_SAMPLES, target_count=TARGET_COUNT
        )
        balanced_sizes[model_name] = len(y_resampled)
        model.model.fit(X_resampled, y_resampled)

        # Predict on the common hold-out test set
        y_pred = model.model.predict(X_test_holdout)

        # Evaluate and save per-model results
        target_names = [name.replace('�', '') for name in label_encoder.inverse_transform(sorted(processed_data['Label'].unique()))]
        evaluation_results = evaluate_model(y_test_holdout, y_pred)
        metrics_list.append((model_name, evaluation_results))

        model_results_dir = os.path.join(RESULTS_DIR, model_key)
        os.makedirs(model_results_dir, exist_ok=True)
        save_metrics_and_report(evaluation_results, y_test_holdout, y_pred, target_names, model_results_dir)

        # Confusion matrices
        plot_confusion_matrix(
            y_test_holdout, y_pred, classes=target_names,
            save_path=os.path.join(model_results_dir, "confusion_matrix.png"), normalize=False
        )
        plot_confusion_matrix(
            y_test_holdout, y_pred, classes=target_names,
            save_path=os.path.join(model_results_dir, "confusion_matrix_normalized.png"), normalize=True
        )

        # Feature importances: delegate computation to evaluation module
        feature_names = X.columns
        importances = get_feature_importances(model.model, X_test_holdout, y_test_holdout, n_repeats=10)

        # Save per-model feature importance plot (plot accepts either model or importances array)
        plot_feature_importance(importances, feature_names_pretty, save_path=os.path.join(model_results_dir, "feature_importance.png"))

        # Store normalized importances for comparative plotting
        if importances.sum() > 0:
            fi_norm = importances / importances.sum()
        else:
            fi_norm = importances
        fi_dict[model_name] = fi_norm

    print("[step] Generating comparative reports and plots...")

    # ----------------
    # Comparative plots
    # ----------------
    # Comparative metrics bar chart
    metrics_df = pd.DataFrame(metrics_list, columns=["model", "metrics"])
    metrics_df = metrics_df.join(pd.DataFrame(metrics_df.pop("metrics").tolist()))
    metrics_df[['accuracy', 'precision', 'recall', 'f1_score']] *= 100.0
    metrics_order = ["accuracy", "precision", "recall", "f1_score"]
    metrics_long = metrics_df.melt(id_vars="model", var_name="metric", value_name="value")
    metrics_long['metric'] = pd.Categorical(metrics_long['metric'], categories=metrics_order, ordered=True)

    # Text report (short) with narrative detail
    write_reports(
        metrics_df=metrics_df,
        save_dir=RESULTS_DIR,
        n_total=n_total,
        n_train=n_train,
        n_test=n_test,
        balanced_sizes=balanced_sizes,
        class_names=class_names,
        feature_names_pretty=feature_names_pretty,
    )

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=metrics_long,
        x="metric",
        y="value",
        hue="model",
        palette="plasma",
        order=metrics_order,
        hue_order=model_order,
        errorbar=None
    )
    ax.set_title('Model comparison - metrics (%)')
    # Let the axis adapt to the actual metric range so bars are not clipped when values drop.
    ymin = max(0.0, metrics_long['value'].min() - 2.0)
    ymax = min(100.0, metrics_long['value'].max() + 2.0)
    ax.set_ylim(ymin, ymax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f%%', rotation=90, padding=2, fontsize=8)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    metrics_chart_path = os.path.join(RESULTS_DIR, "comparative_metrics.png")
    plt.savefig(metrics_chart_path, bbox_inches='tight')
    plt.close()
    print("[done] Comparative metrics plotted.")

    # Comparative feature importance heatmap for top features
    fi_df = pd.DataFrame(fi_dict).T  # rows: models, columns: features
    fi_df.columns = feature_names_pretty
    # Select top 10 features by mean importance across models
    top_features = fi_df.mean(axis=0).sort_values(ascending=False).head(10).index.tolist()
    plt.figure(figsize=(12, 6))
    sns.heatmap(fi_df[top_features], annot=True, cmap='plasma', cbar_kws={"shrink": 0.7}, linewidths=0.3, fmt='.3f')
    plt.title('Normalized feature importances (top 10)')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparative_feature_importance.png"), bbox_inches='tight')
    plt.close()
    print("[done] Comparative feature importances plotted.")

    print("Pipeline completed successfully.")

    if warnings_triggered:
        saw_loky = False
        saw_xgb = False
        saw_lgbm = False

        for msg, _cat in warnings_triggered:
            msg_text = str(msg).lower()
            if ("loky_max_cpu_count" in msg_text) or ("physical cores" in msg_text) or ("loky" in msg_text):
                saw_loky = True
            if "use_label_encoder" in msg_text or "xgboost" in msg_text:
                saw_xgb = True
            if "positive-gain" in msg_text or "positive gain" in msg_text or "min_gain_to_split" in msg_text:
                saw_lgbm = True

        if saw_loky or saw_xgb or saw_lgbm:
            print("Warnings observed are benign:")
            if saw_loky:
                print("- joblib core-count detection: set LOKY_MAX_CPU_COUNT to your logical cores to silence it.")
            if saw_xgb:
                print("- XGBoost use_label_encoder notice: safe to ignore for current training settings.")
            if saw_lgbm:
                print("- LightGBM positive-gain splits: consider increasing min_gain_to_split if desired.")

    # Restore warning handler
    warnings.showwarning = default_showwarning

if __name__ == "__main__":
    main()