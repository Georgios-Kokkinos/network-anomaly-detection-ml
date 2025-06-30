"""
main.py

This script is the main entry point for the network anomaly detection application.

It orchestrates the entire machine learning pipeline for anomaly detection in network traffic, including:
- Loading and preprocessing raw CSV data (data directory is automatically detected, no need to edit paths)
- Feature engineering and label encoding
- Visualizing class distributions and proportions
- Splitting data into training and test sets
- Training a machine learning model for anomaly detection
- Evaluating the model and saving all relevant metrics and plots
- Saving results (plots and CSVs) in a dedicated results directory

All outputs are saved in the 'results' folder at the project root for easy access and reporting.
"""

import os
import shutil
from src.data.data_loader import load_csv_files
from src.features.feature_engineering import preprocess_data
from src.models.model import AnomalyDetectionModel
from src.evaluation.evaluate import evaluate_model, plot_confusion_matrix, plot_feature_importance
from src.utils.helpers import plot_class_distribution, plot_class_pie, save_metrics_and_report
from sklearn.model_selection import train_test_split

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

def main():
    prepare_results_dir()

    # Load and preprocess data
    data = load_csv_files(DATA_DIR, NROWS)
    data.columns = data.columns.str.replace('�', '', regex=False)
    data[' Label'] = data[' Label'].str.replace('�', '', regex=False)
    processed_data, label_encoder = preprocess_data(data)

    # Feature/label split
    X = processed_data.iloc[:, :-1]
    y = processed_data.iloc[:, -1]

    # Visualize class distribution and proportions
    class_names = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
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

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = AnomalyDetectionModel()
    model.train(X_train, y_train, N_SAMPLES, TARGET_COUNT)

    # Model evaluation and saving metrics
    y_pred = model.model.predict(X_test)
    target_names = [name.replace('�', '') for name in label_encoder.inverse_transform(sorted(processed_data['Label'].unique()))]
    evaluation_results = evaluate_model(y_test, y_pred)
    save_metrics_and_report(evaluation_results, y_test, y_pred, target_names, RESULTS_DIR)

    # Save evaluation plots
    plot_confusion_matrix(
        y_test, y_pred, classes=target_names,
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"), normalize=False
    )
    plot_confusion_matrix(
        y_test, y_pred, classes=target_names,
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix_normalized.png"), normalize=True
    )
    plot_feature_importance(
        model.model, X.columns,
        save_path=os.path.join(RESULTS_DIR, "feature_importance.png")
    )

    # Optionally display result images (comment out if not needed)
    from PIL import Image
    for img_file in [
        "class_distribution.png",
        "class_proportion.png",
        "confusion_matrix.png",
        "feature_importance.png",
        "confusion_matrix_normalized.png"
    ]:
        img_path = os.path.join(RESULTS_DIR, img_file)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img.show()

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()