"""
model.py

This module defines the AnomalyDetectionModel class for network anomaly detection.
It provides:
- A Random Forest-based model for classification.
- Methods for handling class imbalance (downsampling majority and upsampling minority classes).
- Model training, saving, and loading utilities.

The class ensures robust training by balancing the dataset, which is crucial for real-world network data where benign traffic often dominates. The model and utilities are designed for easy integration into a machine learning pipeline.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

class AnomalyDetectionModel:
    def __init__(self, random_state=42):
        # Initialize the Random Forest classifier with a fixed random state for reproducibility
        self.model = RandomForestClassifier(random_state=random_state)
    
    def train(self, X, y, n_samples=100000, target_count=20000):
        """
        Train the Random Forest model after handling class imbalance.

        Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.
        """
        X_resampled, y_resampled = self.handle_class_imbalance(X, y, n_samples=n_samples, target_count=target_count)
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        # Optionally, you can print or log evaluation metrics here if needed
    
    def handle_class_imbalance(self, X, y, n_samples=None, target_count=None):
        """
        Balance the dataset by downsampling the majority class and upsampling minority classes using SMOTE.

        Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.

        Returns:
        tuple: (X_resampled, y_resampled) - Balanced feature matrix and labels.
        """
        from sklearn.utils import resample

        # Combine X and y for easier manipulation
        df = pd.concat([X, y.rename("Label")], axis=1)
        class_counts = df["Label"].value_counts()

        # 1. Downsample the majority class (e.g., to 5000 samples)
        majority_class = class_counts.idxmax()
        df_majority = df[df["Label"] == majority_class]
        df_minority = df[df["Label"] != majority_class]
        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=n_samples,  # Set your desired max for the majority class
            random_state=42
        )
        df_balanced = pd.concat([df_majority_downsampled, df_minority])

        X_balanced = df_balanced.drop("Label", axis=1)
        y_balanced = df_balanced["Label"]

        # 2. Upsample minority classes to a target count (e.g., 500)
        target_count = target_count
        class_counts = y_balanced.value_counts()
        sampling_strategy = {cls: target_count for cls, count in class_counts.items() if count < target_count}
        if sampling_strategy:
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_balanced, y_balanced)
        else:
            X_resampled, y_resampled = X_balanced, y_balanced

        return X_resampled, y_resampled
    
    def save_model(self, filename):
        """
        Save the trained model to a file.

        Parameters:
        filename (str): Path to save the model.
        """
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}.")
    
    def load_model(self, filename):
        """
        Load a trained model from a file.

        Parameters:
        filename (str): Path to the saved model.
        """
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}.")