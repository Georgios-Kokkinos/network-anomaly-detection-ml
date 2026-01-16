"""
model.py

This module defines the AnomalyDetectionModel class for network anomaly detection.
It provides:
- Model selection for RandomForest, XGBoost, and LightGBM classifiers.
- Methods for handling class imbalance (downsampling majority and upsampling minority classes).
- Model training, saving, and loading utilities.

The class helps balance datasets where benign traffic dominates and keeps model wiring centralized for the pipeline.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

class AnomalyDetectionModel:
    def __init__(self, model_type="rf", random_state=42):
        """
        Initialize the model.

        model_type: 'rf' (RandomForest), 'xgboost'/'xgb' (XGBoost), or 'lightgbm'/'lgb' (LightGBM).
        """
        self.model_type = model_type
        self.random_state = random_state

        if model_type in ("rf", "random_forest"):
            from sklearn.ensemble import RandomForestClassifier

            self.model = RandomForestClassifier(random_state=random_state)
        elif model_type in ("xgboost", "xgb"):
            try:
                from xgboost import XGBClassifier
            except Exception as e:
                raise ImportError("xgboost is not installed. Install with 'pip install xgboost'.") from e

            # `use_label_encoder=False` and `eval_metric` avoid deprecation warnings
            self.model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="mlogloss")
        elif model_type in ("lightgbm", "lgb"):
            try:
                from lightgbm import LGBMClassifier
            except Exception as e:
                raise ImportError("lightgbm is not installed. Install with 'pip install lightgbm'.") from e
            # Slightly more expressive defaults for multiclass, with basic regularization
            self.model = LGBMClassifier(
                random_state=random_state,
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multiclass",
                class_weight="balanced",
                verbosity=-1,  # suppress repeated info/warning spam in stdout
            )
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'rf', 'xgboost' or 'lightgbm'.")
    
    def train(self, X, y, n_samples=100000, target_count=20000):
        """
        Train the selected model after handling class imbalance.

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

        # 1. Downsample the majority class to n_samples (if provided)
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

        # 2. Upsample minority classes to target_count (if needed)
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