"""
preprocess.py

Data cleaning and preprocessing:
- sanitize feature/label names
- coerce numeric features
- handle inf/NaN
- encode labels
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.helpers import sanitize_text


def preprocess_dataframe(
    df: pd.DataFrame, label_col: str
) -> tuple[np.ndarray, np.ndarray, list[str], LabelEncoder]:
    """
    Clean the raw dataframe, encode labels, and return numeric features.

    Steps:
    - sanitize column names
    - split label column
    - coerce features to numeric
    - handle inf/NaN
    - label encode targets

    Parameters:
    df (pd.DataFrame): Raw input dataframe.
    label_col (str): Name of the label column.

    Returns:
    tuple: (X, y, feature_names, label_encoder)
    """
    df = df.copy()
    df.columns = [sanitize_text(c.strip()) for c in df.columns]

    # Validate label column.
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data")

    # Guard against accidental target leakage if multiple label-like columns exist.
    leakage_candidates = {"label", "attack_cat", "attack", "class", "target", "y"}
    keep_label_norm = sanitize_text(label_col).lower()
    drop_cols: list[str] = []
    for col in df.columns:
        if col == label_col:
            continue
        col_norm = sanitize_text(col).lower()
        if col_norm in leakage_candidates and col_norm != keep_label_norm:
            drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Extract and sanitize label values.
    y_raw = df[label_col].astype(str).map(sanitize_text)
    X_df = df.drop(columns=[label_col])

    # Coerce all features to numeric
    # Convert all features to numeric values.
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    # Replace inf with NaN and fill missing values
    # Replace inf with NaN and fill missing values with median.
    X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    feature_names = [sanitize_text(c) for c in X_df.columns]

    # Encode class labels to integers.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Convert to dense float32 arrays for TensorFlow.
    X = X_df.values.astype(np.float32)

    return X, y, feature_names, label_encoder
