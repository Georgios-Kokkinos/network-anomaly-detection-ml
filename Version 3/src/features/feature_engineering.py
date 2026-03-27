"""
feature_engineering.py

This module provides functions for preprocessing and feature engineering on network traffic data.
It includes:
- Handling missing values and duplicates
- Encoding categorical features (including the target label)
- Coercing numeric-looking object columns into numeric types
- Cleaning infinite values introduced during conversion
- Scaling numerical features for model compatibility

These steps prepare raw network data for machine learning models and reduce common data-quality issues.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    """
    Preprocess the input DataFrame by handling null values, removing duplicates,
    and encoding categorical features.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing network traffic data.

    Returns:
    tuple: (pd.DataFrame, LabelEncoder) The preprocessed DataFrame and the fitted label encoder.
    """
    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()
    # Drop rows with null values and duplicates
    df = df.dropna()
    df = df.drop_duplicates()

    # Encode the target label
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])

    # Try to coerce object columns that look numeric into numeric types
    obj_cols = df.select_dtypes(include=['object']).columns.drop('Label', errors='ignore')
    for col in obj_cols:
        # remove common thousands separators and whitespace then attempt conversion
        coerced = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
        # if conversion succeeded for a majority of rows, keep numeric conversion
        if coerced.notna().sum() >= 0.5 * len(coerced):
            df[col] = coerced

    # Encode remaining categorical (object) columns with a per-column LabelEncoder
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))

    # Replace infinite values with NA and drop any new missing values
    df = df.replace([float('inf'), float('-inf')], pd.NA)
    df = df.dropna()

    return df, label_encoder

def scale_features(df):
    """
    Scale the numerical features of the DataFrame using StandardScaler.

    Parameters:
    df (pd.DataFrame): The input DataFrame with features to scale.

    Returns:
    pd.DataFrame: The scaled DataFrame.
    """
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def feature_engineering_pipeline(df):
    """
    Execute the feature engineering pipeline on the input DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing network traffic data.

    Returns:
    pd.DataFrame: The final DataFrame after preprocessing and scaling.
    """
    df, _ = preprocess_data(df)
    df = scale_features(df)
    return df