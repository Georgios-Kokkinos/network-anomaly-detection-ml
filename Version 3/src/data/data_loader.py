"""
data_loader.py

Data ingestion utilities for Version 3 (UNSW-NB15).

Supported input layouts in data/:
1) Separate features + labels files (preferred):
   - Data.csv
   - Label.csv
2) A single labeled CSV (e.g., CICFlowMeter_out.csv) that already includes a label column.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _find_label_column(columns: list[str]) -> str | None:
    """Best-effort detection of a label column name.

    Parameters:
    columns (list[str]): Available column names.

    Returns:
    str | None: Detected label column or None.
    """
    normalized = {c.lower().strip(): c for c in columns}
    for candidate in ["label", "attack_cat", "attack", "class", "target", "y"]:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _load_data_plus_label(data_path: Path, label_path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Load UNSW features and labels from separate files and merge them.

    Parameters:
    data_path (Path): Path to Data.csv.
    label_path (Path): Path to Label.csv.
    nrows (int | None): Optional row cap.

    Returns:
    pd.DataFrame: Merged dataframe with a final Label column.
    """
    x_df = pd.read_csv(data_path, nrows=nrows, low_memory=False)
    y_df = pd.read_csv(label_path, nrows=nrows, low_memory=False)

    if y_df.shape[1] == 1:
        label_series = y_df.iloc[:, 0]
    else:
        detected = _find_label_column(list(y_df.columns))
        label_series = y_df[detected] if detected else y_df.iloc[:, -1]

    if len(x_df) != len(label_series):
        raise ValueError(
            f"Data.csv rows ({len(x_df)}) and Label.csv rows ({len(label_series)}) do not match"
        )

    merged = x_df.copy()
    merged["Label"] = label_series.values
    return merged


def load_csv_files(data_directory: str, nrows: int | None = None) -> pd.DataFrame:
    """Load UNSW-NB15 data from data/ and return a single dataframe.

    Parameters:
    data_directory (str): Directory containing UNSW files.
    nrows (int | None): Optional row cap for faster experiments.

    Returns:
    pd.DataFrame: Dataset with features and a Label column.
    """
    data_dir = Path(data_directory)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_directory}")

    data_csv = data_dir / "Data.csv"
    label_csv = data_dir / "Label.csv"
    if data_csv.exists() and label_csv.exists():
        return _load_data_plus_label(data_csv, label_csv, nrows=nrows)

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in data/. Expected at least Data.csv + Label.csv or a labeled CSV."
        )

    frames: list[pd.DataFrame] = []
    for file_path in csv_files:
        df = pd.read_csv(file_path, nrows=nrows, low_memory=False)
        label_col = _find_label_column(list(df.columns))
        if label_col:
            df = df.rename(columns={label_col: "Label"})
            frames.append(df)

    if not frames:
        raise ValueError(
            "Could not detect a label column in provided CSV files."
            " Use Data.csv + Label.csv or include a label-like column (Label/attack_cat/class)."
        )

    return pd.concat(frames, ignore_index=True)

def initial_data_checks(df):
    """
    Perform initial checks on the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to check.

    Returns:
    None
    """
    print("Initial Data Checks:")
    print(f"Shape of the DataFrame: {df.shape}")
    print(f"Columns in the DataFrame: {df.columns.tolist()}")
    print(f"Missing values in each column:\n{df.isnull().sum()}")
    print(f"Data types of each column:\n{df.dtypes}")

# Example usage:
# data_directory = 'path_to_your_data_directory'
# data = load_csv_files(data_directory)
# initial_data_checks(data)