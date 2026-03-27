"""
data_loader.py

Load and merge UNSW-NB15/CIC-UNSW-NB15 files into a single dataframe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _csv_files(data_dir: Path) -> Iterable[Path]:
    """Return all CSV files in the provided directory (sorted).

    Parameters:
    data_dir (Path): Directory containing CSV files.

    Returns:
    Iterable[Path]: Sorted CSV file paths.
    """
    return sorted(data_dir.glob("*.csv"))


def _find_label_column(columns: list[str]) -> str | None:
    """Detect likely label column names for UNSW/CICFlowMeter files.

    Parameters:
    columns (list[str]): Column names.

    Returns:
    str | None: Matching label column if found.
    """
    normalized = {c.lower().strip(): c for c in columns}
    for candidate in ["label", "attack_cat", "attack", "class", "target", "y"]:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _load_data_plus_label(data_dir: Path, max_rows_per_file: int | None = None) -> pd.DataFrame:
    """Load Data.csv + Label.csv and merge into one frame.

    Parameters:
    data_dir (Path): Data directory.
    max_rows_per_file (int | None): Optional row cap.

    Returns:
    pd.DataFrame: Merged dataframe with Label column.
    """
    x_df = pd.read_csv(data_dir / "Data.csv", low_memory=False, nrows=max_rows_per_file)
    y_df = pd.read_csv(data_dir / "Label.csv", low_memory=False, nrows=max_rows_per_file)

    if y_df.shape[1] == 1:
        labels = y_df.iloc[:, 0]
    else:
        label_col = _find_label_column(list(y_df.columns))
        labels = y_df[label_col] if label_col else y_df.iloc[:, -1]

    if len(x_df) != len(labels):
        raise ValueError(
            f"Data.csv rows ({len(x_df)}) and Label.csv rows ({len(labels)}) do not match"
        )

    merged = x_df.copy()
    merged["Label"] = labels.values
    return merged


def load_unsw_data(
    data_dir: Path,
    max_rows_per_file: int | None = None,
    sample_fraction: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load and concatenate UNSW-NB15 CSVs.

    Supports optional row limits and fractional sampling to control memory use.

    Parameters:
    data_dir (Path): Directory containing UNSW/CIC-UNSW files.
    max_rows_per_file (int | None): Optional max rows per CSV file.
    sample_fraction (float): Fraction of rows to sample after concatenation.
    random_state (int): Random seed for sampling.

    Returns:
    pd.DataFrame: Concatenated (and optionally sampled) dataset.
    """
    if (data_dir / "Data.csv").exists() and (data_dir / "Label.csv").exists():
        data = _load_data_plus_label(data_dir, max_rows_per_file=max_rows_per_file)
    else:
        files = list(_csv_files(data_dir))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        dfs: list[pd.DataFrame] = []
        # Read each CSV and keep only labeled ones.
        for file_path in files:
            df = pd.read_csv(file_path, low_memory=False, nrows=max_rows_per_file)
            label_col = _find_label_column(list(df.columns))
            if label_col:
                df = df.rename(columns={label_col: "Label"})
                dfs.append(df)

        if not dfs:
            raise ValueError(
                "Could not detect label column in provided CSVs. "
                "Use Data.csv + Label.csv or include a label-like column (Label/attack_cat/class)."
            )

        data = pd.concat(dfs, ignore_index=True)

    # Optional downsampling for faster experiments.
    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction, random_state=random_state)

    return data


def load_cicids_data(
    data_dir: Path,
    max_rows_per_file: int | None = None,
    sample_fraction: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Backward-compatible alias used by existing pipeline imports."""
    return load_unsw_data(
        data_dir,
        max_rows_per_file=max_rows_per_file,
        sample_fraction=sample_fraction,
        random_state=random_state,
    )
