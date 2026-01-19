"""
data_loader.py

Load and merge CICIDS2017 CSVs into a single dataframe.
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


def load_cicids_data(
    data_dir: Path,
    max_rows_per_file: int | None = None,
    sample_fraction: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load and concatenate CICIDS2017 CSVs.

    Supports optional row limits and fractional sampling to control memory use.

    Parameters:
    data_dir (Path): Directory containing CICIDS2017 CSVs.
    max_rows_per_file (int | None): Optional max rows per CSV file.
    sample_fraction (float): Fraction of rows to sample after concatenation.
    random_state (int): Random seed for sampling.

    Returns:
    pd.DataFrame: Concatenated (and optionally sampled) dataset.
    """
    files = list(_csv_files(data_dir))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs: list[pd.DataFrame] = []
    # Read each CSV and append to the list.
    for file_path in files:
        df = pd.read_csv(file_path, low_memory=False, nrows=max_rows_per_file)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # Optional downsampling for faster experiments.
    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction, random_state=random_state)

    return data
