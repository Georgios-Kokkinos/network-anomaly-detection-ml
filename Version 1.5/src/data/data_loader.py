"""
data_loader.py

Data ingestion utilities for Version 1.5. Responsibilities:
- Load and concatenate all CICIDS CSVs from a directory into one DataFrame (optional row cap for memory safety).
- Offer quick initial checks (shape, columns, missing values, dtypes) so we can validate inputs before modeling.

These helpers keep the main pipeline focused on training and evaluation while standardizing how raw files are read.
"""

import pandas as pd
import os

def load_csv_files(data_directory, nrows=None):
    """
    Load and merge CSV files from the specified directory into a single DataFrame.

    Parameters:
    data_directory (str): The path to the directory containing the CSV files.
    nrows (int, optional): Number of rows to read from each CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the merged data from all CSV files.
    """
    all_data = []
    # Iterate through all files in the directory
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_directory, filename)
            try:
                # Read the CSV file into a DataFrame (limit rows if nrows is set)
                df = pd.read_csv(file_path, nrows=nrows)
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Concatenate all DataFrames into a single DataFrame
    merged_data = pd.concat(all_data, ignore_index=True)
    return merged_data

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