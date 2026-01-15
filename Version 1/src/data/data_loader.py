"""
data_loader.py

This module provides utility functions for loading and performing initial checks on network traffic CSV data files.

Functionality:
- Loads and merges all CSV files from a specified directory into a single pandas DataFrame.
- Optionally limits the number of rows read from each file for memory efficiency.
- Provides a function to perform initial data checks (shape, columns, missing values, data types).

These utilities are intended to streamline the data ingestion and exploration process for the network anomaly detection pipeline.
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