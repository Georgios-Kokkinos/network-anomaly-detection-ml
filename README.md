# Network Anomaly Detection - Version 1 Checkpoint

**Date:** 30/6/2025

---

## Project Overview

This project implements a machine learning pipeline for network anomaly detection using the [CICIDS 2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html). The pipeline automates data ingestion, preprocessing, feature engineering, class balancing, model training, evaluation, and result visualization. All outputs (plots, metrics, reports) are saved in a dedicated `results` folder at the project root.

---

## What Has Been Done So Far

1. **Data Loading**
    - All CSV files from the `data` directory are loaded and merged into a single DataFrame.
    - Initial data checks are performed (shape, columns, missing values, dtypes).

2. **Preprocessing & Feature Engineering**
    - Leading/trailing spaces and special characters are removed from column names and labels.
    - Rows with missing values and duplicates are dropped.
    - All categorical features (including the target label) are encoded using `LabelEncoder`.
    - Infinite values are replaced with NA and dropped.
    - Numerical features can be scaled using `StandardScaler` (pipeline ready).

3. **Class Balancing**
    - The majority class is downsampled to a configurable maximum (`N_SAMPLES`).
    - Minority classes are upsampled using SMOTE to a configurable target (`TARGET_COUNT`).

4. **Model Training**
    - A `RandomForestClassifier` is trained on the balanced dataset.
    - The model is evaluated on a held-out test set (20% split).

5. **Evaluation & Visualization**
    - Metrics (accuracy, precision, recall, F1-score) are calculated and saved as CSV.
    - A detailed classification report is saved as CSV.
    - Confusion matrices (raw and normalized) and feature importance plots are generated and saved.
    - Class distribution and class proportion plots are generated and saved.
    - All results are stored in the `results` folder at the project root.

6. **Code Organization**
    - The codebase is modular: `main.py` orchestrates the pipeline, with helpers for plotting and exporting, and separate modules for data loading, feature engineering, model, and evaluation.
    - All plotting and reporting helpers are in `src/utils/helpers.py`.

---

## Current Problem / Limitation

- **Performance Plateau:** Despite increasing the amount of data loaded and adjusting class balancing parameters, the model's confusion matrix and overall performance metrics do not improve. The results are stable and high, but do not reflect further gains from more data or balancing.

**Possible Causes:**
- The current feature engineering approach may not be extracting new or more informative patterns from the data.
- Using the same `LabelEncoder` for all categorical features (including the target) may cause category collisions and limit model learning.
- The model may be overfitting to a small set of highly predictive features, or the features may be too redundant or correlated.
- The test set may not be sufficiently challenging or diverse.

---

## What We Intend To Do Next (Version 2 Plan)

1. **Improve Categorical Feature Encoding**
    - Use a separate `LabelEncoder` for each categorical feature (except the target label) to avoid category collisions and improve feature representation.

2. **Feature Selection**
    - Analyze feature importance and remove features with very low importance.
    - Check for and remove highly correlated features to reduce redundancy.

3. **Feature Transformation**
    - Apply log or other transformations to highly skewed numerical features to improve model learning.

4. **Create New Features**
    - Engineer new features by combining existing ones (e.g., ratios, differences, aggregations) to provide the model with more informative inputs.

5. **Try Alternative Encodings**
    - Experiment with one-hot encoding or target encoding for categorical features with many unique values.

6. **Evaluation Improvements**
    - Use cross-validation to better assess model generalization.
    - Try different random seeds for train/test splits to ensure robustness.

7. **Documentation and Versioning**
    - Continue to checkpoint each major change, documenting the rationale, implementation, and observed effects in a new version folder.

---

## How to Use This Version

- This folder contains a snapshot of the code (`src/`) and results (`results/`) as of this checkpoint.
- To restore this version, copy the contents of this folder back to the project root.
- Refer to this file for a summary of the project state, known issues, and planned improvements.

---

## How to Run

1. **Download the CICIDS 2017 dataset CSVs:**

   - **Official site:**  
     [CICIDS 2017 Dataset Download Page](https://www.unb.ca/cic/datasets/ids-2017.html)  
     Direct CSVs: [MachineLearningCSV.zip](http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs)  
     > ⚠️ **Note:** The direct download link may show a "doesn't support a secure connection" warning in your browser. If prompted, choose "Continue to site" to access the file.

   - **Alternate Google Drive mirror:**  
     [Google Drive Folder](https://drive.google.com/drive/folders/10BsMt0t-tgch2o9nUeJB5eHMIK9XmNkX?usp=sharing)

   - **Instructions:**  
     Download and extract `MachineLearningCSV.zip`. Place all the extracted `.csv` files into the `data` folder at the project root.

2. **Install dependencies:**  
   Open a terminal in the project root (`Version 1`) and run:  
   ```
   pip install -r requirements.txt
   ```

3. **Run the pipeline:**  
   ```
   python -m src.main
   ```

---

## Notes

- **Data folder included, but empty:**  
  The `data` folder is included in this repository, but does not contain any CSV files due to dataset size. Please download the CICIDS CSV files as described above and place them in the `data` folder at the project root.
- **Results folder included:**  
  The `results` folder is included for reference. It contains example output plots and small result CSVs generated by the pipeline.
- **Dependencies:**  
  All required Python packages are listed in `requirements.txt`.

---
