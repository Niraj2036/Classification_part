# Classification Workflow

## Overview

This repository implements a comprehensive machine learning workflow for preprocessing, feature selection, and classification of a dataset.

## Workflow Steps

### 1. Preprocessing
1. **Handle Missing Values:** Missing numeric values are filled with the median of the column.
2. **Handle Non-Numeric Values:** Attempts to convert object-type columns to numeric; if unsuccessful, they remain unchanged.
3. **Drop High Cardinality Columns:** Removes columns with too many unique categorical values (more than `max_unique_values`).
4. **Encode Categorical Variables:** Binary variables are factorized, while others are one-hot encoded.
5. **Remove Outliers:** Uses Z-score to identify and remove rows with more than 20% of values as outliers (threshold > 3).

### 2. Feature Selection
1. **Initial Model Training:** Trains a Random Forest classifier to assess feature importance.
2. **Drop Least Important Features:** Iteratively removes features with the lowest importance, retraining the model until the change in accuracy is below a specified threshold.

### 3. Model Selection
1. **Initial Model Pool:** Considers six classifiers:
    - **Logistic Regression**
    - **GaussianNB**
    - **KNeighborsClassifier**
    - **RandomForestClassifier**
    - **SVC**
    - **XGBoostClassifier**
2. **Rejection Criteria:**
    - **Dataset Size:** Removes Logistic Regression, GaussianNB, KNeighbors if the dataset has >= 1000 samples.
    - **Dimensionality:** Removes Logistic Regression, GaussianNB, KNeighbors for high-dimensional data (features > 100).
    - **Linearity:** Removes Logistic Regression if mean correlation < 0.5.
    - **Sparsity:** Removes Random Forest and XGBoost if sparsity > 0.7.
    - **Class Imbalance:** Removes Logistic Regression, GaussianNB, KNeighbors if class imbalance is severe (class < 10%).
    - **Categorical Columns:** Removes Logistic Regression, SVC if categorical columns > 10.
    - **Continuous Columns:** Removes KNeighbors if continuous columns > 10.

### 4. Model Evaluation
1. **Train-Test Split:** Splits the data into training and testing sets.
2. **Feature Scaling:** Standardizes features.
3. **Evaluate Models:** Trains and evaluates each model, selecting the one with the highest accuracy.

### 5. Visualization
1. **Class Distribution Pie Charts:** Displays class distributions before and after preprocessing.
2. **Outlier Pie Charts:** Shows the percentage of outliers per class.
3. **UMAP Scatter Plots:**
    - **Before Preprocessing:** Visualizes the dataset in 2D, colored by class.
    - **After Preprocessing:** Visualizes the cleaned dataset in 2D, colored by class.
4. **Pair Plots:**
    - **Before Preprocessing:** Shows pairwise relationships in the dataset.
    - **After Preprocessing:** Shows pairwise relationships in the cleaned dataset.
5. **3D Scatter Plots:**
    - **Before Preprocessing:** Visualizes high-dimensional data in 3D using UMAP, colored by class.
    - **After Preprocessing:** Visualizes cleaned data in 3D using UMAP, colored by class.

