# Tugas-Besar-Kecerdasan-Buatan - STudent Performance Analysis
Dokumentasi Tugas Besar Kecerdasan Buatan TK-47-04 Kelompok 8, dengan Anggota
- Muhammad Raffi Ibrahim
- Muh. Abdi Izzan Ismail
- Muhammad Irhan Fadil Hapidi


# Student Performance Prediction — Baseline Linear Regression

A from-scratch implementation of linear regression using gradient descent to predict student academic performance, with full EDA, preprocessing, training, and evaluation pipeline.

---

## Overview

This project builds a **baseline linear regression model** without using scikit-learn's built-in regression estimators. It demonstrates the full machine learning workflow:

1. Load raw dataset from a remote URL
2. Exploratory Data Analysis (EDA)
3. Data preprocessing & encoding
4. Train/test split
5. Manual gradient descent training
6. Model evaluation with multiple metrics
7. Coefficient analysis & regression equation

---

## Dataset

- **Source**: [Mendeley Data — Student Performance Dataset](https://data.mendeley.com/public-files/datasets/5b82ytz489/files/21461ab8-2eb2-4768-9551-5636024f2989/file_downloaded)
- **Target column**: `Overall` (student's overall academic score)
- The dataset is fetched directly from the URL at runtime — no manual download needed.

---

## Requirements

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy requests
```

| Library | Purpose |
|---|---|
| `numpy` | Matrix math & gradient descent |
| `pandas` | Data loading & manipulation |
| `matplotlib` / `seaborn` | Visualizations |
| `scikit-learn` | Label encoding & evaluation metrics |
| `scipy` | Q-Q plot for residual normality check |
| `requests` | Fetching dataset from URL |

---

## Usage

Run the script directly:

```bash
python baseline_linear_regression.py
```

The script will automatically:
- Download the dataset
- Run EDA and display plots
- Train the model
- Print evaluation metrics
- Show coefficient analysis

---

## Configuration

Hyperparameters and settings are defined at the top of the file:

```python
TARGET_COLUMN  = "Overall"   # Column to predict
LEARNING_RATE  = 0.01        # Gradient descent step size
N_ITERATIONS   = 1000        # Training iterations
RANDOM_SEED    = 42          # Reproducibility seed
```

---

## Model Details

The `BaselineLinearRegression` class implements:

- **Forward pass**: `ŷ = X·W + b`
- **Loss function**: Mean Squared Error (MSE/2)
- **Gradient updates**:
  - `dW = (Xᵀ · error) / n`
  - `db = sum(error) / n`
- **Input normalization**: Z-score standardization applied before training to prevent gradient overflow

---

## Pipeline Steps

### 1. Load Data
Fetches the CSV from Mendeley Data using a browser-like User-Agent header.

### 2. Exploratory Data Analysis
- Dataset shape, dtypes, and `.describe()` statistics
- Missing value report
- Duplicate detection and removal
- Target variable distribution (histogram + boxplot)
- Correlation heatmap across all features
- Top 10 features correlated with the target

### 3. Preprocessing
- Drops rows with missing values
- Label-encodes all categorical columns
- Splits data 80/20 into train and test sets (shuffled)

### 4. Training
- Normalizes input features using training set mean/std
- Runs gradient descent for `N_ITERATIONS` epochs
- Logs loss every 200 iterations
- Detects and halts on NaN/Inf overflow

### 5. Evaluation

Reports the following metrics on the test set:

| Metric | Description |
|---|---|
| R² Score | Proportion of variance explained |
| Adjusted R² | R² penalized for number of features |
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |

Generates six plots:
- Training loss curve
- Actual vs. Predicted scatter
- Residual plot
- Residual distribution histogram
- Q-Q normality plot
- Absolute error distribution

### 6. Coefficient Analysis
Ranks features by the absolute magnitude of their learned coefficients, indicating their relative influence on the prediction.

### 7. Regression Equation
Prints the full linear equation in the form:

```
Ŷ = b
    + (w₁ × Feature₁)
    + (w₂ × Feature₂)
    ...
```

---

## Output Example

```
================================================================================
BASELINE LINEAR REGRESSION - STUDENT PERFORMANCE PREDICTION
================================================================================

Loading dataset...
✓ Dataset loaded: XXXX rows, XX columns

...

Performance Metrics:
--------------------------------------------------------------------------------
R² Score        : 0.XXXX
Adjusted R²     : 0.XXXX
MAE             : X.XXXX
MSE             : X.XXXX
RMSE            : X.XXXX
```

---

## Project Structure

```
.
└── baseline_linear_regression.py   # Main script (model + pipeline)
```

---

## Notes

- This implementation uses **only NumPy for math** — no `sklearn.linear_model` is used for the actual regression.
- scikit-learn is only used for `LabelEncoder` and metric functions (`r2_score`, `mean_absolute_error`, `mean_squared_error`).
- The model applies Z-score normalization internally; you do not need to preprocess data before calling `train()`.

---

## License

This project is for educational purposes. Dataset is provided by Mendeley Data under their respective terms of use.
