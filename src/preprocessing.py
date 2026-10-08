"""
preprocessing.py

Handles CLEANING of the raw dataset (missing values, infinite values) and
building a leakage-safe train/test split + scaling pipeline.

Design decisions (documented here so you can explain them in interviews):

1. Financial ratio datasets like this one very commonly contain infinite
   values. This happens when a ratio's denominator is 0 (e.g. a company
   with zero short-term liabilities makes "current assets / short-term
   liabilities" undefined, which pandas/numpy represent as inf). We treat
   +inf/-inf as missing values (NaN), because "undefined ratio" is
   information-wise closer to "unknown" than to a real numeric value.

2. Some columns in this dataset have a high fraction of missing values.
   Rather than silently imputing a column that is mostly missing (which
   would mean we are mostly making up its values), we DROP any column
   whose missingness exceeds a configurable threshold (default 30%). This
   threshold is applied per-column and is computed AFTER inspecting the
   actual data -- not guessed in advance.

3. For the remaining columns, missing values are imputed using the
   TRAINING SET median only. The same fitted imputer is then applied to
   the test set. This is the standard way to avoid leakage: the test set
   must never influence how we fill in missing values.

4. Scaling (StandardScaler) is likewise fit only on the training data and
   applied to the test data.

Nothing in this file writes to data/raw/ -- it only reads from it and
writes derived output to data/processed/.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import TARGET_COLUMN, RANDOM_SEED, TEST_SIZE, CLEANED_CSV_PATH


# Any column with more than this fraction of missing values (after
# converting inf -> NaN) is dropped instead of imputed. 30% is a common,
# defensible rule of thumb: beyond that, imputation is mostly guessing.
MAX_MISSING_FRACTION = 0.30


def replace_infinite_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Financial ratios can be +inf/-inf when a denominator is zero. Convert
    these to NaN so they are handled by the same missing-value logic as
    genuinely missing values, instead of silently poisoning downstream
    statistics (mean, std, etc. all break in the presence of inf).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.copy()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def drop_high_missingness_columns(
    df: pd.DataFrame, max_missing_fraction: float = MAX_MISSING_FRACTION
) -> tuple:
    """
    Drop feature columns whose fraction of missing values exceeds
    `max_missing_fraction`. The target column is never dropped.

    Returns
    -------
    (cleaned_df, dropped_columns, missingness_report)
        dropped_columns: list of column names that were dropped
        missingness_report: a pandas Series with the missing fraction of
            every original feature column, sorted descending -- useful to
            print/inspect so you can explain the decision.
    """
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    missing_fraction = df[feature_cols].isna().mean().sort_values(ascending=False)

    dropped_columns = missing_fraction[missing_fraction > max_missing_fraction].index.tolist()

    cleaned_df = df.drop(columns=dropped_columns)

    return cleaned_df, dropped_columns, missing_fraction


def clean_dataset(df: pd.DataFrame, save_csv: bool = True) -> pd.DataFrame:
    """
    Full cleaning pipeline (no leakage-sensitive fitting happens here --
    this step only removes bad values/columns, it does not impute or
    scale, because imputation/scaling must be fit on the training split
    only, which happens later in `fit_preprocessor`).

    Steps:
        1. Replace +inf/-inf with NaN.
        2. Drop columns with excessive missingness.
        3. Save the result to data/processed/cleaned_5year.csv for
           inspection (this is a DERIVED file, the raw file is untouched).
    """
    df = replace_infinite_with_nan(df)
    cleaned_df, dropped_columns, missing_report = drop_high_missingness_columns(df)

    print(f"Dropped {len(dropped_columns)} column(s) with >"
          f"{MAX_MISSING_FRACTION * 100:.0f}% missing values: {dropped_columns}")
    remaining_missing = cleaned_df.drop(columns=[TARGET_COLUMN]).isna().mean()
    print(f"Remaining columns still have between "
          f"{remaining_missing.min() * 100:.1f}% and "
          f"{remaining_missing.max() * 100:.1f}% missing values "
          "(these will be median-imputed using training data only).")

    if save_csv:
        cleaned_df.to_csv(CLEANED_CSV_PATH, index=False)
        print(f"Saved cleaned dataset to: {CLEANED_CSV_PATH}")

    return cleaned_df


def split_features_target(df: pd.DataFrame) -> tuple:
    """Split a DataFrame into X (features) and y (target Series)."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def train_test_split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Stratified train/test split. Stratification matters here because the
    dataset is imbalanced -- without it, a small test set could end up
    with very few (or zero) bankruptcy cases, making evaluation unreliable.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def fit_preprocessor(X_train: pd.DataFrame) -> dict:
    """
    Fit the imputer and scaler on TRAINING DATA ONLY.

    Returns a small dict bundling the fitted objects and the exact column
    order used during fitting. This dict is what gets saved to
    models/preprocessor.joblib, and it is the single source of truth for
    how to transform new data (train, test, or a single company record at
    prediction time) consistently.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train)
    scaler.fit(X_train_imputed)

    return {
        "imputer": imputer,
        "scaler": scaler,
        "feature_columns": list(X_train.columns),
    }


def apply_preprocessor(preprocessor: dict, X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply an already-fitted preprocessor (from fit_preprocessor) to any
    DataFrame X. Used for the test set, and reused identically in
    predict.py for new/unseen company records.
    """
    # Make sure columns are in the exact order the preprocessor was fit on.
    X = X[preprocessor["feature_columns"]]

    X_imputed = preprocessor["imputer"].transform(X)
    X_scaled = preprocessor["scaler"].transform(X_imputed)

    return pd.DataFrame(X_scaled, columns=preprocessor["feature_columns"], index=X.index)


if __name__ == "__main__":
    # Quick manual check: `python src/preprocessing.py` from project root.
    from data_loader import load_raw_dataset

    raw_df = load_raw_dataset()
    cleaned_df = clean_dataset(raw_df)
    print("\nCleaned dataset shape:", cleaned_df.shape)
