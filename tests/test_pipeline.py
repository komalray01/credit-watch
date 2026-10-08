"""
test_pipeline.py

A lightweight sanity-check script (not a full pytest suite, per the
project's "keep it simple" principle) that verifies the pipeline actually
works end-to-end. Run it after downloading the dataset and training:

    python tests/test_pipeline.py

It checks:
    1. The raw dataset loads and has the expected target column.
    2. The target only contains 0/1 values.
    3. Cleaning removes inf values and does not silently drop the target.
    4. Feature engineering only adds columns, never removes existing ones.
    5. The preprocessor produces no NaNs and the right column count.
    6. (If models have been trained) saved models reload correctly and
       produce probabilities in [0, 1].

Each check prints PASS/FAIL. The script exits with a non-zero status code
if anything fails, so it can be used as a simple CI gate too.
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for path in (PROJECT_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import TARGET_COLUMN, XGBOOST_PATH, PREPROCESSOR_PATH


failures = []


def check(condition: bool, description: str):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}")
    if not condition:
        failures.append(description)


def main():
    from data_loader import load_raw_dataset
    from preprocessing import (
        clean_dataset,
        split_features_target,
        train_test_split_data,
        fit_preprocessor,
        apply_preprocessor,
    )
    from feature_engineering import engineer_features

    # 1. Raw data loads and has the target column.
    try:
        raw_df = load_raw_dataset()
        check(TARGET_COLUMN in raw_df.columns, "Raw dataset contains the target column")
    except FileNotFoundError as e:
        print(f"[FAIL] Could not load raw dataset: {e}")
        failures.append("Raw dataset loading")
        print(f"\n{len(failures)} check(s) failed. Download the dataset first.")
        sys.exit(1)

    # 2. Target only contains 0/1.
    unique_targets = set(raw_df[TARGET_COLUMN].unique().tolist())
    check(unique_targets.issubset({0, 1}), "Target column only contains values 0 and 1")

    # 3. Cleaning removes inf and keeps the target.
    cleaned_df = clean_dataset(raw_df, save_csv=False)
    has_inf = np.isinf(cleaned_df.drop(columns=[TARGET_COLUMN]).select_dtypes(include=[np.number])).values.any()
    check(not has_inf, "Cleaned dataset contains no infinite values")
    check(TARGET_COLUMN in cleaned_df.columns, "Target column survives cleaning")

    # 4. Feature engineering only adds columns.
    features_df = engineer_features(cleaned_df, save_csv=False)
    original_cols = set(cleaned_df.columns)
    new_cols = set(features_df.columns)
    check(original_cols.issubset(new_cols), "Feature engineering did not remove any existing column")
    check(len(new_cols) >= len(original_cols), "Feature engineering added zero or more columns (never negative)")

    # 5. Preprocessor produces no NaNs.
    X, y = split_features_target(features_df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    preprocessor = fit_preprocessor(X_train)
    X_train_processed = apply_preprocessor(preprocessor, X_train)
    X_test_processed = apply_preprocessor(preprocessor, X_test)
    check(not X_train_processed.isna().any().any(), "Processed training data contains no NaNs")
    check(not X_test_processed.isna().any().any(), "Processed test data contains no NaNs")
    check(
        X_train_processed.shape[1] == X_test_processed.shape[1],
        "Train and test have the same number of columns after preprocessing",
    )

    # 6. Saved models reload correctly, if they exist.
    if os.path.exists(XGBOOST_PATH) and os.path.exists(PREPROCESSOR_PATH):
        import joblib

        model = joblib.load(XGBOOST_PATH)
        saved_preprocessor = joblib.load(PREPROCESSOR_PATH)
        X_test_via_saved = apply_preprocessor(saved_preprocessor, X_test)
        probabilities = model.predict_proba(X_test_via_saved)[:, 1]
        check(
            bool(((probabilities >= 0) & (probabilities <= 1)).all()),
            "Reloaded XGBoost model produces valid probabilities in [0, 1]",
        )
    else:
        print(
            "[SKIP] Saved model/preprocessor not found -- run "
            "`python src/train_models.py` to include model-reload checks."
        )

    print(f"\n{len(failures)} check(s) failed out of the checks that ran.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
