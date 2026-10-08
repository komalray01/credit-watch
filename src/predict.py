"""
predict.py

Applies the saved preprocessing pipeline + XGBoost model to new company
records (or existing test-set rows, for demo purposes in the Streamlit
app) and converts the predicted probability into a project-defined risk
category.

CRITICAL: this file must use the EXACT SAME preprocessing steps (same
imputer, same scaler, same feature column order) as training. That is why
it loads models/preprocessor.joblib instead of re-fitting anything -- a
mismatch here would silently make predictions meaningless (train/serve
skew).
"""

import os
import sys

import joblib
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (
    XGBOOST_PATH,
    PREPROCESSOR_PATH,
    RISK_LOW_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
)
from preprocessing import apply_preprocessor
from feature_engineering import engineer_features


def load_prediction_artifacts():
    """Load the trained XGBoost model and the fitted preprocessor."""
    model = joblib.load(XGBOOST_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def probability_to_risk_category(probability: float) -> str:
    """
    Convert a predicted probability of bankruptcy into one of three
    PROJECT-DEFINED risk categories (NOT an official/industry-standard
    scale -- see README "Risk Score" section for how these thresholds
    were chosen):

        probability < RISK_LOW_THRESHOLD                -> "Low Risk"
        RISK_LOW_THRESHOLD <= probability < RISK_MEDIUM_THRESHOLD
                                                          -> "Medium Risk"
        probability >= RISK_MEDIUM_THRESHOLD             -> "High Risk"
    """
    if probability < RISK_LOW_THRESHOLD:
        return "Low Risk"
    elif probability < RISK_MEDIUM_THRESHOLD:
        return "Medium Risk"
    else:
        return "High Risk"


def predict_for_companies(raw_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full prediction pipeline for one or more companies.

    Parameters
    ----------
    raw_features_df : pd.DataFrame
        A DataFrame with the SAME raw columns produced by
        feature_engineering.engineer_features() (i.e. the cleaned Attr*
        columns plus the engineered ratio columns). This function does
        NOT re-run cleaning (missing-value/column-dropping) because that
        step is dataset-level, not something you'd redo per prediction --
        in the Streamlit app, rows are taken directly from the already
        cleaned + engineered feature table.

    Returns
    -------
    pd.DataFrame
        Same rows, with two new columns: "predicted_probability" and
        "risk_category".
    """
    model, preprocessor = load_prediction_artifacts()

    X_processed = apply_preprocessor(preprocessor, raw_features_df)
    probabilities = model.predict_proba(X_processed)[:, 1]

    result_df = raw_features_df.copy()
    result_df["predicted_probability"] = probabilities
    result_df["risk_category"] = [probability_to_risk_category(p) for p in probabilities]

    return result_df


if __name__ == "__main__":
    # Quick manual check: predict on a handful of rows from the processed
    # feature table (data/processed/features_5year.csv must already exist
    # -- run src/feature_engineering.py or src/train_models.py first).
    from config import FEATURES_CSV_PATH, TARGET_COLUMN

    sample_df = pd.read_csv(FEATURES_CSV_PATH).drop(columns=[TARGET_COLUMN]).head(5)
    predictions = predict_for_companies(sample_df)
    print(predictions[["predicted_probability", "risk_category"]])
