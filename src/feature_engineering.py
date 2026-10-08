"""
feature_engineering.py

IMPORTANT CONTEXT ABOUT THIS DATASET:

Unlike a "raw" financial statement dataset (which would have line items
like "total assets", "cash", "current liabilities" in currency units), the
UCI Polish Companies Bankruptcy Dataset ships with 64 attributes that are
ALREADY pre-computed financial ratios (Attr1, Attr2, ..., Attr64). We do
not have access to the underlying raw balance-sheet/income-statement
figures, only these ratios.

This means genuine "feature engineering" here cannot mean recomputing new
ratios from dollar-value line items (those don't exist in this dataset).
Instead, we do the honest, defensible version of feature engineering that
IS possible: combining a small number of EXISTING, well-understood ratio
columns into a few higher-level composite indicators that are easier for
an analyst to interpret at a glance.

The meanings of Attr1, Attr2, Attr4, Attr7, and Attr10 below are the
definitions publicly documented for this dataset (Zięba, Tomczak &
Tomczak, 2016, "Ensemble Boosted Trees with Synthetic Features Generation
in Application to Bankruptcy Prediction", and the accompanying UCI
documentation). Before running this on your actual downloaded copy of the
dataset, double check these definitions against the UCI attribute list --
if anything doesn't match, do NOT blindly trust this docstring.

    Attr1  = net profit / total assets                (profitability)
    Attr2  = total liabilities / total assets          (leverage)
    Attr4  = current assets / short-term liabilities   (liquidity, "current ratio")
    Attr7  = EBIT / total assets                       (operating profitability)
    Attr10 = equity / total assets                     (capital structure)

Every function below FIRST checks that its required source column(s) are
actually present in the DataFrame before creating anything. If a required
column was dropped earlier (e.g. due to excessive missingness), the
corresponding engineered feature is simply skipped -- we never invent a
column that doesn't exist.
"""

import os
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import TARGET_COLUMN, FEATURES_CSV_PATH

# Small epsilon used to avoid division-by-zero when a ratio is used as a
# denominator in an engineered feature. This does NOT touch the original
# Attr columns -- it is only used inside the formulas below.
EPSILON = 1e-6


def add_liquidity_buffer(df: pd.DataFrame) -> pd.DataFrame:
    """
    liquidity_buffer = Attr4 - 1.0

    Attr4 (current assets / short-term liabilities) is the classic
    "current ratio". A value of exactly 1.0 means current assets exactly
    cover short-term liabilities. Subtracting 1.0 re-centers the ratio
    around the break-even point, so:
        liquidity_buffer > 0  -> company has a cushion of current assets
        liquidity_buffer < 0  -> current assets fall short of short-term
                                  obligations (a classic distress signal)
    This does not use the target column, so there is no leakage.
    """
    df = df.copy()
    if "Attr4" in df.columns:
        df["liquidity_buffer"] = df["Attr4"] - 1.0
    return df


def add_profitability_to_leverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    profitability_to_leverage = Attr7 / (Attr2 + EPSILON)

    Attr7 (EBIT / total assets) measures operating profitability, and
    Attr2 (total liabilities / total assets) measures how leveraged the
    company is. Dividing profitability by leverage gives a rough sense of
    "how much operating profit is this company generating relative to how
    indebted it is". A low or negative value suggests a company carrying
    significant debt without the operating profitability to support it --
    a pattern often associated with financial distress.

    EPSILON avoids a divide-by-zero when Attr2 is 0 (a debt-free company).
    """
    df = df.copy()
    if "Attr7" in df.columns and "Attr2" in df.columns:
        df["profitability_to_leverage"] = df["Attr7"] / (df["Attr2"] + EPSILON)
    return df


def add_financial_pressure_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    financial_pressure_indicator = Attr2 - Attr7

    A simple combination of leverage (Attr2) minus operating profitability
    (Attr7). High leverage combined with low/negative profitability pushes
    this value up, which is the textbook definition of financial pressure:
    a company that owes a lot relative to its assets AND is not generating
    enough operating profit to offset that debt burden.
    """
    df = df.copy()
    if "Attr2" in df.columns and "Attr7" in df.columns:
        df["financial_pressure_indicator"] = df["Attr2"] - df["Attr7"]
    return df


def add_equity_cushion(df: pd.DataFrame) -> pd.DataFrame:
    """
    equity_cushion = Attr10 - Attr2

    Attr10 (equity / total assets) and Attr2 (total liabilities / total
    assets) both describe capital structure from opposite sides of the
    balance sheet. Their difference gives a single number: positive means
    the company is funded relatively more by equity than debt (generally
    safer), negative means the opposite.
    """
    df = df.copy()
    if "Attr10" in df.columns and "Attr2" in df.columns:
        df["equity_cushion"] = df["Attr10"] - df["Attr2"]
    return df


def engineer_features(df: pd.DataFrame, save_csv: bool = True) -> pd.DataFrame:
    """
    Apply the small set of engineered features above, in order, and
    replace any resulting +-inf (from divisions) with NaN so they flow
    through the same imputation logic as everything else.

    Note: this function is applied to the FULL cleaned dataset (before the
    train/test split) because these formulas are simple deterministic
    arithmetic on existing columns -- they do not "learn" anything from
    the data (no fitting, no statistics computed across rows), so there is
    no leakage risk in computing them before the split. Contrast this with
    imputation/scaling in preprocessing.py, which DO learn statistics from
    data and therefore MUST be fit after the split, on training data only.
    """
    engineered_df = df.copy()
    engineered_df = add_liquidity_buffer(engineered_df)
    engineered_df = add_profitability_to_leverage(engineered_df)
    engineered_df = add_financial_pressure_indicator(engineered_df)
    engineered_df = add_equity_cushion(engineered_df)

    # Divisions above can still produce +-inf (e.g. Attr2 + EPSILON is
    # effectively 0 only in pathological float cases) -- guard anyway.
    numeric_cols = engineered_df.select_dtypes(include=[np.number]).columns
    engineered_df[numeric_cols] = engineered_df[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    new_columns = [c for c in engineered_df.columns if c not in df.columns]
    print(f"Engineered {len(new_columns)} new feature(s): {new_columns}")

    if save_csv:
        engineered_df.to_csv(FEATURES_CSV_PATH, index=False)
        print(f"Saved feature-engineered dataset to: {FEATURES_CSV_PATH}")

    return engineered_df


if __name__ == "__main__":
    # Quick manual check: `python src/feature_engineering.py` from root.
    from data_loader import load_raw_dataset
    from preprocessing import clean_dataset

    raw_df = load_raw_dataset()
    cleaned_df = clean_dataset(raw_df, save_csv=False)
    features_df = engineer_features(cleaned_df)
    print("\nFinal feature set shape:", features_df.shape)
    print(features_df.head())
