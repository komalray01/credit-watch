"""
config.py

Central place for paths, constants, and settings used across the CreditWatch
project. Every script imports from here instead of hardcoding paths, so if
you move a folder you only need to change it in one place.
"""

import os

# -----------------------------------------------------------------------
# PROJECT ROOT
# -----------------------------------------------------------------------
# This is the folder that contains this config.py file.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------
# DATA PATHS
# -----------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# The exact file you need to download from the UCI Machine Learning
# Repository ("Polish companies bankruptcy data" dataset, 5year.arff).
# See data/README.md for the download instructions.
RAW_ARFF_FILENAME = "5year.arff"
RAW_ARFF_PATH = os.path.join(RAW_DATA_DIR, RAW_ARFF_FILENAME)

# Cleaned (but not yet feature-engineered) data, saved as CSV so it is easy
# to inspect. This is a DERIVED file -- it is never the raw dataset.
CLEANED_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_5year.csv")

# Final modeling table (cleaned data + engineered features).
FEATURES_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "features_5year.csv")

# -----------------------------------------------------------------------
# MODEL / ARTIFACT PATHS
# -----------------------------------------------------------------------
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

LOGISTIC_REGRESSION_PATH = os.path.join(MODELS_DIR, "logistic_regression.joblib")
RANDOM_FOREST_PATH = os.path.join(MODELS_DIR, "random_forest.joblib")
XGBOOST_PATH = os.path.join(MODELS_DIR, "xgboost_model.joblib")
ISOLATION_FOREST_PATH = os.path.join(MODELS_DIR, "isolation_forest.joblib")

# Fitted preprocessing objects (scaler, imputer, feature list) are saved
# together so predict.py can reproduce the EXACT same transformation used
# during training. This is what prevents train/serve skew.
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")

# Evaluation results (metrics table used by the Streamlit dashboard).
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.joblib")

# -----------------------------------------------------------------------
# TARGET COLUMN
# -----------------------------------------------------------------------
# In the UCI Polish Companies Bankruptcy Dataset, the target column in the
# raw ARFF file is literally named "class". It is 1 if the company went
# bankrupt within the forecast horizon (this file = ~1 year ahead) and 0
# otherwise. We rename it to something more readable after loading.
RAW_TARGET_COLUMN = "class"
TARGET_COLUMN = "bankrupt"

# -----------------------------------------------------------------------
# REPRODUCIBILITY
# -----------------------------------------------------------------------
RANDOM_SEED = 42

# -----------------------------------------------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------------------------------------------
TEST_SIZE = 0.2

# -----------------------------------------------------------------------
# RISK SCORE THRESHOLDS
# -----------------------------------------------------------------------
# These are PROJECT-DEFINED thresholds on the model's predicted probability
# of bankruptcy, NOT official Vivriti (or any institution's) categories.
# See README.md "Risk Score" section for how these were chosen.
RISK_LOW_THRESHOLD = 0.30   # probability < 0.30  -> Low Risk
RISK_MEDIUM_THRESHOLD = 0.60  # 0.30 <= probability < 0.60 -> Medium Risk
                               # probability >= 0.60 -> High Risk
