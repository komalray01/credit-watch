"""
train_models.py

Trains all four models used in this project:

    1. Logistic Regression  -- simple linear baseline
    2. Random Forest        -- nonlinear tree-based baseline
    3. XGBoost               -- primary strong supervised model
    4. Isolation Forest      -- unsupervised anomaly detector

CLASS IMBALANCE STRATEGY (read this before changing anything):

This dataset has far more healthy companies than bankrupt ones. Instead of
optimizing for accuracy (which a model could "game" by predicting
"not bankrupt" for everyone), we handle imbalance with CLASS WEIGHTS:

    - Logistic Regression & Random Forest: class_weight="balanced", which
      tells sklearn to automatically weight the minority (bankrupt) class
      more heavily during training, proportional to how rare it is.
    - XGBoost: the equivalent lever is `scale_pos_weight`, set to
      (count of negative class) / (count of positive class) on the
      TRAINING data only.

Class weighting was chosen over resampling (e.g. SMOTE) as the primary
strategy because it is simpler, does not synthesize fake data points, and
is a standard, defensible first approach for imbalanced tabular data. An
optional SMOTE comparison is provided in `run_smote_comparison()` below --
it is NOT used for the models that get saved/deployed, it exists purely to
let you compare the two strategies on paper (see README "Class Imbalance
Strategy" section for the results once you run this).

Isolation Forest is trained differently: it is UNSUPERVISED. We train it
only on the feature matrix (no labels) and treat companies it flags as
"anomalous" as candidates worth investigating, independent of whether they
match historical bankruptcy patterns. We still use the labels afterwards,
only for evaluation (to see whether flagged anomalies correlate with
actual bankruptcies) -- never during Isolation Forest's own fitting.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (
    RANDOM_SEED,
    MODELS_DIR,
    LOGISTIC_REGRESSION_PATH,
    RANDOM_FOREST_PATH,
    XGBOOST_PATH,
    ISOLATION_FOREST_PATH,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
)
from data_loader import load_raw_dataset
from preprocessing import (
    clean_dataset,
    split_features_target,
    train_test_split_data,
    fit_preprocessor,
    apply_preprocessor,
)
from feature_engineering import engineer_features


def prepare_data():
    """
    Run the full data pipeline: load raw -> clean -> engineer features ->
    split -> fit preprocessor on train -> transform train & test.

    Returns a dict with everything downstream code (training, evaluation,
    explainability) needs, so every script uses the EXACT same processed
    data rather than re-deriving it slightly differently.
    """
    raw_df = load_raw_dataset()
    cleaned_df = clean_dataset(raw_df)
    features_df = engineer_features(cleaned_df)

    X, y = split_features_target(features_df)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split_data(X, y)

    # Fit preprocessing (imputer + scaler) ONLY on the training split.
    preprocessor = fit_preprocessor(X_train_raw)
    X_train = apply_preprocessor(preprocessor, X_train_raw)
    X_test = apply_preprocessor(preprocessor, X_test_raw)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
    }


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    # scale_pos_weight = (# negative / # positive), computed on TRAINING
    # data only -- this is the XGBoost equivalent of class_weight="balanced".
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / max(n_positive, 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_isolation_forest(X_train, y_train):
    """
    Isolation Forest is unsupervised -- it never sees y_train during
    fitting. `contamination` is set to the observed fraction of bankrupt
    companies in the training data, so the model flags roughly that same
    proportion of points as anomalies. This is a reasonable, data-driven
    choice rather than an arbitrary guess.
    """
    contamination = float((y_train == 1).mean())
    contamination = min(max(contamination, 0.01), 0.5)  # keep in a sane range

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train)  # no y_train passed on purpose
    return model


def run_smote_comparison(X_train, y_train, X_test, y_test):
    """
    OPTIONAL comparison: train a class-weighted Logistic Regression vs a
    SMOTE-resampled (no class weight) Logistic Regression, and print
    both models' precision/recall/F1 for the bankrupt class side by side.

    This function requires the `imbalanced-learn` package. It is not part
    of the main training flow (main() below does not call it) because the
    project's primary, saved/deployed strategy is class weighting. Run it
    manually if you want to see the comparison:

        python -c "from src.train_models import *; d = prepare_data(); \
            run_smote_comparison(d['X_train'], d['y_train'], d['X_test'], d['y_test'])"
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print(
            "imbalanced-learn is not installed. Run "
            "`pip install imbalanced-learn` if you want to try the SMOTE "
            "comparison. Skipping."
        )
        return

    from sklearn.metrics import classification_report

    # SMOTE is applied ONLY to the training data. The test set is left
    # exactly as-is (real, unresampled data) -- resampling the test set
    # would give a misleadingly easy evaluation.
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    weighted_model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
    )
    weighted_model.fit(X_train, y_train)

    smote_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    smote_model.fit(X_train_smote, y_train_smote)

    print("=== Class-weighted Logistic Regression ===")
    print(classification_report(y_test, weighted_model.predict(X_test), digits=3))

    print("=== SMOTE-resampled Logistic Regression ===")
    print(classification_report(y_test, smote_model.predict(X_test), digits=3))


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Preparing data...")
    data = prepare_data()
    X_train, y_train = data["X_train"], data["y_train"]

    print("\nTraining Logistic Regression...")
    log_reg = train_logistic_regression(X_train, y_train)
    joblib.dump(log_reg, LOGISTIC_REGRESSION_PATH)

    print("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)
    joblib.dump(rf, RANDOM_FOREST_PATH)

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    joblib.dump(xgb_model, XGBOOST_PATH)

    print("Training Isolation Forest...")
    iso_forest = train_isolation_forest(X_train, y_train)
    joblib.dump(iso_forest, ISOLATION_FOREST_PATH)

    print("Saving fitted preprocessor...")
    joblib.dump(data["preprocessor"], PREPROCESSOR_PATH)

    print(f"\nAll models and the preprocessor were saved to: {MODELS_DIR}")
    print("Next step: run `python src/evaluate_models.py` to see metrics.")


if __name__ == "__main__":
    main()
