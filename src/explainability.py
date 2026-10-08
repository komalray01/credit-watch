"""
explainability.py

Provides SHAP-based explainability for the XGBoost model, since it is the
primary supervised model in this project. SHAP (SHapley Additive
exPlanations) explains predictions by attributing each feature's
contribution to the model's output for a given prediction, grounded in
game-theoretic Shapley values.

Two things are provided:

1. Global explanations: which features matter most, on average, across
   the whole test set (a bar chart of mean absolute SHAP values).
2. Individual explanations: for one specific company/row, which features
   pushed its predicted risk up or down, and by how much.

Everything here is generated FROM the actual trained model's SHAP values.
No natural-language explanation text is hardcoded -- the top features and
their direction (risk-increasing vs risk-decreasing) come directly from
the SHAP output for that specific row.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import XGBOOST_PATH


def load_xgboost_model():
    return joblib.load(XGBOOST_PATH)


def compute_shap_values(model, X: pd.DataFrame):
    """
    Compute SHAP values for XGBoost using shap.TreeExplainer, which is
    fast and exact for tree-based models (no sampling approximation
    needed, unlike the general-purpose KernelExplainer).

    Returns a shap.Explanation object, which bundles the values together
    with the feature names and the base (expected) value.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values


def get_global_feature_importance(shap_values, top_n: int = 10) -> pd.DataFrame:
    """
    Rank features by mean absolute SHAP value across all rows -- this is
    the standard way to get a single "global importance" number per
    feature out of per-row SHAP values.
    """
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": shap_values.feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return importance_df.head(top_n).reset_index(drop=True)


def explain_single_prediction(shap_values, row_index: int, top_n: int = 3) -> dict:
    """
    Build a simple, human-readable explanation for ONE row (one company),
    e.g. "High predicted risk was primarily associated with X, Y, Z."

    We take the top `top_n` features by absolute SHAP value FOR THIS ROW
    (not the global average), and separate them into features that pushed
    the prediction toward higher risk (positive SHAP value) vs lower risk
    (negative SHAP value).

    Returns a dict with the raw contributing features (so the Streamlit
    app can display exact numbers) and a plain-English sentence built only
    from those real values -- nothing is invented.
    """
    row_values = shap_values.values[row_index]
    feature_names = shap_values.feature_names

    contributions = pd.DataFrame(
        {"feature": feature_names, "shap_value": row_values}
    )
    contributions["abs_shap_value"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values("abs_shap_value", ascending=False)

    top_features = contributions.head(top_n)

    risk_increasing = top_features[top_features["shap_value"] > 0]["feature"].tolist()

    if risk_increasing:
        feature_list_str = ", ".join(risk_increasing)
        sentence = (
            f"Predicted risk for this company was primarily associated with: "
            f"{feature_list_str}."
        )
    else:
        sentence = (
            "This company's top contributing factors mostly pushed the "
            "predicted risk DOWN rather than up."
        )

    return {
        "top_features": top_features.reset_index(drop=True),
        "explanation_sentence": sentence,
    }


if __name__ == "__main__":
    # Quick manual check: `python src/explainability.py` from project root.
    from train_models import prepare_data

    data = prepare_data()
    model = load_xgboost_model()

    shap_vals = compute_shap_values(model, data["X_test"])

    print("Top global features by mean |SHAP value|:")
    print(get_global_feature_importance(shap_vals))

    print("\nExample explanation for the first test-set company:")
    explanation = explain_single_prediction(shap_vals, row_index=0)
    print(explanation["explanation_sentence"])
    print(explanation["top_features"])
