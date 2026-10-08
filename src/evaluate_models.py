"""
evaluate_models.py

Evaluates all four trained models on the held-out test set and saves a
metrics dictionary to models/metrics.joblib, which the Streamlit dashboard
reads directly (so the dashboard never has to retrain anything).

WHY NOT JUST ACCURACY?
Because bankruptcies are rare, a model that predicts "not bankrupt" for
every single company could still score >90% accuracy while being
completely useless as an early-warning tool (it would have 0% recall on
the class we actually care about). So instead we focus on:

    - Recall for the bankrupt class: of all companies that actually went
      bankrupt, what fraction did we correctly flag? In an early-warning
      system, a FALSE NEGATIVE (a distressed company we failed to flag)
      is the worst outcome -- it means a company that needed attention
      slipped through unnoticed. A false positive, by contrast, just
      means an analyst spends a bit of extra time reviewing a company
      that turns out to be fine. This asymmetry is why recall matters
      more here than in many other classification problems.

    - Precision for the bankrupt class: of all companies we flagged as
      high-risk, what fraction were actually distressed? If precision is
      too low, analysts get flooded with false alarms and may start
      ignoring the system altogether ("alert fatigue"). So precision
      still matters -- it's a genuine trade-off, not something to ignore.

    - PR-AUC (average precision): summarizes precision/recall trade-off
      across all thresholds. More informative than ROC-AUC when the
      positive class is rare, because ROC-AUC can look overly optimistic
      on imbalanced data.

    - ROC-AUC: reported too, since it's a standard, widely understood
      metric, but it is not the primary metric we optimize for here.

THRESHOLD TUNING
By default, classifiers use a 0.5 probability threshold to decide
"bankrupt" vs "not bankrupt". For an early-warning system, that default
is not automatically the right choice -- since we care more about recall,
LOWERING the threshold (flagging more companies as risky) trades some
precision for higher recall. We explicitly show this trade-off across a
range of thresholds instead of silently picking one.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so this works headless / on servers
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

from config import (
    MODELS_DIR,
    LOGISTIC_REGRESSION_PATH,
    RANDOM_FOREST_PATH,
    XGBOOST_PATH,
    ISOLATION_FOREST_PATH,
    PREPROCESSOR_PATH,
    METRICS_PATH,
)
from train_models import prepare_data

# Thresholds we sweep through when demonstrating the precision/recall
# trade-off. 0.5 is included as the "default" reference point.
CANDIDATE_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def get_bankruptcy_probabilities(model, X):
    """
    Return the model's predicted probability of the bankrupt class (1)
    for every row in X. Works for any of the three supervised
    scikit-learn/XGBoost classifiers used in this project.
    """
    return model.predict_proba(X)[:, 1]


def get_isolation_forest_anomaly_scores(iso_forest, X):
    """
    Isolation Forest does not output a probability. Instead it outputs a
    "decision function" score where LOWER values mean MORE anomalous. We
    flip and min-max normalize this to a 0-1 "risk-like" score so it can
    be compared on the same axes as the supervised models' probabilities.
    This is an approximation for comparison purposes only -- it is not a
    calibrated probability, and the dashboard/README are explicit about
    that distinction.
    """
    raw_scores = iso_forest.decision_function(X)  # higher = more normal
    anomaly_score = -raw_scores  # flip: higher = more anomalous
    normalized = (anomaly_score - anomaly_score.min()) / (
        anomaly_score.max() - anomaly_score.min() + 1e-9
    )
    return normalized


def evaluate_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": cm,
    }


def evaluate_model(name, y_true, y_prob):
    """
    Compute the full metric set for one model at the default 0.5
    threshold, plus PR-AUC/ROC-AUC (threshold-independent), plus the
    threshold sweep for the precision/recall trade-off table.
    """
    default_result = evaluate_at_threshold(y_true, y_prob, 0.5)

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    threshold_sweep = [
        evaluate_at_threshold(y_true, y_prob, t) for t in CANDIDATE_THRESHOLDS
    ]

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)

    print(f"\n=== {name} (threshold = 0.5) ===")
    print(f"Precision: {default_result['precision']:.3f}")
    print(f"Recall:    {default_result['recall']:.3f}")
    print(f"F1-score:  {default_result['f1']:.3f}")
    print(f"PR-AUC:    {pr_auc:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{default_result['confusion_matrix']}")
    print(f"False negatives (missed distressed companies): {default_result['false_negatives']}")

    return {
        "name": name,
        "default": default_result,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold_sweep": threshold_sweep,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "y_true": y_true,
        "y_prob": y_prob,
    }


def plot_precision_recall_curves(results, save_path):
    """Overlay PR curves for all supervised models on one plot."""
    plt.figure(figsize=(7, 6))
    for result in results:
        if result["name"] == "Isolation Forest":
            continue  # anomaly score isn't a calibrated probability
        plt.plot(
            result["recall_curve"],
            result["precision_curve"],
            label=f"{result['name']} (PR-AUC={result['pr_auc']:.3f})",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (bankrupt class)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved precision-recall curve plot to: {save_path}")


def main():
    print("Reproducing the train/test split and loading saved models...")
    data = prepare_data()
    X_test, y_test = data["X_test"], data["y_test"]

    log_reg = joblib.load(LOGISTIC_REGRESSION_PATH)
    rf = joblib.load(RANDOM_FOREST_PATH)
    xgb_model = joblib.load(XGBOOST_PATH)
    iso_forest = joblib.load(ISOLATION_FOREST_PATH)

    results = []
    results.append(
        evaluate_model("Logistic Regression", y_test, get_bankruptcy_probabilities(log_reg, X_test))
    )
    results.append(
        evaluate_model("Random Forest", y_test, get_bankruptcy_probabilities(rf, X_test))
    )
    results.append(
        evaluate_model("XGBoost", y_test, get_bankruptcy_probabilities(xgb_model, X_test))
    )
    results.append(
        evaluate_model(
            "Isolation Forest",
            y_test,
            get_isolation_forest_anomaly_scores(iso_forest, X_test),
        )
    )

    plot_path = os.path.join(MODELS_DIR, "precision_recall_curves.png")
    plot_precision_recall_curves(results, plot_path)

    # Save everything the Streamlit dashboard needs, so it never has to
    # re-run training or evaluation itself.
    metrics_bundle = {
        "results": results,
        "candidate_thresholds": CANDIDATE_THRESHOLDS,
        "class_distribution": data["y_train"].value_counts().to_dict(),
        "test_class_distribution": y_test.value_counts().to_dict(),
        "n_train": len(data["X_train"]),
        "n_test": len(X_test),
        "n_features": data["X_train"].shape[1],
    }
    joblib.dump(metrics_bundle, METRICS_PATH)
    print(f"\nSaved full metrics bundle to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
