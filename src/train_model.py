from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)

from .config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, TARGET_COL
from .features import build_pipeline
from .validation import validate_training_dataframe


def load_processed_data():
    train_path = PROCESSED_DATA_DIR / "transactions_train.csv"
    test_path = PROCESSED_DATA_DIR / "transactions_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed train/test files not found in {PROCESSED_DATA_DIR}. "
            f"Run data_prep.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    validate_training_dataframe(train_df, context="processed training data")
    validate_training_dataframe(test_df, context="processed test data")
    return train_df, test_df


def train_and_evaluate() -> dict:
    print("Loading processed training data...")
    train_df, test_df = load_processed_data()

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    print(f"\nTraining RandomForest on {len(X_train):,} samples...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("Computing predictions on test set...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    average_precision = average_precision_score(y_test, y_proba)
    brier_score = brier_score_loss(y_test, y_proba)

    cls_report = classification_report(
        y_test,
        y_pred_default,
        output_dict=True,
        digits=3,
    )

    metrics = {
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "brier_score": float(brier_score),
        "classification_report_default_threshold": cls_report,
        "n_train_samples": int(len(y_train)),
        "n_test_samples": int(len(y_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "note": (
            "Trained on the real-world ULB Credit Card Fraud Detection dataset. "
            "284,807 transactions with 0.172% fraud rate. "
            "V1-V28 are PCA-transformed features. Time and Amount are scaled. "
            "Class imbalance handled via class_weight='balanced' in RandomForest."
        ),
    }

    model_path = MODELS_DIR / "fraud_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    metrics_path = METRICS_DIR / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[SAVED] Model to:   {model_path}")
    print(f"[SAVED] Metrics to: {metrics_path}")
    print("\n=== Model Performance (test set) ===")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"Average precision: {average_precision:.4f}")
    print(f"Brier score:       {brier_score:.4f}")
    print(f"\nPositive rate (train): {y_train.mean():.4%}")
    print(f"Positive rate (test):  {y_test.mean():.4%}")

    return metrics


def main() -> None:
    metrics = train_and_evaluate()
    print("\n=== Full metrics summary ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
