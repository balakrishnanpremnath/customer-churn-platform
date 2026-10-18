import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "customer_churn_dataset.csv"
OUTPUT = ROOT / "data" / "processed" / "baseline_metrics.json"

FEATURES = [
    "frequency_180d",
    "monetary_180d",
    "quantity_180d",
    "active_days_180d",
    "unique_products_180d",
    "recency_days",
    "tenure_180d",
    "avg_order_value_180d",
    "orders_30d",
    "spend_30d",
    "orders_90d",
    "spend_90d",
    "spend_share_30d",
    "has_repeat_purchase_day_180d",
    "avg_repeat_gap_days_180d",
]

TARGET = "churn_90d"


def evaluate(model, x, y):
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)[:, 1]

    return {
        "roc_auc": round(float(roc_auc_score(y, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y, probabilities)), 4),
        "precision": round(float(precision_score(y, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y, predictions, zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, predictions)), 4),
        "brier_score": round(float(brier_score_loss(y, probabilities)), 4),
    }


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    data = pd.read_csv(INPUT)

    required = set(FEATURES + [TARGET, "split"])
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    train = data.loc[data["split"] == "train"].copy()
    validation = data.loc[data["split"] == "validation"].copy()

    if train.empty or validation.empty:
        raise ValueError("Training or validation data is empty")

    x_train = train[FEATURES].astype(float)
    y_train = train[TARGET].astype(int)

    x_validation = validation[FEATURES].astype(float)
    y_validation = validation[TARGET].astype(int)

    if not np.isfinite(x_train.to_numpy()).all():
        raise ValueError("Training features contain invalid values")

    if not np.isfinite(x_validation.to_numpy()).all():
        raise ValueError("Validation features contain invalid values")

    if (x_train.to_numpy() < 0).any() or (x_validation.to_numpy() < 0).any():
        raise ValueError("Negative feature values are not supported")

    if y_train.nunique() != 2 or y_validation.nunique() != 2:
        raise ValueError("Both target classes must be present")

    models = {
        "dummy_prior": DummyClassifier(
            strategy="prior"
        ),
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log_transform", FunctionTransformer(np.log1p)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                random_state=42,
            )),
        ]),
    }

    report = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_churn_rate": round(float(y_train.mean()), 4),
        "validation_churn_rate": round(float(y_validation.mean()), 4),
        "feature_count": len(FEATURES),
        "validation_metrics": {},
    }

    print("=" * 60)
    print("CUSTOMER CHURN - BASELINE ML EVALUATION")
    print("=" * 60)
    print(f"Train rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Features: {len(FEATURES)}")

    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics = evaluate(model, x_validation, y_validation)
        report["validation_metrics"][name] = metrics

        print(f"\nMODEL: {name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=4), encoding="utf-8")

    print(f"\nValidation report: {OUTPUT}")
    print("Held-out test set was not evaluated.")
    print("Baseline evaluation completed!")


if __name__ == "__main__":
    main()
