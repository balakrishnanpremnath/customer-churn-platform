import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from scripts.train_baseline import FEATURES, TARGET, evaluate


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT / "data" / "processed"
    / "customer_churn_dataset.csv"
)

BASELINE_REPORT = (
    ROOT / "data" / "processed"
    / "baseline_metrics.json"
)

OUTPUT = (
    ROOT / "data" / "processed"
    / "advanced_metrics.json"
)


def build_models(rf_trees=200, xgb_trees=250):
    """Create reproducible tree-based classifiers."""

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=rf_trees,
            max_depth=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=4,
        ),

        "xgboost": XGBClassifier(
            n_estimators=xgb_trees,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=4,
        ),
    }


def load_train_validation():
    if not DATASET.is_file():
        raise FileNotFoundError(DATASET)

    data = pd.read_csv(DATASET)

    required = set(FEATURES + [TARGET, "split"])
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    train = data.loc[data["split"] == "train"]
    validation = data.loc[data["split"] == "validation"]

    if train.empty or validation.empty:
        raise ValueError("Missing train or validation data")

    x_train = train[FEATURES].astype(float)
    y_train = train[TARGET].astype(int)

    x_val = validation[FEATURES].astype(float)
    y_val = validation[TARGET].astype(int)

    for features in [x_train, x_val]:
        if not np.isfinite(features.to_numpy()).all():
            raise ValueError("Non-finite features detected")

    for labels in [y_train, y_val]:
        if set(labels.unique()) != {0, 1}:
            raise ValueError("Both churn classes required")

    return x_train, y_train, x_val, y_val


def main():
    x_train, y_train, x_val, y_val = (
        load_train_validation()
    )

    print("=" * 60)
    print("CUSTOMER CHURN - ADVANCED ML COMPARISON")
    print("=" * 60)

    print(f"Training rows: {len(x_train):,}")
    print(f"Validation rows: {len(x_val):,}")
    print(f"Features: {len(FEATURES)}")

    report = {
        "train_rows": len(x_train),
        "validation_rows": len(x_val),
        "feature_count": len(FEATURES),
        "random_state": 42,
        "advanced_models": {},
    }

    for name, model in build_models().items():
        print(f"\nTraining: {name}")

        model.fit(x_train, y_train)

        metrics = evaluate(model, x_val, y_val)

        report["advanced_models"][name] = metrics

        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    if BASELINE_REPORT.is_file():
        baseline = json.loads(
            BASELINE_REPORT.read_text(encoding="utf-8")
        )

        if (
            baseline["train_rows"] != len(x_train)
            or baseline["validation_rows"] != len(x_val)
            or baseline["feature_count"] != len(FEATURES)
        ):
            raise ValueError("Baseline dataset mismatch")

        report["baseline_models"] = baseline[
            "validation_metrics"
        ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("Advanced model comparison completed!")
    print(f"Report saved: {OUTPUT}")
    print("Held-out test data NOT evaluated.")


if __name__ == "__main__":
    main()