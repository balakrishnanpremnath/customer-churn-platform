import json
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from scripts.train_baseline import FEATURES, TARGET, evaluate


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "customer_churn_dataset.csv"
OUTPUT = ROOT / "data" / "processed" / "rf_tuning_results.json"

HORIZON_DAYS = 90

def validate_temporal_fold(train_dates, validation_date):
    if len(train_dates) == 0:
        raise ValueError("Training dates cannot be empty")

    latest_train = max(pd.Timestamp(d) for d in train_dates)
    validation_date = pd.Timestamp(validation_date)

    if latest_train >= validation_date:
        raise ValueError("Validation must come after training")

    label_end = latest_train + pd.Timedelta(
        days=HORIZON_DAYS
    )

    if label_end > validation_date:
        raise ValueError(
            "Temporal label leakage: overlapping outcome window"
        )


def build_model(params):
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features="sqrt",
        random_state=42,
        n_jobs=4,
    )


def main():
    data = pd.read_csv(
        INPUT,
        usecols=FEATURES + [TARGET, "as_of_date", "split"],
    )

    # Never use test records during model selection.
    data = data[data["split"].isin(["train", "validation"])].copy()
    data["as_of_date"] = pd.to_datetime(data["as_of_date"])

    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "validation"].copy()

    dates = sorted(train["as_of_date"].unique())

    if len(dates) != 4:
        raise ValueError("Expected four training snapshot dates")

    if validation["as_of_date"].nunique() != 1:
        raise ValueError("Expected one external validation date")

    validation_date = validation["as_of_date"].iloc[0]

    if (
        pd.Timestamp(dates[-1])
        + pd.Timedelta(days=HORIZON_DAYS)
        > validation_date
    ):
        raise ValueError("Training labels overlap validation cutoff")

    configurations = [
        {"max_depth": depth, "min_samples_leaf": leaf}
        for depth, leaf in product([6, 10], [5, 15])
    ]

    results = []

    print("=" * 60)
    print("RANDOM FOREST - TEMPORAL CROSS-VALIDATION")
    print("=" * 60)

    for params in configurations:
        fold_results = []

        for i in range(1, len(dates)):
            train_dates = dates[:i]
            fold_date = dates[i]

            validate_temporal_fold(train_dates, fold_date)

            fold_train = train[
                train["as_of_date"].isin(train_dates)
            ]

            fold_val = train[
                train["as_of_date"] == fold_date
            ]

            x_train = fold_train[FEATURES]
            y_train = fold_train[TARGET]

            x_val = fold_val[FEATURES]
            y_val = fold_val[TARGET]

            model = build_model(params)
            model.fit(x_train, y_train)

            probabilities = model.predict_proba(x_val)[:, 1]

            fold_results.append({
                "validation_date": str(
                    pd.Timestamp(fold_date).date()
                ),
                "roc_auc": float(
                    roc_auc_score(y_val, probabilities)
                ),
                "pr_auc": float(
                    average_precision_score(y_val, probabilities)
                ),
                "brier_score": float(
                    brier_score_loss(y_val, probabilities)
                ),
            })

        mean_roc = sum(
            fold["roc_auc"] for fold in fold_results
        ) / len(fold_results)

        mean_pr = sum(
            fold["pr_auc"] for fold in fold_results
        ) / len(fold_results)

        mean_brier = sum(
            fold["brier_score"] for fold in fold_results
        ) / len(fold_results)

        result = {
            "params": params,
            "mean_roc_auc": round(mean_roc, 4),
            "mean_pr_auc": round(mean_pr, 4),
            "mean_brier_score": round(mean_brier, 4),
            "folds": fold_results,
        }

        results.append(result)

        print(
            f"depth={params['max_depth']}, "
            f"leaf={params['min_samples_leaf']} | "
            f"ROC-AUC={mean_roc:.4f} | "
            f"PR-AUC={mean_pr:.4f} | "
            f"Brier={mean_brier:.4f}"
        )

    # Select using training-period CV only.
    best = max(
        results,
        key=lambda r: (
            r["mean_roc_auc"],
            r["mean_pr_auc"],
            -r["mean_brier_score"],
        ),
    )

    print("\nBEST CROSS-VALIDATION CONFIGURATION")
    print(best["params"])

    # Refit using all training snapshots.
    final_model = build_model(best["params"])
    final_model.fit(train[FEATURES], train[TARGET])

    # Evaluate once on the separate validation snapshot.
    validation_metrics = evaluate(
        final_model,
        validation[FEATURES],
        validation[TARGET],
    )

    print("\nJUNE 2011 VALIDATION RESULTS")

    for name, value in validation_metrics.items():
        print(f"{name}: {value:.4f}")

    report = {
        "selection_metric": "mean_temporal_cv_roc_auc",
        "configurations": results,
        "selected_parameters": best["params"],
        "validation_metrics": validation_metrics,
        "held_out_test_evaluated": False,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print(f"\nReport saved: {OUTPUT}")
    print("Held-out test set not evaluated.")
    print("Temporal optimization completed!")


if __name__ == "__main__":
    main()