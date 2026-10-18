import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_baseline import FEATURES, TARGET
from scripts.tune_random_forest import (
    build_model,
    validate_temporal_fold,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

INPUT = PROCESSED / "customer_churn_dataset.csv"
TUNING = PROCESSED / "rf_tuning_results.json"
OUTPUT = PROCESSED / "retention_results.json"

BUDGETS = [50, 100, 250, 500]


def ranking_metrics(y_true, risk_scores, budget):
    """Evaluate how well a ranking captures inactive customers."""

    y = np.asarray(y_true)
    scores = np.asarray(risk_scores, dtype=float)

    if (
        y.ndim != 1
        or scores.ndim != 1
        or len(y) == 0
        or len(y) != len(scores)
    ):
        raise ValueError("Invalid ranking input dimensions")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")

    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")

    if not isinstance(budget, (int, np.integer)):
        raise ValueError("Budget must be an integer")

    if not 1 <= budget <= len(y):
        raise ValueError("Invalid customer contact budget")

    # Highest-risk customers first.
    ranked = np.argsort(
        -scores,
        kind="stable",
    )[:budget]

    captured = int(y[ranked].sum())
    total_positive = int(y.sum())

    base_rate = total_positive / len(y)
    precision = captured / budget

    recall = (
        captured / total_positive
        if total_positive > 0
        else 0.0
    )

    lift = (
        precision / base_rate
        if base_rate > 0
        else 0.0
    )

    return {
        "budget": int(budget),
        "captured": captured,
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "lift_at_k": round(lift, 4),
    }


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    if not TUNING.is_file():
        raise FileNotFoundError(TUNING)

    tuning = json.loads(
        TUNING.read_text(encoding="utf-8")
    )

    data = pd.read_csv(
        INPUT,
        usecols=FEATURES + [
            TARGET,
            "as_of_date",
            "split",
        ],
    )

    data["as_of_date"] = pd.to_datetime(
        data["as_of_date"]
    )

    train = data.loc[
        data["split"] == "train"
    ].copy()

    validation = data.loc[
        data["split"] == "validation"
    ].copy()

    if train.empty or validation.empty:
        raise ValueError("Train or validation data is empty")

    if validation["as_of_date"].nunique() != 1:
        raise ValueError("Expected one validation cutoff")

    validate_temporal_fold(
        train["as_of_date"].unique(),
        validation["as_of_date"].iloc[0],
    )

    model = build_model(
        tuning["selected_parameters"]
    )

    model.fit(
        train[FEATURES],
        train[TARGET],
    )

    # Test data is never loaded into this evaluation.
    y_validation = validation[TARGET].to_numpy()

    rf_scores = model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    # Higher recency means more days since last purchase.
    recency_scores = validation[
        "recency_days"
    ].to_numpy()

    base_rate = float(y_validation.mean())

    results = []

    print("=" * 65)
    print("CUSTOMER RETENTION PRIORITIZATION")
    print("=" * 65)

    print(f"Validation customers: {len(validation):,}")
    print(f"Inactivity rate: {base_rate:.2%}")

    for budget in BUDGETS:
        rf = ranking_metrics(
            y_validation,
            rf_scores,
            budget,
        )

        recency = ranking_metrics(
            y_validation,
            recency_scores,
            budget,
        )

        results.append({
            "budget": budget,
            "random_expected_captures": round(
                budget * base_rate, 2
            ),
            "random_forest": rf,
            "recency_baseline": recency,
        })

        print(f"\nCONTACT BUDGET: {budget}")
        print(f"RF captures: {rf['captured']}")
        print(f"RF Precision@K: {rf['precision_at_k']}")
        print(f"RF Recall@K: {rf['recall_at_k']}")
        print(f"RF Lift@K: {rf['lift_at_k']}")
        print(f"Recency captures: {recency['captured']}")

    report = {
        "evaluation_period": "2011-06-01",
        "validation_customers": int(len(validation)),
        "actual_inactive_customers": int(
            y_validation.sum()
        ),
        "inactivity_rate": round(base_rate, 4),
        "model": "uncalibrated_tuned_random_forest",
        "results": results,
        "held_out_test_evaluated": False,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nRetention evaluation completed!")
    print(f"Report saved: {OUTPUT}")
    print("Held-out test set NOT evaluated.")


if __name__ == "__main__":
    main()