import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.train_baseline import FEATURES, TARGET
from scripts.tune_random_forest import (
    build_model,
    validate_temporal_fold,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

INPUT = PROCESSED / "customer_churn_dataset.csv"
TUNING = PROCESSED / "rf_tuning_results.json"
OUTPUT = PROCESSED / "bootstrap_results.json"

N_BOOTSTRAP = 1000
RANDOM_STATE = 42
BUDGETS = [100, 500]


def captures_at_k(labels, scores, k):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("Invalid label or score dimensions")

    if not 1 <= k <= len(labels):
        raise ValueError("Invalid budget")

    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")

    ranking = np.argsort(-scores, kind="stable")

    return int(labels[ranking[:k]].sum())


def confidence_interval(values):
    lower, upper = np.quantile(
        values, [0.025, 0.975]
    )

    return [
        round(float(lower), 3),
        round(float(upper), 3),
    ]


def main():
    if not INPUT.is_file() or not TUNING.is_file():
        raise FileNotFoundError(
            "Dataset or tuning report is missing"
        )

    tuning = json.loads(
        TUNING.read_text(encoding="utf-8")
    )

    data = pd.read_csv(
        INPUT,
        usecols=FEATURES + [
            TARGET,
            "customer_id",
            "as_of_date",
            "split",
        ],
    )

    train = data[data["split"] == "train"].copy()
    validation = data[
        data["split"] == "validation"
    ].copy()

    if train.empty or validation.empty:
        raise ValueError("Missing data split")

    if validation["customer_id"].duplicated().any():
        raise ValueError("Duplicate validation customers")

    if validation["as_of_date"].nunique() != 1:
        raise ValueError("Expected one validation date")

    validate_temporal_fold(
        train["as_of_date"].unique(),
        validation["as_of_date"].iloc[0],
    )

    model = build_model(
        tuning["selected_parameters"]
    )

    model.fit(train[FEATURES], train[TARGET])

    labels = validation[TARGET].to_numpy()

    rf_scores = model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    recency_scores = validation[
        "recency_days"
    ].to_numpy()

    n = len(labels)
    rng = np.random.default_rng(RANDOM_STATE)

    samples = {
        budget: []
        for budget in BUDGETS
    }

    auc_samples = []

    print("=" * 60)
    print("CUSTOMER CHURN - BOOTSTRAP UNCERTAINTY")
    print("=" * 60)
    print(f"Validation customers: {n:,}")
    print(f"Bootstrap repetitions: {N_BOOTSTRAP}")

    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        y = labels[indices]
        rf = rf_scores[indices]
        recency = recency_scores[indices]

        if len(np.unique(y)) < 2:
            continue

        auc_samples.append(
            roc_auc_score(y, rf)
        )

        for budget in BUDGETS:
            rf_captured = captures_at_k(
                y, rf, budget
            )

            recency_captured = captures_at_k(
                y, recency, budget
            )

            samples[budget].append(
                rf_captured - recency_captured
            )

    report = {
        "validation_customers": n,
        "bootstrap_repetitions": len(auc_samples),
        "confidence_level": 0.95,
        "random_state": RANDOM_STATE,
        "roc_auc": {
            "observed": round(
                float(roc_auc_score(labels, rf_scores)),
                4,
            ),
            "ci_95": confidence_interval(auc_samples),
        },
        "ranking_comparison": {},
        "held_out_test_evaluated": False,
    }

    for budget in BUDGETS:
        rf_actual = captures_at_k(
            labels, rf_scores, budget
        )

        recency_actual = captures_at_k(
            labels, recency_scores, budget
        )

        differences = samples[budget]

        report["ranking_comparison"][
            f"top_{budget}"
        ] = {
            "rf_captured": rf_actual,
            "recency_captured": recency_actual,
            "observed_difference": (
                rf_actual - recency_actual
            ),
            "difference_ci_95": confidence_interval(
                differences
            ),
        }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=4))
    print("\nBootstrap analysis completed!")
    print("Held-out test set NOT evaluated.")


if __name__ == "__main__":
    main()