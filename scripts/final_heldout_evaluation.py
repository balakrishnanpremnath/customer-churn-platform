import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from scripts.train_baseline import FEATURES, TARGET


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "final_model.json"
INPUT = ROOT / "data" / "processed" / "customer_churn_dataset.csv"
OUTPUT = ROOT / "data" / "processed" / "final_test_results.json"


def load_config():
    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    if config["features"] != FEATURES:
        raise ValueError("Frozen features do not match code")

    if config["target"] != TARGET:
        raise ValueError("Frozen target mismatch")

    if config["status"] != "locked_before_held_out_test":
        raise ValueError("Model is not frozen")

    if config["model"]["probability_calibration"]:
        raise ValueError("Calibration must remain disabled")

    if config["positive_class"] != 1:
        raise ValueError("Unexpected positive class")

    return config


def check_metadata(config):
    metadata = pd.read_csv(
        INPUT,
        usecols=["split", "as_of_date"],
    )

    splits = config["splits"]

    expected_dates = {
        "train": splits["training_dates"],
        "validation": [splits["validation_date"]],
        "test": [splits["test_date"]],
    }

    if set(metadata["split"]) != set(expected_dates):
        raise ValueError("Unexpected dataset splits")

    for split, expected in expected_dates.items():
        actual = sorted(
            metadata.loc[
                metadata["split"] == split,
                "as_of_date"
            ].unique().tolist()
        )

        if actual != sorted(expected):
            raise ValueError(
                f"Unexpected dates in {split}"
            )

    train_count = int(
        metadata["split"].isin(
            splits["final_training_splits"]
        ).sum()
    )

    test_count = int(
        (metadata["split"] == "test").sum()
    )

    if train_count != splits["expected_final_training_rows"]:
        raise ValueError("Final training count mismatch")

    if test_count != splits["expected_test_rows"]:
        raise ValueError("Held-out test count mismatch")

    latest_training_date = pd.Timestamp(
        splits["validation_date"]
    )
    test_date = pd.Timestamp(splits["test_date"])

    horizon = pd.Timedelta(
        days=config["prediction_horizon_days"]
    )

    if latest_training_date + horizon > test_date:
        raise ValueError("Overlapping label windows")

    model = RandomForestClassifier(
        **config["model"]["parameters"]
    )

    for key, value in config["model"]["parameters"].items():
        if model.get_params()[key] != value:
            raise ValueError(f"Frozen parameter mismatch: {key}")

    return train_count, test_count


def ranking_result(labels, scores, customer_ids, budget):
    # Highest scores first; customer ID resolves score ties.
    order = np.lexsort((
        np.asarray(customer_ids, dtype=str),
        -np.asarray(scores, dtype=float),
    ))

    captured = int(labels[order[:budget]].sum())
    total_inactive = int(labels.sum())
    base_rate = total_inactive / len(labels)

    precision_at_k = captured / budget
    recall_at_k = captured / total_inactive

    return {
        "captured": captured,
        "precision_at_k": round(precision_at_k, 4),
        "recall_at_k": round(recall_at_k, 4),
        "lift_at_k": round(
            precision_at_k / base_rate, 4
        ),
    }


def evaluate(config, train_count, test_count):
    if OUTPUT.exists():
        raise FileExistsError(
            "Final test report already exists. "
            "Do not overwrite or repeatedly evaluate."
        )

    columns = FEATURES + [
        TARGET,
        "customer_id",
        "as_of_date",
        "split",
    ]

    data = pd.read_csv(
        INPUT,
        usecols=columns,
        dtype={"customer_id": "string"},
    )

    if data[columns].isna().any().any():
        raise ValueError("Unexpected missing data")

    if not np.isfinite(
        data[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite features detected")

    train = data[
        data["split"].isin(
            config["splits"]["final_training_splits"]
        )
    ].copy()

    test = data[data["split"] == "test"].copy()

    if len(train) != train_count or len(test) != test_count:
        raise ValueError("Snapshot counts changed")

    if test["customer_id"].duplicated().any():
        raise ValueError("Duplicate test customer IDs")

    if set(train[TARGET].unique()) != {0, 1}:
        raise ValueError("Training labels must be binary")

    if set(test[TARGET].unique()) != {0, 1}:
        raise ValueError("Test labels must be binary")

    model = RandomForestClassifier(
        **config["model"]["parameters"]
    )

    print("Fitting frozen model on train + validation...")
    model.fit(train[FEATURES], train[TARGET])

    print("Evaluating September 2011 held-out test...")

    y = test[TARGET].to_numpy(dtype=int)

    scores = model.predict_proba(
        test[FEATURES]
    )[:, list(model.classes_).index(1)]

    threshold = config["model"]["classification_threshold"]
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    budget = config["evaluation"]["contact_budget"]
    ids = test["customer_id"].to_numpy()

    rf_ranking = ranking_result(
        y, scores, ids, budget
    )

    recency_ranking = ranking_result(
        y,
        test["recency_days"].to_numpy(),
        ids,
        budget,
    )

    with INPUT.open("rb") as file:
        dataset_hash = hashlib.file_digest(
            file, "sha256"
        ).hexdigest()

    report = {
        "evaluation_type": "final_held_out_test",
        "dataset_sha256": dataset_hash,
        "frozen_config": "configs/final_model.json",
        "training_rows": len(train),
        "test_rows": len(test),
        "test_date": config["splits"]["test_date"],
        "test_inactivity_rate": round(float(y.mean()), 4),
        "model": config["model"],
        "metrics": {
            "roc_auc": round(float(
                roc_auc_score(y, scores)
            ), 4),
            "average_precision": round(float(
                average_precision_score(y, scores)
            ), 4),
            "brier_score": round(float(
                brier_score_loss(y, scores)
            ), 4),
            "balanced_accuracy": round(float(
                balanced_accuracy_score(y, predictions)
            ), 4),
            "precision": round(float(
                precision_score(y, predictions, zero_division=0)
            ), 4),
            "recall": round(float(
                recall_score(y, predictions, zero_division=0)
            ), 4),
            "f1": round(float(
                f1_score(y, predictions, zero_division=0)
            ), 4),
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "ranking_at_500": {
            "random_forest": rf_ranking,
            "recency_baseline": recency_ranking,
            "captured_difference": (
                rf_ranking["captured"]
                - recency_ranking["captured"]
            ),
            "tie_breaker": "customer_id_ascending",
        },
        "test_driven_tuning": False,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))
    print("\nFinal held-out evaluation complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["check", "evaluate"],
    )

    args = parser.parse_args()

    config = load_config()
    train_count, test_count = check_metadata(config)

    print("=" * 60)
    print("FINAL MODEL - FROZEN EVALUATION")
    print("=" * 60)
    print(f"Training snapshots: {train_count:,}")
    print(f"Test snapshots: {test_count:,}")
    print("Frozen parameters verified.")

    if args.mode == "check":
        print("Preflight passed.")
        print("Test labels and predictions NOT evaluated.")
        return

    evaluate(config, train_count, test_count)


if __name__ == "__main__":
    main()