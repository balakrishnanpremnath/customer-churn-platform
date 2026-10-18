import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from scripts.train_baseline import FEATURES, TARGET
from scripts.tune_random_forest import (
    build_model,
    validate_temporal_fold,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

DATASET = PROCESSED / "customer_churn_dataset.csv"
TUNING = PROCESSED / "rf_tuning_results.json"
OUTPUT = PROCESSED / "calibration_results.json"


def logit_transform(probabilities):
    p = np.asarray(probabilities, dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    return np.log(p / (1 - p)).reshape(-1, 1)


def calculate_metrics(y_true, probabilities):
    return {
        "roc_auc": round(
            float(roc_auc_score(y_true, probabilities)), 4
        ),
        "pr_auc": round(
            float(average_precision_score(y_true, probabilities)), 4
        ),
        "brier_score": round(
            float(brier_score_loss(y_true, probabilities)), 4
        ),
        "log_loss": round(
            float(log_loss(y_true, probabilities)), 4
        ),
    }


def main():
    if not DATASET.is_file():
        raise FileNotFoundError(DATASET)

    if not TUNING.is_file():
        raise FileNotFoundError(TUNING)

    tuning = json.loads(TUNING.read_text(encoding="utf-8"))
    params = tuning["selected_parameters"]

    data = pd.read_csv(
        DATASET,
        usecols=FEATURES + [TARGET, "as_of_date", "split"],
    )

    data["as_of_date"] = pd.to_datetime(data["as_of_date"])

    train = data.loc[data["split"] == "train"].copy()
    validation = data.loc[
        data["split"] == "validation"
    ].copy()

    if train.empty or validation.empty:
        raise ValueError("Missing training or validation records")

    for frame in (train, validation):
        if not np.isfinite(
            frame[FEATURES].to_numpy(dtype=float)
        ).all():
            raise ValueError("Invalid feature values")

        if set(frame[TARGET].unique()) != {0, 1}:
            raise ValueError("Both churn classes are required")

    dates = sorted(train["as_of_date"].unique())

    if len(dates) != 4 or validation["as_of_date"].nunique() != 1:
        raise ValueError("Unexpected temporal split structure")

    validation_date = validation["as_of_date"].iloc[0]
    validate_temporal_fold(dates, validation_date)

    oof_probabilities = []
    oof_labels = []

    print("=" * 60)
    print("CUSTOMER CHURN - PROBABILITY CALIBRATION")
    print("=" * 60)

    # Generate chronological out-of-fold predictions.
    for i in range(1, len(dates)):
        past_dates = dates[:i]
        future_date = dates[i]

        validate_temporal_fold(past_dates, future_date)

        fold_train = train.loc[
            train["as_of_date"].isin(past_dates)
        ]

        fold_val = train.loc[
            train["as_of_date"] == future_date
        ]

        model = build_model(params)
        model.fit(
            fold_train[FEATURES],
            fold_train[TARGET],
        )

        probabilities = model.predict_proba(
            fold_val[FEATURES]
        )[:, 1]

        oof_probabilities.append(probabilities)
        oof_labels.append(fold_val[TARGET].to_numpy())

        print(
            f"Calibration fold: "
            f"{pd.Timestamp(future_date).date()} | "
            f"Rows: {len(fold_val):,}"
        )

    oof_p = np.concatenate(oof_probabilities)
    oof_y = np.concatenate(oof_labels)

    if len(np.unique(oof_y)) != 2:
        raise ValueError("Calibration requires both classes")

    # Fit sigmoid calibration using out-of-fold scores.
    calibrator = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    calibrator.fit(logit_transform(oof_p), oof_y)

    if calibrator.coef_[0, 0] <= 0:
        raise ValueError("Unexpected inverse calibration slope")

    # Fit final candidate using all training snapshots.
    final_model = build_model(params)
    final_model.fit(train[FEATURES], train[TARGET])

    raw_probabilities = final_model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    calibrated_probabilities = calibrator.predict_proba(
        logit_transform(raw_probabilities)
    )[:, 1]

    y_validation = validation[TARGET]

    raw_metrics = calculate_metrics(
        y_validation, raw_probabilities
    )

    calibrated_metrics = calculate_metrics(
        y_validation, calibrated_probabilities
    )

    report = {
        "method": "chronological_oof_sigmoid_calibration",
        "selected_rf_parameters": params,
        "calibration_rows": int(len(oof_y)),
        "validation_rows": int(len(validation)),
        "uncalibrated_validation": raw_metrics,
        "calibrated_validation": calibrated_metrics,
        "calibrator_coefficient": float(
            calibrator.coef_[0, 0]
        ),
        "calibrator_intercept": float(
            calibrator.intercept_[0]
        ),
        "held_out_test_evaluated": False,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nUNCALIBRATED VALIDATION")
    print(json.dumps(raw_metrics, indent=4))

    print("\nCALIBRATED VALIDATION")
    print(json.dumps(calibrated_metrics, indent=4))

    print(f"\nCalibration rows: {len(oof_y):,}")
    print(f"Report saved: {OUTPUT}")
    print("Held-out test set not evaluated.")
    print("Calibration completed successfully!")


if __name__ == "__main__":
    main()