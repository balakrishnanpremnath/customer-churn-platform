import json
from pathlib import Path

import pandas as pd

from scripts.train_baseline import FEATURES, TARGET
from scripts.tune_random_forest import build_model


ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "configs" / "final_model.json"

DATASET = (
    ROOT / "data" / "processed"
    / "customer_churn_dataset.csv"
)


def load_config():
    return json.loads(
        CONFIG.read_text(encoding="utf-8")
    )


def test_frozen_features_and_target():
    config = load_config()

    assert config["features"] == FEATURES
    assert config["target"] == TARGET
    assert len(config["features"]) == 15


def test_frozen_model_parameters():
    config = load_config()

    params = config["model"]["parameters"]

    model = build_model({
        "max_depth": params["max_depth"],
        "min_samples_leaf": params["min_samples_leaf"],
    })

    actual_params = model.get_params()

    for name, expected in params.items():
        assert actual_params[name] == expected

    assert config["model"]["probability_calibration"] is False


def test_chronological_outcome_windows():
    config = load_config()

    splits = config["splits"]
    horizon = pd.Timedelta(
        days=config["prediction_horizon_days"]
    )

    train_dates = [
        pd.Timestamp(date)
        for date in splits["training_dates"]
    ]

    validation_date = pd.Timestamp(
        splits["validation_date"]
    )

    test_date = pd.Timestamp(
        splits["test_date"]
    )

    assert max(train_dates) + horizon <= validation_date
    assert validation_date + horizon <= test_date

    assert splits["final_training_splits"] == [
        "train",
        "validation",
    ]


def test_frozen_snapshot_counts():
    config = load_config()

    metadata = pd.read_csv(
        DATASET,
        usecols=["split", "as_of_date"],
    )

    counts = metadata.groupby("split").size().to_dict()

    assert counts == {
        "train": 12328,
        "validation": 2659,
        "test": 2772,
    }

    assert (
        counts["train"] + counts["validation"]
        == config["splits"]["expected_final_training_rows"]
    )

    assert (
        counts["test"]
        == config["splits"]["expected_test_rows"]
    )