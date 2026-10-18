
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier

from scripts.final_heldout_evaluation import load_config
from scripts.train_baseline import FEATURES, TARGET


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "processed" / "customer_churn_dataset.csv"
FINAL_REPORT = ROOT / "data" / "processed" / "final_test_results.json"

MODEL_PATH = ROOT / "models" / "final_random_forest.joblib"
MANIFEST_PATH = ROOT / "models" / "final_random_forest_metadata.json"


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def main():
    config = load_config()

    if not DATASET.is_file() or not FINAL_REPORT.is_file():
        raise FileNotFoundError(
            "Dataset or final evaluation record is missing"
        )

    if MODEL_PATH.exists() or MANIFEST_PATH.exists():
        raise FileExistsError(
            "Export already exists. Do not overwrite it."
        )

    evaluation = json.loads(
        FINAL_REPORT.read_text(encoding="utf-8")
    )

    dataset_hash = sha256_file(DATASET)

    if evaluation["dataset_sha256"] != dataset_hash:
        raise ValueError(
            "Dataset differs from the final evaluation dataset"
        )

    required_columns = FEATURES + [
        TARGET, "split", "as_of_date"
    ]

    allowed_splits = set(
        config["splits"]["final_training_splits"]
    )

    parts = []

    for chunk in pd.read_csv(
        DATASET,
        usecols=required_columns,
        chunksize=50000,
    ):
        if not set(chunk["split"].unique()).issubset(
            allowed_splits | {"test"}
        ):
            raise ValueError("Unexpected dataset split")

        # Explicitly exclude held-out test rows.
        selected = chunk.loc[
            chunk["split"].isin(allowed_splits)
        ].copy()

        if not selected.empty:
            parts.append(selected)

    if not parts:
        raise ValueError("No training data found")

    training = pd.concat(parts, ignore_index=True)

    expected_rows = config["splits"][
        "expected_final_training_rows"
    ]

    if len(training) != expected_rows:
        raise ValueError("Unexpected training row count")

    expected_dates = {
        "train": config["splits"]["training_dates"],
        "validation": [
            config["splits"]["validation_date"]
        ],
    }

    for split, dates in expected_dates.items():
        actual = sorted(
            training.loc[
                training["split"] == split,
                "as_of_date"
            ].unique().tolist()
        )

        if actual != sorted(dates):
            raise ValueError(
                f"Unexpected {split} snapshot dates"
            )

    if training[required_columns].isna().any().any():
        raise ValueError("Missing training data")

    features = training[FEATURES].astype(float)
    labels = training[TARGET].astype(int)

    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Invalid numeric features")

    if (features.to_numpy() < 0).any():
        raise ValueError("Negative features detected")

    if set(labels.unique()) != {0, 1}:
        raise ValueError("Unexpected target classes")

    model = RandomForestClassifier(
        **config["model"]["parameters"]
    )

    print("=" * 60)
    print("EXPORT FROZEN RANDOM FOREST")
    print("=" * 60)
    print(f"Training snapshots: {len(training):,}")
    print(f"Features: {len(FEATURES)}")
    print("Training final model...")

    model.fit(features, labels)

    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected model classes")

    bundle = {
        "model": model,
        "features": list(FEATURES),
        "positive_class": 1,
        "threshold": config["model"][
            "classification_threshold"
        ],
        "training_rows": len(training),
        "dataset_sha256": dataset_hash,
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MODEL_PATH.open("xb") as file:
        joblib.dump(bundle, file)

    artifact_hash = sha256_file(MODEL_PATH)

    metadata = {
        "model": "frozen_random_forest",
        "training_rows": len(training),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "parameters": config["model"]["parameters"],
        "positive_class": 1,
        "dataset_sha256": dataset_hash,
        "artifact_sha256": artifact_hash,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "held_out_test_used_for_training": False,
    }

    with MANIFEST_PATH.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"Model saved: {MODEL_PATH}")
    print(f"Metadata saved: {MANIFEST_PATH}")
    print("Held-out test excluded from training.")
    print("Frozen model export completed!")


if __name__ == "__main__":
    main()
