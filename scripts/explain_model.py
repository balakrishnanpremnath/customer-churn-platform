import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier

from scripts.final_heldout_evaluation import load_config
from scripts.train_baseline import FEATURES, TARGET


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "customer_churn_dataset.csv"
OUTPUT = ROOT / "data" / "processed" / "shap_explanations.json"

SAMPLE_SIZE = 100
RANDOM_STATE = 42


def extract_positive_class(explainer, features, model):
    """Return SHAP contributions for inactivity class 1."""

    values = np.asarray(
        explainer.shap_values(features)
    )

    expected = np.asarray(
        explainer.expected_value
    )

    class_index = list(model.classes_).index(1)

    expected_shape = (
        len(features),
        len(FEATURES),
        len(model.classes_),
    )

    if values.shape != expected_shape:
        raise ValueError(
            f"Unexpected SHAP shape: {values.shape}"
        )

    if expected.shape != (len(model.classes_),):
        raise ValueError("Unexpected SHAP base values")

    contributions = values[:, :, class_index]
    base_value = float(expected[class_index])

    return contributions, base_value


def main():
    config = load_config()

    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    if config["model"]["probability_calibration"]:
        raise ValueError("Unexpected calibration setting")

    # Explain the frozen model using train + validation only.
    data = pd.read_csv(
        INPUT,
        usecols=FEATURES + [TARGET, "split"],
    )

    training = data.loc[
        data["split"].isin(
            config["splits"]["final_training_splits"]
        ),
        FEATURES + [TARGET],
    ].copy()

    expected_rows = config["splits"][
        "expected_final_training_rows"
    ]

    if len(training) != expected_rows:
        raise ValueError("Training snapshot count mismatch")

    if training.isna().any().any():
        raise ValueError("Missing training values")

    if not np.isfinite(
        training[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid feature values")

    if set(training[TARGET].unique()) != {0, 1}:
        raise ValueError("Invalid target classes")

    model = RandomForestClassifier(
        **config["model"]["parameters"]
    )

    print("=" * 60)
    print("SHAP EXPLAINABLE AI ANALYSIS")
    print("=" * 60)

    print(f"Training snapshots: {len(training):,}")
    print(f"Features: {len(FEATURES)}")
    print("Training frozen Random Forest...")

    model.fit(
        training[FEATURES],
        training[TARGET],
    )

    # A reproducible sample, not the held-out test.
    sample = (
        training[FEATURES]
        .sample(
            n=min(SAMPLE_SIZE, len(training)),
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    print(f"Explaining {len(sample)} training snapshots...")

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )

    contributions, base_value = extract_positive_class(
        explainer,
        sample,
        model,
    )

    class_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(
        sample
    )[:, class_index]

    # SHAP contributions must reconstruct probabilities.
    reconstructed = (
        base_value
        + contributions.sum(axis=1)
    )

    max_error = float(
        np.max(np.abs(reconstructed - probabilities))
    )

    if not np.allclose(
        reconstructed,
        probabilities,
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("SHAP additivity check failed")

    mean_importances = np.mean(
        np.abs(contributions),
        axis=0,
    )

    global_importance = sorted(
        [
            {
                "feature": feature,
                "mean_abs_shap": round(
                    float(importance), 6
                ),
            }
            for feature, importance in zip(
                FEATURES, mean_importances
            )
        ],
        key=lambda item: item["mean_abs_shap"],
        reverse=True,
    )

    example_contributions = sorted(
        [
            {
                "feature": feature,
                "feature_value": float(
                    sample.iloc[0][feature]
                ),
                "shap_contribution": round(
                    float(contributions[0, index]), 6
                ),
            }
            for index, feature in enumerate(FEATURES)
        ],
        key=lambda item: abs(
            item["shap_contribution"]
        ),
        reverse=True,
    )

    report = {
        "model": "frozen_random_forest",
        "explained_class": "90_day_inactivity",
        "source": "train_and_validation_only",
        "training_snapshots": len(training),
        "explained_snapshots": len(sample),
        "sample_random_state": RANDOM_STATE,
        "shap_method": "tree_path_dependent",
        "max_additivity_error": max_error,
        "global_feature_importance": global_importance,
        "example_prediction": {
            "base_probability": round(base_value, 6),
            "predicted_probability": round(
                float(probabilities[0]), 6
            ),
            "feature_contributions": example_contributions,
        },
        "held_out_test_explained": False,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nTOP 5 GLOBAL FEATURES")

    for index, item in enumerate(
        global_importance[:5],
        start=1,
    ):
        print(
            f"{index}. {item['feature']}: "
            f"{item['mean_abs_shap']:.6f}"
        )

    print("\nEXAMPLE CUSTOMER RISK")
    print(f"Base probability: {base_value:.4f}")
    print(f"Predicted probability: {probabilities[0]:.4f}")
    print(f"Maximum additivity error: {max_error:.8f}")

    print(f"\nReport saved: {OUTPUT}")
    print("Held-out test NOT used for SHAP analysis.")
    print("SHAP analysis completed!")


if __name__ == "__main__":
    main()