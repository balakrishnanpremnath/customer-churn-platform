
import numpy as np
import pandas as pd
import pytest

from scripts import explain_model, plot_shap
from scripts.train_baseline import FEATURES


def example_report():
    contributions = [
        {
            "feature": feature,
            "feature_value": float(i + 1),
            "shap_contribution": 0.01,
        }
        for i, feature in enumerate(FEATURES)
    ]

    importance = [
        {
            "feature": feature,
            "mean_abs_shap": float(15 - i) / 1000,
        }
        for i, feature in enumerate(FEATURES)
    ]

    return {
        "global_feature_importance": importance,
        "example_prediction": {
            "base_probability": 0.50,
            "predicted_probability": 0.65,
            "feature_contributions": contributions,
        },
        "source": "train_and_validation_only",
        "held_out_test_explained": False,
    }


def test_positive_class_shap_extraction():
    class FakeModel:
        classes_ = np.array([0, 1])

    class FakeExplainer:
        expected_value = np.array([0.45, 0.55])

        def shap_values(self, features):
            values = np.zeros(
                (len(features), len(FEATURES), 2)
            )
            values[:, :, 1] = 0.01
            return values

    features = pd.DataFrame(
        np.zeros((2, len(FEATURES))),
        columns=FEATURES,
    )

    contributions, base = (
        explain_model.extract_positive_class(
            FakeExplainer(),
            features,
            FakeModel(),
        )
    )

    assert contributions.shape == (2, 15)
    assert np.allclose(contributions, 0.01)
    assert base == pytest.approx(0.55)


def test_global_importance_chart(tmp_path, monkeypatch):
    monkeypatch.setattr(
        plot_shap,
        "OUTPUT_DIR",
        tmp_path,
    )

    output = plot_shap.create_global_chart(
        example_report()
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_local_waterfall_chart(tmp_path, monkeypatch):
    monkeypatch.setattr(
        plot_shap,
        "OUTPUT_DIR",
        tmp_path,
    )

    output = plot_shap.create_waterfall_chart(
        example_report()
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_invalid_shap_additivity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        plot_shap,
        "OUTPUT_DIR",
        tmp_path,
    )

    report = example_report()
    report["example_prediction"][
        "predicted_probability"
    ] = 0.90

    with pytest.raises(
        ValueError,
        match="do not reconstruct prediction",
    ):
        plot_shap.create_waterfall_chart(report)
