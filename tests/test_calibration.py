import numpy as np

from scripts.calibrate_random_forest import (
    calculate_metrics,
    logit_transform,
)


def test_logit_transform_shape():
    probabilities = [0.2, 0.5, 0.8]

    result = logit_transform(probabilities)

    assert result.shape == (3, 1)
    assert np.isfinite(result).all()


def test_logit_transform_clips_extreme_values():
    probabilities = [0.0, 0.5, 1.0]

    result = logit_transform(probabilities)

    assert np.isfinite(result).all()
    assert result[0, 0] < result[1, 0]
    assert result[1, 0] < result[2, 0]
    assert np.isclose(result[1, 0], 0.0)


def test_calibration_metrics():
    y_true = np.array([0, 0, 1, 1])

    probabilities = np.array([
        0.1, 0.2, 0.8, 0.9
    ])

    metrics = calculate_metrics(
        y_true, probabilities
    )

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert np.isclose(
        metrics["brier_score"], 0.025
    )
    assert 0 < metrics["log_loss"] < 0.3