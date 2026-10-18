import numpy as np
import pytest

from scripts.bootstrap_uncertainty import (
    captures_at_k,
    confidence_interval,
)


def test_top_k_captures():
    labels = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.2])

    assert captures_at_k(labels, scores, 2) == 2
    assert captures_at_k(labels, scores, 1) == 1


def test_tied_scores_are_deterministic():
    labels = np.array([1, 0, 1])
    scores = np.array([0.9, 0.9, 0.1])

    assert captures_at_k(labels, scores, 1) == 1


def test_invalid_bootstrap_inputs():
    labels = np.array([0, 1, 1])
    scores = np.array([0.1, 0.8, 0.9])

    with pytest.raises(ValueError):
        captures_at_k(labels, scores, 0)

    with pytest.raises(ValueError):
        captures_at_k(labels, scores, 4)

    with pytest.raises(ValueError):
        captures_at_k(labels, np.array([0.1]), 1)

    with pytest.raises(ValueError):
        captures_at_k(
            labels,
            np.array([0.1, np.nan, 0.9]),
            2,
        )


def test_confidence_interval_calculation():
    values = np.array([0, 1, 2, 3, 4])

    result = confidence_interval(values)

    assert result == [0.1, 3.9]
    assert result[0] <= result[1]