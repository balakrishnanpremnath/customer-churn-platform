import numpy as np
import pytest

from scripts.evaluate_retention import ranking_metrics


def test_perfect_top_two_ranking():
    labels = np.array([0, 1, 0, 1, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3])

    result = ranking_metrics(labels, scores, 2)

    assert result["captured"] == 2
    assert result["precision_at_k"] == 1.0
    assert result["recall_at_k"] == 0.6667
    assert result["lift_at_k"] == 1.6667


def test_poor_ranking():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.9, 0.8])

    result = ranking_metrics(labels, scores, 2)

    assert result["captured"] == 0
    assert result["precision_at_k"] == 0.0
    assert result["lift_at_k"] == 0.0


def test_invalid_contact_budget():
    labels = [0, 1, 1]
    scores = [0.2, 0.8, 0.7]

    with pytest.raises(ValueError):
        ranking_metrics(labels, scores, 0)

    with pytest.raises(ValueError):
        ranking_metrics(labels, scores, 10)


def test_monotonic_scores_preserve_ranking():
    labels = [0, 1, 0, 1]
    scores = np.array([0.1, 0.9, 0.3, 0.7])

    original = ranking_metrics(labels, scores, 2)

    transformed = ranking_metrics(
        labels,
        scores * 2 + 1,
        2,
    )

    assert original == transformed