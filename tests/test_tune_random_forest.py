import pytest

from scripts.tune_random_forest import (
    build_model,
    validate_temporal_fold,
)


def test_valid_temporal_fold():
    validate_temporal_fold(
        ["2010-06-01"],
        "2010-09-01",
    )


def test_exact_horizon_boundary_allowed():
    validate_temporal_fold(
        ["2010-12-01"],
        "2011-03-01",
    )


def test_overlapping_label_window_rejected():
    with pytest.raises(ValueError, match="overlapping"):
        validate_temporal_fold(
            ["2011-03-01"],
            "2011-05-01",
        )


def test_random_forest_configuration():
    model = build_model({
        "max_depth": 6,
        "min_samples_leaf": 5,
    })

    assert model.n_estimators == 200
    assert model.max_depth == 6
    assert model.min_samples_leaf == 5
    assert model.random_state == 42