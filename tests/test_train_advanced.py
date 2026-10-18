import numpy as np

from scripts.train_advanced import build_models
from scripts.train_baseline import evaluate


def test_advanced_model_configuration():
    models = build_models(rf_trees=10, xgb_trees=10)

    assert set(models) == {
        "random_forest",
        "xgboost",
    }

    assert models["random_forest"].random_state == 42
    assert models["xgboost"].random_state == 42


def test_advanced_models_can_train_and_predict():
    rng = np.random.default_rng(42)

    x = rng.uniform(0, 10, size=(40, 15))

    y = np.array([0, 1] * 20)

    for model in build_models(
        rf_trees=10,
        xgb_trees=10,
    ).values():

        model.fit(x, y)

        probabilities = model.predict_proba(x)

        assert probabilities.shape == (40, 2)

        assert np.isfinite(probabilities).all()

        assert (
            (probabilities >= 0)
            & (probabilities <= 1)
        ).all()

        metrics = evaluate(model, x, y)

        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1
        assert 0 <= metrics["brier_score"] <= 1