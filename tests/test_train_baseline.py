import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from scripts.train_baseline import FEATURES, TARGET, evaluate


def test_feature_list_excludes_leakage():
    forbidden = {
        "customer_id",
        "churn_90d",
        "as_of_date",
        "split",
        "invoice_date",
    }

    assert len(FEATURES) == 15
    assert len(FEATURES) == len(set(FEATURES))
    assert not forbidden.intersection(FEATURES)
    assert TARGET == "churn_90d"


def test_dummy_classifier_is_random_ranker():
    x = np.arange(6).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1])

    model = DummyClassifier(strategy="prior")
    model.fit(x, y)

    metrics = evaluate(model, x, y)

    assert metrics["roc_auc"] == 0.5
    assert metrics["pr_auc"] == 0.5
    assert metrics["brier_score"] == 0.25


def test_logistic_regression_model_evaluation():
    x = np.arange(6).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression(max_iter=1000)
    model.fit(x, y)

    metrics = evaluate(model, x, y)

    required = {
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "brier_score",
    }

    assert set(metrics) == required
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["roc_auc"] > 0.5
    assert 0 <= metrics["brier_score"] <= 1