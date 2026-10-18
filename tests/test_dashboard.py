
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "app" / "dashboard.py"


def launch_dashboard():
    return AppTest.from_file(
        DASHBOARD,
        default_timeout=20,
    ).run()


def test_overview_page():
    app = launch_dashboard()

    assert len(app.exception) == 0

    assert app.title[0].value == (
        "Customer Churn Intelligence Platform"
    )

    assert len(app.metric) == 4


def test_prediction_page():
    app = launch_dashboard()

    app.radio[0].set_value(
        "Customer Risk Prediction"
    ).run()

    assert len(app.exception) == 0
    assert app.title[0].value == (
        "Customer Risk Prediction"
    )

    assert len(app.number_input) == 15


def test_explainable_ai_page():
    app = launch_dashboard()

    app.radio[0].set_value(
        "Explainable AI"
    ).run()

    assert len(app.exception) == 0
    assert app.title[0].value == (
        "Explainable AI — SHAP Analysis"
    )
