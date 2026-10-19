
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



def test_historical_analytics_page():
    app = launch_dashboard()

    app.radio[0].set_value(
        "Historical Analytics"
    ).run()

    assert len(app.exception) == 0

    assert app.title[0].value == (
        "Historical Retail Analytics"
    )

    summary_path = (
        ROOT / "data" / "processed"
        / "analytics_summary.json"
    )

    if summary_path.is_file():
        assert len(app.metric) == 3
    else:
        # Fresh GitHub clones may not contain
        # locally generated analytics reports.
        assert len(app.warning) >= 1

    
def test_historical_analytics_chart_sections():
    app = launch_dashboard()

    app.radio[0].set_value(
        "Historical Analytics"
    ).run()

    assert len(app.exception) == 0

    data_dir = ROOT / "data" / "processed"

    summary_path = data_dir / "analytics_summary.json"
    monthly_path = data_dir / "analytics_monthly.csv"
    country_path = data_dir / "analytics_countries.csv"

    if not summary_path.is_file():
        # Fresh clones may not include generated data.
        assert len(app.warning) >= 1
        return

    headings = {
        section.value
        for section in app.subheader
    }

    assert "Monthly Revenue Trend" in headings
    assert "Top 10 Countries by Revenue" in headings

    if monthly_path.is_file():
        assert "Monthly Purchasing Customers" in headings

    if country_path.is_file():
        assert len(app.exception) == 0

