
import pandas as pd
import pytest

from scripts.build_analytics import build_analytics


@pytest.fixture
def sample_purchases():
    return pd.DataFrame({
        "invoice": ["A", "A", "B", "A", "C"],
        "invoice_date": [
            "2011-01-05",
            "2011-01-05",
            "2011-01-18",
            "2011-02-02",
            "2011-02-07",
        ],
        "customer_id": [
            "C1", "C1", "C2", "C1", "C2"
        ],
        "country": [
            "UK", "UK", "France", "UK", "France"
        ],
        "quantity": [2, 1, 3, 4, 1],
        "line_revenue": [20, 5, 30, 40, 10],
    })


def test_overall_summary(sample_purchases):
    summary, _, _ = build_analytics(sample_purchases)

    assert summary["purchase_lines"] == 5
    assert summary["unique_customers"] == 2
    assert summary["invoice_month_combinations"] == 4
    assert summary["total_units"] == 11
    assert summary["total_revenue"] == 105.0
    assert summary["countries"] == 2


def test_monthly_aggregation(sample_purchases):
    _, monthly, _ = build_analytics(sample_purchases)

    assert len(monthly) == 2

    january = monthly.loc[
        monthly["month"] == "2011-01"
    ].iloc[0]

    february = monthly.loc[
        monthly["month"] == "2011-02"
    ].iloc[0]

    assert january["revenue"] == 55.0
    assert january["invoices"] == 2
    assert january["customers"] == 2

    assert february["revenue"] == 50.0
    assert february["invoices"] == 2


def test_country_aggregation(sample_purchases):
    _, _, countries = build_analytics(
        sample_purchases
    )

    assert countries.iloc[0]["country"] == "UK"
    assert countries.iloc[0]["revenue"] == 65.0

    france = countries.loc[
        countries["country"] == "France"
    ].iloc[0]

    assert france["revenue"] == 40.0


def test_empty_dataset_rejected():
    with pytest.raises(ValueError):
        build_analytics(pd.DataFrame())


def test_missing_column_rejected(sample_purchases):
    invalid = sample_purchases.drop(
        columns=["customer_id"]
    )

    with pytest.raises(ValueError, match="Missing columns"):
        build_analytics(invalid)


@pytest.mark.parametrize(
    "column,value",
    [
        ("quantity", 0),
        ("line_revenue", -10),
        ("country", None),
    ],
)
def test_invalid_data_rejected(
    sample_purchases, column, value
):
    invalid = sample_purchases.copy()
    invalid.loc[0, column] = value

    with pytest.raises(ValueError):
        build_analytics(invalid)
