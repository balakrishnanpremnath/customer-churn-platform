import pandas as pd
import pytest

from scripts.customer_features import (
    build_customer_features,
)


def transaction(customer, invoice, timestamp, product, revenue):
    return {
        "customer_id": customer,
        "invoice": invoice,
        "invoice_date": timestamp,
        "stock_code": product,
        "quantity": 1,
        "line_revenue": revenue,
    }


def test_rfm_feature_calculations():
    data = pd.DataFrame([
        transaction("A", "A1", "2011-02-21", "X", 10),
        transaction("A", "A2", "2011-05-12", "Y", 30),
        transaction("A", "A2", "2011-05-12", "Z", 20),
        transaction("B", "B1", "2011-05-22", "P", 40),
    ])

    result = build_customer_features(
        data, "2011-06-01"
    ).set_index("customer_id")

    assert result.loc["A", "frequency_180d"] == 2
    assert result.loc["A", "monetary_180d"] == 60
    assert result.loc["A", "unique_products_180d"] == 3
    assert result.loc["A", "recency_days"] == 20
    assert result.loc["A", "avg_order_value_180d"] == 30
    assert result.loc["A", "orders_30d"] == 1
    assert result.loc["A", "spend_30d"] == 50
    assert result.loc["A", "avg_repeat_gap_days_180d"] == 80

    assert result.loc["B", "frequency_180d"] == 1
    assert result.loc["B", "has_repeat_purchase_day_180d"] == 0


def test_future_purchases_do_not_leak():
    history = pd.DataFrame([
        transaction("A", "A1", "2011-05-01", "X", 10),
    ])

    future = pd.DataFrame([
        transaction("A", "A2", "2011-06-02", "X", 9999),
    ])

    before = build_customer_features(
        history, "2011-06-01"
    )

    after = build_customer_features(
        pd.concat([history, future]),
        "2011-06-01",
    )

    pd.testing.assert_frame_equal(
        before.reset_index(drop=True),
        after.reset_index(drop=True),
    )


def test_empty_history_is_rejected():
    data = pd.DataFrame([
        transaction("A", "A1", "2011-07-01", "X", 10),
    ])

    with pytest.raises(ValueError):
        build_customer_features(data, "2011-06-01")