from pathlib import Path

import pandas as pd

from scripts.build_churn_dataset import (
    build_snapshot,
    SNAPSHOTS,
    LOOKBACK_DAYS,
    HORIZON_DAYS,
)


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT / "data" / "processed"
    / "customer_churn_dataset.csv"
)


def purchase(customer, invoice, timestamp, revenue=10):
    timestamp = pd.Timestamp(timestamp)

    return {
        "customer_id": customer,
        "invoice": invoice,
        "invoice_date": timestamp,
        "purchase_date": timestamp.normalize(),
        "quantity": 1,
        "line_revenue": revenue,
        "stock_code": "TEST-SKU",
    }


def test_temporal_boundaries_and_leakage():
    cutoff = pd.Timestamp("2011-06-01")
    start = cutoff - pd.Timedelta(days=LOOKBACK_DAYS)
    end = cutoff + pd.Timedelta(days=HORIZON_DAYS)

    rows = [
        # A: Old purchase excluded from features.
        purchase("A", "old", start - pd.Timedelta(seconds=1), 999),

        # A: First permitted historical purchase.
        purchase("A", "a1", start, 10),

        # A: Purchase at prediction time affects label only.
        purchase("A", "a2", cutoff, 5000),

        # B: Purchase exactly at horizon end is too late.
        purchase("B", "b1", cutoff - pd.Timedelta(days=1)),
        purchase("B", "b2", end),

        # C: Only future activity, not eligible.
        purchase("C", "c1", cutoff),

        # D: Purchase just before horizon end counts.
        purchase("D", "d1", cutoff - pd.Timedelta(days=3)),
        purchase("D", "d2", end - pd.Timedelta(seconds=1)),
    ]

    data = pd.DataFrame(rows)

    result = build_snapshot(
        data,
        "2011-06-01",
        "validation",
    ).set_index("customer_id")

    assert set(result.index) == {"A", "B", "D"}

    # Only historical values can enter features.
    assert result.loc["A", "monetary_180d"] == 10
    assert result.loc["A", "frequency_180d"] == 1
    assert result.loc["A", "recency_days"] == 180

    # Future 90-day purchase outcomes.
    assert result.loc["A", "churn_90d"] == 0
    assert result.loc["B", "churn_90d"] == 1
    assert result.loc["D", "churn_90d"] == 0


def test_snapshot_dates_are_chronological():
    dates = [pd.Timestamp(date) for date, _ in SNAPSHOTS]

    assert dates == sorted(dates)
    assert len(dates) == len(set(dates))

    for current, following in zip(dates, dates[1:]):
        assert (
            current + pd.Timedelta(days=HORIZON_DAYS)
            <= following
        )


def test_generated_dataset_integrity():
    assert DATASET.is_file()

    df = pd.read_csv(
        DATASET,
        dtype={"customer_id": "string"},
    )

    assert not df.empty

    assert not df.duplicated(
        ["customer_id", "as_of_date"]
    ).any()

    assert df["churn_90d"].isin([0, 1]).all()

    assert df["recency_days"].between(
        1, LOOKBACK_DAYS
    ).all()

    assert (df["frequency_180d"] >= 1).all()
    assert (df["monetary_180d"] > 0).all()
    assert (df["active_days_180d"] >= 1).all()

    assert not df.isna().any().any()

    assert set(df["split"]) == {
        "train", "validation", "test"
    }

    expected_dates = {
        date for date, _ in SNAPSHOTS
    }
    assert set(df["as_of_date"]) == expected_dates


def test_expected_split_counts():
    df = pd.read_csv(DATASET)

    assert len(df) == 17759

    counts = df["split"].value_counts()

    assert counts["train"] == 12328
    assert counts["validation"] == 2659
    assert counts["test"] == 2772

def test_advanced_feature_integrity():
    df = pd.read_csv(DATASET)

    required = {
        "unique_products_180d",
        "avg_order_value_180d",
        "tenure_180d",
        "orders_30d",
        "orders_90d",
        "spend_30d",
        "spend_90d",
        "spend_share_30d",
        "has_repeat_purchase_day_180d",
        "avg_repeat_gap_days_180d",
    }

    assert required.issubset(df.columns)

    assert (
        df["orders_30d"] <= df["orders_90d"]
    ).all()

    assert (
        df["orders_90d"] <= df["frequency_180d"]
    ).all()

    assert df["spend_share_30d"].between(
        0, 1.000001
    ).all()

    assert (
        df["tenure_180d"] >= df["recency_days"]
    ).all()

    assert not df[list(required)].isna().any().any()