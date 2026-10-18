import pandas as pd


def build_customer_features(
    transactions,
    as_of_date,
    lookback_days=180,
):
    """Build past-only customer behavior features."""

    cutoff = pd.Timestamp(as_of_date)
    start = cutoff - pd.Timedelta(days=lookback_days)

    required = {
        "customer_id",
        "invoice",
        "invoice_date",
        "stock_code",
        "quantity",
        "line_revenue",
    }

    missing = required - set(transactions.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = transactions.copy()

    data["invoice_date"] = pd.to_datetime(
        data["invoice_date"],
        errors="raise",
    )

    for col in ["quantity", "line_revenue"]:
        data[col] = pd.to_numeric(data[col], errors="raise")

    # Strictly historical data only.
    data = data.loc[
        (data["invoice_date"] >= start)
        & (data["invoice_date"] < cutoff)
    ].copy()

    if data.empty:
        raise ValueError("No purchases in observation window")

    if data["customer_id"].isna().any():
        raise ValueError("Missing customer IDs")

    if (
        (data["quantity"] <= 0).any()
        or (data["line_revenue"] <= 0).any()
    ):
        raise ValueError("Invalid purchase values")

    data["purchase_date"] = (
        data["invoice_date"].dt.normalize()
    )

    features = data.groupby("customer_id").agg(
        frequency_180d=("invoice", "nunique"),
        monetary_180d=("line_revenue", "sum"),
        quantity_180d=("quantity", "sum"),
        active_days_180d=("purchase_date", "nunique"),
        unique_products_180d=("stock_code", "nunique"),
        first_purchase=("purchase_date", "min"),
        last_purchase=("purchase_date", "max"),
    )

    features["recency_days"] = (
        cutoff - features.pop("last_purchase")
    ).dt.days

    features["tenure_180d"] = (
        cutoff - features.pop("first_purchase")
    ).dt.days

    features["avg_order_value_180d"] = (
        features["monetary_180d"]
        / features["frequency_180d"]
    )

    # Recent purchase activity.
    for days in [30, 90]:
        recent = data.loc[
            data["invoice_date"]
            >= cutoff - pd.Timedelta(days=days)
        ]

        group = recent.groupby("customer_id")

        features[f"orders_{days}d"] = (
            group["invoice"]
            .nunique()
            .reindex(features.index, fill_value=0)
        )

        features[f"spend_{days}d"] = (
            group["line_revenue"]
            .sum()
            .reindex(features.index, fill_value=0)
        )

    # Spending concentration in the last 30 days.
    features["spend_share_30d"] = (
        features["spend_30d"]
        / features["monetary_180d"]
    )

    # Repeat purchase behavior.
    purchase_days = (
        data[["customer_id", "purchase_date"]]
        .drop_duplicates()
        .sort_values(["customer_id", "purchase_date"])
    )

    purchase_days["gap_days"] = (
        purchase_days.groupby("customer_id")[
            "purchase_date"
        ].diff().dt.days
    )

    mean_gaps = (
        purchase_days.groupby("customer_id")["gap_days"]
        .mean()
        .reindex(features.index)
    )

    features["has_repeat_purchase_day_180d"] = (
        features["active_days_180d"] > 1
    ).astype("int8")

    # 0 means no observed repeat gap; use with indicator.
    features["avg_repeat_gap_days_180d"] = (
        mean_gaps.fillna(0.0)
    )

    return features.reset_index()