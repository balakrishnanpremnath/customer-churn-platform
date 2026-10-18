from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "purchases_clean.csv"
OUTPUT = ROOT / "data" / "processed" / "customer_churn_dataset.csv"

LOOKBACK_DAYS = 180
HORIZON_DAYS = 90

SNAPSHOTS = [
    ("2010-06-01", "train"),
    ("2010-09-01", "train"),
    ("2010-12-01", "train"),
    ("2011-03-01", "train"),
    ("2011-06-01", "validation"),
    ("2011-09-01", "test"),
]


def build_snapshot(data, date, split):
    cutoff = pd.Timestamp(date)

    history_start = cutoff - pd.Timedelta(
        days=LOOKBACK_DAYS
    )
    outcome_end = cutoff + pd.Timedelta(
        days=HORIZON_DAYS
    )

    # Only past purchases are used for ML features.
    history = data.loc[
        (data["invoice_date"] >= history_start)
        & (data["invoice_date"] < cutoff)
    ]

    # Future purchases are used only for labels.
    future_customers = data.loc[
        (data["invoice_date"] >= cutoff)
        & (data["invoice_date"] < outcome_end),
        "customer_id",
    ].unique()

    if history.empty:
        raise ValueError(f"No history for {date}")

    features = history.groupby("customer_id").agg(
        last_purchase=("invoice_date", "max"),
        frequency_180d=("invoice", "nunique"),
        monetary_180d=("line_revenue", "sum"),
        quantity_180d=("quantity", "sum"),
        active_days_180d=("purchase_date", "nunique"),
    )

    features["recency_days"] = (
        cutoff - features["last_purchase"].dt.normalize()
    ).dt.days

    features["churn_90d"] = (
        ~features.index.isin(future_customers)
    ).astype("int8")

    features["as_of_date"] = date
    features["split"] = split

    return (
        features.drop(columns=["last_purchase"])
        .reset_index()
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    data = pd.read_csv(
        INPUT,
        usecols=[
            "customer_id",
            "invoice",
            "invoice_date",
            "quantity",
            "line_revenue",
        ],
        dtype={
            "customer_id": "string",
            "invoice": "string",
        },
    )

    data["invoice_date"] = pd.to_datetime(
        data["invoice_date"],
        format="ISO8601",
        errors="raise",
    )

    for column in ["quantity", "line_revenue"]:
        data[column] = pd.to_numeric(
            data[column], errors="raise"
        )

    if data.isna().any().any():
        raise ValueError("Unexpected missing values detected")

    if (
        (data["quantity"] <= 0).any()
        or (data["line_revenue"] <= 0).any()
    ):
        raise ValueError("Invalid purchase values detected")

    data["purchase_date"] = (
        data["invoice_date"].dt.normalize()
    )

    earliest_required = (
        pd.Timestamp(SNAPSHOTS[0][0])
        - pd.Timedelta(days=LOOKBACK_DAYS)
    )
    latest_required = (
        pd.Timestamp(SNAPSHOTS[-1][0])
        + pd.Timedelta(days=HORIZON_DAYS)
    )

    if data["invoice_date"].min() > earliest_required:
        raise ValueError("Incomplete observation history")

    if data["invoice_date"].max() < latest_required:
        raise ValueError("Incomplete outcome period")

    snapshots = [
        build_snapshot(data, date, split)
        for date, split in SNAPSHOTS
    ]

    result = pd.concat(snapshots, ignore_index=True)

    if result.duplicated(
        ["as_of_date", "customer_id"]
    ).any():
        raise ValueError("Duplicate customer snapshots")

    if not result["churn_90d"].isin([0, 1]).all():
        raise ValueError("Invalid churn labels")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    summary = result.groupby(
        ["split", "as_of_date"], sort=False
    ).agg(
        customers=("customer_id", "size"),
        churn_rate=("churn_90d", "mean"),
    )

    summary["churn_rate"] = (
        summary["churn_rate"] * 100
    ).round(2)

    print("=" * 55)
    print("CUSTOMER CHURN DATASET")
    print("=" * 55)
    print(summary.to_string())
    print(f"\nTotal customer snapshots: {len(result):,}")
    print(f"Output: {OUTPUT}")
    print("Dataset generation successful!")


if __name__ == "__main__":
    main()