
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "purchases_clean.csv"
OUTPUT = ROOT / "data" / "processed"

COLUMNS = [
    "invoice",
    "invoice_date",
    "customer_id",
    "country",
    "quantity",
    "line_revenue",
]


def build_analytics(data):
    if data.empty:
        raise ValueError("Purchase dataset is empty")

    missing = set(COLUMNS) - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    data = data.copy()

    if data[COLUMNS].isna().any().any():
        raise ValueError("Unexpected missing values")

    data["invoice_date"] = pd.to_datetime(
        data["invoice_date"],
        format="ISO8601",
        errors="raise",
    )

    for column in ["quantity", "line_revenue"]:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

        if not (data[column] > 0).all():
            raise ValueError(
                f"Invalid values in {column}"
            )

    data["month"] = (
        data["invoice_date"]
        .dt.strftime("%Y-%m")
    )

    # Invoice identifiers may recur across years.
    # Count an invoice within its transaction month.
    data["invoice_month_key"] = (
        data["month"] + "|" + data["invoice"]
    )

    summary = {
        "data_source": "cleaned_historical_purchases",
        "first_purchase_date": str(
            data["invoice_date"].min().date()
        ),
        "last_purchase_date": str(
            data["invoice_date"].max().date()
        ),
        "purchase_lines": int(len(data)),
        "unique_customers": int(
            data["customer_id"].nunique()
        ),
        "invoice_month_combinations": int(
            data["invoice_month_key"].nunique()
        ),
        "total_units": int(
            data["quantity"].sum()
        ),
        "total_revenue": round(
            float(data["line_revenue"].sum()), 2
        ),
        "countries": int(
            data["country"].nunique()
        ),
    }

    monthly = (
        data.groupby("month", as_index=False)
        .agg(
            purchase_lines=("invoice", "size"),
            invoices=("invoice_month_key", "nunique"),
            customers=("customer_id", "nunique"),
            units=("quantity", "sum"),
            revenue=("line_revenue", "sum"),
        )
        .sort_values("month")
    )

    monthly["revenue"] = (
        monthly["revenue"].round(2)
    )

    monthly["average_invoice_revenue"] = (
        monthly["revenue"] / monthly["invoices"]
    ).round(2)

    countries = (
        data.groupby("country", as_index=False)
        .agg(
            purchase_lines=("invoice", "size"),
            invoices=("invoice_month_key", "nunique"),
            customers=("customer_id", "nunique"),
            revenue=("line_revenue", "sum"),
        )
        .sort_values("revenue", ascending=False)
    )

    countries["revenue"] = (
        countries["revenue"].round(2)
    )

    return summary, monthly, countries


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    print("=" * 60)
    print("CUSTOMER ANALYTICS ENGINE")
    print("=" * 60)

    data = pd.read_csv(
        INPUT,
        usecols=COLUMNS,
        dtype={
            "invoice": "string",
            "customer_id": "string",
            "country": "string",
        },
    )

    summary, monthly, countries = build_analytics(
        data
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT / "analytics_summary.json"
    )

    monthly_path = (
        OUTPUT / "analytics_monthly.csv"
    )

    country_path = (
        OUTPUT / "analytics_countries.csv"
    )

    summary_path.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    monthly.to_csv(
        monthly_path,
        index=False,
    )

    countries.to_csv(
        country_path,
        index=False,
    )

    print(json.dumps(summary, indent=4))

    print("\nLATEST MONTHLY METRICS")
    print(monthly.tail(3).to_string(index=False))

    print("\nTOP 5 COUNTRIES BY REVENUE")
    print(countries.head(5).to_string(index=False))

    print("\nCustomer analytics completed!")
    print("Only aggregated reports were generated.")


if __name__ == "__main__":
    main()
