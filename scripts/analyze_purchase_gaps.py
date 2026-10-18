import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "purchases_clean.csv"
OUTPUT = ROOT / "data" / "processed" / "purchase_gap_report.json"


def analyze_purchase_gaps():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(
        INPUT,
        usecols=["customer_id", "invoice_date"],
        dtype={"customer_id": "string"},
    )

    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"],
        format="ISO8601",
        errors="coerce",
    )

    if df["invoice_date"].isna().any():
        raise ValueError("Invalid transaction dates detected.")

    if df["customer_id"].isna().any():
        raise ValueError("Missing customer IDs detected.")

    df["purchase_date"] = df["invoice_date"].dt.normalize()

    purchases = (
        df[["customer_id", "purchase_date"]]
        .drop_duplicates()
        .sort_values(["customer_id", "purchase_date"])
    )

    customer_counts = purchases.groupby("customer_id").size()

    gaps = (
        purchases.groupby("customer_id")["purchase_date"]
        .diff()
        .dt.days
        .dropna()
        .astype("int64")
    )

    if gaps.empty:
        raise ValueError("No repeat purchase gaps available.")

    report = {
        "transaction_rows": int(len(df)),
        "unique_customers": int(len(customer_counts)),
        "distinct_customer_purchase_days": int(len(purchases)),
        "one_day_customers": int((customer_counts == 1).sum()),
        "repeat_customers": int((customer_counts >= 2).sum()),
        "observed_purchase_gaps": int(len(gaps)),
        "gap_statistics_days": {
            "mean": round(float(gaps.mean()), 2),
            "median": round(float(gaps.median()), 2),
            "p75": round(float(gaps.quantile(0.75)), 2),
            "p90": round(float(gaps.quantile(0.90)), 2),
            "p95": round(float(gaps.quantile(0.95)), 2),
            "maximum": int(gaps.max()),
        },
        "gap_thresholds": {
            "at_least_60_days_pct": round(
                float((gaps >= 60).mean() * 100), 2
            ),
            "at_least_90_days_pct": round(
                float((gaps >= 90).mean() * 100), 2
            ),
            "at_least_120_days_pct": round(
                float((gaps >= 120).mean() * 100), 2
            ),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("=" * 55)
    print("CUSTOMER PURCHASE BEHAVIOR ANALYSIS")
    print("=" * 55)
    print(json.dumps(report, indent=4))
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    analyze_purchase_gaps()
