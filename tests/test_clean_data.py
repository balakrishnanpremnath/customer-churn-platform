import csv
import json
from decimal import Decimal
from pathlib import Path

from scripts.clean_data import (
    customer_id_as_text,
    positive_decimal,
)


ROOT = Path(__file__).resolve().parents[1]
CLEANED_FILE = ROOT / "data" / "processed" / "purchases_clean.csv"
REPORT_FILE = ROOT / "data" / "processed" / "cleaning_report.json"


def test_customer_id_normalization():
    assert customer_id_as_text(None) == ""
    assert customer_id_as_text(12345.0) == "12345"
    assert customer_id_as_text(" 12345 ") == "12345"


def test_positive_decimal_validation():
    assert positive_decimal(10) == Decimal("10")
    assert positive_decimal("6.95") == Decimal("6.95")
    assert positive_decimal(0) is None
    assert positive_decimal(-5) is None
    assert positive_decimal("invalid") is None
    assert positive_decimal(float("nan")) is None


def test_cleaned_dataset_integrity():
    assert CLEANED_FILE.is_file()
    assert REPORT_FILE.is_file()

    report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))

    expected_columns = [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country",
        "line_revenue",
    ]

    rows_checked = 0

    with CLEANED_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        assert reader.fieldnames == expected_columns

        for row in reader:
            rows_checked += 1

            assert row["customer_id"].strip()
            assert not row["invoice"].upper().startswith("C")

            quantity = Decimal(row["quantity"])
            price = Decimal(row["unit_price"])
            revenue = Decimal(row["line_revenue"])

            assert quantity > 0
            assert price > 0
            assert revenue == quantity * price

    assert rows_checked > 0
    assert rows_checked == report["clean_rows"]

    excluded = sum(
        value
        for key, value in report.items()
        if key.startswith("excluded_")
    )

    assert report["total_rows"] == rows_checked + excluded
