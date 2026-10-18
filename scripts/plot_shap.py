
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed"
    / "shap_explanations.json"
)

OUTPUT_DIR = ROOT / "docs" / "figures"


def create_global_chart(report):
    rows = report["global_feature_importance"][:10]

    names = [
        row["feature"]
        for row in rows
    ][::-1]

    importance = [
        float(row["mean_abs_shap"])
        for row in rows
    ][::-1]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.barh(names, importance)

    ax.set_xlabel(
        "Mean absolute SHAP contribution "
        "(probability units)"
    )

    ax.set_title(
        "Top 10 Features Influencing "
        "90-Day Inactivity Predictions"
    )

    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()

    output = (
        OUTPUT_DIR / "shap_global_importance.png"
    )

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output


def create_waterfall_chart(report):
    example = report["example_prediction"]

    contributions = example["feature_contributions"]

    names = [
        item["feature"]
        for item in contributions
    ]

    values = np.asarray(
        [
            item["shap_contribution"]
            for item in contributions
        ],
        dtype=float,
    )

    feature_values = np.asarray(
        [
            item["feature_value"]
            for item in contributions
        ],
        dtype=float,
    )

    base = float(example["base_probability"])
    predicted = float(example["predicted_probability"])

    reconstructed = base + values.sum()

    # Serialized SHAP contributions are rounded.
    if not np.isclose(
        reconstructed,
        predicted,
        atol=1e-4,
        rtol=0,
    ):
        raise ValueError(
            "Saved SHAP values do not reconstruct prediction"
        )

    explanation = shap.Explanation(
        values=values,
        base_values=base,
        data=feature_values,
        feature_names=names,
    )

    plt.figure(figsize=(12, 8))

    shap.plots.waterfall(
        explanation,
        max_display=10,
        show=False,
    )

    fig = plt.gcf()

    output = (
        OUTPUT_DIR / "shap_local_waterfall.png"
    )

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(
            "Run SHAP explanation script first"
        )

    report = json.loads(
        INPUT.read_text(encoding="utf-8")
    )

    if report["source"] != "train_and_validation_only":
        raise ValueError("Unexpected SHAP data source")

    if report["held_out_test_explained"]:
        raise ValueError("Held-out test explanations not allowed")

    if len(report["global_feature_importance"]) != 15:
        raise ValueError("Expected 15 model features")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_path = create_global_chart(report)
    local_path = create_waterfall_chart(report)

    print("=" * 60)
    print("SHAP VISUALIZATIONS COMPLETED")
    print("=" * 60)

    print(f"Global importance: {global_path}")
    print(f"Local waterfall: {local_path}")
    print("Frozen model configuration unchanged.")


if __name__ == "__main__":
    main()
