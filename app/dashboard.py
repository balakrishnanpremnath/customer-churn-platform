
import streamlit as st
import plotly.graph_objects as go
import json
from urllib import request, error
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("📊 Churn Intelligence")
    st.caption("Customer Analytics Platform")

    page = st.radio(
        "Navigation",
        [
    "Overview",
    "Historical Analytics",
    "Customer Risk Prediction",
    "Explainable AI",
        ],
    )

    st.divider()
    st.caption("Model: Frozen Random Forest")
    st.caption("Research evaluation: September 2011")


if page == "Overview":

    st.title("Customer Churn Intelligence Platform")
    st.write(
        "Machine Learning • Customer Analytics • "
        "Retention Risk Prioritization"
    )

    st.info(
        "The figures below are historical held-out research "
        "results, not live business metrics."
    )

    st.subheader("Final Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("ROC-AUC", "0.7361")
    col2.metric("Average Precision", "0.5956")
    col3.metric("Recall", "81.63%")
    col4.metric("F1 Score", "0.6475")

    st.divider()

    st.subheader("Customer Risk Prioritization")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Random Forest", "Recency Baseline"],
            y=[312, 311],
            text=[312, 311],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Inactive Customers Captured — Top 500",
        xaxis_title="Ranking Method",
        yaxis_title="Inactive Customers Captured",
        yaxis_range=[0, 360],
        height=420,
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.warning(
        "Random Forest captured only one additional "
        "inactive customer compared with recency "
        "on the held-out test."
    )


    st.subheader("Research Summary")

    st.write(
        "The model uses 15 historical customer behavior "
        "features to predict 90-day purchase inactivity. "
        "The held-out evaluation includes 2,772 customer "
        "snapshots. Model scores are not calibrated "
        "probabilities, and risk predictions do not "
        "establish the effect of retention interventions."
    )


elif page == "Historical Analytics":

    st.title("Historical Retail Analytics")

    st.info(
        "These figures describe historical UCI Online Retail II "
        "purchase activity, not live business performance. "
        "Revenue uses source-data currency units, not LKR, "
        "and excludes transactions removed during cleaning."
    )

    root = Path(__file__).resolve().parents[1]

    summary_path = (
        root / "data" / "processed"
        / "analytics_summary.json"
    )

    if not summary_path.is_file():
        st.warning(
            "Analytics summary not found. "
            "Run the analytics generation script first."
        )

    else:
        try:
            with summary_path.open(
                "r", encoding="utf-8"
            ) as file:
                summary = json.load(file)

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Historical Revenue",
                f"{summary['total_revenue']:,.2f}",
            )

            col2.metric(
                "Unique Customers",
                f"{summary['unique_customers']:,}",
            )

            col3.metric(
                "Purchase Lines",
                f"{summary['purchase_lines']:,}",
            )

            st.divider()

            st.caption(
                f"Data period: "
                f"{summary['first_purchase_date']} to "
                f"{summary['last_purchase_date']}"
            )


            st.warning(
                "December 2011 is incomplete: "
                "data ends on December 9. "
                "Do not compare it directly with full months."
            )

# Historical monthly revenue visualization
            monthly_path = (
                root / "data" / "processed"
                / "analytics_monthly.csv"
            )

            st.divider()
            st.subheader("Monthly Revenue Trend")

            if monthly_path.is_file():
                monthly = pd.read_csv(monthly_path)

                # December 2011 contains only nine days.

                # Exclude it from full-month comparisons.
                complete_months = monthly.loc[
                    monthly["month"] < "2011-12"
                ].sort_values("month")

                if complete_months.empty:
                    st.warning(
                        "No complete monthly data available."
                    )
                else:
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=complete_months["month"],
                            y=complete_months["revenue"],
                            mode="lines+markers",
                            name="Monthly Revenue",
                            hovertemplate=(
                                "Month: %{x}<br>"
                                "Revenue: %{y:,.2f}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    fig.update_layout(
                        title="Historical Monthly Revenue",
                        xaxis_title="Month",
                        yaxis_title="Revenue (source currency units)",
                        height=450,
                        hovermode="x unified",
                    )

                    st.plotly_chart(
                        fig,
                        width="stretch",
                    )

                    st.caption(
                        "December 2011 is excluded because "
                        "it contains only partial-month data. "
                        "Revenue is historical, not live."
                    )


                    st.divider()
                    st.subheader("Monthly Purchasing Customers")

                    customer_fig = go.Figure(
                        go.Bar(
                            x=complete_months["month"],
                            y=complete_months["customers"],
                            name="Purchasing Customers",
                            hovertemplate=(
                                "Month: %{x}<br>"
                                "Customers: %{y:,}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    customer_fig.update_layout(
                        xaxis_title="Month",
                        yaxis_title="Unique Purchasing Customers",
                        height=400,
                    )

                    st.plotly_chart(
                        customer_fig,
                        width="stretch",
                    )

                    st.caption(
                        "Customers are counted uniquely within "
                        "each month. A customer may appear in "
                        "multiple months. December 2011 is excluded."
                    )

            else:
                st.warning(
                    "Monthly analytics file not found. "
                    "Run the analytics generation script."
                )

                
            # Country-wise historical revenue analytics
            st.divider()
            st.subheader("Top 10 Countries by Revenue")

            country_path = (
                root / "data" / "processed"
                / "analytics_countries.csv"
            )

            if country_path.is_file():
                country_data = pd.read_csv(country_path)

                required_columns = {"country", "revenue"}

                if not required_columns.issubset(
                    country_data.columns
                ):
                    raise ValueError(
                        "Country analytics columns are missing"
                    )

                if country_data.empty:
                    st.warning(
                        "No country analytics available."
                    )
                else:
                    top_countries = (
                        country_data
                        .nlargest(10, "revenue")
                        .sort_values("revenue")
                    )

                    country_fig = go.Figure(
                        go.Bar(
                            x=top_countries["revenue"],
                            y=top_countries["country"],
                            orientation="h",
                            name="Revenue",
                            hovertemplate=(
                                "Country: %{y}<br>"
                                "Revenue: %{x:,.2f}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    country_fig.update_layout(
                        xaxis_title=(
                            "Revenue (source currency units)"
                        ),
                        yaxis_title="Country",
                        height=500,
                        margin=dict(l=100, r=30, t=30, b=60),
                    )

                    st.plotly_chart(
                        country_fig,
                        width="stretch",
                    )

                    st.caption(
                        "Top 10 countries by total historical "
                        "revenue across the available dataset. "
                        "Totals include partial months and "
                        "are not live business figures."
                    )

            else:
                st.warning(
                    "Country analytics file not found. "
                    "Run the analytics generation script."
                )


        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ):
            st.error(
                "Analytics summary could not be loaded. "
                "Regenerate the analytics reports."
            )

elif page == "Customer Risk Prediction":

    st.title("Customer Risk Prediction")
    st.caption("AI-powered 90-day purchase inactivity risk assessment")

    st.info(
        "Enter customer behavior features from the "
        "last 180 days to generate a risk prediction."
    )

    defaults = {
        "frequency_180d": 12,
        "monetary_180d": 6000,
        "quantity_180d": 40,
        "active_days_180d": 10,
        "unique_products_180d": 15,
        "recency_days": 20,
        "tenure_180d": 150,
        "avg_order_value_180d": 500,
        "orders_30d": 2,
        "spend_30d": 1000,
        "orders_90d": 6,
        "spend_90d": 3000,
        "spend_share_30d": 0.1667,
        "has_repeat_purchase_day_180d": 1,
        "avg_repeat_gap_days_180d": 14,
    }

    st.subheader("Customer Information")

    with st.form("customer_prediction_form"):

        columns = st.columns(3)
        customer = {}

        for index, (feature, default) in enumerate(
            defaults.items()
        ):
            with columns[index % 3]:
                options = {
                    "label": feature.replace("_", " ").title(),
                    "min_value": 0.0,
                    "value": float(default),
                    "key": f"input_{feature}",
                    "format": "%.4f",
                }

                if feature == "has_repeat_purchase_day_180d":
                    options["max_value"] = 1.0

                customer[feature] = st.number_input(
                    **options
                )

        submitted = st.form_submit_button(
            "Predict Customer Risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        try:
            payload = json.dumps(customer).encode("utf-8")

            api_request = request.Request(
                "http://127.0.0.1:8000/predict",
                data=payload,
                headers={
                    "Content-Type": "application/json"
                },
                method="POST",
            )

            with request.urlopen(
                api_request,
                timeout=10,
            ) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )

            score = result["inactivity_risk_score"]
            predicted = result["predicted_inactive"]

            st.divider()
            st.subheader("AI Prediction Result")

            st.metric(
                "90-Day Inactivity Risk Score",
                f"{score * 100:.2f}%",
            )

            st.progress(float(score))

            if predicted:
                st.warning(
                    "The model classifies this customer "
                    "as potentially inactive."
                )
            else:
                st.success(
                    "The model classifies this customer "
                    "as not inactive."
                )

            st.caption(
                "Model threshold: 50%. Scores are "
                "uncalibrated and should not be treated "
                "as actual probabilities or guarantees."
            )

        except error.HTTPError as exc:
            st.error(
                f"API rejected the request "
                f"(HTTP {exc.code})."
            )

        except (error.URLError, TimeoutError):
            st.error(
                "Cannot connect to FastAPI. "
                "Check that the API server is running "
                "on port 8000."
            )

        except (ValueError, KeyError):
            st.error(
                "Unexpected API response received."
            )


elif page == "Explainable AI":

    st.title("Explainable AI — SHAP Analysis")

    st.write(
        "Understand which customer behavior features "
        "influence the frozen Random Forest model."
    )

    st.info(
        "These explanations use 100 sampled training "
        "snapshots. The waterfall is a historical example, "
        "not an explanation of the customer entered "
        "on the Prediction page."
    )

    root = Path(__file__).resolve().parents[1]

    global_chart = (
        root / "docs" / "figures"
        / "shap_global_importance.png"
    )

    local_chart = (
        root / "docs" / "figures"
        / "shap_local_waterfall.png"
    )

    st.subheader("Global Feature Importance")

    if global_chart.exists():
        st.image(str(global_chart), width="stretch")
    else:
        st.warning("Global SHAP chart not found.")

    st.divider()

    st.subheader("Example Customer Explanation")

    if local_chart.exists():
        st.image(str(local_chart), width="stretch")
    else:
        st.warning("SHAP waterfall chart not found.")

    st.caption(
        "SHAP explains model behavior, not causal "
        "reasons for customer inactivity."
    )
