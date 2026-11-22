
import plotly.express as px
import streamlit as st

from app.components.header import render_header
from app.components.charts import render_time_series, render_sensor_weights
from app.utils.data_loader import load_all_datasets, prepare_timestamp


def render_sensor_fusion():
    """Render multi-sensor fusion analytics."""

    datasets = load_all_datasets()

    fused = datasets["fused"]
    comparison = datasets["fusion_comparison"]
    performance = datasets["fusion_performance"]
    weights = datasets["sensor_weights"]

    render_header(
        title="Sensor Fusion Analytics",
        subtitle=(
            "Analyze weighted sensor fusion performance, "
            "reliability weights, and measurement improvements."
        ),
        section="MULTI-SENSOR INTELLIGENCE",
    )

    if any(data is None for data in [fused, comparison, weights]):
        st.error("Required sensor fusion datasets are unavailable.")
        return

    # -----------------------------------------------------
    # VARIABLE SELECTION
    # -----------------------------------------------------

    variables = {
        "Temperature": "temperature",
        "Humidity": "humidity",
        "Pressure": "pressure",
        "PM2.5": "pm25",
    }

    selected_label = st.selectbox(
        "Environmental Variable",
        list(variables.keys()),
    )

    variable = variables[selected_label]

    # -----------------------------------------------------
    # PERFORMANCE METRICS
    # -----------------------------------------------------

    filtered = comparison[
        comparison["Variable"] == variable
    ].copy()

    if filtered.empty:
        st.warning("No fusion comparison data found.")
        return

    best_individual = filtered[
        filtered["Method"].str.contains(
            "Sensor",
            case=False,
            na=False,
        )
    ]

    fusion_result = filtered[
        filtered["Method"].str.contains(
            "Fusion",
            case=False,
            na=False,
        )
    ]

    if best_individual.empty or fusion_result.empty:
        st.warning("Unable to identify individual sensor and fusion results.")
        return

    best_individual_mae = best_individual["MAE"].min()
    fusion_mae = fusion_result["MAE"].min()

    improvement = (
        (best_individual_mae - fusion_mae)
        / best_individual_mae
        * 100
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Fusion MAE",
            f"{fusion_mae:.4f}",
        )

    with col2:
        st.metric(
            "Best Individual MAE",
            f"{best_individual_mae:.4f}",
        )

    with col3:
        st.metric(
            "MAE Improvement",
            f"{improvement:.2f}%",
        )

    st.divider()

    # -----------------------------------------------------
    # FUSION PERFORMANCE COMPARISON
    # -----------------------------------------------------

    st.subheader("Fusion Performance Comparison")

    metric = st.selectbox(
        "Performance Metric",
        ["MAE", "RMSE"],
    )

    fig = px.bar(
        filtered,
        x="Method",
        y=metric,
        color="Method",
        text_auto=".4f",
        title=f"{metric} Comparison — {selected_label}",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#CBD5E1"},
        yaxis_title=metric,
        xaxis_title="Fusion Method",
    )

    st.plotly_chart(fig, width="stretch")

    # -----------------------------------------------------
    # SENSOR RELIABILITY WEIGHTS
    # -----------------------------------------------------

    st.divider()
    st.subheader("Sensor Reliability Weights")

    render_sensor_weights(
        data=weights,
        variable=variable,
    )

    # -----------------------------------------------------
    # TIME SERIES COMPARISON
    # -----------------------------------------------------

    st.divider()
    st.subheader("Ground Truth vs Fused Measurement")

    fused = prepare_timestamp(fused)

    chart_columns = [
        f"{variable}_true",
        f"{variable}_fused",
    ]

    render_time_series(
        data=fused,
        x="timestamp",
        y_columns=chart_columns,
        title=f"{selected_label} — Fusion Time Series",
    )

    # -----------------------------------------------------
    # DETAILED PERFORMANCE TABLE
    # -----------------------------------------------------

    st.divider()
    st.subheader("Detailed Fusion Results")

    st.dataframe(
        filtered.sort_values("MAE"),
        width="stretch",
        hide_index=True,
    )

    # -----------------------------------------------------
    # OVERALL FUSION PERFORMANCE
    # -----------------------------------------------------

    if performance is not None:
        st.divider()
        st.subheader("Overall Fusion Performance")

        st.dataframe(
            performance,
            width="stretch",
            hide_index=True,
        )
