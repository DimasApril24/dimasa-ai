
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.header import render_header
from app.utils.data_loader import load_all_datasets


def render_uncertainty_intelligence():
    """Render conformal prediction and uncertainty analytics."""

    datasets = load_all_datasets()

    metrics = datasets.get("uncertainty_metrics")
    predictions = datasets.get("uncertainty_predictions")

    render_header(
        title="Uncertainty Intelligence",
        subtitle=(
            "Evaluate prediction intervals, empirical coverage, "
            "and uncertainty-aware machine learning predictions."
        ),
        section="UNCERTAINTY QUANTIFICATION",
    )

    if metrics is None or predictions is None:
        st.error("Uncertainty datasets are unavailable.")
        return

    if metrics.empty or predictions.empty:
        st.warning("Uncertainty datasets contain no records.")
        return

    variables = {
        "Temperature": "temperature",
        "Humidity": "humidity",
        "Pressure": "pressure",
        "PM2.5": "pm25",
    }

    required_metric_columns = {
        "Target",
        "Model",
        "Conformal Quantile",
        "Target Coverage",
        "Actual Coverage",
        "Mean Interval Width",
        "MAE",
        "RMSE",
    }

    missing_metrics = (
        required_metric_columns - set(metrics.columns)
    )

    if missing_metrics:
        st.error(
            f"Missing uncertainty metric columns: "
            f"{sorted(missing_metrics)}"
        )
        return

    metrics = metrics.copy()
    metrics["Target"] = (
        metrics["Target"].astype(str).str.lower()
    )

    # -----------------------------------------------------
    # OVERALL COVERAGE SUMMARY
    # -----------------------------------------------------

    st.subheader("Overall Coverage Summary")

    summary = metrics[
        metrics["Target"].isin(variables.values())
    ].copy()

    if summary.empty:
        st.warning("No valid uncertainty targets found.")
        return

    summary["Actual Coverage (%)"] = (
        summary["Actual Coverage"] * 100
    )

    summary["Target Coverage (%)"] = (
        summary["Target Coverage"] * 100
    )

    summary["Coverage Difference (pp)"] = (
        summary["Actual Coverage (%)"]
        - summary["Target Coverage (%)"]
    )

    summary["Coverage Status"] = np.where(
        summary["Actual Coverage (%)"]
        >= summary["Target Coverage (%)"],
        "Meets target",
        "Below target",
    )

    summary_display = summary[
        [
            "Target",
            "Model",
            "Target Coverage (%)",
            "Actual Coverage (%)",
            "Coverage Difference (pp)",
            "Mean Interval Width",
            "Coverage Status",
        ]
    ].copy()

    summary_display["Target"] = (
        summary_display["Target"].map(
            {
                value: label
                for label, value in variables.items()
            }
        )
    )

    st.dataframe(
        summary_display.round(3),
        width="stretch",
        hide_index=True,
    )

    # Overall coverage comparison chart.
    fig_overall = go.Figure()

    fig_overall.add_trace(
        go.Bar(
            x=summary["Target"].map(
                {
                    value: label
                    for label, value in variables.items()
                }
            ),
            y=summary["Actual Coverage (%)"],
            name="Actual Coverage",
            marker_color="#38BDF8",
            text=[
                f"{value:.1f}%"
                for value in summary["Actual Coverage (%)"]
            ],
            textposition="outside",
        )
    )

    fig_overall.add_trace(
        go.Scatter(
            x=summary["Target"].map(
                {
                    value: label
                    for label, value in variables.items()
                }
            ),
            y=summary["Target Coverage (%)"],
            name="Target Coverage",
            mode="lines+markers",
            line={
                "color": "#F59E0B",
                "width": 2,
                "dash": "dash",
            },
        )
    )

    fig_overall.update_layout(
        title="Actual vs Target Coverage — All Variables",
        yaxis_title="Coverage (%)",
        yaxis_range=[0, 110],
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#CBD5E1"},
        hovermode="x unified",
        legend={"orientation": "h"},
    )

    st.plotly_chart(
        fig_overall,
        width="stretch",
    )

    st.divider()

    # -----------------------------------------------------
    # TARGET SELECTION
    # -----------------------------------------------------

    st.subheader("Target-Level Analysis")

    selected_label = st.selectbox(
        "Environmental Target",
        list(variables.keys()),
    )

    target = variables[selected_label]

    filtered = metrics[
        metrics["Target"] == target
    ].copy()

    if len(filtered) != 1:
        st.warning(
            f"Expected one uncertainty result for {target}, "
            f"but found {len(filtered)}."
        )
        return

    result = filtered.iloc[0]

    actual_coverage = (
        float(result["Actual Coverage"]) * 100
    )

    target_coverage = (
        float(result["Target Coverage"]) * 100
    )

    interval_width = float(
        result["Mean Interval Width"]
    )

    mae = float(result["MAE"])
    rmse = float(result["RMSE"])

    model_name = str(result["Model"])

    conformal_quantile = float(
        result["Conformal Quantile"]
    )

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    st.subheader("Uncertainty Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Actual Coverage",
            f"{actual_coverage:.2f}%",
            delta=(
                f"{actual_coverage - target_coverage:+.2f} pp"
            ),
        )

    with col2:
        st.metric(
            "Target Coverage",
            f"{target_coverage:.2f}%",
        )

    with col3:
        st.metric(
            "Mean Interval Width",
            f"{interval_width:.4f}",
        )

    with col4:
        st.metric(
            "Prediction MAE",
            f"{mae:.4f}",
        )

    col5, col6, col7 = st.columns(3)

    with col5:
        st.metric(
            "Prediction RMSE",
            f"{rmse:.4f}",
        )

    with col6:
        st.metric(
            "Conformal Quantile",
            f"{conformal_quantile:.4f}",
        )

    with col7:
        st.metric(
            "Selected Model",
            model_name,
        )

    # Sample information.
    sample_columns = [
        "Training Samples",
        "Calibration Samples",
        "Test Samples",
    ]

    if all(
        column in result.index
        for column in sample_columns
    ):
        sample_col1, sample_col2, sample_col3 = (
            st.columns(3)
        )

        with sample_col1:
            st.metric(
                "Training Samples",
                int(result["Training Samples"]),
            )

        with sample_col2:
            st.metric(
                "Calibration Samples",
                int(result["Calibration Samples"]),
            )

        with sample_col3:
            st.metric(
                "Test Samples",
                int(result["Test Samples"]),
            )

    st.divider()

    # -----------------------------------------------------
    # COVERAGE EVALUATION
    # -----------------------------------------------------

    st.subheader("Prediction Coverage Evaluation")

    coverage_difference = (
        actual_coverage - target_coverage
    )

    if coverage_difference >= 0:
        st.success(
            "Empirical coverage meets or exceeds the nominal target."
        )
    else:
        st.warning(
            f"Empirical coverage is {abs(coverage_difference):.2f} "
            "percentage points below the nominal target."
        )

    fig_coverage = go.Figure()

    fig_coverage.add_trace(
        go.Bar(
            x=["Target Coverage", "Actual Coverage"],
            y=[target_coverage, actual_coverage],
            marker_color=["#F59E0B", "#38BDF8"],
            text=[
                f"{target_coverage:.2f}%",
                f"{actual_coverage:.2f}%",
            ],
            textposition="outside",
        )
    )

    fig_coverage.update_layout(
        title=f"Coverage Comparison — {selected_label}",
        yaxis_title="Coverage (%)",
        yaxis_range=[0, 110],
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#CBD5E1"},
    )

    st.plotly_chart(
        fig_coverage,
        width="stretch",
    )

    st.divider()

    # -----------------------------------------------------
    # PREDICTION INTERVAL
    # -----------------------------------------------------

    st.subheader(
        "Actual vs Prediction with Uncertainty Interval"
    )

    prediction_columns = [
        "timestamp",
        f"{target}_actual",
        f"{target}_prediction",
        f"{target}_lower",
        f"{target}_upper",
    ]

    missing_columns = [
        column
        for column in prediction_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        st.warning(
            f"Prediction columns unavailable: {missing_columns}"
        )
        return

    plot_data = predictions[
        prediction_columns
    ].copy()

    plot_data["timestamp"] = pd.to_datetime(
        plot_data["timestamp"],
        errors="coerce",
    )

    numeric_columns = prediction_columns[1:]

    for column in numeric_columns:
        plot_data[column] = pd.to_numeric(
            plot_data[column],
            errors="coerce",
        )

    plot_data = plot_data.dropna(
        subset=prediction_columns
    )

    if plot_data.empty:
        st.warning(
            "No valid prediction interval records available."
        )
        return

    fig_interval = go.Figure()

    # Upper interval boundary.
    fig_interval.add_trace(
        go.Scatter(
            x=plot_data["timestamp"],
            y=plot_data[f"{target}_upper"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Lower boundary and shaded interval.
    fig_interval.add_trace(
        go.Scatter(
            x=plot_data["timestamp"],
            y=plot_data[f"{target}_lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(56,189,248,0.18)",
            line={"width": 0},
            name="Prediction Interval",
        )
    )

    # Actual observations.
    fig_interval.add_trace(
        go.Scatter(
            x=plot_data["timestamp"],
            y=plot_data[f"{target}_actual"],
            mode="lines",
            name="Actual",
            line={
                "color": "#F8FAFC",
                "width": 2,
            },
        )
    )

    # Model predictions.
    fig_interval.add_trace(
        go.Scatter(
            x=plot_data["timestamp"],
            y=plot_data[f"{target}_prediction"],
            mode="lines",
            name="Prediction",
            line={
                "color": "#38BDF8",
                "width": 2,
            },
        )
    )

    fig_interval.update_layout(
        title=f"Prediction Interval — {selected_label}",
        xaxis_title="Timestamp",
        yaxis_title="Predicted Value",
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#CBD5E1"},
        legend={"orientation": "h"},
    )

    st.plotly_chart(
        fig_interval,
        width="stretch",
    )

    st.caption(
        "The shaded region represents the conformal prediction "
        "interval around the model predictions."
    )

    st.divider()

    # -----------------------------------------------------
    # DETAILED METRICS
    # -----------------------------------------------------

    st.subheader("Detailed Uncertainty Metrics")

    st.dataframe(
        filtered,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Coverage is measured empirically on the held-out test set. "
        "The dataset contains simulated environmental sensor data. "
        "Classical split conformal coverage guarantees rely on "
        "exchangeability and are not automatically guaranteed "
        "for chronologically dependent observations."
    )
