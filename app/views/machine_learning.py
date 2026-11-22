
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.header import render_header
from app.utils.data_loader import load_all_datasets


def render_machine_learning():
    """Render machine learning validation and test analytics."""

    datasets = load_all_datasets()

    validation_results = datasets.get("ml_results")

    render_header(
        title="Machine Learning Performance",
        subtitle=(
            "Evaluate predictive model performance across "
            "environmental variables and algorithms."
        ),
        section="PREDICTIVE INTELLIGENCE",
    )

    if validation_results is None:
        st.error("Machine learning validation results are unavailable.")
        return

    # -----------------------------------------------------
    # LOAD FINAL TEST RESULTS
    # -----------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    test_results_path = (
        project_root
        / "data"
        / "processed"
        / "machine_learning_test_results.csv"
    )

    if not test_results_path.exists():
        st.error(
            "Final test results are unavailable. "
            "Please run the machine learning training pipeline first."
        )
        return

    try:
        test_results = pd.read_csv(test_results_path)
    except Exception as error:
        st.error(f"Unable to load final test results: {error}")
        return

    # -----------------------------------------------------
    # VALIDATE DATA STRUCTURE
    # -----------------------------------------------------

    validation_columns = {
        "Target",
        "Model",
        "Validation_MAE",
        "Validation_RMSE",
        "Validation_R2",
    }

    test_columns = {
        "Target",
        "Selected_Model",
        "Validation_MAE",
        "MAE",
        "RMSE",
        "R2",
    }

    missing_validation = (
        validation_columns - set(validation_results.columns)
    )

    missing_test = (
        test_columns - set(test_results.columns)
    )

    if missing_validation:
        st.error(
            "Validation dataset is missing columns: "
            f"{sorted(missing_validation)}"
        )
        return

    if missing_test:
        st.error(
            "Final test dataset is missing columns: "
            f"{sorted(missing_test)}"
        )
        return

    validation_results = validation_results.copy()
    test_results = test_results.copy()

    validation_results["Target"] = (
        validation_results["Target"].astype(str).str.lower()
    )

    test_results["Target"] = (
        test_results["Target"].astype(str).str.lower()
    )

    variables = {
        "Temperature": "temperature",
        "Humidity": "humidity",
        "Pressure": "pressure",
        "PM2.5": "pm25",
    }

    validation_targets = set(
        validation_results["Target"].unique()
    )

    test_targets = set(
        test_results["Target"].unique()
    )

    expected_targets = set(variables.values())

    if not expected_targets.issubset(validation_targets):
        st.error(
            "Some environmental targets are missing "
            "from the validation results."
        )
        return

    if not expected_targets.issubset(test_targets):
        st.error(
            "Some environmental targets are missing "
            "from the final test results."
        )
        return

    # -----------------------------------------------------
    # ENVIRONMENTAL TARGET SELECTION
    # -----------------------------------------------------

    selected_label = st.selectbox(
        "Environmental Target",
        list(variables.keys()),
    )

    target = variables[selected_label]

    validation_filtered = validation_results[
        validation_results["Target"] == target
    ].copy()

    test_filtered = test_results[
        test_results["Target"] == target
    ].copy()

    if validation_filtered.empty or len(test_filtered) != 1:
        st.warning(
            f"Incomplete model results for {selected_label}."
        )
        return

    # -----------------------------------------------------
    # SELECTED MODEL AND FINAL TEST PERFORMANCE
    # -----------------------------------------------------

    final_result = test_filtered.iloc[0]

    selected_model = str(
        final_result["Selected_Model"]
    )

    test_mae = float(final_result["MAE"])
    test_rmse = float(final_result["RMSE"])
    test_r2 = float(final_result["R2"])

    validation_mae = float(
        final_result["Validation_MAE"]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Selected Model",
            selected_model,
        )

    with col2:
        st.metric(
            "Final Test MAE",
            f"{test_mae:.4f}",
        )

    with col3:
        st.metric(
            "Final Test RMSE",
            f"{test_rmse:.4f}",
        )

    with col4:
        st.metric(
            "Final Test R²",
            f"{test_r2:.4f}",
        )

    st.caption(
        f"Selected using validation MAE: {validation_mae:.4f}. "
        "Final metrics are calculated on the held-out test dataset."
    )

    st.divider()

    # -----------------------------------------------------
    # VALIDATION MODEL COMPARISON
    # -----------------------------------------------------

    st.subheader("Validation Model Comparison")

    st.caption(
        "Three algorithms are compared using validation data. "
        "The best model is selected based on the lowest Validation MAE."
    )

    metric_mapping = {
        "MAE": "Validation_MAE",
        "RMSE": "Validation_RMSE",
        "R²": "Validation_R2",
    }

    selected_metric = st.selectbox(
        "Select Validation Metric",
        list(metric_mapping.keys()),
    )

    metric_column = metric_mapping[selected_metric]

    fig_validation = px.bar(
        validation_filtered,
        x="Model",
        y=metric_column,
        color="Model",
        text_auto=".4f",
        title=(
            f"{selected_metric} Comparison — "
            f"{selected_label} Validation"
        ),
    )

    fig_validation.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#CBD5E1"},
        xaxis_title="Machine Learning Algorithm",
        yaxis_title=selected_metric,
        showlegend=False,
    )

    st.plotly_chart(
        fig_validation,
        width="stretch",
    )

    if selected_metric in ["MAE", "RMSE"]:
        st.caption(
            "Lower values indicate smaller prediction errors."
        )
    else:
        st.caption(
            "Higher R² indicates a greater proportion of "
            "target variance explained by the model."
        )

    # -----------------------------------------------------
    # VALIDATION RESULTS TABLE
    # -----------------------------------------------------

    st.divider()

    st.subheader("Detailed Validation Results")

    validation_display = validation_filtered.rename(
        columns={
            "Validation_MAE": "MAE",
            "Validation_RMSE": "RMSE",
            "Validation_R2": "R2",
        }
    )

    st.dataframe(
        validation_display[
            ["Target", "Model", "MAE", "RMSE", "R2"]
        ].sort_values("MAE"),
        width="stretch",
        hide_index=True,
    )

    # -----------------------------------------------------
    # FINAL TEST RESULTS
    # -----------------------------------------------------

    st.divider()

    st.subheader("Final Test Evaluation")

    st.caption(
        "Final performance of the selected model on the "
        "chronological test dataset."
    )

    test_display = test_results[
        [
            "Target",
            "Selected_Model",
            "Validation_MAE",
            "MAE",
            "RMSE",
            "R2",
        ]
    ].copy()

    test_display = test_display.rename(
        columns={
            "Selected_Model": "Selected Model",
            "Validation_MAE": "Validation MAE",
            "MAE": "Test MAE",
            "RMSE": "Test RMSE",
            "R2": "Test R2",
        }
    )

    st.dataframe(
        test_display,
        width="stretch",
        hide_index=True,
    )

    # -----------------------------------------------------
    # OVERALL MODEL PERFORMANCE
    # -----------------------------------------------------

    st.divider()

    st.subheader("Overall Model Performance")

    overall = test_results[
        [
            "Target",
            "Selected_Model",
            "Validation_MAE",
            "MAE",
            "RMSE",
            "R2",
        ]
    ].copy()

    overall = overall.rename(
        columns={
            "Selected_Model": "Selected Model",
            "Validation_MAE": "Validation MAE",
            "MAE": "Test MAE",
            "RMSE": "Test RMSE",
            "R2": "Test R2",
        }
    )

    st.dataframe(
        overall.round(4),
        width="stretch",
        hide_index=True,
    )

    # -----------------------------------------------------
    # MODEL DISTRIBUTION
    # -----------------------------------------------------

    st.divider()

    st.subheader("Experiment Distribution")

    distribution = (
        validation_results.groupby("Model")
        .size()
        .reset_index(name="Experiments")
    )

    fig_distribution = px.pie(
        distribution,
        names="Model",
        values="Experiments",
        hole=0.55,
        title="Validation Model Experiment Distribution",
    )

    fig_distribution.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#CBD5E1"},
    )

    st.plotly_chart(
        fig_distribution,
        width="stretch",
    )

    st.caption(
        "Validation results are used for model selection. "
        "Test results are reserved for final performance evaluation. "
        "The dataset consists of simulated environmental sensor data."
    )
