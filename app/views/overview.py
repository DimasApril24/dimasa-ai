
import streamlit as st

from app.components.header import render_header
from app.components.kpi_cards import render_kpi_cards
from app.components.charts import render_time_series
from app.utils.data_loader import (
    load_all_datasets,
    prepare_timestamp,
    get_dataset_status,
)


def render_overview():
    """Render DIMASA AI executive overview."""

    datasets = load_all_datasets()

    fused = datasets["fused"]
    ml_results = datasets["ml_results"]
    uncertainty = datasets["uncertainty_metrics"]

    render_header(
        title="Executive Overview",
        subtitle=(
            "A unified view of environmental sensor intelligence, "
            "machine learning, and prediction uncertainty."
        ),
        section="ENVIRONMENTAL INTELLIGENCE",
    )

    if fused is None:
        st.error("Fused environmental dataset is unavailable.")
        return

    # -----------------------------------------------------
    # KPI METRICS
    # -----------------------------------------------------

    sensor_columns = [
        column
        for column in fused.columns
        if "_sensor_" in column
    ]

    metrics = [
        {
            "label": "Total Observations",
            "value": f"{len(fused):,}",
            "delta": "Environmental records",
            "icon": "◈",
        },
        {
            "label": "Sensor Channels",
            "value": str(len(sensor_columns)),
            "delta": "Across 4 variables",
            "icon": "◉",
        },
        {
            "label": "ML Experiments",
            "value": str(len(ml_results)) if ml_results is not None else "N/A",
            "delta": "Model evaluations",
            "icon": "⌘",
        },
        {
            "label": "Uncertainty Targets",
            "value": str(len(uncertainty)) if uncertainty is not None else "N/A",
            "delta": "Prediction evaluations",
            "icon": "◇",
        },
    ]

    render_kpi_cards(metrics)

    st.write("")

    # -----------------------------------------------------
    # ENVIRONMENTAL TIME SERIES
    # -----------------------------------------------------

    st.subheader("Environmental Monitoring")

    variables = {
        "Temperature": "temperature",
        "Humidity": "humidity",
        "Pressure": "pressure",
        "PM2.5": "pm25",
    }

    selected_label = st.selectbox(
        "Select environmental variable",
        list(variables.keys()),
    )

    selected_variable = variables[selected_label]

    fused = prepare_timestamp(fused)

    chart_columns = [
        f"{selected_variable}_true",
        f"{selected_variable}_fused",
    ]

    available_columns = [
        column
        for column in chart_columns
        if column in fused.columns
    ]

    render_time_series(
        data=fused,
        x="timestamp",
        y_columns=available_columns,
        title=f"{selected_label}: Ground Truth vs Sensor Fusion",
    )

    # -----------------------------------------------------
    # DATASET STATUS
    # -----------------------------------------------------

    st.divider()
    st.subheader("Data Infrastructure Status")

    status = get_dataset_status(datasets)

    available_count = sum(
        item["available"] for item in status.values()
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Available Datasets",
            f"{available_count}/{len(status)}",
        )

    with col2:
        st.metric(
            "Dataset Coverage",
            f"{available_count / len(status) * 100:.0f}%",
        )

    status_rows = []

    for name, details in status.items():
        status_rows.append({
            "Dataset": name.replace("_", " ").title(),
            "Status": "Available" if details["available"] else "Missing",
            "Rows": details["rows"],
            "Columns": details["columns"],
        })

    st.dataframe(
        status_rows,
        width="stretch",
        hide_index=True,
    )
