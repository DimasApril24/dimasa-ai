
import plotly.express as px
import streamlit as st

from app.components.header import render_header
from app.components.charts import render_time_series
from app.utils.data_loader import load_all_datasets, prepare_timestamp


# =========================================================
# DIMASA AI — SENSOR VISUAL CONFIGURATION
# =========================================================

SENSOR_COLORS = {
    "Sensor 1": "#38BDF8",
    "Sensor 2": "#2563EB",
    "Sensor 3": "#F59E0B",
}


def format_sensor_name(sensor):
    """Convert database sensor names into readable labels."""

    parts = str(sensor).split("_sensor_")

    if len(parts) == 2:
        variable = parts[0].replace("pm25", "PM2.5").title()
        return f"{variable} Sensor {parts[1]}"

    return str(sensor).replace("_", " ").title()


def get_sensor_label(sensor):
    """Get consistent short sensor labels."""

    if "_sensor_" in str(sensor):
        number = str(sensor).split("_sensor_")[-1]
        return f"Sensor {number}"

    return format_sensor_name(sensor)


def render_sensor_intelligence():
    """Analyze sensor accuracy, reliability, and missing readings."""

    datasets = load_all_datasets()

    fused = datasets["fused"]
    sensor_errors = datasets["sensor_errors"]

    render_header(
        title="Sensor Intelligence",
        subtitle=(
            "Evaluate sensor accuracy, measurement errors, "
            "and data availability."
        ),
        section="SENSOR ANALYTICS",
    )

    if sensor_errors is None or fused is None:
        st.error("Required sensor datasets are unavailable.")
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

    filtered_errors = sensor_errors[
        sensor_errors["Variable"] == variable
    ].copy()

    if filtered_errors.empty:
        st.warning("No sensor error records found.")
        return

    filtered_errors["Sensor Label"] = (
        filtered_errors["Sensor"].apply(get_sensor_label)
    )

    filtered_errors = filtered_errors.sort_values("MAE")

    # -----------------------------------------------------
    # SENSOR PERFORMANCE METRICS
    # -----------------------------------------------------

    best_sensor = filtered_errors.iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Sensors Evaluated",
            len(filtered_errors),
        )

    with col2:
        st.metric(
            "Best Sensor",
            format_sensor_name(best_sensor["Sensor"]),
        )

    with col3:
        st.metric(
            "Lowest MAE",
            f'{best_sensor["MAE"]:.4f}',
        )

    st.divider()

    # -----------------------------------------------------
    # SENSOR ERROR COMPARISON
    # -----------------------------------------------------

    st.subheader("Sensor Accuracy Comparison")

    fig = px.bar(
        filtered_errors,
        x="Sensor Label",
        y="MAE",
        color="Sensor Label",
        color_discrete_map=SENSOR_COLORS,
        text_auto=".4f",
        title=f"Mean Absolute Error — {selected_label}",
        category_orders={
            "Sensor Label": ["Sensor 1", "Sensor 2", "Sensor 3"]
        },
    )

    fig.update_traces(
        marker_line_width=0,
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#CBD5E1"},
        title={
            "font": {"size": 17, "color": "#F8FAFC"},
            "x": 0,
        },
        yaxis_title="Mean Absolute Error",
        xaxis_title="Sensor",
        showlegend=False,
        margin={"l": 55, "r": 25, "t": 75, "b": 65},
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.15)",
        rangemode="tozero",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )

    # -----------------------------------------------------
    # ERROR ANALYSIS TABLE
    # -----------------------------------------------------

    st.subheader("Detailed Sensor Error Analysis")

    display_errors = filtered_errors.drop(
        columns=["Sensor Label"]
    )

    display_errors["Sensor"] = (
        display_errors["Sensor"].apply(format_sensor_name)
    )

    st.dataframe(
        display_errors,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    # -----------------------------------------------------
    # SENSOR TIME SERIES
    # -----------------------------------------------------

    st.subheader("Individual Sensor Measurements")

    fused = prepare_timestamp(fused)

    sensor_columns = [
        f"{variable}_sensor_1",
        f"{variable}_sensor_2",
        f"{variable}_sensor_3",
    ]

    available_columns = [
        column
        for column in sensor_columns
        if column in fused.columns
    ]

    if available_columns:
        render_time_series(
            data=fused,
            x="timestamp",
            y_columns=available_columns,
            title=f"Individual Sensor Readings — {selected_label}",
        )
    else:
        st.warning("No individual sensor measurements are available.")

    # -----------------------------------------------------
    # MISSING DATA ANALYSIS
    # -----------------------------------------------------

    st.subheader("Sensor Data Availability")

    available_missing_columns = [
        column
        for column in sensor_columns
        if column in fused.columns
    ]

    if not available_missing_columns:
        st.warning("No sensor columns available for missing data analysis.")
        return

    missing_counts = fused[available_missing_columns].isna().sum()

    missing_data = missing_counts.rename_axis(
        "Sensor"
    ).reset_index(name="Missing Readings")

    missing_data["Sensor Label"] = (
        missing_data["Sensor"].apply(get_sensor_label)
    )

    missing_data["Missing Percentage"] = (
        missing_data["Missing Readings"] / len(fused) * 100
    )

    fig_missing = px.bar(
        missing_data,
        x="Sensor Label",
        y="Missing Percentage",
        color="Sensor Label",
        color_discrete_map=SENSOR_COLORS,
        text_auto=".2f",
        title="Missing Readings by Sensor",
    )

    fig_missing.update_traces(
        marker_line_width=0,
        textposition="outside",
        cliponaxis=False,
    )

    fig_missing.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#CBD5E1"},
        title={
            "font": {"size": 17, "color": "#F8FAFC"},
            "x": 0,
        },
        yaxis_title="Missing Readings (%)",
        xaxis_title="Sensor",
        showlegend=False,
        margin={"l": 55, "r": 25, "t": 75, "b": 65},
    )

    fig_missing.update_xaxes(showgrid=False)
    fig_missing.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.15)",
        rangemode="tozero",
    )

    st.plotly_chart(
        fig_missing,
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )
