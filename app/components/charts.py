
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# DIMASA AI — ENTERPRISE CHART CONFIGURATION
# =========================================================

COLORS = {
    "primary": "#38BDF8",
    "secondary": "#2563EB",
    "tertiary": "#F59E0B",
    "success": "#10B981",
    "danger": "#EF4444",
    "purple": "#8B5CF6",
    "muted": "#94A3B8",
    "actual": "#F8FAFC",
    "prediction": "#38BDF8",
}

PALETTE = [
    "#38BDF8",
    "#2563EB",
    "#F59E0B",
    "#10B981",
    "#8B5CF6",
    "#EF4444",
]

CHART_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {
        "family": "Arial, sans-serif",
        "color": "#CBD5E1",
        "size": 11,
    },
    "title": {
        "font": {
            "size": 17,
            "color": "#F8FAFC",
        },
        "x": 0,
        "xanchor": "left",
    },
    "margin": {
        "l": 55,
        "r": 25,
        "t": 75,
        "b": 85,
    },
    "legend": {
        "orientation": "h",
        "yanchor": "top",
        "y": -0.20,
        "xanchor": "left",
        "x": 0,
        "font": {"size": 10},
    },
    "hoverlabel": {
        "bgcolor": "#14243A",
        "font": {"color": "#F8FAFC"},
        "bordercolor": "#38BDF8",
    },
    "hovermode": "x unified",
}


def _format_name(value):
    """Convert database-style names into readable labels."""

    return (
        str(value)
        .replace("_", " ")
        .replace("pm25", "PM2.5")
        .title()
    )



def _apply_layout(fig, title=None):
    """Apply consistent DIMASA AI chart styling."""

    layout = CHART_LAYOUT.copy()

    if title is not None:
        layout["title"] = {
            "text": title,
            "font": {
                "size": 17,
                "color": "#F8FAFC",
            },
            "x": 0,
            "xanchor": "left",
        }

    fig.update_layout(**layout)

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#263852",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.15)",
        zeroline=False,
    )

    return fig

# =========================================================
# 1. TIME SERIES
# =========================================================

def render_time_series(data, x, y_columns, title):
    """Display environmental sensor time-series."""

    fig = go.Figure()

    for index, column in enumerate(y_columns):
        if column not in data.columns:
            continue

        label = _format_name(column)

        fig.add_trace(
            go.Scatter(
                x=data[x],
                y=data[column],
                mode="lines",
                name=label,
                line={
                    "color": PALETTE[index % len(PALETTE)],
                    "width": 2,
                },
                connectgaps=False,
            )
        )

    if not fig.data:
        st.warning("No valid columns available for this chart.")
        return

    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Measurement",
    )

    _apply_layout(fig, title)

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )


# =========================================================
# 2. MACHINE LEARNING MODEL COMPARISON
# =========================================================

def render_model_comparison(
    data,
    variable,
    metric="MAE",
    target_col="Target",
    model_col="Model",
):
    """Compare machine learning models."""

    required = {target_col, model_col, metric}

    if not required.issubset(data.columns):
        st.warning(
            f"Missing required columns: {required - set(data.columns)}"
        )
        return

    filtered = data[data[target_col] == variable].copy()

    if filtered.empty:
        st.warning(f"No results found for {variable}.")
        return

    filtered[model_col] = filtered[model_col].astype(str)

    fig = px.bar(
        filtered,
        x=model_col,
        y=metric,
        color=model_col,
        text_auto=".3f",
        color_discrete_sequence=PALETTE,
    )

    fig.update_traces(
        marker_line_width=0,
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(
        xaxis_title="Machine Learning Algorithm",
        yaxis_title=metric,
        showlegend=False,
        margin={"t": 65, "b": 65, "l": 55, "r": 25},
    )

    _apply_layout(fig, f"{metric} Comparison — {_format_name(variable)}")

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )


# =========================================================
# 3. SENSOR FUSION COMPARISON
# =========================================================

def render_fusion_comparison(data, variable, metric="MAE"):
    """Compare individual sensors against weighted fusion."""

    render_model_comparison(
        data=data,
        variable=variable,
        metric=metric,
        target_col="Variable",
        model_col="Method",
    )


# =========================================================
# 4. SENSOR RELIABILITY WEIGHTS
# =========================================================

def render_sensor_weights(data, variable):
    """Visualize inverse-variance sensor reliability weights."""

    required = {"Variable", "Sensor", "Weight"}

    if not required.issubset(data.columns):
        st.warning("Sensor weight dataset has unexpected columns.")
        return

    filtered = data[data["Variable"] == variable].copy()

    if filtered.empty:
        st.warning(f"No sensor weights found for {variable}.")
        return

    filtered["Sensor Label"] = filtered["Sensor"].apply(_format_name)

    fig = px.bar(
        filtered,
        x="Sensor Label",
        y="Weight",
        color="Sensor Label",
        text_auto=".3f",
        color_discrete_sequence=PALETTE,
    )

    fig.update_traces(
        marker_line_width=0,
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(
        xaxis_title="Sensor",
        yaxis_title="Normalized Reliability Weight",
        showlegend=False,
        yaxis_range=[
            0,
            max(filtered["Weight"].max() * 1.2, 0.1),
        ],
    )

    _apply_layout(
        fig,
        f"Sensor Reliability Weights — {_format_name(variable)}",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )


# =========================================================
# 5. PREDICTION INTERVAL
# =========================================================

def render_prediction_interval(data, target):
    """Display actual values, predictions and conformal intervals."""

    columns = {
        "actual": f"{target}_actual",
        "prediction": f"{target}_prediction",
        "lower": f"{target}_lower",
        "upper": f"{target}_upper",
    }

    if not set(columns.values()).issubset(data.columns):
        st.warning(f"Prediction columns unavailable for {target}.")
        return

    x = (
        data["timestamp"]
        if "timestamp" in data.columns
        else data.index
    )

    fig = go.Figure()

    # Upper interval boundary
    fig.add_trace(
        go.Scatter(
            x=x,
            y=data[columns["upper"]],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Lower interval boundary and shaded area
    fig.add_trace(
        go.Scatter(
            x=x,
            y=data[columns["lower"]],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(56,189,248,0.16)",
            line={"width": 1, "color": "rgba(56,189,248,0.5)"},
            name="Prediction Interval",
        )
    )

    # Actual measurements
    fig.add_trace(
        go.Scatter(
            x=x,
            y=data[columns["actual"]],
            mode="lines",
            name="Actual",
            line={
                "color": COLORS["actual"],
                "width": 1.8,
            },
        )
    )

    # Model predictions
    fig.add_trace(
        go.Scatter(
            x=x,
            y=data[columns["prediction"]],
            mode="lines",
            name="Prediction",
            line={
                "color": COLORS["primary"],
                "width": 2,
            },
        )
    )

    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Predicted Value",
    )

    _apply_layout(
        fig,
        f"Prediction Uncertainty — {_format_name(target)}",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )
