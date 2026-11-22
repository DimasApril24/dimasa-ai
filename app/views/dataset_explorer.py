
import streamlit as st

from app.components.header import render_header
from app.utils.data_loader import load_all_datasets


def render_dataset_explorer():
    """Render enterprise dataset exploration workspace."""

    datasets = load_all_datasets()

    render_header(
        title="Dataset Explorer",
        subtitle=(
            "Explore, inspect, filter, and export "
            "environmental intelligence datasets."
        ),
        section="DATA MANAGEMENT",
    )

    # -----------------------------------------------------
    # DATASET REGISTRY
    # -----------------------------------------------------

    dataset_labels = {
        "fused": "Fused Environmental Data",
        "fusion_comparison": "Fusion Comparison",
        "fusion_performance": "Fusion Performance",
        "sensor_errors": "Sensor Error Analysis",
        "sensor_weights": "Sensor Fusion Weights",
        "ml_results": "Machine Learning Results",
        "ml_predictions": "Machine Learning Predictions",
        "uncertainty_metrics": "Uncertainty Metrics",
        "uncertainty_predictions": "Uncertainty Predictions",
    }

    available_datasets = {
        key: value
        for key, value in datasets.items()
        if value is not None
    }

    # -----------------------------------------------------
    # DATASET OVERVIEW
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    total_rows = sum(
        len(data) for data in available_datasets.values()
    )

    total_columns = sum(
        len(data.columns) for data in available_datasets.values()
    )

    with col1:
        st.metric(
            "Available Datasets",
            len(available_datasets),
        )

    with col2:
        st.metric(
            "Combined Records",
            f"{total_rows:,}",
        )

    with col3:
        st.metric(
            "Total Data Fields",
            total_columns,
        )

    st.divider()

    if not available_datasets:
        st.error("No datasets are currently available.")
        return

    # -----------------------------------------------------
    # SELECT DATASET
    # -----------------------------------------------------

    st.subheader("Dataset Workspace")

    selected_key = st.selectbox(
        "Select Dataset",
        list(available_datasets.keys()),
        format_func=lambda key: dataset_labels.get(
            key, key.replace("_", " ").title()
        ),
    )

    selected_data = available_datasets[selected_key].copy()

    # -----------------------------------------------------
    # DATASET METADATA
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", f"{len(selected_data):,}")

    with col2:
        st.metric("Columns", len(selected_data.columns))

    with col3:
        st.metric(
            "Missing Values",
            f"{int(selected_data.isna().sum().sum()):,}",
        )

    # -----------------------------------------------------
    # COLUMN FILTER
    # -----------------------------------------------------

    st.divider()
    st.subheader("Data Filtering")

    selected_columns = st.multiselect(
        "Choose columns to display",
        options=list(selected_data.columns),
        default=list(selected_data.columns),
    )

    if not selected_columns:
        st.warning("Select at least one column.")
        return

    filtered_data = selected_data[selected_columns].copy()

    # -----------------------------------------------------
    # SEARCH FILTER
    # -----------------------------------------------------

    searchable_columns = filtered_data.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if searchable_columns:
        search_column = st.selectbox(
            "Search within text column",
            searchable_columns,
        )

        search_term = st.text_input(
            "Search keyword",
            placeholder="Enter a keyword...",
        )

        if search_term:
            filtered_data = filtered_data[
                filtered_data[search_column]
                .astype(str)
                .str.contains(
                    search_term,
                    case=False,
                    na=False,
                    regex=False,
                )
            ]

    # -----------------------------------------------------
    # DATA PREVIEW
    # -----------------------------------------------------

    st.divider()
    st.subheader("Data Preview")

    st.caption(
        f"Displaying {len(filtered_data):,} records "
        f"and {len(filtered_data.columns)} columns."
    )

    st.dataframe(
        filtered_data,
        width="stretch",
        height=450,
        hide_index=True,
    )

    # -----------------------------------------------------
    # MISSING VALUE ANALYSIS
    # -----------------------------------------------------

    st.divider()
    st.subheader("Data Quality Analysis")

    missing_summary = (
        selected_data.isna()
        .sum()
        .reset_index()
    )

    missing_summary.columns = [
        "Column",
        "Missing Values",
    ]

    missing_summary["Missing Percentage"] = (
        missing_summary["Missing Values"]
        / max(len(selected_data), 1)
        * 100
    )

    st.dataframe(
        missing_summary.sort_values(
            "Missing Percentage",
            ascending=False,
        ),
        width="stretch",
        hide_index=True,
    )

    # -----------------------------------------------------
    # EXPORT DATA
    # -----------------------------------------------------

    st.divider()
    st.subheader("Export Dataset")

    csv_data = filtered_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name=f"dimasa_ai_{selected_key}.csv",
        mime="text/csv",
        width="stretch",
    )

    st.caption(
        "Export contains the currently selected columns "
        "and applied search filter."
    )
