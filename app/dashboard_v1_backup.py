
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DIMASA AI | Environmental Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


# =========================================================
# DATA LOADER
# =========================================================

@st.cache_data
def load_csv(file_path):
    if file_path.exists():
        return pd.read_csv(file_path)
    return None


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 1px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
    }

    .section-title {
        color: #38bdf8;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD EXPERIMENT RESULTS
# =========================================================

files = {
    "Fused Environmental Data": "fused_environmental_data.csv",
    "Fusion Comparison": "fusion_comparison.csv",
    "Machine Learning Results": "machine_learning_results.csv",
    "Uncertainty Metrics": "uncertainty_metrics.csv",
    "Sensor Fusion Weights": "sensor_fusion_weights.csv",
}

datasets = {
    name: load_csv(DATA_DIR / filename)
    for name, filename in files.items()
}


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("DIMASA AI")
st.sidebar.caption("Artificial Intelligence & Innovation")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Sensor Fusion",
        "Machine Learning",
        "Uncertainty Analysis",
        "Dataset Explorer",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Where Hope Meets Intelligence.")
st.sidebar.caption("Inspired by Dreams. Built for the Future.")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">DIMASA AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Multi-Sensor Data Fusion & Uncertainty-Aware Prediction System'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.markdown(
        '<h3 class="section-title">Environmental Intelligence Overview</h3>',
        unsafe_allow_html=True,
    )

    st.write(
        "An experimental AI system that combines environmental sensor "
        "measurements, trains machine learning models, and estimates "
        "prediction uncertainty."
    )

    fused_data = datasets["Fused Environmental Data"]
    ml_results = datasets["Machine Learning Results"]
    uncertainty = datasets["Uncertainty Metrics"]

    col1, col2, col3 = st.columns(3)

    with col1:
        if fused_data is not None:
            st.metric(
                "Total Observations",
                f"{len(fused_data):,}",
            )
        else:
            st.metric("Total Observations", "Unavailable")

    with col2:
        if ml_results is not None:
            st.metric(
                "Model Experiments",
                f"{len(ml_results):,}",
            )
        else:
            st.metric("Model Experiments", "Unavailable")

    with col3:
        if uncertainty is not None:
            st.metric(
                "Uncertainty Experiments",
                f"{len(uncertainty):,}",
            )
        else:
            st.metric("Uncertainty Experiments", "Unavailable")

    st.divider()

    st.subheader("System Components")

    component_col1, component_col2 = st.columns(2)

    with component_col1:
        st.markdown("**01 — Sensor Intelligence**")
        st.write(
            "Analyzes sensor errors, missing readings, "
            "and measurement reliability."
        )

        st.markdown("**02 — Multi-Sensor Fusion**")
        st.write(
            "Combines multiple sensors measuring the same "
            "environmental variable."
        )

    with component_col2:
        st.markdown("**03 — Machine Learning**")
        st.write(
            "Trains predictive models for environmental variables."
        )

        st.markdown("**04 — Uncertainty Quantification**")
        st.write(
            "Evaluates prediction intervals and empirical coverage."
        )

    st.divider()

    available = sum(data is not None for data in datasets.values())

    st.caption(
        f"Data availability: {available}/{len(files)} result files found."
    )


# =========================================================
# SENSOR FUSION PAGE
# =========================================================

elif page == "Sensor Fusion":

    st.subheader("Multi-Sensor Fusion")

    fusion_data = datasets["Fusion Comparison"]
    weights_data = datasets["Sensor Fusion Weights"]

    if fusion_data is not None:
        st.markdown("**Fusion Performance Comparison**")
        st.dataframe(fusion_data, use_container_width=True)
    else:
        st.warning("Fusion comparison data was not found.")

    if weights_data is not None:
        st.markdown("**Sensor Reliability Weights**")
        st.dataframe(weights_data, use_container_width=True)
    else:
        st.warning("Sensor fusion weights were not found.")


# =========================================================
# MACHINE LEARNING PAGE
# =========================================================

elif page == "Machine Learning":

    st.subheader("Machine Learning Performance")

    ml_results = datasets["Machine Learning Results"]

    if ml_results is not None:
        st.dataframe(ml_results, use_container_width=True)
    else:
        st.warning("Machine learning results were not found.")


# =========================================================
# UNCERTAINTY ANALYSIS PAGE
# =========================================================

elif page == "Uncertainty Analysis":

    st.subheader("Prediction Uncertainty Analysis")

    uncertainty = datasets["Uncertainty Metrics"]

    if uncertainty is not None:
        st.dataframe(uncertainty, use_container_width=True)
    else:
        st.warning("Uncertainty metrics were not found.")


# =========================================================
# DATASET EXPLORER PAGE
# =========================================================

elif page == "Dataset Explorer":

    st.subheader("Processed Dataset Explorer")

    available_datasets = {
        name: data
        for name, data in datasets.items()
        if data is not None
    }

    if available_datasets:

        selected_dataset = st.selectbox(
            "Select a dataset",
            list(available_datasets.keys()),
        )

        selected_data = available_datasets[selected_dataset]

        st.write(
            f"Rows: {len(selected_data):,} | "
            f"Columns: {len(selected_data.columns):,}"
        )

        st.dataframe(
            selected_data,
            use_container_width=True,
            height=450,
        )

    else:
        st.warning("No processed datasets are currently available.")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "DIMASA AI | Artificial Intelligence & Innovation | "
    "Experimental research using simulated environmental sensor data."
)
