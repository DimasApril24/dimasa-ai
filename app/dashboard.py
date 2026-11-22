
import streamlit as st

from app.components.navigation import render_navigation

from app.views.overview import render_overview
from app.views.sensor_intelligence import render_sensor_intelligence
from app.views.sensor_fusion import render_sensor_fusion
from app.views.machine_learning import render_machine_learning
from app.views.uncertainty_intelligence import render_uncertainty_intelligence
from app.views.dataset_explorer import render_dataset_explorer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DIMASA AI | Enterprise Analytics",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# GLOBAL STYLING
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0B1120;
        color: #E2E8F0;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1E293B;
    }

    [data-testid="stHeader"] {
        background-color: transparent;
    }

    [data-testid="stMetric"] {
        background-color: #111C30;
        border: 1px solid #26364F;
        padding: 16px;
        border-radius: 12px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #26364F;
        border-radius: 10px;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #F1F5F9;
    }

    p, label {
        color: #CBD5E1;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

selected_page = render_navigation()


# =========================================================
# PAGE ROUTER
# =========================================================

pages = {
    "Executive Overview": render_overview,
    "Sensor Intelligence": render_sensor_intelligence,
    "Sensor Fusion": render_sensor_fusion,
    "Machine Learning": render_machine_learning,
    "Uncertainty Intelligence": render_uncertainty_intelligence,
    "Dataset Explorer": render_dataset_explorer,
}


# =========================================================
# RENDER SELECTED PAGE
# =========================================================

if selected_page in pages:
    pages[selected_page]()
else:
    st.error(f"Page not found: {selected_page}")


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "DIMASA AI | Artificial Intelligence & Innovation"
)

st.sidebar.caption(
    "Where Hope Meets Intelligence."
)

st.sidebar.caption(
    "Environmental Sensor Intelligence System"
)
