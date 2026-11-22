
import streamlit as st


PAGES = [
    {
        "name": "Executive Overview",
        "icon": "◈",
    },
    {
        "name": "Sensor Intelligence",
        "icon": "◉",
    },
    {
        "name": "Sensor Fusion",
        "icon": "⟡",
    },
    {
        "name": "Machine Learning",
        "icon": "⌘",
    },
    {
        "name": "Uncertainty Intelligence",
        "icon": "◇",
    },
    {
        "name": "Dataset Explorer",
        "icon": "▤",
    },
]


def render_navigation():
    """Render DIMASA AI enterprise sidebar navigation."""

    st.sidebar.markdown(
        """
        <div style="padding: 12px 0 20px 0;">
            <div style="
                font-size: 25px;
                font-weight: 800;
                color: #F8FAFC;
                letter-spacing: 1px;
            ">
                DIMASA <span style="color:#38BDF8;">AI</span>
            </div>
            <div style="
                color: #94A3B8;
                font-size: 11px;
                letter-spacing: 1px;
            ">
                ARTIFICIAL INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.divider()

    st.sidebar.caption("ANALYTICS WORKSPACE")

    selected = st.sidebar.radio(
        "Navigation",
        options=[page["name"] for page in PAGES],
        format_func=lambda name: (
            next(
                page["icon"] + "  " + page["name"]
                for page in PAGES
                if page["name"] == name
            )
        ),
        label_visibility="collapsed",
        key="dimasa_navigation",
    )

    st.sidebar.divider()

    st.sidebar.caption("Where Hope Meets Intelligence.")

    st.sidebar.caption("Enterprise Edition · v2.0")

    return selected
