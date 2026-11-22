
import streamlit as st
from datetime import datetime


def render_header(title, subtitle, section="DIMASA AI ANALYTICS"):
    """Render consistent DIMASA AI enterprise header."""

    left, right = st.columns([5, 1])

    with left:
        st.markdown(
            f'<div style="color:#38BDF8;font-size:11px;font-weight:700;letter-spacing:2px;margin-bottom:9px;">{section}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="color:#F8FAFC;font-size:30px;font-weight:800;letter-spacing:-0.8px;">{title}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="color:#94A3B8;font-size:14px;margin-top:7px;">{subtitle}</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div style="background:#14243A;border:1px solid #263852;border-radius:9px;padding:9px 13px;color:#BAE6FD;font-size:11px;white-space:nowrap;text-align:center;margin-top:15px;">● SYSTEM ONLINE</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"DIMASA AI Enterprise Edition | "
        f"{datetime.now().astimezone().strftime('%d %b %Y, %H:%M')}"
    )

    st.divider()
