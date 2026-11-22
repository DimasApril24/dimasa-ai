
import streamlit as st


def render_kpi_cards(metrics):
    """
    Render responsive enterprise KPI cards.

    Expected metrics:
    [
        {
            "label": "Total Observations",
            "value": "1,000",
            "delta": "12 sensor channels",
            "icon": "◈"
        }
    ]
    """

    st.markdown(
        """
        <style>
        .kpi-card {
            background: linear-gradient(
                145deg,
                #17243a 0%,
                #111c2e 100%
            );
            border: 1px solid #263852;
            border-radius: 12px;
            padding: 20px;
            min-height: 145px;
            transition: border-color 0.2s ease;
        }

        .kpi-card:hover {
            border-color: #38bdf8;
        }

        .kpi-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 18px;
        }

        .kpi-label {
            color: #a6b5c9;
            font-size: 13px;
            font-weight: 500;
        }

        .kpi-icon {
            color: #38bdf8;
            font-size: 21px;
        }

        .kpi-value {
            color: #f8fafc;
            font-size: 30px;
            font-weight: 750;
            letter-spacing: -0.8px;
        }

        .kpi-delta {
            color: #8fa3bb;
            font-size: 12px;
            margin-top: 7px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(len(metrics))

    for column, metric in zip(columns, metrics):
        with column:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-header">
                        <span class="kpi-label">
                            {metric["label"]}
                        </span>
                        <span class="kpi-icon">
                            {metric.get("icon", "◈")}
                        </span>
                    </div>
                    <div class="kpi-value">
                        {metric["value"]}
                    </div>
                    <div class="kpi-delta">
                        {metric.get("delta", "")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
