import streamlit as st
import pandas as pd
from ..utils.charts import create_gauge

_LIGHT = "#94A3B8"
_TEXT  = "#0F172A"
_MID   = "#475569"


def _section(title):
    st.markdown(
        f"<p style='font-size:0.7rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.1em;color:{_LIGHT};margin:1rem 0 0.4rem;'>{title}</p>",
        unsafe_allow_html=True,
    )


def render(avg_daily: float, EOQ: float, ROP: float,
           safety_stock: float, lead_time: int):

    _section("Inventory Management Dashboard")

    c1, c2 = st.columns(2)

    with c1:
        slider_max     = max(int(EOQ * 1.5), 100)
        slider_default = min(int(ROP * 1.1), slider_max)
        curr_stock     = st.slider("Current Stock Level", 0, slider_max, slider_default)

        fig_g, status = create_gauge(curr_stock, ROP, EOQ)
        st.plotly_chart(fig_g, use_container_width=True)

        if curr_stock <= ROP * 0.5:
            st.markdown(
                f'<div class="warning-box"><h4>{status}</h4>'
                f'<p>Order <strong>{int(EOQ)}</strong> units immediately!</p></div>',
                unsafe_allow_html=True,
            )
        elif curr_stock <= ROP:
            st.markdown(
                f'<div class="warning-box"><h4>{status}</h4>'
                f'<p>Consider ordering <strong>{int(EOQ)}</strong> units soon.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="success-box"><h4>{status}</h4>'
                f'<p>Stock levels are adequate. Monitor regularly.</p></div>',
                unsafe_allow_html=True,
            )

    with c2:
        _section("Inventory Metrics")
        st.dataframe(
            pd.DataFrame({
                "Metric": [
                    "EOQ", "Reorder Point", "Safety Stock",
                    "Avg Daily", "Lead Time Demand", "Service Level",
                ],
                "Value": [
                    f"{int(EOQ)} units", f"{int(ROP)} units",
                    f"{int(safety_stock)} units", f"{avg_daily:.1f}/day",
                    f"{int(avg_daily * lead_time)} units", "—",
                ],
                "Description": [
                    "Optimal order size", "When to reorder",
                    "Buffer stock", "Daily average",
                    f"{lead_time}-day demand", "Availability target",
                ],
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
        _section("Recommendations")

        days_out    = curr_stock / avg_daily if avg_daily > 0 else 0
        order_cycle = int(EOQ / avg_daily) if avg_daily > 0 else 0

        st.info(
            f"**Current Status**\n"
            f"- Days until stockout: **{days_out:.1f} days**\n"
            f"- Recommended order: **{int(EOQ)} units**\n"
            f"- Order every: **{order_cycle} days**"
        )