import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ..utils.charts import chart_layout

_BLUE   = "#2563EB"
_TEAL   = "#0D9488"
_TEXT   = "#0F172A"
_MID    = "#475569"
_LIGHT  = "#94A3B8"
_BORDER = "#E2E8F0"
_BG     = "#F1F5F9"
_WHITE  = "#FFFFFF"


def _section(title):
    st.markdown(
        f"<p style='font-size:0.7rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.1em;color:{_LIGHT};margin:1rem 0 0.4rem;'>{title}</p>",
        unsafe_allow_html=True,
    )


def render(df: pd.DataFrame, selected_product: str, avg_daily: float,
           EOQ: float, ROP: float):

    # ── Metric cards ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Product",
                  selected_product if selected_product != "All Products" else "All")
    with c2:
        st.metric("Avg Daily Demand", f"{avg_daily:.1f} units")
    with c3:
        st.metric("EOQ", f"{int(EOQ)} units")
    with c4:
        st.metric("Reorder Point", f"{int(ROP)} units")

    st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)

    # ── Historical Demand Trend ───────────────────────────────────────────────
    _section("Historical Demand Trend")

    fig_h = go.Figure()

    fig_h.add_trace(go.Bar(
        x=df["date"],
        y=df["quantity_sold"],
        name="Units Sold",
        marker=dict(color=_BLUE, opacity=0.3, line=dict(width=0)),
        hovertemplate="<b>%{x|%b %d, %Y}</b>  Sold: <b>%{y}</b><extra></extra>",
    ))

    fig_h.add_trace(go.Scatter(
        x=df["date"],
        y=df["quantity_sold"].rolling(30, min_periods=1).mean(),
        mode="lines",
        line=dict(color=_BLUE, width=2.5),
        name="30-day MA",
    ))

    fig_h.add_trace(go.Scatter(
        x=df["date"],
        y=df["quantity_sold"].rolling(7, min_periods=1).mean(),
        mode="lines",
        line=dict(color=_TEAL, width=1.5, dash="dot"),
        name="7-day MA",
        opacity=0.8,
    ))

    fig_h.update_layout(**chart_layout(380, ""))
    st.plotly_chart(fig_h, width='stretch')

    # ── Dataset Info ──────────────────────────────────────────────────────────
    _section("Dataset Info")
    st.dataframe(
        pd.DataFrame({
            "Attribute": ["Total Records", "Date Range", "Latest Date", "Quality"],
            "Value": [
                f"{len(df)} days",
                f"{df['date'].min().strftime('%Y-%m-%d')} → "
                f"{df['date'].max().strftime('%Y-%m-%d')}",
                df["date"].max().strftime("%Y-%m-%d"),
                "Good ✅",
            ],
        }),
        width='stretch',
        hide_index=True,
    )