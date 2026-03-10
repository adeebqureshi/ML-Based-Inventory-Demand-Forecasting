import streamlit as st
import pandas as pd
from ..utils.charts import create_forecast_chart

_LIGHT = "#94A3B8"


def _section(title):
    st.markdown(
        f"<p style='font-size:0.7rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.1em;color:{_LIGHT};margin:1rem 0 0.4rem;'>{title}</p>",
        unsafe_allow_html=True,
    )


def render(df_hist: pd.DataFrame, forecast_7_days: pd.DataFrame):

    _section("7-Day Demand Forecast")
    st.plotly_chart(
        create_forecast_chart(df_hist.tail(30), forecast_7_days),
        use_container_width=True,
    )

    c1, c2 = st.columns([2, 1])

    with c1:
        _section("Forecast Table")
        fd = forecast_7_days.copy()
        fd["date"] = fd["date"].dt.strftime("%Y-%m-%d  (%A)")
        fd.columns = ["Date", "Predicted Demand (units)"]
        st.dataframe(fd, use_container_width=True, hide_index=True)

    with c2:
        _section("Summary")
        st.metric("Total 7-Day",  f"{forecast_7_days['predicted_demand'].sum():.0f} units")
        st.metric("Daily Avg",    f"{forecast_7_days['predicted_demand'].mean():.1f} units")
        st.metric("Peak Day",     f"{forecast_7_days['predicted_demand'].max():.0f} units")
        st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
        st.download_button(
            "↓ Download Forecast CSV",
            forecast_7_days.to_csv(index=False),
            "forecast_7days.csv",
            "text/csv",
            use_container_width=True,
        )