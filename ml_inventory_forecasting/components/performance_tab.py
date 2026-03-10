import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ..utils.charts import chart_layout, create_feature_importance_chart

_BLUE   = "#2563EB"
_TEAL   = "#0D9488"
_LIGHT  = "#94A3B8"
_TEXT   = "#0F172A"

MODEL_FEATURE_NAMES = [
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_7", "rolling_std_7", "trend",
]


def _section(title):
    st.markdown(
        f"<p style='font-size:0.7rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.1em;color:{_LIGHT};margin:1rem 0 0.4rem;'>{title}</p>",
        unsafe_allow_html=True,
    )


def render(df, model, y, split, y_pred, metrics):

    # ── KPI row ───────────────────────────────────────────────────────────────
    _section("Model Performance Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Mean Absolute Error",    f"{metrics.get('MAE',  0):.2f} units")
    with c2:
        st.metric("Root Mean Square Error", f"{metrics.get('RMSE', 0):.2f} units")
    with c3:
        st.metric("Mean Abs % Error",       f"{metrics.get('MAPE', 0):.2f}%")

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    with c1:
        _section("Actual vs Predicted")
        actual    = np.array(y.iloc[split:])
        predicted = np.array(y_pred)
        n         = min(len(actual), len(predicted))
        actual    = actual[:n]
        predicted = predicted[:n]
        x_axis    = list(range(n))

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=x_axis, y=actual,
            mode="lines", name="Actual",
            line=dict(color=_BLUE, width=2),
        ))
        fig_pred.add_trace(go.Scatter(
            x=x_axis, y=predicted,
            mode="lines", name="Predicted",
            line=dict(color=_TEAL, width=2, dash="dot"),
        ))
        fig_pred.update_layout(**chart_layout(360, "Actual vs Predicted Demand"))
        st.plotly_chart(fig_pred, use_container_width=True)

    # ── Residual Distribution ─────────────────────────────────────────────────
    with c2:
        _section("Residual Distribution")
        residuals = actual - predicted
        fig_r = px.histogram(
            residuals, nbins=30,
            title="Prediction Error Distribution",
            labels={"value": "Residual (units)", "count": "Frequency"},
            color_discrete_sequence=[_BLUE],
        )
        fig_r.update_layout(**chart_layout(360))
        st.plotly_chart(fig_r, use_container_width=True)

    # ── Model info ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="info-box">'
        '<h4>Algorithm Details</h4>'
        '<p><strong>Algorithm:</strong> Random Forest Regressor &nbsp;·&nbsp; '
        '<strong>Features:</strong> Lag-1 · Lag-2 · Lag-3 · Rolling Mean · Rolling Std · Trend &nbsp;·&nbsp; '
        '<strong>Split:</strong> 80/20 &nbsp;·&nbsp; '
        '<strong>Trees:</strong> 100 &nbsp;·&nbsp; '
        '<strong>Max Depth:</strong> 10</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Feature Importance ────────────────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        _section("Feature Importance")
        st.plotly_chart(
            create_feature_importance_chart(model, MODEL_FEATURE_NAMES),
            use_container_width=True,
        )