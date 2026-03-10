import streamlit as st
import numpy as np
from ..utils.charts import (
    chart_layout,
    create_feature_importance_chart,
)

_BLUE    = "#2563EB"
_TEAL    = "#0D9488"
_PURPLE  = "#7C3AED"
_LIGHT   = "#94A3B8"
_TEXT    = "#0F172A"
_MID     = "#475569"
_BORDER  = "#E2E8F0"
_BG      = "#F1F5F9"
_WHITE   = "#FFFFFF"
_SUCCESS = "#10B981"
_WARN    = "#F59E0B"

MODEL_FEATURE_NAMES = [
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_7", "rolling_std_7", "trend",
]


def _section(title):
    st.markdown(
        f"<p style='font-size:0.7rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.1em;color:{_LIGHT};margin:1.25rem 0 0.5rem;'>{title}</p>",
        unsafe_allow_html=True,
    )


def _metric_card(col, label, rf_val, xgb_val, fmt=".2f", lower_is_better=True, unit=""):
    """Render a side-by-side metric comparison card."""
    rf_better  = (rf_val  <= xgb_val) if lower_is_better else (rf_val  >= xgb_val)
    xgb_better = (not rf_better) if abs(rf_val - xgb_val) > 1e-6 else False

    rf_border  = f"border:2px solid {_BLUE};" if rf_better  else f"border:1px solid {_BORDER};"
    xgb_border = f"border:2px solid {_TEAL};" if xgb_better else f"border:1px solid {_BORDER};"

    rf_badge  = f"<span style='font-size:0.65rem;font-weight:700;color:{_BLUE};'>▲ BEST</span>"  if rf_better  else ""
    xgb_badge = f"<span style='font-size:0.65rem;font-weight:700;color:{_TEAL};'>▲ BEST</span>" if xgb_better else ""

    with col:
        st.markdown(
            f"<div style='background:{_WHITE};border-radius:14px;padding:1.1rem 1rem;"
            f"box-shadow:0 1px 4px rgba(15,23,42,0.06);margin-bottom:0.5rem;'>"

            f"<p style='font-size:0.68rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.1em;color:{_LIGHT};margin:0 0 0.75rem;'>{label}</p>"

            f"<div style='background:#F8FAFF;{rf_border}border-radius:10px;"
            f"padding:0.65rem 0.85rem;margin-bottom:0.4rem;"
            f"display:flex;justify-content:space-between;align-items:center;'>"
            f"<div>"
            f"<p style='font-size:0.72rem;font-weight:600;color:{_MID};margin:0 0 2px;'>🌲 Random Forest</p>"
            f"<p style='font-size:1.3rem;font-weight:800;color:{_BLUE};margin:0;"
            f"font-family:DM Mono,monospace;'>{rf_val:{fmt}}{unit}</p>"
            f"</div>{rf_badge}</div>"

            f"<div style='background:rgba(13,148,136,0.04);{xgb_border}border-radius:10px;"
            f"padding:0.65rem 0.85rem;"
            f"display:flex;justify-content:space-between;align-items:center;'>"
            f"<div>"
            f"<p style='font-size:0.72rem;font-weight:600;color:{_MID};margin:0 0 2px;'>⚡ XGBoost</p>"
            f"<p style='font-size:1.3rem;font-weight:800;color:{_TEAL};margin:0;"
            f"font-family:DM Mono,monospace;'>{xgb_val:{fmt}}{unit}</p>"
            f"</div>{xgb_badge}</div>"

            f"</div>",
            unsafe_allow_html=True,
        )


def _overall_winner_banner(rf_metrics, xgb_metrics):
    """Show a prominent banner declaring the overall winner."""
    keys  = ["MAE", "RMSE", "MAPE"]
    rf_w  = sum(1 for k in keys if rf_metrics.get(k, 0) <= xgb_metrics.get(k, 0))
    xgb_w = len(keys) - rf_w

    if rf_w > xgb_w:
        winner     = "Random Forest"
        icon       = "🌲"
        color      = _BLUE
        bg         = "#EFF6FF"
        border     = "rgba(37,99,235,0.3)"
        score_line = f"Won {rf_w}/{len(keys)} metrics"
    elif xgb_w > rf_w:
        winner     = "XGBoost"
        icon       = "⚡"
        color      = _TEAL
        bg         = "rgba(13,148,136,0.08)"
        border     = "rgba(13,148,136,0.3)"
        score_line = f"Won {xgb_w}/{len(keys)} metrics"
    else:
        winner     = "It's a Tie"
        icon       = "🤝"
        color      = _PURPLE
        bg         = "rgba(124,58,237,0.06)"
        border     = "rgba(124,58,237,0.25)"
        score_line = "Equal performance across metrics"

    st.markdown(
        f"<div style='background:{bg};border:1.5px solid {border};"
        f"border-radius:14px;padding:1.1rem 1.5rem;"
        f"display:flex;align-items:center;gap:1rem;margin-bottom:0.25rem;'>"
        f"<div style='font-size:2rem;'>{icon}</div>"
        f"<div>"
        f"<p style='font-size:0.68rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.1em;color:{_LIGHT};margin:0;'>Overall Winner</p>"
        f"<p style='font-size:1.2rem;font-weight:800;color:{color};margin:0 0 1px;"
        f"font-family:Plus Jakarta Sans,sans-serif;'>{winner}</p>"
        f"<p style='font-size:0.8rem;color:{_MID};margin:0;'>{score_line}</p>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render(df,
           rf_model,  y,  rf_split,  rf_y_pred,  rf_metrics,
           xgb_model,     xgb_split, xgb_y_pred, xgb_metrics):

    xgb_available = xgb_model is not None and xgb_y_pred is not None

    # ── Section header ────────────────────────────────────────────────────────
    _section("Model Performance & Comparison")

    # ── Overall winner banner ─────────────────────────────────────────────────
    if xgb_available:
        _overall_winner_banner(rf_metrics, xgb_metrics)
        st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)

    # ── Metric comparison cards ───────────────────────────────────────────────
    if xgb_available:
        c1, c2, c3 = st.columns(3)
        _metric_card(c1, "Mean Absolute Error",
                     rf_metrics.get("MAE",  0), xgb_metrics.get("MAE",  0),
                     fmt=".2f", lower_is_better=True, unit=" u")
        _metric_card(c2, "Root Mean Square Error",
                     rf_metrics.get("RMSE", 0), xgb_metrics.get("RMSE", 0),
                     fmt=".2f", lower_is_better=True, unit=" u")
        _metric_card(c3, "Mean Abs % Error",
                     rf_metrics.get("MAPE", 0), xgb_metrics.get("MAPE", 0),
                     fmt=".2f", lower_is_better=True, unit="%")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Mean Absolute Error",    f"{rf_metrics.get('MAE',  0):.2f} units")
        with c2:
            st.metric("Root Mean Square Error", f"{rf_metrics.get('RMSE', 0):.2f} units")
        with c3:
            st.metric("Mean Abs % Error",       f"{rf_metrics.get('MAPE', 0):.2f}%")

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # ── Algorithm info cards ──────────────────────────────────────────────────
    _section("Algorithm Details")

    if xgb_available:
        ca, cb = st.columns(2)
        with ca:
            st.markdown(
                f"<div style='background:#EFF6FF;border:1px solid rgba(37,99,235,0.2);"
                f"border-left:4px solid {_BLUE};border-radius:10px;padding:1rem 1.25rem;'>"
                f"<p style='font-weight:700;font-size:0.9rem;color:{_BLUE};margin:0 0 0.4rem;'>"
                f"🌲 Random Forest Regressor</p>"
                f"<p style='font-size:0.82rem;color:{_MID};margin:0;line-height:1.7;'>"
                f"<b>Trees:</b> 100 &nbsp;·&nbsp; <b>Max Depth:</b> 10<br>"
                f"<b>Features:</b> Lag-1·2·3, Rolling Mean/Std, Trend<br>"
                f"<b>Split:</b> 80/20 train/test</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with cb:
            st.markdown(
                f"<div style='background:rgba(13,148,136,0.06);"
                f"border:1px solid rgba(13,148,136,0.2);"
                f"border-left:4px solid {_TEAL};border-radius:10px;padding:1rem 1.25rem;'>"
                f"<p style='font-weight:700;font-size:0.9rem;color:{_TEAL};margin:0 0 0.4rem;'>"
                f"⚡ XGBoost Regressor</p>"
                f"<p style='font-size:0.82rem;color:{_MID};margin:0;line-height:1.7;'>"
                f"<b>Trees:</b> 300 &nbsp;·&nbsp; <b>Max Depth:</b> 6 &nbsp;·&nbsp; <b>LR:</b> 0.05<br>"
                f"<b>Subsample:</b> 0.8 &nbsp;·&nbsp; <b>ColSample:</b> 0.8<br>"
                f"<b>Split:</b> 80/20 train/test</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"<div class='info-box'><h4>Algorithm Details</h4>"
            f"<p><strong>Algorithm:</strong> Random Forest Regressor &nbsp;·&nbsp; "
            f"<strong>Trees:</strong> 100 &nbsp;·&nbsp; <strong>Max Depth:</strong> 10 &nbsp;·&nbsp; "
            f"<strong>Split:</strong> 80/20</p>"
            f"<p style='color:{_WARN};font-size:0.82rem;margin-top:0.4rem;'>"
            f"⚠️ XGBoost not installed — run <code>pip install xgboost</code> to enable comparison.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Feature importance side by side ───────────────────────────────────────
    if hasattr(rf_model, "feature_importances_"):
        _section("Feature Importance")

        if xgb_available and hasattr(xgb_model, "feature_importances_"):
            fi_c1, fi_c2 = st.columns(2)
            with fi_c1:
                st.markdown(
                    f"<p style='font-size:0.75rem;font-weight:700;color:{_BLUE};"
                    f"margin-bottom:0.25rem;'>🌲 Random Forest</p>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    create_feature_importance_chart(rf_model, MODEL_FEATURE_NAMES),
                    use_container_width=True,
                )
            with fi_c2:
                st.markdown(
                    f"<p style='font-size:0.75rem;font-weight:700;color:{_TEAL};"
                    f"margin-bottom:0.25rem;'>⚡ XGBoost</p>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    create_feature_importance_chart(xgb_model, MODEL_FEATURE_NAMES),
                    use_container_width=True,
                )
        else:
            st.plotly_chart(
                create_feature_importance_chart(rf_model, MODEL_FEATURE_NAMES),
                use_container_width=True,
            )