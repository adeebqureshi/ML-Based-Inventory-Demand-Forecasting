import streamlit as st
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_inventory_forecasting.config.settings import page_setup, load_css, Z_SCORES
from ml_inventory_forecasting.components.header import render as render_header
from ml_inventory_forecasting.components.sidebar import render as render_sidebar
from ml_inventory_forecasting.components.overview_tab import render as render_overview
from ml_inventory_forecasting.components.forecast_tab import render as render_forecast
from ml_inventory_forecasting.components.inventory_tab import render as render_inventory
from ml_inventory_forecasting.components.performance_tab import render as render_performance
from ml_inventory_forecasting.ml.features import prepare_features
from ml_inventory_forecasting.ml.model import train_model, train_xgboost, forecast_next_7
from ml_inventory_forecasting.ml.metrics import calc_metrics


@st.cache_resource
def train_model_cached(_df):
    return train_model(_df, split_ratio=0.8)


page_setup()
load_css()

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

render_header()
side = render_sidebar()

# Validate sidebar return value
if side is None or not isinstance(side, dict):
    st.error("Sidebar initialization failed")
    st.stop()

if side["data_loaded"] and side["df_raw"] is not None:
    st.session_state.df_raw      = side["df_raw"]
    st.session_state.data_loaded = True

# ── Welcome Screen ────────────────────────────────────────────────────────────
if not st.session_state.data_loaded:

    # Greeting banner
    st.markdown(
        "<div style='background:#ffffff;border:1px solid #E2E8F0;border-radius:14px;"
        "padding:1.75rem 2rem 1.5rem;margin-bottom:1.25rem;"
        "box-shadow:0 1px 4px rgba(15,23,42,0.06);'>"
        "<p style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.1em;color:#94A3B8;margin:0 0 0.4rem;'>GET STARTED</p>"
        "<h2 style='font-family:\"Plus Jakarta Sans\",sans-serif;font-size:1.65rem;"
        "font-weight:800;color:#0F172A;margin:0 0 0.5rem;letter-spacing:-0.02em;'>"
        "AI-Driven Inventory Demand Forecasting<br>For Smarter Stock Planning</h2>"
        "<p style='font-size:0.9rem;color:#475569;margin:0;'>"
        "Load your sales data to begin intelligent demand forecasting and inventory optimisation.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Feature cards — HRdream quick-action style
    c1, c2, c3 = st.columns(3)

    cards = [
        ("#2563EB", "rgba(37,99,235,0.1)", "", "Data Analysis",
         "Visualise historical demand patterns and seasonal trends."),
        ("#0D9488", "rgba(13,148,136,0.1)", "", "ML Forecasting",
         "Predict future drug demand with Random Forest models."),
        ("#7C3AED", "rgba(124,58,237,0.1)", "", "Inventory Optimisation",
         "Calculate EOQ, safety stock, and reorder points automatically."),
    ]

    for col, (accent, bg, icon, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f"<div style='background:#ffffff;border:1px solid #E2E8F0;"
                f"border-radius:12px;padding:1.4rem 1.25rem;"
                f"box-shadow:0 1px 4px rgba(15,23,42,0.05);height:100%;'>"
                f"<p style='font-weight:700;font-size:0.95rem;color:#0F172A;"
                f"margin:0 0 0.3rem;'>{title}</p>"
                f"<p style='font-size:0.83rem;color:#475569;margin:0;line-height:1.5;'>"
                f"{desc}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.stop()


# ── Main App ──────────────────────────────────────────────────────────────────
try:
    df = st.session_state.df_raw.copy()

    required_cols = {"date", "quantity_sold", "product"}
    if not required_cols.issubset(df.columns):
        st.error("Dataset must contain: date, quantity_sold, product")
        st.stop()

    product_counts = df["product"].value_counts()
    product_list   = ["All Products"] + product_counts.index.tolist()

    selected_product = st.selectbox(
        "Select Product to Analyse",
        product_list,
        format_func=lambda x: (
            f"All Products ({len(df)} records)"
            if x == "All Products"
            else f"{x} ({product_counts.get(x, 0)} records)"
        ),
    )

    if selected_product != "All Products":
        df = df[df["product"] == selected_product].copy()

    df = prepare_features(df)

    if len(df) < 10:
        st.error("Dataset must contain at least 10 records.")
        st.stop()
    
    # Warn if dataset is small
    if len(df) < 50:
        st.warning(f"Warning: Small dataset detected ({len(df)} records). Predictions may be unreliable.")
    
    # Warn if dataset contains zero quantities
    if (df["quantity_sold"] == 0).any():
        st.warning("Warning: Dataset contains zero quantities. MAPE metric may be unreliable.")

    model, split, y, y_pred, model_metrics = train_model_cached(df)

    metrics = calc_metrics(y.iloc[split:], y_pred)
    if metrics is None:
        metrics = {}
    metrics["MAPE"] = model_metrics.get("MAPE", 0)

    # Train XGBoost model for comparison
    xgb_model, xgb_split, xgb_y, xgb_y_pred, xgb_model_metrics = train_xgboost(df)
    xgb_metrics = {}
    if xgb_y_pred is not None:
        xgb_metrics = calc_metrics(xgb_y.iloc[xgb_split:], xgb_y_pred)
        if xgb_metrics is None:
            xgb_metrics = {}
        xgb_metrics["MAPE"] = xgb_model_metrics.get("MAPE", 0)

    forecast_7_days = forecast_next_7(df, model)

    avg_daily = float(np.mean(y_pred))
    std_daily = float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else 0.0

    z_score      = Z_SCORES.get(side["service_level"], 1.65)
    EOQ          = (np.sqrt((2 * avg_daily * 365 * side["order_cost"]) /
                   max(side["holding_cost"], 1e-6)) if avg_daily > 0 else 0)
    safety_stock = z_score * std_daily * np.sqrt(side["lead_time"])
    ROP          = (avg_daily * side["lead_time"]) + safety_stock

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Forecasting", "Inventory", "Model Performance"]
    )

    with tab1:
        render_overview(df, selected_product, avg_daily, EOQ, ROP)
    with tab2:
        render_forecast(df, forecast_7_days)
    with tab3:
        render_inventory(avg_daily, EOQ, ROP, safety_stock, side["lead_time"])
    with tab4:
        render_performance(df, model, y, split, y_pred, metrics,
                          xgb_model, xgb_split, xgb_y_pred, xgb_metrics)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check your data format and try again.")