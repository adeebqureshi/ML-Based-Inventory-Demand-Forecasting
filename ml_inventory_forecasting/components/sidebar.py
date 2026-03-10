import streamlit as st
import pandas as pd
import importlib.util
from pathlib import Path
from ..utils.data_processor import detect_columns, process_data


def generate_sample_data():
    file_path = Path(__file__).resolve().parents[1] / "synthetic data" / "sample_generator.py"
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))
    spec = importlib.util.spec_from_file_location("synthetic_sample_generator", file_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError("Unable to load sample data module")
    spec.loader.exec_module(module)
    if hasattr(module, "generate_sample_data"):
        return module.generate_sample_data()
    if hasattr(module, "get_sample_data"):
        return module.get_sample_data()
    if hasattr(module, "SAMPLE_DATA"):
        return module.SAMPLE_DATA.copy()
    if hasattr(module, "df"):
        return module.df.copy()
    raise AttributeError("No sample data generator found in synthetic data module")


def _nav_label(icon, text):
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;padding:8px 12px;"
        f"border-radius:8px;margin-bottom:2px;color:#475569;"
        f"font-size:0.875rem;font-weight:600;'>"
        f"<span style='font-size:1rem;'>{icon}</span>{text}</div>",
        unsafe_allow_html=True,
    )


def render():
    df_raw        = None
    data_loaded   = False
    order_cost    = 500
    holding_cost  = 2.0
    lead_time     = 5
    service_level = 95

    with st.sidebar:
        # ── Nav header ───────────────────────────────────────────────────────
        st.markdown(
            "<div style='padding:0.75rem 0.5rem 0.25rem;'>"
            "<span style='font-family:\"Plus Jakarta Sans\",sans-serif;"
            "font-size:1.1rem;font-weight:800;color:#0F172A;"
            "letter-spacing:-0.02em;'>📦 InvForecast</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── Data Source ──────────────────────────────────────────────────────
        st.markdown("#### DATA SOURCE")
        data_option = st.radio("", ["Sample Data", "Upload CSV"],
                               label_visibility="collapsed")

        if data_option == "Sample Data":
            if st.button("Generate Sample Dataset", use_container_width=True):
                with st.spinner("Generating…"):
                    try:
                        df_raw = generate_sample_data()
                        detection   = detect_columns(df_raw)
                        date_col    = detection["suggestions"]["date"]
                        qty_col     = detection["suggestions"]["quantity"]
                        product_col = detection["suggestions"]["product"]
                        processed, error = process_data(df_raw, date_col, qty_col, product_col)
                        if error:
                            st.error(error)
                        else:
                            df_raw      = processed
                            data_loaded = True
                            st.success("Sample data ready!")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            uploaded = st.file_uploader("Upload CSV File", type=["csv"],
                                        help="Upload any CSV file")
            if uploaded:
                try:
                    df_raw = pd.read_csv(uploaded)
                    st.success(f"{len(df_raw):,} records loaded")
                    st.markdown("#### Preview")
                    st.dataframe(df_raw.head(5), use_container_width=True)

                    detection       = detect_columns(df_raw)
                    date_candidates = detection["candidates"]["date"]
                    date_options    = list(df_raw.columns)
                    qty_options     = list(df_raw.columns)
                    product_options = ["None"] + list(df_raw.columns)

                    date_default = detection["suggestions"]["date"]
                    qty_default  = detection["suggestions"]["quantity"]
                    prod_default = detection["suggestions"]["product"]

                    date_index = date_options.index(date_default) if date_default in date_options else 0
                    qty_index  = qty_options.index(qty_default)   if qty_default  in qty_options  else 0
                    prod_index = product_options.index(prod_default) if prod_default in product_options else 0

                    st.markdown("#### COLUMN MAPPING")
                    date_col    = st.selectbox("Date Column",               date_options,    index=date_index)
                    qty_col     = st.selectbox("Quantity Column",           qty_options,     index=qty_index)
                    product_col = st.selectbox("Product Column (optional)", product_options, index=prod_index)

                    sequential = False
                    start_date = None
                    if not date_candidates:
                        st.info("No date column detected.")
                        sequential = st.checkbox("Generate sequential dates")
                        if sequential:
                            start_date = st.date_input("Start date")

                    if st.button("Confirm & Process Data", use_container_width=True):
                        date_arg    = None if sequential else date_col
                        product_arg = None if product_col == "None" else product_col
                        processed, error = process_data(
                            df_raw, date_arg, qty_col, product_arg,
                            sequential=sequential,
                            start_date=str(start_date) if start_date else None,
                        )
                        if error:
                            st.error(error)
                        else:
                            df_raw      = processed
                            data_loaded = True
                            st.success("Data processed successfully")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("---")

        # ── Inventory Parameters ─────────────────────────────────────────────
        st.markdown("#### PARAMETERS")
        order_cost    = st.number_input("Order Cost (₹)",           value=order_cost,   min_value=0)
        holding_cost  = st.number_input("Holding Cost (₹/unit/yr)", value=holding_cost, min_value=0.0)
        lead_time     = st.number_input("Lead Time (Days)",         value=lead_time,    min_value=1, max_value=30)
        service_level = st.slider("Service Level (%)", min_value=90, max_value=99, value=service_level)

        st.markdown("---")

        # ── How to Use ───────────────────────────────────────────────────────
        st.markdown("#### GUIDE")
        with st.expander("How to use"):
            st.markdown(
                "1. **Load Data** — Sample or upload CSV  \n"
                "2. **Select Product** — Pick from dropdown  \n"
                "3. **Review Forecast** — 7-day prediction  \n"
                "4. **Monitor Stock** — Use the gauge slider  \n"
                "5. **Act on Insights** — Follow recommendations"
            )

    return {
        "df_raw":        df_raw,
        "data_loaded":   data_loaded,
        "order_cost":    order_cost,
        "holding_cost":  holding_cost,
        "lead_time":     lead_time,
        "service_level": service_level,
    }