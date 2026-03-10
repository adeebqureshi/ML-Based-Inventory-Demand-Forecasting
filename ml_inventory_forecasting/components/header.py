import streamlit as st


def render():
    st.markdown(
        '<h1 class="main-title">ML Inventory Forecasting</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">Intelligent demand prediction & inventory optimisation — powered by machine learning</p>',
        unsafe_allow_html=True,
    )