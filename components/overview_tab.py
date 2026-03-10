import streamlit as st
import pandas as pd


def render_overview_tab(df):

    st.subheader("Dataset Overview")

    st.write("### Dataset Info")

    null_percent = df.isnull().mean().mean() * 100

    if null_percent < 1:
        quality = "Good ✅"
    elif null_percent < 5:
        quality = "Moderate ⚠️"
    else:
        quality = "Poor ❌"

    info_df = pd.DataFrame({
        "Metric": [
            "Rows",
            "Date Range",
            "Data Quality"
        ],
        "Value": [
            len(df),
            f"{df['date'].min()} → {df['date'].max()}",
            quality
        ]
    })

    st.table(info_df)

    st.write("### Summary Statistics")

    st.dataframe(df.describe())