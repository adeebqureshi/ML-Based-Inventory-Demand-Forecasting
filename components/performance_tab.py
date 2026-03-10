import streamlit as st
import plotly.express as px
import pandas as pd
from ..ml.model import MODEL_FEATURE_NAMES


def render_performance_tab(y, split, y_pred, metrics, model):

    st.subheader("Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("MAE", f"{metrics.get('MAE',0):.2f}")

    with c2:
        st.metric("RMSE", f"{metrics.get('RMSE',0):.2f}")

    with c3:
        st.metric("MAPE", f"{metrics.get('MAPE',0):.2f}%")

    with c4:
        st.metric("R² Score", f"{metrics.get('R2',0):.3f}")

    actual = y.iloc[split:].values

    n = min(len(actual), len(y_pred))

    df_chart = pd.DataFrame({
        "Actual": actual[:n],
        "Predicted": y_pred[:n]
    })

    fig = px.line(df_chart)

    st.plotly_chart(fig, use_container_width=True)

    if hasattr(model, "feature_importances_"):

        fi_df = pd.DataFrame({
            "Feature": MODEL_FEATURE_NAMES,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.write("### Feature Importance")

        fig2 = px.bar(fi_df, x="Importance", y="Feature", orientation="h")

        st.plotly_chart(fig2, use_container_width=True)