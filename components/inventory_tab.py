import streamlit as st


def render_inventory_tab(avg_daily, eoq, rop, safety_stock, service_level):

    st.subheader("Inventory Control")

    current_stock = st.slider(
        "Current Stock Level",
        min_value=0,
        max_value=int(eoq * 2),
        value=int(eoq),
    )

    if avg_daily == 0:

        st.warning("Insufficient demand data to estimate stockout timing.")

        days_out = None
        order_cycle = None

    else:

        days_out = current_stock / avg_daily
        order_cycle = eoq / avg_daily

    st.write("### Inventory Metrics")

    st.table({
        "Metric": [
            "EOQ",
            "Reorder Point",
            "Safety Stock",
            "Service Level"
        ],
        "Value": [
            round(eoq, 2),
            round(rop, 2),
            round(safety_stock, 2),
            f"{service_level}%"
        ]
    })

    if days_out is not None:

        st.info(f"Estimated days until stockout: {round(days_out,2)}")

        st.success(f"Recommended order cycle: {round(order_cycle,2)} days")