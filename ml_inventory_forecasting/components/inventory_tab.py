import streamlit as st
from ..utils.charts import create_gauge

_BLUE    = "#2563EB"
_TEAL    = "#0D9488"
_PURPLE  = "#7C3AED"
_LIGHT   = "#94A3B8"
_TEXT    = "#0F172A"
_MID     = "#475569"
_BORDER  = "#E2E8F0"
_WHITE   = "#FFFFFF"
_BG      = "#F1F5F9"
_SUCCESS = "#10B981"
_DANGER  = "#EF4444"
_WARN    = "#F59E0B"


def _section(title):
    st.markdown(
        f"<p style='font-size:0.7rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.1em;color:{_LIGHT};margin:1.25rem 0 0.6rem;'>{title}</p>",
        unsafe_allow_html=True,
    )


def _kpi_card(icon, label, value, sub, accent):
    """Single KPI card with icon, value and subtitle."""
    return (
        f"<div style='background:{_WHITE};border:1px solid {_BORDER};"
        f"border-top:3px solid {accent};border-radius:12px;"
        f"padding:1rem 1.1rem;box-shadow:0 1px 4px rgba(15,23,42,0.05);'>"
        f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem;'>"
        f"<span style='font-size:1.1rem;'>{icon}</span>"
        f"<span style='font-size:0.68rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.08em;color:{_LIGHT};'>{label}</span>"
        f"</div>"
        f"<p style='font-size:1.45rem;font-weight:800;color:{_TEXT};"
        f"font-family:DM Mono,monospace;margin:0 0 2px;'>{value}</p>"
        f"<p style='font-size:0.75rem;color:{_MID};margin:0;'>{sub}</p>"
        f"</div>"
    )


def _status_pill(curr_stock, ROP, EOQ):
    """Top status bar — critical / warning / healthy."""
    if curr_stock <= ROP * 0.5:
        bg      = "rgba(239,68,68,0.08)"
        border  = "rgba(239,68,68,0.3)"
        dot     = _DANGER
        label   = "CRITICAL"
        title   = "Stock Critically Low"
        msg     = f"Order <strong>{int(EOQ)} units</strong> immediately to avoid stockout."
        icon    = "🚨"
    elif curr_stock <= ROP:
        bg      = "rgba(245,158,11,0.08)"
        border  = "rgba(245,158,11,0.3)"
        dot     = _WARN
        label   = "WARNING"
        title   = "Approaching Reorder Point"
        msg     = f"Consider ordering <strong>{int(EOQ)} units</strong> soon."
        icon    = "⚠️"
    else:
        bg      = "rgba(16,185,129,0.08)"
        border  = "rgba(16,185,129,0.25)"
        dot     = _SUCCESS
        label   = "HEALTHY"
        title   = "Stock Levels Adequate"
        msg     = "Inventory is well-stocked. Continue monitoring regularly."
        icon    = "✅"

    st.markdown(
        f"<div style='background:{bg};border:1.5px solid {border};"
        f"border-radius:12px;padding:0.9rem 1.25rem;"
        f"display:flex;align-items:center;gap:1rem;margin-bottom:0.25rem;'>"
        f"<span style='font-size:1.6rem;'>{icon}</span>"
        f"<div style='flex:1;'>"
        f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:2px;'>"
        f"<span style='width:7px;height:7px;border-radius:50%;background:{dot};"
        f"display:inline-block;'></span>"
        f"<span style='font-size:0.65rem;font-weight:800;letter-spacing:0.1em;"
        f"color:{dot};text-transform:uppercase;'>{label}</span>"
        f"</div>"
        f"<p style='font-size:0.92rem;font-weight:700;color:{_TEXT};margin:0 0 2px;'>{title}</p>"
        f"<p style='font-size:0.8rem;color:{_MID};margin:0;'>{msg}</p>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _metric_row(icon, label, value, desc, accent, bg_tint):
    """Single row inside the metrics panel."""
    return (
        f"<div style='display:flex;align-items:center;gap:0.9rem;"
        f"padding:0.75rem 0.9rem;background:{bg_tint};"
        f"border-radius:10px;margin-bottom:0.4rem;'>"
        f"<div style='width:36px;height:36px;border-radius:9px;"
        f"background:{accent}1A;display:flex;align-items:center;"
        f"justify-content:center;font-size:1rem;flex-shrink:0;'>{icon}</div>"
        f"<div style='flex:1;min-width:0;'>"
        f"<p style='font-size:0.72rem;color:{_LIGHT};font-weight:600;margin:0;'>{label}</p>"
        f"<p style='font-size:0.95rem;font-weight:800;color:{_TEXT};"
        f"font-family:DM Mono,monospace;margin:0;'>{value}</p>"
        f"</div>"
        f"<p style='font-size:0.72rem;color:{_MID};margin:0;text-align:right;"
        f"max-width:110px;line-height:1.4;'>{desc}</p>"
        f"</div>"
    )


def render(avg_daily: float, EOQ: float, ROP: float,
           safety_stock: float, lead_time: int):

    # ── Derived values ────────────────────────────────────────────────────────
    slider_max     = max(int(EOQ * 1.5), 100)
    slider_default = min(int(ROP * 1.1), slider_max)

    # ── Page title ────────────────────────────────────────────────────────────
    _section("Inventory Management Dashboard")

    # ── Top KPI strip ─────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, "📦", "EOQ",           f"{int(EOQ)}",          "Optimal order qty",  _BLUE),
        (k2, "🔁", "Reorder Point", f"{int(ROP)}",          "Units trigger",      _PURPLE),
        (k3, "🛡️", "Safety Stock",  f"{int(safety_stock)}", "Buffer units",       _TEAL),
        (k4, "📅", "Lead Time",     f"{lead_time}d",        "Days to receive",    _WARN),
    ]
    for col, icon, label, value, sub, accent in kpis:
        with col:
            st.markdown(_kpi_card(icon, label, value, sub, accent), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)

    # ── Main layout: gauge left | metrics right ───────────────────────────────
    col_gauge, col_metrics = st.columns([1, 1], gap="large")

    with col_gauge:
        _section("Live Stock Monitor")

        curr_stock = st.slider(
            "Adjust Current Stock Level",
            min_value=0,
            max_value=slider_max,
            value=slider_default,
        )

        fig_g, status = create_gauge(curr_stock, ROP, EOQ)
        st.plotly_chart(fig_g, use_container_width=True)

        _status_pill(curr_stock, ROP, EOQ)

    with col_metrics:
        _section("Inventory Metrics")

        days_out    = curr_stock / avg_daily if avg_daily > 0 else 0
        order_cycle = int(EOQ / avg_daily)   if avg_daily > 0 else 0
        ltd         = int(avg_daily * lead_time)

        rows = [
            ("📦", "Economic Order Qty",  f"{int(EOQ)} units",       "Optimal order size",         _BLUE,   _WHITE),
            ("🔁", "Reorder Point",       f"{int(ROP)} units",       "Place order when reached",   _PURPLE, _BG),
            ("🛡️", "Safety Stock",        f"{int(safety_stock)} units","Buffer against variability", _TEAL,   _WHITE),
            ("📈", "Avg Daily Demand",    f"{avg_daily:.1f} / day",  "Mean units per day",         _BLUE,   _BG),
            ("🚚", "Lead Time Demand",    f"{ltd} units",            f"Demand over {lead_time}d",  _WARN,   _WHITE),
            ("⏳", "Days to Stockout",    f"{days_out:.1f} days",    "At current stock level",     _DANGER, _BG),
            ("🔄", "Order Cycle",         f"Every {order_cycle}d",   "Recommended frequency",      _TEAL,   _WHITE),
        ]

        html = "".join(
            _metric_row(icon, label, value, desc, accent, bg)
            for icon, label, value, desc, accent, bg in rows
        )
        st.markdown(
            f"<div style='background:{_WHITE};border:1px solid {_BORDER};"
            f"border-radius:14px;padding:0.75rem;'>{html}</div>",
            unsafe_allow_html=True,
        )

        # ── Action recommendation card ────────────────────────────────────────
        st.markdown("<div style='margin-top:0.85rem;'></div>", unsafe_allow_html=True)
        _section("Recommended Action")

        if curr_stock <= ROP * 0.5:
            accent = _DANGER
            bg     = "rgba(239,68,68,0.06)"
            border = "rgba(239,68,68,0.2)"
            action = f"Place an urgent order of <strong>{int(EOQ)} units</strong> now."
            note   = f"Stock will last approx. <strong>{days_out:.1f} days</strong>. Lead time is {lead_time} days — you are at risk."
        elif curr_stock <= ROP:
            accent = _WARN
            bg     = "rgba(245,158,11,0.06)"
            border = "rgba(245,158,11,0.2)"
            action = f"Schedule an order of <strong>{int(EOQ)} units</strong> within the next few days."
            note   = f"Reorder point reached. Next order cycle recommended every <strong>{order_cycle} days</strong>."
        else:
            accent = _SUCCESS
            bg     = "rgba(16,185,129,0.06)"
            border = "rgba(16,185,129,0.2)"
            action = "No immediate action required."
            note   = f"Next reorder in approx. <strong>{max(0, int(days_out - lead_time))} days</strong>. Order <strong>{int(EOQ)} units</strong> every <strong>{order_cycle} days</strong>."

        st.markdown(
            f"<div style='background:{bg};border:1px solid {border};"
            f"border-left:4px solid {accent};border-radius:10px;padding:1rem 1.2rem;'>"
            f"<p style='font-size:0.85rem;font-weight:700;color:{_TEXT};margin:0 0 0.3rem;'>{action}</p>"
            f"<p style='font-size:0.8rem;color:{_MID};margin:0;line-height:1.6;'>{note}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )