import plotly.graph_objects as go
import pandas as pd

# HRdream palette
_BLUE       = "#2563EB"
_TEAL       = "#0D9488"
_PURPLE     = "#7C3AED"
_BLUE_LIGHT = "rgba(37,99,235,0.08)"
_TEAL_LIGHT = "rgba(13,148,136,0.08)"
_BG         = "rgba(0,0,0,0)"
_GRID       = "rgba(226,232,240,0.8)"
_TEXT       = "#0F172A"
_TEXT_MID   = "#475569"
_TEXT_LIGHT = "#94A3B8"
_BORDER     = "#E2E8F0"
_SUCCESS    = "#10B981"
_DANGER     = "#EF4444"
_WARN       = "#F59E0B"
_FONT       = "Plus Jakarta Sans, sans-serif"
_MONO       = "DM Mono, monospace"

# Legacy aliases
TEAL  = _BLUE
MINT  = _TEAL
BROWN = _TEXT


def chart_layout(height=420, title=""):
    return dict(
        title=dict(
            text=title,
            font=dict(size=14, family=_FONT, color=_TEXT),
        ),
        template="plotly_white",
        height=height,
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, size=12, color=_TEXT_MID),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            font=dict(size=11, family=_FONT),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor=_GRID,
            zeroline=False,
            linecolor=_BORDER,
            tickfont=dict(family=_FONT, size=11, color=_TEXT_LIGHT),
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor=_GRID,
            zeroline=False,
            linecolor=_BORDER,
            tickfont=dict(family=_FONT, size=11, color=_TEXT_LIGHT),
            showgrid=True,
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )


def create_forecast_chart(df_hist, df_fc):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_hist["date"],
        y=df_hist["quantity_sold"],
        name="Historical",
        marker=dict(
            color=_BLUE,
            opacity=0.35,
            line=dict(width=0),
        ),
        hovertemplate="<b>%{x|%b %d}</b>  Sales: <b>%{y:.0f}</b><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_hist["date"],
        y=df_hist["quantity_sold"].rolling(7, min_periods=1).mean(),
        mode="lines",
        name="7-day trend",
        line=dict(color=_BLUE, width=2.5),
        hovertemplate="<b>%{x|%b %d}</b>  Trend: <b>%{y:.1f}</b><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_fc["date"],
        y=df_fc["predicted_demand"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color=_TEAL, width=2.5),
        marker=dict(size=8, symbol="circle", color=_TEAL,
                    line=dict(color="white", width=2)),
        hovertemplate="<b>%{x|%b %d}</b>  Forecast: <b>%{y:.0f}</b><extra></extra>",
    ))

    fig.update_layout(**chart_layout(420, "7-Day Demand Forecast"))
    return fig


def create_gauge(curr, rop, eoq):
    if curr > rop:
        color      = _SUCCESS
        status_key = "healthy"
    elif curr > rop * 0.5:
        color      = _WARN
        status_key = "warning"
    else:
        color      = _DANGER
        status_key = "critical"

    gauge_max = max(eoq * 1.5, 1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=curr,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Current Stock Level",
               "font": {"size": 13, "family": _FONT, "color": _TEXT}},
        delta={"reference": rop,
               "increasing": {"color": _SUCCESS},
               "decreasing": {"color": _DANGER}},
        number={"font": {"color": color, "family": _MONO, "size": 34}},
        gauge={
            "axis": {
                "range": [None, gauge_max],
                "tickfont": {"family": _MONO, "size": 10, "color": _TEXT_LIGHT},
                "tickcolor": _BORDER,
            },
            "bar":         {"color": color, "thickness": 0.7},
            "bgcolor":     _BG,
            "borderwidth": 0,
            "steps": [
                {"range": [0, rop * 0.5],  "color": "rgba(239,68,68,0.07)"},
                {"range": [rop * 0.5, rop], "color": "rgba(245,158,11,0.07)"},
                {"range": [rop, gauge_max], "color": "rgba(16,185,129,0.07)"},
            ],
            "threshold": {
                "line": {"color": _BLUE, "width": 2},
                "thickness": 0.75,
                "value": rop,
            },
        },
    ))

    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=40, b=5),
        paper_bgcolor=_BG,
        font=dict(family=_FONT),
    )

    return fig, status_key


def create_feature_importance_chart(model, feature_names):
    importance = model.feature_importances_
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=True)

    n = len(df)
    colours = [
        f"rgba(37,99,235,{0.3 + 0.7 * i / max(n - 1, 1):.2f})"
        for i in range(n)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(**chart_layout(380, "Feature Importance"))
    return fig
