import streamlit as st
from pathlib import Path

# ── HRdream-inspired palette ──────────────────────────────────────────────────
WHITE       = "#FFFFFF"
BG          = "#F1F5F9"       # page background  (light grey)
SIDEBAR_BG  = "#FFFFFF"       # sidebar white
BLUE        = "#2563EB"       # primary blue
BLUE_DARK   = "#1D4ED8"       # hover blue
BLUE_LIGHT  = "#EFF6FF"       # blue tint
TEAL        = "#0D9488"       # secondary accent (teal cards)
PURPLE      = "#7C3AED"       # tertiary accent
TEXT        = "#0F172A"       # primary text (near-black)
TEXT_MID    = "#475569"       # secondary text
TEXT_LIGHT  = "#94A3B8"       # muted text
BORDER      = "#E2E8F0"       # dividers / borders
SUCCESS     = "#10B981"
DANGER      = "#EF4444"
WARN        = "#F59E0B"

# Legacy aliases — keep all existing chart/helper imports working
TEAL_ALIAS  = BLUE
MINT_ALIAS  = TEAL

Z_SCORES = {90: 1.28, 95: 1.65, 99: 2.33}


def page_setup():
    st.set_page_config(
        page_title="ML Inventory Forecasting",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def load_css():
    css_path = Path(__file__).resolve().parent.parent / "styles" / "style.css"
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass
    # JS: hide only the raw icon text inside the sidebar toggle button.
    # We keep the button itself functional so user can re-open the sidebar.
    st.components.v1.html(
        """
        <script>
        function fixCollapseBtn() {
            var doc = window.parent.document;
            // Find spans whose text is the raw material icon name and blank them
            doc.querySelectorAll(
                '[data-testid="stSidebarCollapsedControl"] span, ' +
                '[data-testid="collapsedControl"] span'
            ).forEach(function(span) {
                if (span.innerText && span.innerText.trim().startsWith('keyboard')) {
                    span.style.fontSize   = '0';
                    span.style.color      = 'transparent';
                    span.style.lineHeight = '0';
                }
            });
        }
        fixCollapseBtn();
        var obs = new MutationObserver(fixCollapseBtn);
        obs.observe(window.parent.document.body, { childList: true, subtree: true });
        </script>
        """,
        height=0,
        width=0,
    )