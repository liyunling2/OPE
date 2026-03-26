"""
app.py - Main entry point
Run with: streamlit run app.py
"""
import base64
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "logo" / "hh_logo.png"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

st.set_page_config(
    page_title="HungryHub Performance Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

:root {
    --bg: #f4f6f9;
    --surface: #ffffff;
    --surface2: #f8f9fa;
    --border: #e0e0e0;
    --accent: #cc0000;
    --text: #111827;
    --muted: #6b7280;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

section.main > div.block-container {
    max-width: 1380px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.35rem;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
}

[data-testid="stMetricLabel"] {
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700;
}

[data-testid="stMetricDelta"] svg {
    display: none;
}

.header-shell {
    position: relative;
    overflow: hidden;
    margin-bottom: 0;
    padding: 1rem 1.25rem 0.9rem;
    border-radius: 26px 26px 0 0;
    background:
        radial-gradient(circle at top left, rgba(239, 68, 68, 0.18), transparent 28%),
        linear-gradient(135deg, #0f172a 0%, #121c34 48%, #0b1220 100%);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-bottom: none;
    box-shadow: 0 20px 45px rgba(15, 23, 42, 0.22);
}

.header-shell::after {
    content: "";
    position: absolute;
    right: -3rem;
    bottom: -4rem;
    width: 180px;
    height: 180px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(248, 113, 113, 0.16), transparent 68%);
    pointer-events: none;
}

.brand-panel {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 0.95rem;
}

.brand-tile {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 auto;
    padding: 0.55rem 0.7rem;
    background: linear-gradient(180deg, #fff7f7 0%, #ffffff 100%);
    border: 1px solid #f3d1d1;
    border-radius: 18px;
    box-shadow: 0 12px 24px rgba(17, 24, 39, 0.18);
}

.brand-tile img {
    display: block;
    width: 56px;
    height: auto;
}

.brand-copy {
    flex: 1 1 auto;
    min-width: 0;
}

.brand-kicker {
    margin-bottom: 0.15rem;
    color: #fca5a5;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.brand-title {
    margin: 0;
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem;
    line-height: 1.02;
    letter-spacing: -0.03em;
}

.brand-subtitle {
    margin: 0.3rem 0 0;
    max-width: none;
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.45;
}

div.stRadio {
    margin-top: -0.1rem;
    margin-bottom: 1rem;
}

div.stRadio > div[role="radiogroup"] {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.4rem;
    padding: 0.4rem;
    background: rgba(15, 23, 42, 0.62);
    border: 1px solid #22304b;
    border-top: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 0 0 24px 24px;
    box-shadow: 0 18px 34px rgba(15, 23, 42, 0.16);
}

div.stRadio > div[role="radiogroup"] > label {
    min-width: 0;
    justify-content: center;
    background: transparent !important;
    padding: 0.78rem 0.7rem;
    border: 1px solid transparent;
    border-radius: 12px;
}

div.stRadio > div[role="radiogroup"] > label > div:first-child {
    display: none;
}

div.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
    background: linear-gradient(180deg, #df2626 0%, #b51212 100%) !important;
    border-color: rgba(255, 255, 255, 0.14);
    box-shadow: 0 10px 20px rgba(204, 0, 0, 0.22);
}

div.stRadio > div[role="radiogroup"] > label[data-checked="true"] p {
    color: #ffffff !important;
    font-weight: 700;
}

div.stRadio p {
    font-size: 1rem;
    color: #dbe4f3 !important;
    font-weight: 600;
    text-align: center;
    white-space: nowrap;
}

[data-testid="stSidebar"] {
    display: none;
}

@media (max-width: 900px) {
    .header-shell {
        padding: 0.95rem 0.95rem 0.85rem;
        border-radius: 22px 22px 0 0;
    }

    .brand-title {
        font-size: 1.4rem;
    }

    .brand-subtitle {
        font-size: 0.88rem;
    }

    div.stRadio > div[role="radiogroup"] {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        border-radius: 0 0 22px 22px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
st.markdown(
    f"""
    <div class="header-shell">
        <div class="brand-panel">
            <div class="brand-tile">
                <img src="data:image/png;base64,{logo_b64}" alt="HungryHub logo" />
            </div>
            <div class="brand-copy">
                <div class="brand-kicker">HungryHub Intelligence</div>
                <h1 class="brand-title">Growth Command Center</h1>
                <p class="brand-subtitle">
                    Monitor restaurant momentum, priority, cluster themes, and Google Ads efficiency in one operating view.
                </p>
            </div>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

page = st.radio(
    "Navigation",
    ["Overview", "Momentum", "Priority", "Clustering", "Strategy"],
    label_visibility="collapsed",
    horizontal=True,
)

st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

if page == "Overview":
    from pages import overview

    overview.render()
elif page == "Momentum":
    from pages import momentum

    momentum.render()
elif page == "Priority":
    from pages import priority

    priority.render()
elif page == "Clustering":
    from pages import clustering

    clustering.render()
elif page == "Strategy":
    from pages import strategy

    strategy.render()