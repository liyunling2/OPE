"""
app.py — Main entry point
Run with: streamlit run app.py
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

st.set_page_config(
    page_title="HungryHub Performance Dashboard",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="collapsed", # Ensures no sidebar is shown
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

/* Hide Hamburger Menu and default Streamlit header */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* HungryHub Light Theme Variables */
:root {
    --bg:        #f4f6f9; 
    --surface:   #ffffff; 
    --surface2:  #f8f9fa;
    --border:    #e0e0e0;
    --accent:    #cc0000; /* HungryHub Red */
    --text:      #111827; /* Dark text */
    --muted:     #6b7280;
    --green:     #10b981;
    --blue:      #3b82f6;
    --orange:    #f59e0b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* Metric cards styling to look like dashboard widgets */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
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
[data-testid="stMetricDelta"] svg { display: none; }

/* Custom Horizontal Top Navigation */
div.stRadio > div[role="radiogroup"] {
    display: flex;
    flex-direction: row;
    gap: 2.5rem;
    padding: 0.5rem 0;
}
div.stRadio > div[role="radiogroup"] > label {
    background: transparent !important;
    padding: 0.5rem 0;
    border-bottom: 3px solid transparent;
    border-radius: 0;
}
div.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
    border-bottom: 3px solid var(--accent);
}
div.stRadio > div[role="radiogroup"] > label[data-checked="true"] p {
    color: var(--accent) !important;
    font-weight: 700;
}
div.stRadio p { 
    font-size: 1.1rem; 
    color: var(--text); 
    font-weight: 500;
}

/* Hide Sidebar completely if triggered */
[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Custom Top Branding ───────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:baseline; gap: 1rem; margin-bottom: 0.5rem; padding-top: 1rem;'>
    <div style='font-family: "DM Sans", sans-serif; font-size: 1.8rem; font-weight: 700; color: #cc0000;'>
        <span style="letter-spacing:-2px;">|H|</span> HungryHub
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation as a Horizontal Top Bar
page = st.radio(
    "Navigation",
    ["Overview", "Momentum", "Priority", "Strategy"],
    label_visibility="collapsed",
    horizontal=True
)

st.markdown("<hr style='margin-top: 0; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)

# ── Routing ───────────────────────────────────────────────────────────────────
# Adjust imports to match your folder structure if they sit in a pages/ directory
if page == "Overview":
    from pages import overview
    overview.render()
elif page == "Momentum":
    from pages import momentum
    momentum.render()
elif page == "Priority":
    from pages import priority
    priority.render()
elif page == "Strategy":
    from pages import strategy
    strategy.render()