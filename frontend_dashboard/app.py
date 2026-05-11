"""
app.py - Main entry point
Run with: streamlit run app.py
"""
import base64
import sys
from pathlib import Path
import pandas as pd
from data.loader import load_cluster_assignments, load_priority, load_momentum, load_momentum_segments, SEGMENT_COLORS
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

priority_df = load_priority()
momentum_df  = load_momentum()
cluster_df = load_cluster_assignments()
logo_b64  = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")

if "selected_restaurant" not in st.session_state:
    st.session_state["selected_restaurant"] = "All"

if "selected_segment" not in st.session_state:
    st.session_state["selected_segment"] = "All"

if "selected_cluster" not in st.session_state:
    st.session_state["selected_cluster"] = "All"

PAGE_OPTIONS = ["Guide", "Overview", "Clustering", "Strategy"]
if st.session_state.get("nav_page") not in PAGE_OPTIONS:
    st.session_state.pop("nav_page", None)


def sync_segment_filter():
    st.session_state["selected_segment"] = st.session_state.get("navbar_segment", "All")


def sync_cluster_filter():
    st.session_state["selected_cluster"] = st.session_state.get("navbar_cluster", "All")


def sync_restaurant_filter():
    st.session_state["selected_restaurant"] = st.session_state.get("navbar_restaurant", "All")


def clear_navbar_filters():
    st.session_state["selected_segment"] = "All"
    st.session_state["selected_cluster"] = "All"
    st.session_state["selected_restaurant"] = "All"
    st.session_state["navbar_segment"] = "All"
    st.session_state["navbar_cluster"] = "All"
    st.session_state["navbar_restaurant"] = "All"
    st.session_state["selected_strategy_family"] = "All"
    st.session_state["selected_ga_campaign_type"] = "All"


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
    padding: 1.5rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #0f172a 0%, #121c34 48%, #0b1220 100%);
    color: #fff;
    margin-bottom: 1.5rem;
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
    display: flex !important;
    justify-content: space-between;
    background: linear-gradient(135deg, #0f172a 0%, #121c34 48%, #0b1220 100%);
    border: 1px solid #22304b;
    border-radius: 16px;
    box-shadow: 0 12px 24px rgba(17, 24, 39, 0.18);
    padding: 0.5rem;
    width: 100%;
    box-sizing: border-box;
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

st.markdown(
    f"""
    <div class="header-shell">
        <div class="brand-panel">
            <div class="brand-tile">
                <img src="data:image/png;base64,{logo_b64}" alt="HungryHub logo" />
            </div>
            <div class="brand-copy">
                <h2>Identify. <span style="color:#cc4c33;">Prioritise.</span> Grow.</h2>
                <p>Monitor restaurant momentum, priority, cluster themes, and Google Ads efficiency in one operating view.</p>
            </div>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)


col1, col2, col3, col4 = st.columns([1, 1, 1.2, 0.55])
current_page = st.session_state.get("nav_page", "Guide")
restaurant_filter_disabled = current_page == "Overview"
if restaurant_filter_disabled:
    st.session_state["selected_restaurant"] = "All"
    st.session_state["navbar_restaurant"] = "All"

segments_available = sorted(momentum_df["latest_segment"].dropna().unique()) if "latest_segment" in momentum_df.columns else []
if cluster_df.empty or "cluster_id" not in cluster_df.columns:
    cluster_options = ["All"]
    cluster_label_map = {}
else:
    cluster_rows = (
        cluster_df[["cluster_id", "cluster_label"]]
        .drop_duplicates()
        .assign(cluster_id=lambda df: pd.to_numeric(df["cluster_id"], errors="coerce"))
        .dropna(subset=["cluster_id"])
        .sort_values(["cluster_id", "cluster_label"])
    )
    cluster_options = ["All"] + cluster_rows["cluster_id"].astype(int).tolist()
    cluster_label_map = dict(zip(cluster_rows["cluster_id"].astype(int), cluster_rows["cluster_label"]))

segment_options = ["All"] + (segments_available if segments_available else list(SEGMENT_COLORS.keys()))
if st.session_state["selected_segment"] not in segment_options:
    st.session_state["selected_segment"] = "All"
st.session_state["navbar_segment"] = st.session_state["selected_segment"]

with col1:
    st.selectbox(
        "Segment",
        segment_options,
        key="navbar_segment",
        on_change=sync_segment_filter,
    )

if st.session_state["selected_cluster"] not in cluster_options:
    st.session_state["selected_cluster"] = "All"
st.session_state["navbar_cluster"] = st.session_state["selected_cluster"]

with col2:
    st.selectbox(
        "Cluster",
        cluster_options,
        format_func=lambda cid: (
            "All" if cid == "All" else f"Cluster {cid}: {cluster_label_map.get(cid, 'Unknown')}"
        ),
        key="navbar_cluster",
        on_change=sync_cluster_filter,
    )

# restaurant_scope = momentum_df.copy()
# if (
#     st.session_state["selected_segment"] != "All"
#     or st.session_state["selected_cluster"] != "All"
# ) and not cluster_df.empty:
#     restaurant_scope = cluster_df.copy()
#     if st.session_state["selected_segment"] != "All" and "latest_segment" in restaurant_scope.columns:
#         restaurant_scope = restaurant_scope[
#             restaurant_scope["latest_segment"].astype(str).eq(st.session_state["selected_segment"])
#         ].copy()
#     if st.session_state["selected_cluster"] != "All" and "cluster_id" in restaurant_scope.columns:
#         cluster_value = pd.to_numeric(pd.Series([st.session_state["selected_cluster"]]), errors="coerce").iloc[0]
#         if pd.notna(cluster_value):
#             restaurant_scope = restaurant_scope[
#                 pd.to_numeric(restaurant_scope["cluster_id"], errors="coerce").eq(int(cluster_value))
#             ].copy()

# all_names = ["All"] + sorted(restaurant_scope["name"].dropna().unique())
# if st.session_state["selected_restaurant"] not in all_names:
#     st.session_state["selected_restaurant"] = "All"
# st.session_state["navbar_restaurant"] = st.session_state["selected_restaurant"]

restaurant_scope = cluster_df.copy()

# Keep only restaurants with a valid cluster
if not restaurant_scope.empty:
    if "cluster_id" in restaurant_scope.columns:
        restaurant_scope = restaurant_scope[
            pd.to_numeric(restaurant_scope["cluster_id"], errors="coerce").notna()
        ].copy()

    if "cluster_label" in restaurant_scope.columns:
        restaurant_scope = restaurant_scope[
            ~restaurant_scope["cluster_label"]
            .astype(str)
            .str.contains("Unclustered", case=False, na=False)
        ].copy()

# Apply navbar segment filter
if (
    st.session_state["selected_segment"] != "All"
    and "latest_segment" in restaurant_scope.columns
):
    restaurant_scope = restaurant_scope[
        restaurant_scope["latest_segment"]
        .astype(str)
        .eq(st.session_state["selected_segment"])
    ].copy()

# Apply navbar cluster filter
if (
    st.session_state["selected_cluster"] != "All"
    and "cluster_id" in restaurant_scope.columns
):
    cluster_value = pd.to_numeric(
        pd.Series([st.session_state["selected_cluster"]]),
        errors="coerce"
    ).iloc[0]

    if pd.notna(cluster_value):
        restaurant_scope = restaurant_scope[
            pd.to_numeric(restaurant_scope["cluster_id"], errors="coerce")
            .eq(int(cluster_value))
        ].copy()

all_names = ["All"] + sorted(restaurant_scope["name"].dropna().unique())

with col3:
    st.selectbox(
        "Restaurant",
        all_names,
        key="navbar_restaurant",
        on_change=sync_restaurant_filter,
        disabled=restaurant_filter_disabled,
        help="Restaurant filtering is disabled on the Overview page so the priority ranking stays broad."
        if restaurant_filter_disabled else None,
    )

with col4:
    st.markdown("<div style='height:1.72rem;'></div>", unsafe_allow_html=True)
    st.button("Clear filters", on_click=clear_navbar_filters, width="stretch")


page = st.radio(
    "Navigation",
    PAGE_OPTIONS,
    key="nav_page",
    label_visibility="collapsed",
    horizontal=True,
)

st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

if page == "Guide":
    from pages import overview

    overview.render()
# elif page == "Momentum":
#     from pages import momentum

    # momentum.render()
elif page == "Overview":
    from pages import priority

    priority.render()
elif page == "Clustering":
    from pages import clustering

    clustering.render()
elif page == "Strategy":
    from pages import strategy

    strategy.render()
