# -*- coding: utf-8 -*-
"""
pages/clustering.py
Cluster exploration dashboard with cross-highlighting and strategy effectiveness ranking.
"""

from __future__ import annotations

import html, re, hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from peer_recommender.build_recommendation_prompt import build_recommendation_prompt
from peer_recommender.generate_ai_playbook import generate_ai_playbook
from data.loader import (
    get_cluster_strategy_recommendations,
    load_cluster_assignments,
    load_cluster_ga_campaign_effectiveness,
    load_ga_campaign_outreach_raw,
    load_ga_campaign_type_monthly,
    load_cluster_keywords,
    load_raw_gmv_view_monthly,
    load_cluster_strategy_outcomes,
    load_cluster_strategy_rankings,
    load_cluster_text_corpus,
    load_priority_list,
)
from theme import BASE_LAYOUT, CHART_THEME, MUTED_TEXT

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

def _normalize_name(value: str) -> str:  # Match restaurant names for selection syncing
    return re.sub(r"\s+", " ", str(value)).strip().lower()

def _fmt_pct(value: float | int | None) -> str: # Format as % with handling for None/NaN
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{value:.1%}"

# Format as THB with commas and no decimals, handling None/NaN
def _fmt_thb(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"THB {value:,.0f}"

# Format year-month with handling for None/NaN and invalid formats
def _fmt_year_month(value) -> str:
    if value is None or pd.isna(value):
        return "Unknown month"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%Y-%m")

# Format decimal with specified digits, handling None/NaN
def _fmt_decimal(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{value:,.{digits}f}"

def _fmt_roi(v):
    if pd.isna(v): return "-"
    return f"{v:.1f}x"

def _fmt_lift(v):
    if pd.isna(v): return "-"
    return f"+{v:.2f}/day"

def _fmt_reliability(v):
    if pd.isna(v): return "-"
    # lift_reliability is typically a ratio: n_positive_lift / n_campaigns
    pct = v * 100 if v <= 1 else v
    color = "🟢" if pct >= 70 else "🟡" if pct >= 40 else "🔴"
    return f"{color} {pct:.0f}%"

# Summarise into concise string with max unique items and "more" indicator, handling empty/invalid values
def _summarize_campaign_types(values: pd.Series, max_items: int = 3) -> str:
    unique_types = []
    for value in values.fillna("Unknown").astype(str).str.strip():
        if not value or value in {"(not set)", "Unknown"} or value in unique_types:
            continue
        unique_types.append(value)
    if not unique_types:
        return "-"
    if len(unique_types) <= max_items:
        return ", ".join(unique_types)
    return ", ".join(unique_types[:max_items]) + f" +{len(unique_types) - max_items} more"

# Build campaign type context by month with primary type and summary of all types, handling various edge cases in the data
def _build_campaign_type_context(campaign_type_monthly: pd.DataFrame) -> pd.DataFrame:
    if campaign_type_monthly.empty: # Handle empty input with correct columns
        return pd.DataFrame(columns=["year_month", "primary_ga_campaign_type", "ga_campaign_types"])

    required_cols = {"year_month", "googleAdsCampaignType"}
    if not required_cols.issubset(campaign_type_monthly.columns): # Handle missing required columns
        return pd.DataFrame(columns=["year_month", "primary_ga_campaign_type", "ga_campaign_types"])

    context = campaign_type_monthly.copy() # Create columns 'year_month', 'googleAdsCampaignType', and optionally 'total_sessions' and 'active_campaigns'
    context["year_month"] = pd.to_datetime(context["year_month"], errors="coerce")
    context["total_sessions"] = ( # Use total_sessions for sorting if available
        pd.to_numeric(context["total_sessions"], errors="coerce").fillna(0)
        if "total_sessions" in context.columns
        else 0 # Default to zero if total_sessions column is missing
    )
    context["active_campaigns"] = (
        pd.to_numeric(context["active_campaigns"], errors="coerce").fillna(0)
        if "active_campaigns" in context.columns
        else 0 # Default to zero if active_campaigns column is missing
    )
    context = context.dropna(subset=["year_month"]) # Drop rows with invalid year_month after coercion
    if context.empty:
        return pd.DataFrame(columns=["year_month", "primary_ga_campaign_type", "ga_campaign_types"])
    # Clean googleAdsCampaignType by filling NaN, stripping whitespace, and excluding empty or "(not set)" values
    context["googleAdsCampaignType"] = context["googleAdsCampaignType"].fillna("Unknown").astype(str).str.strip()
    context = context[
        context["googleAdsCampaignType"].ne("") # Exclude empty strings
        & context["googleAdsCampaignType"].ne("(not set)")
    ]
    if context.empty:
        return pd.DataFrame(columns=["year_month", "primary_ga_campaign_type", "ga_campaign_types"])

    context = context.sort_values(
        ["year_month", "total_sessions", "active_campaigns", "googleAdsCampaignType"],
        ascending=[True, False, False, True],
    )
    return (
        context.groupby("year_month", as_index=False)
        .agg(
            primary_ga_campaign_type=("googleAdsCampaignType", "first"),
            ga_campaign_types=("googleAdsCampaignType", lambda s: _summarize_campaign_types(s)),
        )
        .reset_index(drop=True)
    )

# Extract from plot selection state, handle both dict and Plotly event object formats,
# Purpose is sync selection between scatter plot & restaurant list
def _extract_selected_point(state) -> tuple[str | None, int | None]:
    if state is None:
        return None, None

    selection = None
    if isinstance(state, dict):
        selection = state.get("selection")
    else:
        selection = getattr(state, "selection", None)

    if not selection:
        return None, None
    # selection plots can have multiple points but we only consider the first one for syncing selection
    points = selection.get("points", []) if isinstance(selection, dict) else []
    if not points:
        return None, None

    point = points[0]
    custom = point.get("customdata", [])
    if isinstance(custom, (list, tuple)) and len(custom) >= 2: # customdata: [restaurant_name, cluster_id, ...]
        try:
            return str(custom[0]), int(custom[1])
        except Exception:
            return str(custom[0]), None
    return None, None

# Extract selected row indices from table selection state, handle both dict and Streamlit data frame state formats
def _extract_selected_rows(state) -> list[int]:
    if state is None:
        return []
    selection = None
    if isinstance(state, dict):
        selection = state.get("selection")
    else:
        selection = getattr(state, "selection", None)

    if not selection:
        return []

    rows = selection.get("rows", []) if isinstance(selection, dict) else []
    return rows if isinstance(rows, list) else []

# Prep GA campaign effectiveness data for display, including renaming, formatting, and optional inclusion of restaurant name column
def _prepare_raw_ga_display(df: pd.DataFrame, include_restaurant: bool = False) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    display_cols = [ # Select columns to rename
        c
        for c in [
            "name",
            "cluster_label",
            "year_month",
            "primary_ga_campaign_type",
            "ga_campaign_types",
            "monthly_gmv",
            "monthly_bookings",
            "ga_items_viewed",
            "ga_items_added_to_cart",
            "ga_items_purchased",
            "ga_item_revenue",
            "gmv_per_ga_view",
            "bookings_per_ga_view",
            "ga_add_to_cart_rate",
            "ga_view_to_purchase_rate",
            "ga_purchase_to_cart_rate",
            "ga_revenue_per_view",
        ]
        if c in df.columns
    ]
    out = df[display_cols].copy().rename(
        columns={
            "name": "Restaurant",
            "cluster_label": "Cluster",
            "year_month": "Year Month",
            "primary_ga_campaign_type": "Primary GA Campaign Type",
            "ga_campaign_types": "GA Campaign Types",
            "monthly_gmv": "Monthly GMV",
            "monthly_bookings": "Monthly Bookings",
            "ga_items_viewed": "Items Viewed",
            "ga_items_added_to_cart": "Items Added to Cart",
            "ga_items_purchased": "Items Purchased",
            "ga_item_revenue": "GA Item Revenue",
            "gmv_per_ga_view": "GMV / GA View",
            "bookings_per_ga_view": "Bookings / GA View",
            "ga_add_to_cart_rate": "Add to Cart Rate",
            "ga_view_to_purchase_rate": "View to Purchase Rate",
            "ga_purchase_to_cart_rate": "Purchase to Cart Rate",
            "ga_revenue_per_view": "GA Revenue / View",
        }
    )
    if "Year Month" in out.columns:
        out["Year Month"] = pd.to_datetime(out["Year Month"], errors="coerce").dt.strftime("%Y-%m")
    if not include_restaurant and "Restaurant" in out.columns:
        out = out.drop(columns=["Restaurant"])
    for col in ["Monthly GMV", "GA Item Revenue", "GMV / GA View", "GA Revenue / View"]:
        if col in out.columns:
            out[col] = out[col].apply(_fmt_thb)
    if "Bookings / GA View" in out.columns:
        out["Bookings / GA View"] = pd.to_numeric(out["Bookings / GA View"], errors="coerce").round(3)
    for col in ["Monthly Bookings", "Items Viewed", "Items Added to Cart", "Items Purchased"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(0)
    for col in ["Add to Cart Rate", "View to Purchase Rate", "Purchase to Cart Rate"]:
        if col in out.columns:
            out[col] = out[col].apply(_fmt_pct)
    return out 

# Prep campaign breakdown data for display, including renaming and formatting
def _prepare_campaign_breakdown_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    display_cols = [
        c
        for c in [
            "year_month",
            "googleAdsCampaignType",
            "campaign_name",
            "campaign_id",
            "sessions",
            "session_share",
        ]
        if c in df.columns
    ]
    out = df[display_cols].copy().rename(
        columns={
            "year_month": "Year Month",
            "googleAdsCampaignType": "GA Campaign Type",
            "campaign_name": "Campaign Name",
            "campaign_id": "Campaign ID",
            "sessions": "Sessions",
            "session_share": "Session Share",
        }
    )
    if "Year Month" in out.columns:
        out["Year Month"] = pd.to_datetime(out["Year Month"], errors="coerce").dt.strftime("%Y-%m")
    if "Sessions" in out.columns:
        out["Sessions"] = pd.to_numeric(out["Sessions"], errors="coerce").round(0)
    if "Session Share" in out.columns:
        out["Session Share"] = out["Session Share"].apply(_fmt_pct)
    return out


def _prepare_strategy_campaign_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    display_cols = [
        c
        for c in [
            "applied_date",
            "campaign_name",
            "strategy_family",
            "strategy_name",
            "channel",
            "restaurant_name",
            "cluster_label",
            "latest_segment",
            "bookings_before",
            "bookings_after",
            "bookings_uplift_pct",
            "incremental_revenue_thb",
            "roi",
            "activity_id",
        ]
        if c in df.columns
    ]
    out = df[display_cols].copy().rename(
        columns={
            "applied_date": "Applied Date",
            "campaign_name": "Campaign Name",
            "strategy_family": "Assigned Strategy",
            "strategy_name": "Strategy Label",
            "channel": "Channel",
            "restaurant_name": "Restaurant",
            "cluster_label": "Cluster",
            "latest_segment": "Momentum Category",
            "bookings_before": "Bookings Before",
            "bookings_after": "Bookings After",
            "bookings_uplift_pct": "Bookings Uplift %",
            "incremental_revenue_thb": "Incremental Revenue (THB)",
            "roi": "ROI",
            "activity_id": "Activity ID",
        }
    )
    if "Applied Date" in out.columns:
        out["Applied Date"] = pd.to_datetime(out["Applied Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["Bookings Before", "Bookings After"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(0)
    if "Bookings Uplift %" in out.columns:
        out["Bookings Uplift %"] = out["Bookings Uplift %"].apply(_fmt_pct)
    if "Incremental Revenue (THB)" in out.columns:
        out["Incremental Revenue (THB)"] = out["Incremental Revenue (THB)"].apply(_fmt_thb)
    if "ROI" in out.columns:
        out["ROI"] = pd.to_numeric(out["ROI"], errors="coerce").apply(_fmt_roi)
    return out

# Tokenize text into words, filter out short tokens and common stopwords, and return list of meaningful tokens for theme extraction
def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", str(text).lower())
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "was",
        "are",
        "but",
        "have",
        "had",
        "not",
        "very",
        "from",
        "they",
        "you",
        "our",
        "their",
        "all",
        "just",
        "food",
        "restaurant",
        "good",
        "great",
        "nice",
    }
    return [token for token in tokens if token not in stopwords]

# Extract top N terms from a series of text, counting frequency of meaningful tokens 
# and return a DataFrame of keywords and their weights for word cloud visualization
def _terms_from_texts(text_series: pd.Series, top_n: int = 40) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for text in text_series.fillna(""):
        for token in _tokenize(text):
            counts[token] = counts.get(token, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["keyword", "weight"])

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return pd.DataFrame(items, columns=["keyword", "weight"])

# Build a word cloud figure using Plotly, where word size corresponds to weight and highlighted terms 
# are colored differently, with handling for empty data and customization options
def _build_word_cloud_figure(
    word_df: pd.DataFrame,
    title: str,
    highlight_terms: set[str] | None = None,
    default_color: str = "#3b82f6",
) -> go.Figure:
    fig = go.Figure()

    if word_df.empty:
        fig.update_layout(
            **BASE_LAYOUT,
            height=320,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(text="No text data available", showarrow=False, x=0.5, y=0.5, font=dict(color="#6b7280"))],
            title=title,
        )
        return fig
    # Ensure keywords are strings for display, weights are numeric for sizing
    words = word_df["keyword"].astype(str).tolist()
    weights = pd.to_numeric(word_df["weight"], errors="coerce").fillna(0).to_numpy()

    if np.max(weights) > np.min(weights):
        sizes = 16 + (weights - np.min(weights)) * (44 / (np.max(weights) - np.min(weights)))
    else:
        sizes = np.full(len(weights), 24)

    n_terms = len(words) # Generate positions in spiral pattern to distribute words, with some random noise for better aesthetics
    angles = np.linspace(0.0, 5 * np.pi, n_terms) # More turns for better distribution of larger number of words
    radius = np.linspace(0.15, 1.0, n_terms) # Start with small radius for first word and increase to max radius for last word to create spiral effect
    rng = np.random.default_rng(42) # Fixed seed for reproducibility of random noise
    x = radius * np.cos(angles) + rng.normal(0, 0.08, n_terms) # Add small random noise to x positions to avoid perfect alignment and create more natural word cloud appearance
    y = radius * np.sin(angles) + rng.normal(0, 0.08, n_terms) # y positions to avoid perfect alignment and create more natural word cloud appearance

    highlights = highlight_terms or set() # Highlight terms from the restaurant's own text in a different color to show overlap with cluster themes
    colors = ["#cc0000" if word in highlights else default_color for word in words] # Color highlighted terms in red and others in default color (blue) to visually distinguish them in the word cloud

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="text",
            text=words,
            textfont=dict(size=sizes, color=colors, family="DM Sans"),
            hovertemplate="<b>%{text}</b><br>weight: %{customdata}<extra></extra>",
            customdata=[int(v) for v in weights],
        )
    )

    fig.update_layout(
        **BASE_LAYOUT,
        title=title,
        height=320,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig

# Highlight occurrences of query in the text by wrapping them in <mark> tags with styling, 
# and escape HTML to prevent injection, also handle newlines for proper display in Streamlit
def _highlight_text(value: str, query: str) -> str:
    escaped = html.escape(str(value))
    if not query.strip():
        return escaped.replace("\n", "<br>")
    # Find case-insensitive matches of the query in the text, wrap them in styled <mark> tags for highlighting, 
    # while preserving the original text and escaping HTML to prevent injection
    pattern = re.compile(re.escape(query.strip()), flags=re.IGNORECASE)
    parts: list[str] = []
    last_end = 0
    for match in pattern.finditer(str(value)):
        parts.append(html.escape(str(value)[last_end:match.start()]))
        parts.append(
            "<mark style='background:#fbbf24; color:#111827; padding:0 0.2rem; border-radius:0.2rem;'>"
            f"{html.escape(match.group(0))}"
            "</mark>"
        )
        last_end = match.end()
    parts.append(html.escape(str(value)[last_end:]))
    return "".join(parts).replace("\n", "<br>")

def _render_raw_text_records(text_df: pd.DataFrame, search_text: str, max_rows: int) -> None:
    if text_df.empty:
        st.info("No raw text rows for this restaurant with the current filter.")
        return

    display_df = text_df.copy()
    display_df["_sort_month"] = pd.to_datetime(display_df.get("year_month"), errors="coerce")
    sort_cols = [c for c in ["_sort_month", "text_id"] if c in display_df.columns]
    if sort_cols:
        ascending = [False] * len(sort_cols)
        display_df = display_df.sort_values(sort_cols, ascending=ascending)
    display_df = display_df.head(int(max_rows))

    for _, row in display_df.iterrows(): # Render each text record as a styled card with source and date
        text_id = row.get("text_id", "-")
        year_month = _fmt_year_month(row.get("year_month"))
        text_source = str(row.get("text_source", "Review")).strip() or "Review"
        raw_text = row.get("raw_text", "")
        if pd.isna(raw_text) or str(raw_text).strip() == "":
            raw_text = "(empty text)"
        source_bg = {
            "Review": "#172554",
            "Google": "#14532d",
            "KOL": "#7c2d12",
            "Combined": "#3f3f46",
        }.get(text_source, "#334155")
        card_html = f"""
        <div style="
            background: linear-gradient(180deg, #161b26 0%, #10151f 100%);
            border: 1px solid #2a3344;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
        ">
            <div style="
                display:flex;
                justify-content:space-between;
                gap:1rem;
                margin-bottom:0.75rem;
                flex-wrap:wrap;
                align-items:center;
            ">
                <div style="display:flex; align-items:center; gap:0.55rem; flex-wrap:wrap;">
                    <div style="font-size:0.85rem; font-weight:700; color:#f9fafb;">Text ID {html.escape(str(text_id))}</div>
                    <span style="
                        display:inline-flex;
                        align-items:center;
                        padding:0.18rem 0.55rem;
                        border-radius:999px;
                        background:{source_bg};
                        color:#e5e7eb;
                        font-size:0.72rem;
                        font-weight:700;
                        letter-spacing:0.03em;
                    ">{html.escape(text_source)}</span>
                </div>
                <div style="font-size:0.8rem; color:#cbd5e1;">{html.escape(year_month)}</div>
            </div>
            <div style="
                color:#e5e7eb;
                font-size:0.98rem;
                line-height:1.7;
                white-space:normal;
                word-break:break-word;
            ">{_highlight_text(str(raw_text), search_text)}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)


def render():
    assignments = load_cluster_assignments() # Load from clustering_results
    ga_effectiveness_df = load_cluster_ga_campaign_effectiveness()
    ga_campaign_raw = load_ga_campaign_outreach_raw()
    ga_campaign_type_monthly = load_ga_campaign_type_monthly()
    ga_restaurant_monthly = load_raw_gmv_view_monthly()
    campaign_type_context = _build_campaign_type_context(ga_campaign_type_monthly)
    text_corpus = load_cluster_text_corpus()
    keyword_df = load_cluster_keywords()
    outcomes_df = load_cluster_strategy_outcomes()

    st.markdown("## Clustering Explorer")
    st.markdown(
        f"<p style='color:{MUTED_TEXT}; margin-top:-0.5rem;'>"
        "Explore restaurant clusters, inspect text themes, and benchmark strategy effectiveness by cluster."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if assignments.empty:
        st.warning("No clustering data found. Run clustering.ipynb to generate clustering outputs.")
        return

    assignments = assignments.copy()
    assignments["cluster_id"] = pd.to_numeric(assignments["cluster_id"], errors="coerce").astype(int)
    if "name_norm" not in assignments.columns:
        assignments["name_norm"] = assignments["name"].apply(_normalize_name)

    all_clusters_option = "All Clusters"
    cluster_rows = assignments[["cluster_id", "cluster_label"]].drop_duplicates().sort_values(["cluster_id", "cluster_label"])
    cluster_options = cluster_rows["cluster_id"].tolist() # Get list of unique cluster IDs for selection options, sorted by cluster_id
    cluster_options_ui = [all_clusters_option] + cluster_options # UI options include "All Clusters" plus individual cluster IDs, with mapping to labels for display
    cluster_label_map = dict(zip(cluster_rows["cluster_id"], cluster_rows["cluster_label"])) # Map cluster_id to cluster_label for display in selection box, handle missing labels

    if not cluster_options: # No valid clusters after processing, show warning and exit
        st.warning("No valid clusters available in clustering data.")
        return

    if "cluster_active_cluster" not in st.session_state: # Initialize active cluster selection in session state, default to "All Clusters"
        st.session_state["cluster_active_cluster"] = all_clusters_option

    if st.session_state["cluster_active_cluster"] not in cluster_options_ui:
        st.session_state["cluster_active_cluster"] = all_clusters_option

    def _filter_by_selected_cluster(df: pd.DataFrame) -> pd.DataFrame:
        active_value = st.session_state["cluster_active_cluster"]
        if active_value == all_clusters_option: # If "All Clusters" is selected, return the full DataFrame without filtering, otherwise filter by the selected cluster_id
            return df.copy()
        return df[df["cluster_id"] == int(active_value)].copy() 
# Filter DF to only include rows where cluster_id matches the selected cluster, handle potential issues with non-numeric cluster_id
    cluster_df = _filter_by_selected_cluster(assignments).sort_values("name")

    if "cluster_active_restaurant" not in st.session_state:
        st.session_state["cluster_active_restaurant"] = cluster_df.iloc[0]["name"] if len(cluster_df) else assignments.iloc[0]["name"]

    if _normalize_name(st.session_state["cluster_active_restaurant"]) not in set(cluster_df["name_norm"]):
        st.session_state["cluster_active_restaurant"] = cluster_df.iloc[0]["name"] if len(cluster_df) else assignments.iloc[0]["name"]

    control_a, control_b, control_c = st.columns([1.5, 2.5, 1])

    with control_a: # Layout for cluster selection
        active_cluster = st.selectbox(
            "Cluster",
            options=cluster_options_ui,
            index=cluster_options_ui.index(st.session_state["cluster_active_cluster"]),
            format_func=lambda cid: ( # Display "All Clusters" for the all_clusters_option
                all_clusters_option if cid == all_clusters_option else f"Cluster {cid}: {cluster_label_map.get(cid, 'Unknown')}"
            ),
            key="cluster_selector",
        )
        if active_cluster != st.session_state["cluster_active_cluster"]:
            st.session_state["cluster_active_cluster"] = active_cluster
            new_cluster_df = _filter_by_selected_cluster(assignments).sort_values("name")
            if len(new_cluster_df):
                st.session_state["cluster_active_restaurant"] = new_cluster_df.iloc[0]["name"]

    cluster_df = _filter_by_selected_cluster(assignments).sort_values("name")

    with control_b: # Layout for restaurant selection within selected cluster
        restaurant_options = cluster_df["name"].tolist()
        default_restaurant = st.session_state["cluster_active_restaurant"]
        if default_restaurant not in restaurant_options and restaurant_options:
            default_restaurant = restaurant_options[0]
            st.session_state["cluster_active_restaurant"] = default_restaurant

        active_restaurant = st.selectbox( # Select restaurant from currently selected cluster
            "Restaurant (within selected scope)",
            options=restaurant_options,
            index=restaurant_options.index(default_restaurant) if restaurant_options else 0,
            key="cluster_restaurant_selector",
        )
        if active_restaurant != st.session_state["cluster_active_restaurant"]:
            st.session_state["cluster_active_restaurant"] = active_restaurant
        # Display its cluster label and ID for reference, handle case where restaurant might not be found in the current cluster_df due to data issues
        selected_lookup = assignments[assignments["name"].eq(st.session_state["cluster_active_restaurant"])].head(1)
        if len(selected_lookup):
            selected_cluster_label = selected_lookup.iloc[0].get("cluster_label", "Unknown")
            selected_cluster_id = selected_lookup.iloc[0].get("cluster_id", "Unknown")
            st.caption(f"Selected restaurant cluster: {selected_cluster_label} (ID: {selected_cluster_id})")

    with control_c: # Layout for minimum sample size input for strategy ranking, with help text
        min_sample_size = st.number_input(
            "Min sample",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            help="Minimum number of activities for a strategy to be eligible in rankings.",
        )

    active_cluster = st.session_state["cluster_active_cluster"]
    active_restaurant = st.session_state["cluster_active_restaurant"]
    active_rest_norm = _normalize_name(active_restaurant)
    scope_label = all_clusters_option if active_cluster == all_clusters_option else f"Cluster {active_cluster}"
    scope_gmv_per_ga_view = ( # Avg gmv per view across cluster
        pd.to_numeric(cluster_df["gmv_per_ga_view"], errors="coerce").mean()
        if "gmv_per_ga_view" in cluster_df.columns
        else np.nan
    )
    scope_ga_add_to_cart = ( # Avg add to cart rate across cluster
        pd.to_numeric(cluster_df["ga_add_to_cart_rate"], errors="coerce").mean()
        if "ga_add_to_cart_rate" in cluster_df.columns
        else np.nan
    )
    # Key metrics for selected cluster scope
    metric_a, metric_b, metric_c, metric_d, metric_e, metric_f = st.columns(6)
    metric_a.metric("Restaurants in Scope", f"{len(cluster_df):,}")
    metric_b.metric("Total Scope Bookings", f"{cluster_df['monthly_bookings'].fillna(0).sum():,.0f}")
    metric_c.metric("Avg Scope Revenue", _fmt_thb(cluster_df["monthly_gmv"].mean()))
    metric_d.metric("Scope GMV / GA View", _fmt_thb(scope_gmv_per_ga_view))
    metric_e.metric("Scope GA Add to Cart", _fmt_pct(scope_ga_add_to_cart))
    metric_f.metric("Known Momentum Segment", f"{cluster_df['latest_segment'].notna().sum():,}")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.7, 1.3])

    with left: # cluster visuals, scatter plot of UMAP components colored by cluster label, with hover info and selection syncing to restaurant list, handle case where UMAP components might be missing or non-numeric
        st.markdown("### Cluster Map")
        scatter_df = assignments.copy()

        fig = px.scatter(
            scatter_df,
            x="x",
            y="y",
            color="cluster_label",
            hover_name="name",
            size=scatter_df["monthly_bookings"].fillna(5).clip(lower=5, upper=250),
            size_max=22,
            opacity=0.72,
            custom_data=["name", "cluster_id", "latest_segment"],
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

        fig.update_traces(
            marker=dict(line=dict(width=1, color="#ffffff")),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Cluster: %{customdata[1]}<br>"
                "Momentum: %{customdata[2]}<br>"
                "x: %{x:.2f}, y: %{y:.2f}<extra></extra>"
            ),
        )

        selected_row = scatter_df[scatter_df["name_norm"] == active_rest_norm]
        if len(selected_row):
            fig.add_trace(
                go.Scatter(
                    x=selected_row["x"],
                    y=selected_row["y"],
                    mode="markers+text",
                    text=["Selected"],
                    textposition="top center",
                    name="Selected Restaurant",
                    marker=dict(size=30, color="#cc0000", symbol="diamond", line=dict(width=2, color="#cc0000")),
                    customdata=np.stack(
                        [
                            selected_row["name"].to_numpy(),
                            selected_row["cluster_id"].to_numpy(),
                            selected_row["latest_segment"].fillna("-").to_numpy(),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Cluster: %{customdata[1]}<br>"
                        "Momentum: %{customdata[2]}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            **BASE_LAYOUT,
            height=450,
            xaxis=dict(**CHART_THEME["xaxis"], title="UMAP Component 1"),
            yaxis=dict(**CHART_THEME["yaxis"], title="UMAP Component 2"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )

        plot_state = st.plotly_chart(
            fig,
            width="stretch",
            key="cluster_scatter_plot",
            on_select="rerun",
            selection_mode=("points",),
        )
# Selection point data 
        point_name, point_cluster = _extract_selected_point(plot_state)
        if point_name:
            should_rerun = False
            if point_name != st.session_state["cluster_active_restaurant"]:
                st.session_state["cluster_active_restaurant"] = point_name
                should_rerun = True
            if point_cluster is not None and (
                st.session_state["cluster_active_cluster"] == all_clusters_option
                or int(point_cluster) != int(st.session_state["cluster_active_cluster"])
            ):
                st.session_state["cluster_active_cluster"] = int(point_cluster)
                should_rerun = True
            if should_rerun:
                st.rerun()

    with right: # Cluster restaurant list
        st.markdown(f"### Restaurants in {scope_label}")
        cluster_df = _filter_by_selected_cluster(assignments).sort_values(
            ["latest_segment", "name"], ascending=[True, True]
        )
        active_rest_norm = _normalize_name(st.session_state["cluster_active_restaurant"])
        cluster_df["active"] = np.where(cluster_df["name_norm"] == active_rest_norm, "Selected", "")

        display_cols = [
            "active",
            "name",
            "latest_segment",
            "monthly_bookings",
            "growth_signal_used",
        ]
        list_df = cluster_df[display_cols].rename(
            columns={
                "active": "",
                "name": "Restaurant",
                "latest_segment": "Momentum Category",
                "monthly_bookings": "Monthly Bookings",
                "growth_signal_used": "Growth Signal",
            }
        )

        table_state = st.dataframe(
            list_df,
            hide_index=True,
            width="stretch",
            height=450,
            key="cluster_restaurant_table",
            on_select="rerun",
            selection_mode="single-row",
        )

        selected_rows = _extract_selected_rows(table_state)
        if selected_rows:
            selected_idx = int(selected_rows[0])
            if 0 <= selected_idx < len(cluster_df):
                selected_restaurant = cluster_df.iloc[selected_idx]["name"]
                if selected_restaurant != st.session_state["cluster_active_restaurant"]:
                    st.session_state["cluster_active_restaurant"] = selected_restaurant
                    st.rerun()

    st.markdown("---")

    active_cluster = st.session_state["cluster_active_cluster"]
    active_restaurant = st.session_state["cluster_active_restaurant"]
    active_rest_norm = _normalize_name(active_restaurant)
    # Snapshot metrics for selected restaurant, also filter cluster-level data to selected cluster
    selected_cluster_df = _filter_by_selected_cluster(assignments).copy()
    selected_restaurant_row = assignments[assignments["name_norm"] == active_rest_norm].head(1)

    st.markdown(f"### Selected Restaurant Snapshot: {active_restaurant}")
    if selected_restaurant_row.empty:
        st.info("No restaurant-level cluster snapshot found.")
    else:
        selected_rest = selected_restaurant_row.iloc[0]
        snap_a, snap_b, snap_c, snap_d = st.columns(4)
        snap_a.metric("Cluster", str(selected_rest.get("cluster_label", "-")))
        snap_b.metric("GMV / GA View", _fmt_thb(selected_rest.get("gmv_per_ga_view")))
        snap_c.metric("GA Add to Cart", _fmt_pct(selected_rest.get("ga_add_to_cart_rate")))
        snap_d.metric("GA View to Purchase", _fmt_pct(selected_rest.get("ga_view_to_purchase_rate")))

    section_a, section_b = st.columns(2)

    with section_a:
        st.markdown("### Cluster Word Cloud")
        if active_cluster == all_clusters_option:
            cluster_keywords = (
                keyword_df.groupby("keyword", as_index=False)["weight"]
                .sum()
                .sort_values("weight", ascending=False)
            )
            cloud_title = f"{all_clusters_option} terms (red = terms from {active_restaurant})"
        else:
            cluster_keywords = keyword_df[keyword_df["cluster_id"] == active_cluster].sort_values("weight", ascending=False)
            cloud_title = f"Cluster {active_cluster} terms (red = terms from {active_restaurant})"
        if cluster_keywords.empty:
            if active_cluster == all_clusters_option:
                cluster_text = text_corpus["clean_text"]
            else: # Fall back to extracting terms directly from text corpus
                cluster_text = text_corpus[text_corpus["cluster_id"] == active_cluster]["clean_text"]
            cluster_keywords = _terms_from_texts(cluster_text, top_n=40)

        rest_text_df = text_corpus[text_corpus["name_norm"] == active_rest_norm]
        restaurant_terms = _terms_from_texts(rest_text_df["clean_text"], top_n=30)
        highlight_terms = set(restaurant_terms["keyword"].tolist())

        fig_cluster_wc = _build_word_cloud_figure( # for cluster-level keywords
            cluster_keywords[["keyword", "weight"]].head(40),
            title=cloud_title,
            highlight_terms=highlight_terms,
            default_color="#3b82f6",
        )
        st.plotly_chart(fig_cluster_wc, width="stretch")

    with section_b:
        st.markdown("### Restaurant Word Cloud")
        if rest_text_df.empty:
            st.info("No text records found for selected restaurant.")
        else:
            fig_rest_wc = _build_word_cloud_figure(
                restaurant_terms,
                title=f"{active_restaurant} terms",
                default_color="#cc0000",
            )
            st.plotly_chart(fig_rest_wc, width="stretch")

    st.markdown("### Raw Text Records")
    rest_text_df = text_corpus[text_corpus["name_norm"] == active_rest_norm].copy()
    source_options = ["All Sources"]
    if "text_source" in rest_text_df.columns:
        source_values = sorted(rest_text_df["text_source"].dropna().astype(str).unique().tolist())
        source_options.extend(source_values)

    text_control_a, text_control_b, text_control_c = st.columns([3.4, 1.4, 1.0])
    with text_control_a:
        search_text = st.text_input("Filter selected restaurant text", value="", key="cluster_text_filter")
    with text_control_b:
        source_choice = st.selectbox("Source", options=source_options, index=0, key="cluster_text_source_filter")
    with text_control_c:
        text_limit = st.selectbox(
            "Rows to show",
            options=[5, 10, 15, 20, 30],
            index=2,
            key="cluster_text_limit",
        )

    if search_text.strip():
        mask = rest_text_df["raw_text"].fillna("").str.contains(search_text.strip(), case=False, regex=False)
        rest_text_df = rest_text_df[mask]
    if source_choice != "All Sources" and "text_source" in rest_text_df.columns:
        rest_text_df = rest_text_df[rest_text_df["text_source"] == source_choice].copy()

    if rest_text_df.empty:
        st.info("No raw text rows for this restaurant with the current filter.")
    else:
        st.caption(
            f"Showing {min(len(rest_text_df), int(text_limit))} of {len(rest_text_df)} matched text records for {active_restaurant}."
        )
        _render_raw_text_records(rest_text_df, search_text=search_text, max_rows=int(text_limit))

    st.markdown("---")
    st.markdown(f"### GA Campaign Effectiveness ({scope_label})")
    if active_cluster == all_clusters_option:
        scope_ga = ga_effectiveness_df.copy()
    else:
        scope_ga = ga_effectiveness_df[ga_effectiveness_df["cluster_id"] == int(active_cluster)].copy()

    if scope_ga.empty:
        st.info("No GA campaign effectiveness data available for this scope.")
    else:
        st.caption(
            "Campaign effectiveness is inferred from month-level overlap between cluster GMV per GA view "
            "and the platform Google Ads campaign mix. Use it as directional evidence, not restaurant-level attribution."
        )

        top_ga = scope_ga.sort_values("ga_effectiveness_score", ascending=False).head(3)
        ga_cols = st.columns(min(3, len(top_ga)))
        for idx, (_, rec) in enumerate(top_ga.iterrows()):
            with ga_cols[idx]:
                st.metric(
                    f"#{idx + 1} {rec['googleAdsCampaignType']}",
                    _fmt_thb(rec.get("session_weighted_gmv_per_ga_view")),
                    f"ATC {_fmt_pct(rec.get('session_weighted_add_to_cart_rate'))}",
                )

        ga_display_cols = [
            c
            for c in [
                "cluster_label",
                "googleAdsCampaignType",
                "active_months",
                "total_sessions",
                "active_campaigns",
                "session_weighted_gmv_per_ga_view",
                "session_weighted_add_to_cart_rate",
                "session_weighted_view_to_purchase_rate",
                "sessions_to_gmv_correlation",
                "ga_effectiveness_score",
            ]
            if c in scope_ga.columns
        ]
        ga_display = scope_ga[ga_display_cols].copy().rename(
            columns={
                "cluster_label": "Cluster",
                "googleAdsCampaignType": "GA Campaign Type",
                "active_months": "Active Months",
                "total_sessions": "Sessions",
                "active_campaigns": "Campaign Months",
                "session_weighted_gmv_per_ga_view": "GMV / GA View",
                "session_weighted_add_to_cart_rate": "Add to Cart Rate",
                "session_weighted_view_to_purchase_rate": "View to Purchase Rate",
                "sessions_to_gmv_correlation": "Sessions to GMV Corr",
                "ga_effectiveness_score": "GA Effectiveness Score",
            }
        )
        if active_cluster != all_clusters_option and "Cluster" in ga_display.columns:
            ga_display = ga_display.drop(columns=["Cluster"])

        if "Sessions" in ga_display.columns:
            ga_display["Sessions"] = pd.to_numeric(ga_display["Sessions"], errors="coerce").round(0)
        if "Campaign Months" in ga_display.columns:
            ga_display["Campaign Months"] = pd.to_numeric(ga_display["Campaign Months"], errors="coerce").round(0)
        if "GMV / GA View" in ga_display.columns:
            ga_display["GMV / GA View"] = ga_display["GMV / GA View"].apply(_fmt_thb)
        for col in ["Add to Cart Rate", "View to Purchase Rate"]:
            if col in ga_display.columns:
                ga_display[col] = ga_display[col].apply(_fmt_pct)
        if "Sessions to GMV Corr" in ga_display.columns:
            ga_display["Sessions to GMV Corr"] = pd.to_numeric(ga_display["Sessions to GMV Corr"], errors="coerce").round(2)
        if "GA Effectiveness Score" in ga_display.columns:
            ga_display["GA Effectiveness Score"] = pd.to_numeric(ga_display["GA Effectiveness Score"], errors="coerce").round(2)

        st.dataframe(
            ga_display.sort_values("GA Effectiveness Score", ascending=False),
            hide_index=True,
            width="stretch",
            height=280,
        )

    st.markdown("### Raw GA Monthly View")
    st.caption(
        "Source: `_5_GA_data/data_output/gmv/gmv_view.parquet`. "
        "Use this to inspect the selected restaurant's month-level GA performance and the raw restaurant-month rows for its cluster. "
        "GA campaign type reflects the platform campaign mix for that month, not restaurant-level attribution."
    )

    raw_cluster_id = None
    raw_cluster_label = scope_label
    if not selected_restaurant_row.empty:
        raw_cluster_id_val = pd.to_numeric(
            pd.Series([selected_restaurant_row.iloc[0].get("cluster_id")]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(raw_cluster_id_val):
            raw_cluster_id = int(raw_cluster_id_val)
            raw_cluster_label = str(
                selected_restaurant_row.iloc[0].get("cluster_label", f"Cluster {raw_cluster_id}")
            )
    elif active_cluster != all_clusters_option:
        raw_cluster_id = int(active_cluster)

    cluster_ga_raw = ga_restaurant_monthly.copy()
    if not cluster_ga_raw.empty and "name_norm" in cluster_ga_raw.columns:
        cluster_ref = assignments[["name_norm", "cluster_id", "cluster_label"]].drop_duplicates("name_norm")
        cluster_ga_raw = cluster_ga_raw.merge(cluster_ref, on="name_norm", how="inner")
        if raw_cluster_id is not None:
            cluster_ga_raw = cluster_ga_raw[cluster_ga_raw["cluster_id"] == raw_cluster_id].copy()
    else:
        cluster_ga_raw = cluster_ga_raw.iloc[0:0].copy()
    if not cluster_ga_raw.empty and not campaign_type_context.empty:
        cluster_ga_raw = cluster_ga_raw.merge(campaign_type_context, on="year_month", how="left")

    selected_rest_raw = cluster_ga_raw[cluster_ga_raw["name_norm"] == active_rest_norm].copy() if not cluster_ga_raw.empty else cluster_ga_raw.copy()

    raw_tab_a, raw_tab_b = st.tabs(["Selected Restaurant", "Cluster Restaurants"])

    with raw_tab_a:
        if selected_rest_raw.empty:
            st.info("No raw gmv_view rows found for the selected restaurant.")
        else:
            selected_rest_raw = selected_rest_raw.sort_values("year_month").copy()
            chart_custom = np.column_stack(
                [
                    selected_rest_raw.get(
                        "primary_ga_campaign_type",
                        pd.Series("-", index=selected_rest_raw.index),
                    ).fillna("-").astype(str),
                    selected_rest_raw.get(
                        "ga_campaign_types",
                        pd.Series("-", index=selected_rest_raw.index),
                    ).fillna("-").astype(str),
                ]
            )
            fig_raw = go.Figure()
            fig_raw.add_trace(
                go.Bar(
                    x=selected_rest_raw["year_month"],
                    y=selected_rest_raw["monthly_bookings"],
                    name="Monthly Bookings",
                    customdata=chart_custom,
                    marker_color="#3b82f6",
                    marker_opacity=0.65,
                    hovertemplate=(
                        "<b>%{x|%b %Y}</b><br>"
                        "Bookings: %{y:,.0f}<br>"
                        "Primary GA Campaign Type: %{customdata[0]}<br>"
                        "GA Campaign Types: %{customdata[1]}<extra></extra>"
                    ),
                )
            )
            if "gmv_per_ga_view" in selected_rest_raw.columns:
                fig_raw.add_trace(
                    go.Scatter(
                        x=selected_rest_raw["year_month"],
                        y=pd.to_numeric(selected_rest_raw["gmv_per_ga_view"], errors="coerce"),
                        mode="lines+markers",
                        name="GMV / GA View",
                        customdata=chart_custom,
                        line=dict(color="#2ecc71", width=3),
                        marker=dict(size=7),
                        yaxis="y2",
                        hovertemplate=(
                            "<b>%{x|%b %Y}</b><br>"
                            "GMV / GA View: %{y:.2f} THB<br>"
                            "Primary GA Campaign Type: %{customdata[0]}<br>"
                            "GA Campaign Types: %{customdata[1]}<extra></extra>"
                        ),
                    )
                )
            fig_raw.update_layout(
                **BASE_LAYOUT,
                height=280,
                xaxis=dict(**CHART_THEME["xaxis"], title="Month", tickangle=-30),
                yaxis=dict(**CHART_THEME["yaxis"], title="Monthly Bookings"),
                yaxis2=dict(
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    title="GMV / GA View",
                    color="#2ecc71",
                ),
                legend=dict(orientation="h", y=1.02, x=0, font_size=10),
            )
            st.plotly_chart(fig_raw, width="stretch")

            selected_display = _prepare_raw_ga_display(
                selected_rest_raw.sort_values("year_month", ascending=False),
                include_restaurant=False,
            )
            st.dataframe(selected_display, hide_index=True, width="stretch", height=240)

    with raw_tab_b:
        if cluster_ga_raw.empty:
            st.info("No raw gmv_view rows found for the selected cluster.")
        else:
            st.caption(f"Showing restaurants in {raw_cluster_label}.")
            available_months = sorted(
                pd.to_datetime(cluster_ga_raw["year_month"], errors="coerce").dropna().dt.strftime("%Y-%m").unique().tolist(),
                reverse=True,
            )
            month_choice = st.selectbox(
                "Month filter",
                options=["All Months"] + available_months,
                index=0,
                key=f"cluster_raw_ga_month_{str(active_cluster)}_{active_rest_norm}",
            )
            cluster_raw_filtered = cluster_ga_raw.copy()
            if month_choice != "All Months":
                cluster_raw_filtered = cluster_raw_filtered[
                    pd.to_datetime(cluster_raw_filtered["year_month"], errors="coerce").dt.strftime("%Y-%m") == month_choice
                ].copy()

            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Restaurants", f"{cluster_raw_filtered['name_norm'].nunique():,}")
            metric_b.metric("Rows", f"{len(cluster_raw_filtered):,}")
            metric_c.metric(
                "Avg GMV / GA View",
                _fmt_thb(
                    pd.to_numeric(cluster_raw_filtered["gmv_per_ga_view"], errors="coerce").mean()
                    if "gmv_per_ga_view" in cluster_raw_filtered.columns
                    else np.nan
                ),
            )

            cluster_display = _prepare_raw_ga_display(
                cluster_raw_filtered.sort_values(["year_month", "gmv_per_ga_view"], ascending=[False, False]),
                include_restaurant=True,
            )
            st.dataframe(cluster_display, hide_index=True, width="stretch", height=280)

    st.markdown("#### Raw GA Campaign Breakdown")
    st.caption(
        "Source: `data/marketing/googleAPI/campaigns_outreach.parquet`. "
        "These are platform Google Ads campaigns active in the selected month window, shown for context alongside the restaurant GA rows."
    )

    if ga_campaign_raw.empty:
        st.info("No raw GA campaign outreach rows found.")
    else:
        campaign_scope = ga_campaign_raw.copy()
        if not selected_rest_raw.empty:
            scope_months = pd.to_datetime(selected_rest_raw["year_month"], errors="coerce").dropna().unique().tolist()
        elif not cluster_ga_raw.empty:
            scope_months = pd.to_datetime(cluster_ga_raw["year_month"], errors="coerce").dropna().unique().tolist()
        else:
            scope_months = []
        if scope_months:
            campaign_scope = campaign_scope[campaign_scope["year_month"].isin(scope_months)].copy()

        campaign_month_options = sorted(
            pd.to_datetime(campaign_scope["year_month"], errors="coerce").dropna().dt.strftime("%Y-%m").unique().tolist(),
            reverse=True,
        )
        if not campaign_month_options:
            st.info("No campaign outreach rows match the available month window for this restaurant or cluster.")
        else:
            default_campaign_month = None
            if not selected_rest_raw.empty:
                default_campaign_month = (
                    pd.to_datetime(selected_rest_raw["year_month"], errors="coerce")
                    .dropna()
                    .max()
                )
            elif not cluster_ga_raw.empty:
                default_campaign_month = (
                    pd.to_datetime(cluster_ga_raw["year_month"], errors="coerce")
                    .dropna()
                    .max()
                )
            default_campaign_month_str = (
                default_campaign_month.strftime("%Y-%m")
                if pd.notna(default_campaign_month)
                else campaign_month_options[0]
            )
            default_campaign_idx = (
                campaign_month_options.index(default_campaign_month_str)
                if default_campaign_month_str in campaign_month_options
                else 0
            )

            campaign_control_a, campaign_control_b = st.columns([1.2, 1.2])
            with campaign_control_a:
                campaign_month_choice = st.selectbox(
                    "Campaign month",
                    options=campaign_month_options,
                    index=default_campaign_idx,
                    key=f"cluster_campaign_breakdown_month_{str(active_cluster)}_{active_rest_norm}",
                )
            month_campaigns = campaign_scope[
                pd.to_datetime(campaign_scope["year_month"], errors="coerce").dt.strftime("%Y-%m") == campaign_month_choice
            ].copy()
            type_options = sorted(month_campaigns["googleAdsCampaignType"].dropna().astype(str).unique().tolist())
            with campaign_control_b:
                campaign_type_choice = st.selectbox(
                    "Campaign type",
                    options=["All Types"] + type_options,
                    index=0,
                    key=f"cluster_campaign_breakdown_type_{str(active_cluster)}_{active_rest_norm}",
                )

            if campaign_type_choice != "All Types":
                month_campaigns = month_campaigns[
                    month_campaigns["googleAdsCampaignType"] == campaign_type_choice
                ].copy()

            if month_campaigns.empty:
                st.info("No raw campaigns found for the selected month and campaign type.")
            else:
                total_sessions = pd.to_numeric(month_campaigns["sessions"], errors="coerce").sum()
                month_campaigns["session_share"] = np.where(
                    total_sessions > 0,
                    pd.to_numeric(month_campaigns["sessions"], errors="coerce") / total_sessions,
                    np.nan,
                )

                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("Campaigns", f"{month_campaigns['campaign_id'].nunique():,}")
                metric_b.metric("Campaign Types", f"{month_campaigns['googleAdsCampaignType'].nunique():,}")
                metric_c.metric("Sessions", f"{total_sessions:,.0f}")

                mix_df = (
                    month_campaigns.groupby("googleAdsCampaignType", as_index=False)["sessions"]
                    .sum()
                    .sort_values("sessions", ascending=False)
                )
                mix_df["session_share"] = np.where(
                    mix_df["sessions"].sum() > 0,
                    mix_df["sessions"] / mix_df["sessions"].sum(),
                    np.nan,
                )

                fig_campaign_mix = go.Figure(
                    go.Bar(
                        x=mix_df["googleAdsCampaignType"],
                        y=mix_df["sessions"],
                        marker_color="#3b82f6",
                        text=mix_df["session_share"].apply(lambda v: "-" if pd.isna(v) else f"{v:.0%}"),
                        textposition="outside",
                        hovertemplate="%{x}<br>Sessions: %{y:,.0f}<br>Share: %{text}<extra></extra>",
                    )
                )
                fig_campaign_mix.update_layout(
                    **BASE_LAYOUT,
                    height=260,
                    showlegend=False,
                    xaxis=dict(**CHART_THEME["xaxis"], title="GA Campaign Type", tickangle=-20),
                    yaxis=dict(**CHART_THEME["yaxis"], title="Sessions"),
                )
                st.plotly_chart(fig_campaign_mix, width="stretch")

                campaign_display = _prepare_campaign_breakdown_display(
                    month_campaigns.sort_values(["sessions", "campaign_name"], ascending=[False, True])
                )
                st.dataframe(campaign_display, hide_index=True, width="stretch", height=280)

    st.markdown("---")

    lower_left, lower_right = st.columns([1.1, 1.9])

    with lower_left:
        st.markdown(f"### Momentum Mix in {scope_label}")
        seg_counts = (
            selected_cluster_df["latest_segment"].fillna("Unknown").value_counts().reset_index()
        )
        seg_counts.columns = ["segment", "count"]

        fig_seg = go.Figure(
            go.Bar(
                x=seg_counts["segment"],
                y=seg_counts["count"],
                marker_color="#3b82f6",
                text=seg_counts["count"],
                textposition="outside",
                hovertemplate="%{x}: %{y} restaurants<extra></extra>",
            )
        )
        fig_seg.update_layout(
            **BASE_LAYOUT,
            height=320,
            showlegend=False,
            xaxis=dict(**CHART_THEME["xaxis"], title="Momentum category", tickangle=-20),
            yaxis=dict(**CHART_THEME["yaxis"], title="Restaurants"),
        )
        st.plotly_chart(fig_seg, width="stretch")

    with lower_right:
        st.markdown(f"### Marketing Strategy Effectiveness ({scope_label})")
        rankings_df = load_cluster_strategy_rankings(min_sample_size=int(min_sample_size))
        if active_cluster == all_clusters_option:
            cluster_rank = rankings_df.copy()
        else:
            cluster_rank = rankings_df[rankings_df["cluster_id"] == active_cluster].copy()

        if cluster_rank.empty:
            st.info("No strategy outcomes available for this cluster.")
        else:
            st.caption(
                "Ranking score = 100 x Revenue uplift component + 30 x Bookings uplift + 10 x ROI + 5 x success rate. "
                "If revenue uplift % is unavailable, normalized incremental revenue is used."
            )

            if active_cluster == all_clusters_option:
                best_recs = cluster_rank[cluster_rank["meets_sample_guardrail"]].sort_values("ranking_score", ascending=False).head(3)
            else:
                best_recs = get_cluster_strategy_recommendations(
                    cluster_id=int(active_cluster), top_n=3, min_sample_size=int(min_sample_size)
                )
            if best_recs.empty:
                st.warning("No strategies meet the sample-size guardrail for this cluster.")
            else:
                top_cols = st.columns(min(3, len(best_recs)))
                for idx, (_, rec) in enumerate(best_recs.iterrows()):
                    with top_cols[idx]:
                        st.metric(
                            f"#{idx + 1} {rec['strategy_name'][:26]}",
                            _fmt_pct(rec.get("avg_revenue_uplift_pct")),
                            f"bookings {_fmt_pct(rec.get('avg_bookings_uplift_pct'))}",
                        )

            rank_display = cluster_rank[
                [
                    "cluster_label",
                    "strategy_name",
                    "activities",
                    "restaurants",
                    "avg_revenue_uplift_pct",
                    "avg_bookings_uplift_pct",
                    "avg_roi",
                    "total_incremental_revenue_thb",
                    "ranking_score",
                    "meets_sample_guardrail",
                    "data_quality_note",
                ]
            ].copy()

            rank_display = rank_display.rename(
                columns={
                    "cluster_label": "Cluster",
                    "strategy_name": "Strategy",
                    "activities": "Activities",
                    "restaurants": "Restaurants",
                    "avg_revenue_uplift_pct": "Avg Revenue Uplift %",
                    "avg_bookings_uplift_pct": "Avg Bookings Uplift %",
                    "avg_roi": "Avg ROI",
                    "total_incremental_revenue_thb": "Total Incremental Revenue (THB)",
                    "ranking_score": "Rank Score",
                    "meets_sample_guardrail": "Eligible",
                    "data_quality_note": "Confidence Note",
                }
            )

            for col in ["Avg Revenue Uplift %", "Avg Bookings Uplift %"]:
                rank_display[col] = rank_display[col].apply(_fmt_pct)
            rank_display["Avg ROI"] = rank_display["Avg ROI"].apply(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
            rank_display["Total Incremental Revenue (THB)"] = rank_display["Total Incremental Revenue (THB)"].apply(_fmt_thb)
            rank_display["Rank Score"] = pd.to_numeric(rank_display["Rank Score"], errors="coerce").round(2)

            st.dataframe(rank_display.sort_values(["Eligible", "Rank Score"], ascending=[False, False]), hide_index=True, width="stretch", height=280)

            st.markdown("#### Raw Campaign Strategy Breakdown")
            with st.expander("How Campaign Names and Assigned Strategies Are Named", expanded=False):
                st.markdown(
                    """
                    **Campaign Name** comes from the raw marketing activity fields: CRM uses `crm_campaign_name`,
                    FB uses `fb_campaign`, KOL uses the creator username, and rows fall back to `activity_id`
                    when a source-specific name is unavailable.

                    **Assigned Strategy** is rule-based, not manually labeled. The dashboard reads the channel plus
                    keywords from the campaign name, CRM topic, FB campaign, and KOL username, then assigns one
                    category:

                    - `CRM | Reactivation`: CRM campaigns with words like `reactivat`, `winback`, `inactive`,
                      `lapsed`, `dormant`, `comeback`, or `churn`.
                    - `CRM | Loyalty & Retention`: CRM campaigns with words like `loyal`, `member`, `vip`,
                      `retention`, `repeat`, `reward`, or `point`.
                    - `CRM | Promotional Blast`: CRM campaigns with words like `promo`, `discount`, `voucher`,
                      `coupon`, `deal`, `sale`, `flash`, `bundle`, or `off`.
                    - `CRM | Seasonal Campaign`: CRM campaigns with words like `season`, `festival`, `holiday`,
                      `songkran`, `new year`, `christmas`, `valentine`, `ramadan`, or `lunar`.
                    - `CRM | Lifecycle Nurture`: the default CRM category when the campaign is a normal CRM push
                      or notification and does not match the reactivation, loyalty, promo, or seasonal keywords.

                    For example, `TH_BKK_ctnoti_netcore_single_N_N_active_20260109_1100_yok-chinese-restaurant-jan26`
                    is treated as `CRM | Lifecycle Nurture` because it is a CRM notification campaign for an active
                    audience, and its topic `yok-chinese-restaurant-jan26` does not contain promo, discount,
                    reactivation, loyalty, or seasonal keywords. The restaurant/month slug gives context, but it
                    does not by itself imply a specific promotional strategy.

                    FB campaigns use similar keyword rules: retargeting words map to `FB | Retargeting`, promo/deal
                    words map to `FB | Conversion Offer`, and prospecting/acquisition/reach/awareness words map to
                    `FB | Prospecting & Awareness`; otherwise they become `FB | Performance Campaign`. KOL campaigns
                    map to creator collaboration or promo categories based on influencer/creator/promo keywords.
                    """
                )
            raw_strategy_campaigns = outcomes_df.copy()
            if active_cluster != all_clusters_option:
                raw_strategy_campaigns = raw_strategy_campaigns[
                    pd.to_numeric(raw_strategy_campaigns["cluster_id"], errors="coerce") == int(active_cluster)
                ].copy()

            if raw_strategy_campaigns.empty:
                st.info("No raw campaign rows available for this strategy scope.")
            else:
                raw_strategy_campaigns = raw_strategy_campaigns.sort_values(
                    ["applied_date", "channel", "campaign_name"],
                    ascending=[False, True, True],
                    na_position="last",
                )
                st.caption(
                    f"{len(raw_strategy_campaigns):,} campaign rows mapped to "
                    f"{raw_strategy_campaigns['strategy_family'].nunique():,} assigned strategies."
                )
                st.dataframe(
                    _prepare_strategy_campaign_display(raw_strategy_campaigns),
                    hide_index=True,
                    width="stretch",
                    height=320,
                )

    st.markdown("---")
    st.markdown("### Peer-Based Strategy Recommendations")
    st.caption(
        "Recommendations are dynamically generated by learning from high-performing peers "
        "with identical business models and digital funnel behaviors."
    )
    '''
    Step by step implementing peer recommender logic:
    1. Identify which restaurants have campaign data available for recommendation generation (using outcomes_df as proxy)
    2. Create a priority peer pool of restaurants that are in high-momentum segments (Rising stars & Established players)
    3. If there are enough high-momentum peers with campaign data, use them exclusively. 
    4. If not, top up with closest peers by distance until we have K peers. Calculate distance using UMAP coordinates, GMV per GA view, and cluster membership (same cluster gets a distance bonus).
    5. Conduct similarity-weighted aggregation of peer campaign outcomes, give more weight to peers closer in distance in the same cluster
    6. Surface top recommended strategies based on aggregated peer performance, along with key campaign metrics from the peer group to provide context on why these strategies are recommended.
    '''

    peer_k = st.slider("Number of similar peers to analyze", 3, 20, 6, key="peer_k_slider")
    st.caption("Peers ranked by distance, with priority given to high-momentum segments. " \
    "Distance calculated using UMAP coordinates, GMV per GA view, and cluster membership (same cluster gets a distance bonus).")

    if not selected_restaurant_row.empty and not assignments.empty:
        if outcomes_df.empty:
            st.info("No marketing outcomes data available to generate recommendations.")
        else: # 1. PRE-PROCESS: Identify which restaurants actually have campaign data
            o = outcomes_df.copy() # Avoid editing original df
            o["_name_norm"] = o["restaurant_name"].apply(_normalize_name) # Create normalised name for matching
            restaurants_with_campaigns = set(o["_name_norm"].unique()) # Identify restaurants with campaign data for filtering peers later

            active_gmv = selected_restaurant_row.iloc[0].get("monthly_gmv", 0) # Use montly GMV as key bz metric for momentum categorization and distance calculation
            HIGH_MOMENTUM = {"Rising Stars", "Established Players"} # Only compare peers under these segments

            # 2. Start with high-momentum peers that have campaign data, as they are more likely to yield relevant insights
            high_momentum_peers = assignments[ # Start with Rising Stars & Established Players 
                (assignments["name_norm"] != active_rest_norm) & # Exclude self
                (assignments["latest_segment"].isin(HIGH_MOMENTUM)) & # Focus on high-momentum segments
                (assignments["name_norm"].isin(restaurants_with_campaigns)) # Ensure campaign data is available
            ].copy()
            high_momentum_peers["momentum_bonus"] = 1.0 # Prioritize peers from high-momentum segments, reducing their distance in the ranking steps later

            # 3. Fallback pool: all other segments with campaign data (Last resort)
            fallback_peers = assignments[
                (assignments["name_norm"] != active_rest_norm) &
                (~assignments["latest_segment"].isin(HIGH_MOMENTUM)) &
                (assignments["name_norm"].isin(restaurants_with_campaigns))
            ].copy()
            fallback_peers["momentum_bonus"] = 0.0

            if len(high_momentum_peers) >= peer_k: # Enough high-momentum peers — use exclusively
                peer_df = high_momentum_peers.copy()
            elif len(high_momentum_peers) > 0: # Partial: top up with fallback peers sorted by distance later
                slots_needed = peer_k - len(high_momentum_peers) # Fill in remaining slots with closest peers by distance (no momentum bonus)
                peer_df = pd.concat( 
                    [high_momentum_peers, fallback_peers.head(slots_needed)],
                    ignore_index=True
                )
                st.caption(
                    f"Only {len(high_momentum_peers)} high-performing peer(s) found. "
                    f"{slots_needed} additional peer(s) included by proximity."
                )
            else: # No high-momentum peers at all — use fallback entirely
                peer_df = fallback_peers.copy()
                st.caption(
                    "No Rising Stars or Established Players with campaign data found. "
                    "Showing closest available peers by distance."
                )

            if peer_df.empty:
                st.warning("No peers with campaign data found for this restaurant.")
                top_peers = pd.DataFrame()
                peer_campaigns = pd.DataFrame(columns=["strategy_name", "channel"])
            else:
                DISTANCE_FEATURES = ['x', 'y', 'avg_gmv_perview']
                # Calculate distance using UMAP coordinates
                for col in DISTANCE_FEATURES: # Ensure missing columns don't break the scaler
                    if col not in peer_df.columns: peer_df[col] = 0
                    if col not in selected_restaurant_row.columns: selected_restaurant_row[col] = 0

                valid_features = [ # VARIANCE GUARD: drop any feature whose peer-pool std is ~0.
                    col for col in DISTANCE_FEATURES
                    if peer_df[col].fillna(0).std() > 1e-9
                ]
                if not valid_features: # Fall back to simple unweighted distance if all features are constant within the peer pool
                    valid_features = ["avg_monthly_gmv"]  # last-resort fallback

                scaler = StandardScaler() # Scale features -> fair distance calculation
                pool_scaled = scaler.fit_transform(peer_df[valid_features].fillna(0))
                active_scaled = scaler.transform( # Scale active restaurant's features using the same scaler fitted on the peer pool
                    selected_restaurant_row[valid_features].fillna(0).values
                )
                # Calculate distance from active restaurant to each peer in the pool
                base_distances = euclidean_distances(pool_scaled, active_scaled).flatten()

                # 5. Add same-cluster bonus, cluster membership is strong signal of similarity.
                active_cluster = selected_restaurant_row.iloc[0].get("cluster_id", -1)
                same_cluster = (peer_df["cluster_id"] == active_cluster).astype(float)

                # Adjust distance: smaller is better. momentum_bonus pushes high-momentum peers closer.
                peer_df["distance"] = base_distances - (same_cluster * 0.5) - (peer_df["momentum_bonus"] * 0.5)
                peer_df["distance"] = np.maximum(peer_df["distance"], 0)

                # Take top K — high-momentum peers naturally rank first due to momentum_bonus
                top_peers = peer_df.sort_values("distance").head(peer_k).copy()

                # 6. Compute SIMILARITY-WEIGHTED. Each peer is assigned a similarity weight that is the inverse of its distance
                top_peers["sim_w"] = 1 / (top_peers["distance"] + 1e-6) # Similarity weight: inverse of distance, add small constant to avoid division by zero
                # Kepp only necessary columns for merging with campaign data
                peer_dist = top_peers[["name_norm", "sim_w", "distance"]].copy() # Keep only necessary columns for merging with campaign data
                peer_campaigns = o[o["_name_norm"].isin(set(peer_dist["name_norm"]))].copy() # Get campaign outcomes for the selected peers
                peer_campaigns = peer_campaigns.merge(peer_dist, left_on="_name_norm", right_on="name_norm", how="left")
                
                rec = ( # 7. Join campaign and peer by strategy. Created by grouping peer campaign outcomes by strategy and channel, and calculating weighted performance metrics for each strategy based on the similarity weights of the peers that used it.
                    peer_campaigns.groupby(["strategy_name", "channel"], dropna=False)
                    .agg(
                        campaigns=("activity_id", "nunique"),
                        peers_using=("restaurant_name", "nunique"),
                        med_rev_uplift=("revenue_uplift_pct", "median"),
                        med_roi=("roi", "median"),
                        total_incremental=("incremental_revenue_thb", "sum"),
                        total_sim_weight=("sim_w", "sum")
                    )
                    .reset_index()
                )

                # GUARDRAILS FOR RELIABILITY: Require ≥2 peers
                MIN_PEERS = 2
                rec["is_reliable"] = rec["peers_using"] >= MIN_PEERS
                rec["score"] = (
                    rec["med_rev_uplift"].fillna(0) * 100
                    + rec["med_roi"].fillna(0) * 50
                    + np.log1p(rec["total_sim_weight"]) * 20
                    + rec["is_reliable"].astype(int) * 30
                )
                rec = rec.sort_values(["is_reliable", "score"], ascending=[False, False]) # 8. Score & rank strats by a combination of median revenue uplift, median ROI, total similarity weight (popularity among similar peers), and a reliability boost for strategies used by at least MIN_PEERS peers. This multi-factor scoring helps surface strategies that are not only high-performing but also more likely to be relevant and reliable for the active restaurant.

                # 9: FUNNEL STAGE DIVERSIFICATION (Funnel stage defined by the strategy's primary channel, e.g. Discovery, Consideration, Conversion)
                # Avoid recommending multiple strategies from the same funnel stage or channel to ensure a more holistic growth approach.
                # Pick top few by score alone, 
                diverse_recs, seen_stages = [], set() # Track seen funnel stages or channels to ensure diversity in recommendations
                for _, row in rec.iterrows(): # Iterate through sorted recommendations and pick top ones from different stages/channels
                    stage = row.get("funnel_stage", row.get("channel", "Unknown"))
                    if stage not in seen_stages or len(diverse_recs) == 0:
                        diverse_recs.append(row) # Add recco if its stage / channel hasn't been seen yet, or if it's the first recommendation
                        seen_stages.add(stage) # Mark this stage / channel as seen to avoid recommending another strategy from the same stage
                    if len(diverse_recs) >= 3: # We only want to show top 3 recommendations, so
                        break

                if len(diverse_recs) < 3: # Fill in next best recco if not enought reccos
                    remaining = rec[~rec.index.isin([r.name for r in diverse_recs])]
                    for _, r in remaining.head(3 - len(diverse_recs)).iterrows():
                        diverse_recs.append(r) # Add next best reccos regardless of stage to ensure we show 3 recommendations if possible

                top_recs = pd.DataFrame(diverse_recs) # Convert list of recommendations back to DataFrame for display

                # UI PRESENTATION
                st.markdown("#### Top Strategies Recommended by Peers")
                displayable_recs = top_recs[top_recs["med_rev_uplift"].notna()].head(3)

                if displayable_recs.empty: # No valid reccos to display
                    st.info("No peer-validated strategies with recorded performance data. Try increasing peers.")
                else: # Only show reccos with valid revenue uplift data to ensure we're surfacing actionable insights
                    cols = st.columns(len(displayable_recs))
                    for idx, (_, r) in enumerate(displayable_recs.iterrows()):
                        with cols[idx]:
                            roi_val = r["med_roi"]
                            roi_str = f"ROI: {roi_val:.1f}x" if pd.notna(roi_val) else " " #"ROI: N/A"
                            uplift_val = r["med_rev_uplift"]
                            uplift_str = _fmt_pct(uplift_val)
                            st.metric(
                                label=f"{r['strategy_name']} ({r['channel']})",
                                value=uplift_str,
                                delta=roi_str,
                                help=f"Used by {r['peers_using']} similar peers across {r['campaigns']} campaigns."
                            )

                # 10. ENRICH TOP PEERS with their campaign strats FOR THE LLM
                # calculate peer financial stats here so the LLM has context
                if not o.empty and "_name_norm" in o.columns:
                    # 1. Extract NORMALIZED names from top peers 
                    peer_names_norm = set(top_peers["name_norm"].dropna())
                    
                    # 2. Filter the outcomes dataframe using its NORMALIZED column
                    o_peers = o[o["_name_norm"].isin(peer_names_norm)].copy()
                    
                    if not o_peers.empty:
                        best_ch = (
                            o_peers.groupby(["_name_norm", "channel"])["revenue_uplift_pct"]
                            .median().reset_index()
                            .sort_values("revenue_uplift_pct", ascending=False)
                            .drop_duplicates("_name_norm")[["_name_norm", "channel"]]
                            .rename(columns={"channel": "best_channel"})
                        )
                        peer_stats = (
                            o_peers.groupby("_name_norm")
                            .agg(
                                n_campaigns=("activity_id", "nunique"),
                                avg_roi=("roi", "median"),
                                avg_incremental_rev=("incremental_revenue_thb", "median"),
                                n_positive_lift=("revenue_uplift_pct", lambda x: (x > 0).sum()),
                                n_total=("revenue_uplift_pct", "count"),
                            ).reset_index()
                        )
                        
                        peer_stats["lift_reliability"] = peer_stats["n_positive_lift"] / peer_stats["n_total"].replace(0, pd.NA)
                        peer_stats = peer_stats.merge(best_ch, on="_name_norm", how="left")
                        
                        # 3. Merge using NORMALIZED names on both sides!
                        top_peers = top_peers.merge(peer_stats, left_on="name_norm", right_on="_name_norm", how="inner")

                # ENHANCED PEER EVIDENCE TABLE
                with st.expander("🔍 View Raw Peer Evidence (Metadata & Scale)"):
                    # display_peers ALREADY contains all the stats from Step 10!
                    display_peers = top_peers.copy()

                    # 1. Flag high-quality peers and sort them to the top
                    if "n_campaigns" in display_peers.columns and "avg_incremental_rev" in display_peers.columns:
                        # Create a hidden column that is True if they meet your criteria, False otherwise
                        display_peers["is_reliable_peer"] = (
                            (display_peers["n_campaigns"].fillna(0) > 1) & 
                            (display_peers["avg_incremental_rev"].fillna(0) > 0)
                        )
                        
                        # Sort by reliability first (True at the top), then by similarity distance (closest first)
                        if "distance" in display_peers.columns:
                            display_peers = display_peers.sort_values(
                                by=["is_reliable_peer", "distance"], 
                                ascending=[False, True]  # False means 'True' goes first for boolean; True means smallest distance first
                            )
                        else:
                            display_peers = display_peers.sort_values(
                                by=["is_reliable_peer", "avg_incremental_rev"], 
                                ascending=[False, False]
                            )

                    # 2. Select columns to display
                    display_cols = [
                        "name", "monthly_gmv", "best_channel", "avg_incremental_rev", 
                        "n_campaigns", "lift_reliability", "latest_segment",
                    ]
                    existing_cols = [c for c in display_cols if c in display_peers.columns]

                    # 3. Rename for UI
                    renamed = display_peers[existing_cols].rename(columns={
                        "name":                "Peer Restaurant",
                        "monthly_gmv":         "Monthly GMV",
                        "best_channel":        "Best Channel",
                        "avg_incremental_rev": "Avg Incremental Rev (THB)",
                        "n_campaigns":         "Campaigns Run",
                        "lift_reliability":    "Lift Reliability",
                        "latest_segment":      "Momentum",
                    })

                    # 4. Render Table
                    st.dataframe(
                        renamed.style.format({
                            "Monthly GMV":               lambda x: f"THB {x:,.0f}" if pd.notna(x) else "-",
                            "Avg Incremental Rev (THB)": lambda x: f"THB {x:,.0f}" if pd.notna(x) else "-",
                            "Lift Reliability":          lambda x: f"{x:.1%}" if pd.notna(x) else "-",
                            "Campaigns Run":             lambda x: f"{int(x)}" if pd.notna(x) else "-",
                        }),
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.caption(
                        f"Similarity computed on: {', '.join(valid_features)}. "
                        "Best Channel and ROI sourced from historical campaign outcomes."
                    )

                # ── GROUNDED BRIEF (always renders, no API needed) ──────────────────────────
                active_momentum = selected_restaurant_row.iloc[0].get("latest_segment", "Unknown")

                segment_advice = {
                    "Rising Stars":            ("Scale what's working before momentum plateaus. Double down on your best-performing channel.", "🟢"),
                    "Emerging Opportunities":  ("Conversion efficiency is the priority — get more bookings from existing traffic before increasing spend.", "🟡"),
                    "Established Players":     ("Focus on retention and upsell. Acquisition costs are high; loyal guests have higher LTV.", "🔵"),
                    "Needs Attention":         ("Diagnose the funnel before spending on campaigns. Traffic may not be the problem.", "🔴"),
                }.get(active_momentum, ("Insufficient momentum data to generate directional advice.", "⚪"))

                # st.markdown("#### 💡 Strategic Action Plan")

                # if not top_recs.empty:
                #     active_momentum = selected_restaurant_row.iloc[0].get("latest_segment", "Unknown")
                #     segment_framing = {
                #         "Rising Stars": (
                #             "You are in a high-growth phase. Your priority is **scaling what works** "
                #             "before momentum plateaus."
                #         ),
                #         "Emerging Opportunities": (
                #             "You have strong growth signals but limited current scale. "
                #             "The focus should be **conversion efficiency** — getting more bookings from existing traffic."
                #         ),
                #         "Established Players": (
                #             "Growth has stabilised. The opportunity lies in **retention and upsell** "
                #             "rather than pure acquisition."
                #         ),
                #         "Needs Attention": (
                #             "Performance signals are weak. Before investing in new campaigns, "
                #             "**diagnose the funnel** — traffic may not be the problem."
                #         ),
                #     }.get(active_momentum, "")

                #     top_strat = top_recs.iloc[0]
                #     s_name = top_strat['strategy_name']
                #     s_channel = top_strat['channel']
                #     s_roi = top_strat['med_roi']
                #     s_uplift = top_strat['med_rev_uplift']

                #     top_channels = top_recs['channel'].value_counts()
                #     dominant_channel = top_channels.index[0] if top_channels.iloc[0] > 1 else None

                #     narrative = ""
                #     if segment_framing:
                #         narrative += segment_framing + "\n\n"

                #     narrative += (
                #         f"Based on the behavior of high-momentum restaurants with similar digital conversion rates "
                #         f"and themes, your immediate focus should be **{s_name}** via **{s_channel}**. "
                #     )

                #     if pd.notna(s_roi) and s_roi > 0:
                #         narrative += (
                #             f"Peers utilizing this exact approach have historically seen a median ROI of "
                #             f"**{s_roi:.1f}x** and a revenue uplift of **{_fmt_pct(s_uplift)}**. "
                #         )

                #     if dominant_channel:
                #         narrative += (
                #             f"\n\n**Channel Insight:** Notice that **{dominant_channel}** appears frequently among "
                #             f"your top matches. This strongly suggests that your target demographic is highly responsive "
                #             f"to interventions on this specific platform, making it the safest bet for your ad spend."
                #         )

                #     if len(top_recs) > 1:
                #         second_strat = top_recs.iloc[1]
                #         narrative += (
                #             f"\n\n**Execution Steps:** \n"
                #             f"1. **Anchor:** Launch the `{s_name}` campaign first as your primary growth lever.\n"
                #             f"2. **Layer:** Once baseline metrics stabilize, introduce `{second_strat['strategy_name']}` "
                #             f"({second_strat['channel']}) to capture residual traffic and diversify your funnel."
                #         )

                #     st.success(narrative)
                
                grounded_advice, segment_icon = segment_advice

                # # Derive top strategy stats for the brief
                # top_strat        = top_recs.iloc[0] if not top_recs.empty else None
                # has_data         = top_strat is not None and pd.notna(top_strat.get("med_rev_uplift"))
                # reliable_peers   = int(top_recs["is_reliable"].sum()) if "is_reliable" in top_recs.columns else 0
                # dominant_channel = top_recs["channel"].value_counts().index[0] if not top_recs.empty else "Unknown"

                # grounded_lines = [
                #     f"**{segment_icon} Segment:** {active_momentum} — {grounded_advice}",
                #     "",
                # ]

                # if top_strat is not None:
                #     uplift_str = f"{top_strat['med_rev_uplift']:.1%}" if has_data else "no uplift data recorded"
                #     grounded_lines += [
                #         f"**📌 Priority Campaign:** `{top_strat['strategy_name']}` via **{top_strat['channel']}** "
                #         f"— {uplift_str} median uplift across {int(top_strat.get('peers_using', 0))} peer(s).",
                #         "",
                #         f"**📡 Dominant Channel:** {dominant_channel} appears in {int(top_recs['channel'].value_counts().iloc[0])} of "
                #         f"{len(top_recs)} recommended strategies for this peer group.",
                #         "",
                #         f"**✅ Reliability:** {reliable_peers}/{len(top_recs)} strategies are backed by ≥2 peers with recorded outcomes.",
                #     ]
                # else:
                #     grounded_lines.append("_No peer-validated strategies available. Try increasing the number of peers._")

                # grounded_brief = "\n".join(grounded_lines)

                # st.markdown("---")
                # st.markdown("#### 📋 Strategy Brief")
                # st.caption("Deterministic summary based on peer data — always available regardless of AI status.")
                # with st.container(border=True):
                #     st.markdown(grounded_brief)

                # # Download button for the grounded brief (borrowed from strategy.py)
                # st.download_button(
                #     label="⬇️ Download Strategy Brief",
                #     data="PEER STRATEGY BRIEF\n%s\n\n%s" % (
                #         selected_restaurant_row.iloc[0].get("name", "Unknown"),
                #         grounded_brief.replace("**", "").replace("`", "")   # strip markdown for plain text
                #     ),
                #     file_name="strategy_brief_%s.txt" % selected_restaurant_row.iloc[0].get("name", "Unknown").replace(" ", "_"),
                #     mime="text/plain",
                # )

                # ── OPTIONAL AI NARRATIVE (button-triggered, borrowed from strategy.py) ──────
                st.markdown("---")
                st.markdown("#### 🤖 Optional AI Narrative")
                st.caption("Uses Gemini to synthesise peer campaign evidence into a narrative. The Strategy Brief above is the source of truth.")

                ai_key = "ai_playbook_%s" % selected_restaurant_row.iloc[0].get("name", "")

                gen_col, clr_col = st.columns([3, 1])
                with gen_col:
                    generate_ai = st.button("✨ Generate AI Narrative", key="gen_ai_peer", use_container_width=True)
                with clr_col:
                    if st.button("🗑️ Clear", key="clr_ai_peer", use_container_width=True):
                        st.session_state[ai_key] = None

                if generate_ai:
                    prompt = build_recommendation_prompt(
                        restaurant_name=selected_restaurant_row.iloc[0].get("name", "Unknown"),
                        momentum=active_momentum,
                        theme=selected_restaurant_row.iloc[0].get("theme", "Unknown"),
                        active_gmv=active_gmv,
                        active_conv=selected_restaurant_row.iloc[0].get("avg_view_to_purchase", 0.0) or 0.0,
                        top_recs=top_recs,
                        top_peers=display_peers,   # enriched version with best_channel, reliability etc.
                    )
                    _phash = hashlib.sha256(prompt.encode()).hexdigest()
                    with st.spinner("Gemini is analysing peer strategies..."):
                        try:
                            st.session_state[ai_key] = generate_ai_playbook(prompt, _prompt_hash=_phash)
                        except Exception as e:
                            st.error("AI generation failed: %s" % e)
                            st.session_state[ai_key] = None

                if st.session_state.get(ai_key):
                    with st.container(border=True):
                        st.markdown(st.session_state[ai_key])
                    st.download_button(
                        label="⬇️ Download AI Narrative",
                        data="AI STRATEGY NARRATIVE\n%s\n\n%s" % (
                            selected_restaurant_row.iloc[0].get("name", "Unknown"),
                            st.session_state[ai_key]
                        ),
                        file_name="ai_narrative_%s.txt" % selected_restaurant_row.iloc[0].get("name", "Unknown").replace(" ", "_"),
                        mime="text/plain",
                    )

                # st.markdown("---")
                # st.markdown("#### 🧠 Strategic Playbook")
                # st.caption(
                #     f"Generated based on {len(top_peers)} similar peers · "
                #     f"{len(top_recs)} validated strategies · Segment: {active_momentum}"
                # )

                # # Step 1: build the structured prompt from peer + strategy data
                # prompt = build_recommendation_prompt(
                #     restaurant_name=selected_restaurant_row.iloc[0].get("name", "Unknown"),
                #     momentum=active_momentum,
                #     theme=selected_restaurant_row.iloc[0].get("theme", "Unknown"),
                #     active_gmv=active_gmv,
                #     active_conv=selected_restaurant_row.iloc[0].get("avg_view_to_purchase", 0.0) or 0.0,
                #     top_recs=top_recs,
                #     top_peers=top_peers,
                # )
 
                # # Step 2: call the LLM and render the response
                # # _prompt_hash ensures the cache key is based on prompt content, not call position
                # _phash = hashlib.sha256(prompt.encode()).hexdigest()
                # with st.spinner("Generating strategic playbook..."):
                #     playbook_text = generate_ai_playbook(prompt, _prompt_hash=_phash)
 
                # with st.container(border=True):
                #     st.markdown(playbook_text)
 
                # # Optional: let manager flag if it's useful
                # col_fb1, col_fb2, _ = st.columns([1, 1, 6])
                # with col_fb1:
                #     if st.button("👍 Useful", key="playbook_up"):
                #         st.toast("Feedback recorded — thanks!")
                # with col_fb2:
                #     if st.button("👎 Not relevant", key="playbook_down"):
                #         st.toast("Noted — we'll improve the model.")
