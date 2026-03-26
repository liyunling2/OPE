# -*- coding: utf-8 -*-
"""
pages/clustering.py
Cluster exploration dashboard with cross-highlighting and strategy effectiveness ranking.
"""

from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
)

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e8eaf0", family="DM Sans"),
    xaxis=dict(gridcolor="#2e3350", showline=False, zeroline=False, color="#9ca3c4"),
    yaxis=dict(gridcolor="#2e3350", showline=False, zeroline=False, color="#9ca3c4"),
    margin=dict(l=0, r=0, t=30, b=0),
)
BASE_LAYOUT = {k: v for k, v in CHART_THEME.items() if k not in ("xaxis", "yaxis")}


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _fmt_pct(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{value:.1%}"


def _fmt_thb(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"THB {value:,.0f}"


def _fmt_year_month(value) -> str:
    if value is None or pd.isna(value):
        return "Unknown month"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%Y-%m")


def _fmt_decimal(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{value:,.{digits}f}"


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


def _build_campaign_type_context(campaign_type_monthly: pd.DataFrame) -> pd.DataFrame:
    if campaign_type_monthly.empty:
        return pd.DataFrame(columns=["year_month", "primary_ga_campaign_type", "ga_campaign_types"])

    required_cols = {"year_month", "googleAdsCampaignType"}
    if not required_cols.issubset(campaign_type_monthly.columns):
        return pd.DataFrame(columns=["year_month", "primary_ga_campaign_type", "ga_campaign_types"])

    context = campaign_type_monthly.copy()
    context["year_month"] = pd.to_datetime(context["year_month"], errors="coerce")
    context["total_sessions"] = (
        pd.to_numeric(context["total_sessions"], errors="coerce").fillna(0)
        if "total_sessions" in context.columns
        else 0
    )
    context["active_campaigns"] = (
        pd.to_numeric(context["active_campaigns"], errors="coerce").fillna(0)
        if "active_campaigns" in context.columns
        else 0
    )
    context = context.dropna(subset=["year_month"])
    if context.empty:
        return pd.DataFrame(columns=["year_month", "primary_ga_campaign_type", "ga_campaign_types"])

    context["googleAdsCampaignType"] = context["googleAdsCampaignType"].fillna("Unknown").astype(str).str.strip()
    context = context[
        context["googleAdsCampaignType"].ne("")
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

    points = selection.get("points", []) if isinstance(selection, dict) else []
    if not points:
        return None, None

    point = points[0]
    custom = point.get("customdata", [])
    if isinstance(custom, (list, tuple)) and len(custom) >= 2:
        try:
            return str(custom[0]), int(custom[1])
        except Exception:
            return str(custom[0]), None
    return None, None


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


def _prepare_raw_ga_display(df: pd.DataFrame, include_restaurant: bool = False) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    display_cols = [
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


def _terms_from_texts(text_series: pd.Series, top_n: int = 40) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for text in text_series.fillna(""):
        for token in _tokenize(text):
            counts[token] = counts.get(token, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["keyword", "weight"])

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return pd.DataFrame(items, columns=["keyword", "weight"])


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

    words = word_df["keyword"].astype(str).tolist()
    weights = pd.to_numeric(word_df["weight"], errors="coerce").fillna(0).to_numpy()

    if np.max(weights) > np.min(weights):
        sizes = 16 + (weights - np.min(weights)) * (44 / (np.max(weights) - np.min(weights)))
    else:
        sizes = np.full(len(weights), 24)

    n_terms = len(words)
    angles = np.linspace(0.0, 5 * np.pi, n_terms)
    radius = np.linspace(0.15, 1.0, n_terms)
    rng = np.random.default_rng(42)
    x = radius * np.cos(angles) + rng.normal(0, 0.08, n_terms)
    y = radius * np.sin(angles) + rng.normal(0, 0.08, n_terms)

    highlights = highlight_terms or set()
    colors = ["#cc0000" if word in highlights else default_color for word in words]

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


def _highlight_text(value: str, query: str) -> str:
    escaped = html.escape(str(value))
    if not query.strip():
        return escaped.replace("\n", "<br>")

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

    for _, row in display_df.iterrows():
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
    assignments = load_cluster_assignments()
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
        "<p style='color:#9ca3c4; margin-top:-0.5rem;'>"
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
    cluster_options = cluster_rows["cluster_id"].tolist()
    cluster_options_ui = [all_clusters_option] + cluster_options
    cluster_label_map = dict(zip(cluster_rows["cluster_id"], cluster_rows["cluster_label"]))

    if not cluster_options:
        st.warning("No valid clusters available in clustering data.")
        return

    if "cluster_active_cluster" not in st.session_state:
        st.session_state["cluster_active_cluster"] = all_clusters_option

    if st.session_state["cluster_active_cluster"] not in cluster_options_ui:
        st.session_state["cluster_active_cluster"] = all_clusters_option

    def _filter_by_selected_cluster(df: pd.DataFrame) -> pd.DataFrame:
        active_value = st.session_state["cluster_active_cluster"]
        if active_value == all_clusters_option:
            return df.copy()
        return df[df["cluster_id"] == int(active_value)].copy()

    cluster_df = _filter_by_selected_cluster(assignments).sort_values("name")

    if "cluster_active_restaurant" not in st.session_state:
        st.session_state["cluster_active_restaurant"] = cluster_df.iloc[0]["name"] if len(cluster_df) else assignments.iloc[0]["name"]

    if _normalize_name(st.session_state["cluster_active_restaurant"]) not in set(cluster_df["name_norm"]):
        st.session_state["cluster_active_restaurant"] = cluster_df.iloc[0]["name"] if len(cluster_df) else assignments.iloc[0]["name"]

    control_a, control_b, control_c = st.columns([1.5, 2.5, 1])

    with control_a:
        active_cluster = st.selectbox(
            "Cluster",
            options=cluster_options_ui,
            index=cluster_options_ui.index(st.session_state["cluster_active_cluster"]),
            format_func=lambda cid: (
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

    with control_b:
        restaurant_options = cluster_df["name"].tolist()
        default_restaurant = st.session_state["cluster_active_restaurant"]
        if default_restaurant not in restaurant_options and restaurant_options:
            default_restaurant = restaurant_options[0]
            st.session_state["cluster_active_restaurant"] = default_restaurant

        active_restaurant = st.selectbox(
            "Restaurant (within selected scope)",
            options=restaurant_options,
            index=restaurant_options.index(default_restaurant) if restaurant_options else 0,
            key="cluster_restaurant_selector",
        )
        if active_restaurant != st.session_state["cluster_active_restaurant"]:
            st.session_state["cluster_active_restaurant"] = active_restaurant

        selected_lookup = assignments[assignments["name"].eq(st.session_state["cluster_active_restaurant"])].head(1)
        if len(selected_lookup):
            selected_cluster_label = selected_lookup.iloc[0].get("cluster_label", "Unknown")
            selected_cluster_id = selected_lookup.iloc[0].get("cluster_id", "Unknown")
            st.caption(f"Selected restaurant cluster: {selected_cluster_label} (ID: {selected_cluster_id})")

    with control_c:
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
    scope_gmv_per_ga_view = (
        pd.to_numeric(cluster_df["gmv_per_ga_view"], errors="coerce").mean()
        if "gmv_per_ga_view" in cluster_df.columns
        else np.nan
    )
    scope_ga_add_to_cart = (
        pd.to_numeric(cluster_df["ga_add_to_cart_rate"], errors="coerce").mean()
        if "ga_add_to_cart_rate" in cluster_df.columns
        else np.nan
    )

    metric_a, metric_b, metric_c, metric_d, metric_e, metric_f = st.columns(6)
    metric_a.metric("Restaurants in Scope", f"{len(cluster_df):,}")
    metric_b.metric("Total Scope Bookings", f"{cluster_df['monthly_bookings'].fillna(0).sum():,.0f}")
    metric_c.metric("Avg Scope Revenue", _fmt_thb(cluster_df["monthly_gmv"].mean()))
    metric_d.metric("Scope GMV / GA View", _fmt_thb(scope_gmv_per_ga_view))
    metric_e.metric("Scope GA Add to Cart", _fmt_pct(scope_ga_add_to_cart))
    metric_f.metric("Known Momentum Segment", f"{cluster_df['latest_segment'].notna().sum():,}")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.7, 1.3])

    with left:
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

    with right:
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
            else:
                cluster_text = text_corpus[text_corpus["cluster_id"] == active_cluster]["clean_text"]
            cluster_keywords = _terms_from_texts(cluster_text, top_n=40)

        rest_text_df = text_corpus[text_corpus["name_norm"] == active_rest_norm]
        restaurant_terms = _terms_from_texts(rest_text_df["clean_text"], top_n=30)
        highlight_terms = set(restaurant_terms["keyword"].tolist())

        fig_cluster_wc = _build_word_cloud_figure(
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

    st.markdown(f"### Strategy Activity Detail ({scope_label})")
    if active_cluster == all_clusters_option or "cluster_id" not in outcomes_df.columns:
        cluster_outcomes = outcomes_df.copy()
    else:
        cluster_outcomes = outcomes_df[outcomes_df["cluster_id"] == active_cluster].copy()

    if cluster_outcomes.empty:
        st.info("No campaign-level records found for this cluster.")
    else:
        if "strategy_family" in cluster_outcomes.columns:
            family_options = sorted(cluster_outcomes["strategy_family"].dropna().astype(str).unique().tolist())
            family_choice = st.selectbox(
                "Filter by strategy family",
                options=["All Strategy Families"] + family_options,
                key=f"cluster_activity_family_filter_{str(active_cluster)}",
            )
            if family_choice != "All Strategy Families":
                cluster_outcomes = cluster_outcomes[
                    cluster_outcomes["strategy_family"].fillna("Unknown") == family_choice
                ].copy()

        if cluster_outcomes.empty:
            st.info("No campaign-level records found for the selected strategy family.")
        else:
            detail_cols = [
                c
                for c in [
                    "strategy_family",
                    "strategy_name",
                    "cluster_label",
                    "restaurant_name",
                    "channel",
                    "applied_date",
                    "bookings_before",
                    "bookings_after",
                    "bookings_uplift_pct",
                    "incremental_revenue_thb",
                    "revenue_uplift_pct",
                    "roi",
                ]
                if c in cluster_outcomes.columns
            ]
            detail_df = cluster_outcomes[detail_cols].copy()
            detail_df = detail_df.rename(
                columns={
                    "strategy_family": "Strategy Family",
                    "strategy_name": "Strategy",
                    "cluster_label": "Cluster",
                    "restaurant_name": "Restaurant",
                    "channel": "Channel",
                    "applied_date": "Applied Date",
                    "bookings_before": "Bookings Before",
                    "bookings_after": "Bookings After",
                    "bookings_uplift_pct": "Bookings Uplift %",
                    "incremental_revenue_thb": "Incremental Revenue (THB)",
                    "revenue_uplift_pct": "Revenue Uplift %",
                    "roi": "ROI",
                }
            )

            if "Bookings Uplift %" in detail_df.columns:
                detail_df["Bookings Uplift %"] = detail_df["Bookings Uplift %"].apply(_fmt_pct)
            if "Revenue Uplift %" in detail_df.columns:
                detail_df["Revenue Uplift %"] = detail_df["Revenue Uplift %"].apply(_fmt_pct)
            if "Incremental Revenue (THB)" in detail_df.columns:
                detail_df["Incremental Revenue (THB)"] = detail_df["Incremental Revenue (THB)"].apply(_fmt_thb)
            if "ROI" in detail_df.columns:
                detail_df["ROI"] = detail_df["ROI"].apply(lambda v: "-" if pd.isna(v) else f"{v:.2f}")

            st.dataframe(detail_df.sort_values("Applied Date", ascending=False), hide_index=True, width="stretch", height=300)

    if not selected_restaurant_row.empty:
        st.caption(
            f"Active restaurant: {active_restaurant} | "
            f"Cluster: {selected_restaurant_row.iloc[0]['cluster_label']} | "
            f"Momentum: {selected_restaurant_row.iloc[0].get('latest_segment', 'Unknown')}"
        )
