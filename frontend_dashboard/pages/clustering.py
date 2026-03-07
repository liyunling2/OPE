# -*- coding: utf-8 -*-
"""
pages/clustering.py
Cluster exploration dashboard with cross-highlighting and strategy effectiveness ranking.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.loader import (
    get_cluster_strategy_recommendations,
    load_cluster_assignments,
    load_cluster_keywords,
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


def render():
    assignments = load_cluster_assignments()
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

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Restaurants in Scope", f"{len(cluster_df):,}")
    metric_b.metric("Total Scope Bookings", f"{cluster_df['monthly_bookings'].fillna(0).sum():,.0f}")
    metric_c.metric("Avg Scope Revenue", _fmt_thb(cluster_df["monthly_revenue"].mean()))
    metric_d.metric("Known Momentum Segment", f"{cluster_df['latest_segment'].notna().sum():,}")

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
                    marker=dict(size=20, color="#cc0000", symbol="diamond-open", line=dict(width=2, color="#cc0000")),
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
        cluster_df["active"] = np.where(cluster_df["name"].apply(_normalize_name) == active_rest_norm, "Selected", "")

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
    search_text = st.text_input("Filter selected restaurant text", value="", key="cluster_text_filter")
    rest_text_df = text_corpus[text_corpus["name_norm"] == active_rest_norm].copy()

    if search_text.strip():
        mask = rest_text_df["raw_text"].fillna("").str.contains(search_text.strip(), case=False, regex=False)
        rest_text_df = rest_text_df[mask]

    if rest_text_df.empty:
        st.info("No raw text rows for this restaurant with the current filter.")
    else:
        raw_display = rest_text_df[[c for c in ["text_id", "year_month", "raw_text"] if c in rest_text_df.columns]].rename(
            columns={"text_id": "Text ID", "year_month": "Year Month", "raw_text": "Raw Text"}
        )
        st.dataframe(raw_display, hide_index=True, width="stretch", height=240)

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
    if active_cluster == all_clusters_option:
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
