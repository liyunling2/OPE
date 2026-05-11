from __future__ import annotations

import json
import os
import re
from textwrap import dedent
from html import escape
import numpy as np
import pandas as pd
import streamlit as st
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

from data.loader import (
    load_priority,
    load_momentum,
    get_restaurant_history,
    get_restaurant_priority_row,
    load_cluster_assignments,
    load_ga_restaurant_monthly,
    load_ga_campaign_type_monthly,
    load_cluster_strategy_outcomes,
)
from theme import BORDER_COLOR, MUTED_TEXT, SURFACE_COLOR, TEXT_COLOR

load_dotenv()

from peer_recommender.llm_strategy import call_cohere
from peer_recommender.build_recommendation_prompt import build_recommendation_prompt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# =============================================================================
# Formatting helpers
# =============================================================================

def _fmt_pct(value: float | int | None) -> str: # Format as % with handling for None/NaN
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{value:.1%}"

def extract_json(text):
    if not text or not text.strip():
        raise ValueError("extract_json received empty input")

    text = text.strip()

    # extract ```json ... ``` block if present
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # fallback: sometimes LLM returns ``` ... ``` without "json"
    else:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON after extraction: {text[:200]}") from e

    return json.loads(text)
def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _to_float(value, default=np.nan):
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(val) else float(val)


def fmt_thb(val):
    if val is None or pd.isna(val):
        return "-"
    val = float(val)
    if abs(val) >= 1_000_000:
        return "%.1fM THB" % (val / 1_000_000)
    if abs(val) >= 1_000:
        return "%.0fK THB" % (val / 1_000)
    return "%.0f THB" % val


def fmt_pct(val):
    if val is None or pd.isna(val):
        return "-"
    return "%.1f%%" % (float(val) * 100)


def fmt_num(val, digits=3):
    if val is None or pd.isna(val):
        return "-"
    return f"{float(val):,.{digits}f}"


def fmt_int(val):
    if val is None or pd.isna(val):
        return "-"
    return f"{int(round(float(val))):,}"


def display_value(value):
    if value is None or pd.isna(value):
        return "Not available in data"
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "nan", "none", "-", "<na>"}:
        return "Not available in data"
    return text


METRIC_HIGHLIGHT_TERMS = [
    "View-to-purchase rate",
    "View to Purchase",
    "Add-to-cart rate",
    "Add to Cart",
    "Revenue per view",
    "Revenue Uplift %",
    "Revenue uplift",
    "Bookings Uplift %",
    "Bookings uplift",
    "Booking uplift",
    "Priority Score",
    "Priority Tier",
    "Google Ads Strategy Score",
    "GMV/GA view",
    "GMV / GA View",
    "GMV per GA view",
    "GMV/GA",
    "GA4",
    "items viewed",
    "conversion",
    "bookings",
    "GMV",
]


def highlight_metrics_html(text: str) -> str:
    escaped_text = escape("" if text is None else str(text))
    if not escaped_text:
        return escaped_text

    terms = sorted({escape(term) for term in METRIC_HIGHLIGHT_TERMS}, key=len, reverse=True)
    pattern = re.compile(r"(?<![\w/])(" + "|".join(re.escape(term) for term in terms) + r")(?![\w/])", re.IGNORECASE)

    def repl(match: re.Match) -> str:
        term = match.group(0)
        return (
            "<span style='display:inline-block;background:#fff3cd;color:#7a5200;"
            "border:1px solid #f2d37b;border-radius:6px;padding:0.02rem 0.34rem;"
            "font-weight:800;line-height:1.35;white-space:nowrap;'>%s</span>" % term
        )

    return pattern.sub(repl, escaped_text)


def highlighted_bullets(items: list[str]) -> str:
    return "".join(f"<li>{highlight_metrics_html(item)}</li>" for item in items)


def factual_segment(row: dict, hist: pd.DataFrame | None = None):
    candidates = [row.get("latest_segment"), row.get("segment")]
    if hist is not None and len(hist):
        latest_hist = hist.sort_values("year_month").iloc[-1]
        candidates.extend([latest_hist.get("latest_segment"), latest_hist.get("segment")])

    for value in candidates:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"unknown", "nan", "none", "-", "<na>"}:
            return text
    return None


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    valid = values.notna() & weights.gt(0)
    if valid.any():
        return float(np.average(values[valid], weights=weights[valid]))
    fallback = values.mean()
    return float(fallback) if pd.notna(fallback) else np.nan


# =============================================================================
# Formula display
# =============================================================================

def formula_card(title: str, formulas: list[str]) -> None:
    formula_html = "".join(
        [
            f"<div style='font-family:monospace;font-size:0.9rem;color:{TEXT_COLOR};"
            f"background:#f8fafc;border:1px solid {BORDER_COLOR};border-radius:8px;"
            f"padding:0.55rem 0.75rem;margin-top:0.45rem;'>{formula}</div>"
            for formula in formulas
        ]
    )
    st.markdown(
        f"""
        <div style='background:{SURFACE_COLOR};border:1px solid {BORDER_COLOR};
                    border-radius:10px;padding:0.9rem 1rem;margin-bottom:1rem;'>
            <div style='font-size:0.95rem;font-weight:700;color:{TEXT_COLOR};'>{title}</div>
            {formula_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# GA ranking logic
# =============================================================================

def _prepare_ga_inputs(
    ga_monthly: pd.DataFrame,
    campaign_monthly: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean and merge GA restaurant-month data with cluster/segment labels."""
    ga = ga_monthly.copy()
    campaigns = campaign_monthly.copy()

    if ga.empty or campaigns.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "year_month" in ga.columns:
        ga["year_month"] = pd.to_datetime(ga["year_month"], errors="coerce")
    if "year_month" in campaigns.columns:
        campaigns["year_month"] = pd.to_datetime(campaigns["year_month"], errors="coerce")

    if "name_norm" not in ga.columns and "name" in ga.columns:
        ga["name_norm"] = ga["name"].apply(_normalize_name)

    if not assignments.empty:
        a = assignments.copy()
        if "name_norm" not in a.columns and "name" in a.columns:
            a["name_norm"] = a["name"].apply(_normalize_name)
        merge_cols = [c for c in ["name_norm", "cluster_id", "cluster_label", "latest_segment"] if c in a.columns]
        if "name_norm" in ga.columns and merge_cols:
            ga = ga.drop(columns=[c for c in ["cluster_id", "cluster_label", "latest_segment"] if c in ga.columns], errors="ignore")
            ga = ga.merge(a[merge_cols].drop_duplicates("name_norm"), on="name_norm", how="left")

    required_ga_cols = ["year_month", "monthly_gmv", "ga_items_viewed", "ga_items_added_to_cart", "ga_items_purchased"]
    required_campaign_cols = ["year_month", "googleAdsCampaignType", "total_sessions"]
    if not all(c in ga.columns for c in required_ga_cols) or not all(c in campaigns.columns for c in required_campaign_cols):
        return pd.DataFrame(), pd.DataFrame()

    for col in ["monthly_gmv", "ga_items_viewed", "ga_items_added_to_cart", "ga_items_purchased"]:
        ga[col] = pd.to_numeric(ga[col], errors="coerce")
    campaigns["total_sessions"] = pd.to_numeric(campaigns["total_sessions"], errors="coerce").fillna(0)
    campaigns["googleAdsCampaignType"] = campaigns["googleAdsCampaignType"].fillna("Unknown").astype(str).str.strip()
    campaigns = campaigns[
        campaigns["googleAdsCampaignType"].ne("")
        & campaigns["googleAdsCampaignType"].ne("(not set)")
        & campaigns["googleAdsCampaignType"].ne("Unknown")
    ].copy()

    return ga, campaigns


def build_ga_scope_rankings(
    ga_monthly: pd.DataFrame,
    campaign_monthly: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build Google Ads campaign type ranking inputs for cluster, segment, and global scopes.

    Correct Count logic:
    - Google Ads campaign sessions are platform-month level, not restaurant-level.
    - So for cluster/segment, we estimate the scope's campaign sessions by allocating
      platform campaign sessions using the scope's share of GA item views in that month.

    Estimated Sessions for one scope-month-campaign type:
        campaign_total_sessions_for_month_type
        × (scope_ga_item_views_that_month / global_ga_item_views_that_month)

    Why this is better:
    - Cluster, segment, and global no longer reuse the same platform session total.
    - Counts stay in the original column meaning: estimated sessions.
    - We do not add misleading extra columns.
    """
    empty = {"cluster": pd.DataFrame(), "segment": pd.DataFrame(), "global": pd.DataFrame()}
    ga, campaigns = _prepare_ga_inputs(ga_monthly, campaign_monthly, assignments)
    if ga.empty or campaigns.empty:
        return empty

    # Global monthly GA views are the denominator used to allocate platform campaign sessions.
    global_monthly_views = (
        ga.groupby("year_month", as_index=False)["ga_items_viewed"]
        .sum()
        .rename(columns={"ga_items_viewed": "global_ga_items_viewed"})
    )

    def _scope_rollup(scope_cols: list[str], scope_type: str) -> pd.DataFrame:
        needed_cols = [c for c in scope_cols if c in ga.columns]
        group_cols = needed_cols + ["year_month"]
        if "year_month" not in group_cols:
            return pd.DataFrame()

        monthly = (
            ga.groupby(group_cols, dropna=False, as_index=False)
            .agg(
                total_monthly_gmv=("monthly_gmv", "sum"),
                total_ga_items_viewed=("ga_items_viewed", "sum"),
                total_ga_items_added_to_cart=("ga_items_added_to_cart", "sum"),
                total_ga_items_purchased=("ga_items_purchased", "sum"),
            )
        )
        monthly = monthly.merge(global_monthly_views, on="year_month", how="left")
        monthly["scope_view_share"] = monthly["total_ga_items_viewed"] / monthly["global_ga_items_viewed"].replace(0, np.nan)
        monthly["gmv_per_ga_view"] = monthly["total_monthly_gmv"] / monthly["total_ga_items_viewed"].replace(0, np.nan)
        monthly["ga_add_to_cart_rate"] = monthly["total_ga_items_added_to_cart"] / monthly["total_ga_items_viewed"].replace(0, np.nan)
        monthly["ga_view_to_purchase_rate"] = monthly["total_ga_items_purchased"] / monthly["total_ga_items_viewed"].replace(0, np.nan)

        merged = monthly.merge(campaigns, on="year_month", how="inner")
        if merged.empty:
            return pd.DataFrame()

        merged["estimated_scope_sessions"] = (
            pd.to_numeric(merged["total_sessions"], errors="coerce").fillna(0)
            * pd.to_numeric(merged["scope_view_share"], errors="coerce").fillna(0)
        )

        output_rows = []
        group_keys = needed_cols + ["googleAdsCampaignType"]
        for keys, grp in merged.groupby(group_keys, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_keys, keys))
            row["scope_type"] = scope_type
            row["estimated_sessions"] = float(pd.to_numeric(grp["estimated_scope_sessions"], errors="coerce").sum())

            # Count unique restaurants in this exact scope during the months where the campaign type appeared.
            months = set(pd.to_datetime(grp["year_month"], errors="coerce").dropna())
            rest_df = ga[ga["year_month"].isin(months)].copy()
            for col in needed_cols:
                rest_df = rest_df[rest_df[col].astype(str) == str(row.get(col))]
            row["restaurants"] = int(rest_df["name_norm"].nunique()) if "name_norm" in rest_df.columns else 0

            weights = grp["estimated_scope_sessions"]
            # Fallback to GA views if estimated sessions are zero for some reason.
            if pd.to_numeric(weights, errors="coerce").fillna(0).sum() <= 0:
                weights = grp["total_ga_items_viewed"]

            row["weighted_gmv_per_ga_view"] = _weighted_mean(grp["gmv_per_ga_view"], weights)
            row["weighted_add_to_cart_rate"] = _weighted_mean(grp["ga_add_to_cart_rate"], weights)
            row["weighted_view_to_purchase_rate"] = _weighted_mean(grp["ga_view_to_purchase_rate"], weights)
            output_rows.append(row)

        return pd.DataFrame(output_rows)

    return {
        "cluster": _scope_rollup(["cluster_id", "cluster_label"], "cluster"),
        "segment": _scope_rollup(["latest_segment"], "segment"),
        "global": _scope_rollup([], "global"),
    }

def build_ga_rank_table(scope_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks Google Ads campaign types using:
    Google Ads Strategy Score = 0.40 x GMV/GA + 0.30 x Add to Cart + 0.30 x View to Purchase.
    """
    if scope_df.empty or "googleAdsCampaignType" not in scope_df.columns:
        return pd.DataFrame()

    df = scope_df.copy()
    df["googleAdsCampaignType"] = df["googleAdsCampaignType"].fillna("Unknown").astype(str).str.strip()
    df = df[(df["googleAdsCampaignType"] != "") & (df["googleAdsCampaignType"] != "Unknown")]
    if df.empty:
        return pd.DataFrame()

    metric_cols = [
        "weighted_gmv_per_ga_view",
        "weighted_add_to_cart_rate",
        "weighted_view_to_purchase_rate",
    ]
    if not all(c in df.columns for c in metric_cols):
        return pd.DataFrame()

    grouped = (
        df.groupby("googleAdsCampaignType", as_index=False)
        .agg(
            gmv_per_ga=("weighted_gmv_per_ga_view", "mean"),
            add_to_cart=("weighted_add_to_cart_rate", "mean"),
            view_to_purchase=("weighted_view_to_purchase_rate", "mean"),
        )
        .rename(columns={"googleAdsCampaignType": "strategy"})
    )

    for col in ["gmv_per_ga", "add_to_cart", "view_to_purchase"]:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce")

    max_gmv = grouped["gmv_per_ga"].max(skipna=True)
    grouped["gmv_per_ga_norm"] = np.where(
        pd.notna(max_gmv) & (max_gmv > 0),
        grouped["gmv_per_ga"] / max_gmv,
        0,
    )

    grouped["ga_strategy_score"] = (
        grouped["gmv_per_ga_norm"].fillna(0) * 0.40
        + grouped["add_to_cart"].fillna(0) * 0.30
        + grouped["view_to_purchase"].fillna(0) * 0.30
    )

    grouped = grouped.sort_values("ga_strategy_score", ascending=False).reset_index(drop=True)
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    return grouped[["rank", "strategy", "gmv_per_ga", "add_to_cart", "view_to_purchase", "ga_strategy_score"]]

def display_ga_rank_table(title: str, table: pd.DataFrame) -> None:
    st.markdown(f"#### {title}")
    if table.empty:
        st.info("No Google Ads strategy data available for this scope.")
        return

    out = table.copy().rename(
        columns={
            "rank": "Rank",
            "strategy": "Activity",
            "gmv_per_ga": "GMV/GA",
            "add_to_cart": "Add to Cart",
            "view_to_purchase": "View to Purchase",
            "ga_strategy_score": "Google Ads Strategy Score",
        }
    )
    out["GMV/GA"] = out["GMV/GA"].apply(fmt_thb)
    out["Add to Cart"] = out["Add to Cart"].apply(fmt_pct)
    out["View to Purchase"] = out["View to Purchase"].apply(fmt_pct)
    out["Google Ads Strategy Score"] = out["Google Ads Strategy Score"].apply(lambda v: fmt_num(v, 3))

    st.dataframe(out, hide_index=True, width="stretch", height=min(320, 72 + len(out) * 36))


# =============================================================================
# CRM/KOL/FB ranking logic
# =============================================================================

def build_marketing_rank_table(scope_df: pd.DataFrame) -> pd.DataFrame:
    """Ranks CRM/KOL/FB strategies from raw campaign outcomes."""
    if scope_df.empty:
        return pd.DataFrame()

    df = scope_df.copy()
    strategy_col = "strategy_family" if "strategy_family" in df.columns else "strategy_name"
    required = [strategy_col, "revenue_uplift_pct", "bookings_uplift_pct"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df[strategy_col] = df[strategy_col].fillna("Unknown").astype(str).str.strip()
    df = df[(df[strategy_col] != "") & (df[strategy_col].str.lower() != "unknown")]
    if df.empty:
        return pd.DataFrame()

    df["revenue_uplift_pct"] = pd.to_numeric(df["revenue_uplift_pct"], errors="coerce")
    df["bookings_uplift_pct"] = pd.to_numeric(df["bookings_uplift_pct"], errors="coerce")
    df = df[
        df["revenue_uplift_pct"].notna()
        & df["bookings_uplift_pct"].notna()
        & df["revenue_uplift_pct"].ge(0)
        & df["bookings_uplift_pct"].ge(0)
    ].copy()
    if df.empty:
        return pd.DataFrame()

    count_col = "activity_id" if "activity_id" in df.columns else strategy_col
    restaurant_col = "restaurant_name" if "restaurant_name" in df.columns else None

    agg_dict = {
        "count": (count_col, "nunique" if count_col == "activity_id" else "size"),
        "avg_revenue_uplift_pct": ("revenue_uplift_pct", "mean"),
        "avg_bookings_uplift_pct": ("bookings_uplift_pct", "mean"),
    }
    if restaurant_col:
        agg_dict["restaurants"] = (restaurant_col, "nunique")

    grouped = df.groupby(strategy_col, as_index=False).agg(**agg_dict)
    grouped = grouped.rename(columns={strategy_col: "strategy_name"})

    grouped["marketing_strategy_score"] = (
        grouped["avg_revenue_uplift_pct"].fillna(0) * 0.60
        + grouped["avg_bookings_uplift_pct"].fillna(0) * 0.40
    )

    grouped = grouped.sort_values(["marketing_strategy_score", "count"], ascending=[False, False]).reset_index(drop=True)
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    return grouped


def display_marketing_rank_table(title: str, table: pd.DataFrame) -> None:
    st.markdown(f"#### {title}")
    st.caption(
        "Revenue uplift and booking uplift are average percentage changes versus the campaign baseline. "
        "Only strategies with nonnegative revenue and booking uplift are ranked."
    )
    with st.expander("How score is calculated", expanded=False):
        st.markdown(
            """
            `Strategy Score = (Revenue uplift % × 0.60) + (Booking uplift % × 0.40)`

            The table first removes campaigns with negative revenue or booking uplift, then averages the remaining
            uplift values by activity. Higher score means the strategy has historically improved both GMV and bookings.
            """
        )
    if table.empty:
        st.info("No CRM/KOL/FB strategies with nonnegative revenue and booking uplift are available for this scope.")
        return

    display_cols = [
        "rank",
        "strategy_name",
        "count",
        "avg_revenue_uplift_pct",
        "avg_bookings_uplift_pct",
        "marketing_strategy_score",
    ]
    out = table[display_cols].copy().rename(
        columns={
            "rank": "Rank",
            "strategy_name": "Activity",
            "count": "Count",
            "avg_revenue_uplift_pct": "Revenue Uplift %",
            "avg_bookings_uplift_pct": "Booking Uplift %",
            "marketing_strategy_score": "Strategy Score",
        }
    )
    out["Revenue Uplift %"] = out["Revenue Uplift %"].apply(fmt_pct)
    out["Booking Uplift %"] = out["Booking Uplift %"].apply(fmt_pct)
    out["Strategy Score"] = out["Strategy Score"].apply(lambda v: fmt_num(v, 3))

    st.dataframe(out, hide_index=True, width="stretch", height=min(320, 72 + len(out) * 36))


# =============================================================================
# Strategy dashboard cards
# =============================================================================

def _badge_html(number: int) -> str:
    return (
        "<span style='display:inline-flex;align-items:center;justify-content:center;"
        "width:1.95rem;height:1.95rem;border-radius:999px;background:#cc0000;color:#fff;"
        "font-weight:800;font-size:0.95rem;margin-right:0.75rem;flex:0 0 auto;'>%d</span>" % number
    )


def _strategy_box_header(number: int, title: str, subtitle: str) -> None:
    st.markdown(
        dedent(f"""
        <div style='display:flex;align-items:flex-start;margin-bottom:1rem;'>
            {_badge_html(number)}
            <div>
                <div style='font-size:1.18rem;font-weight:800;color:{TEXT_COLOR};line-height:1.2;'>
                    {escape(title)}
                </div>
                <div style='font-size:0.92rem;color:{MUTED_TEXT};margin-top:0.2rem;line-height:1.35;'>
                    {escape(subtitle)}
                </div>
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def render_placeholder_list_box(items: list[tuple[str, str]], accent: str = "#cc0000") -> None:
    rows = []
    for label, body in items:
        rows.append(
            dedent(f"""
            <div style='border:1px solid {BORDER_COLOR};border-left:4px solid {accent};
                        border-radius:10px;padding:0.95rem 1rem;background:#fff;min-height:8rem;'>
                <div style='font-size:0.78rem;font-weight:850;color:{accent};
                            text-transform:uppercase;letter-spacing:0.02em;'>{escape(label)}</div>
                <div style='font-size:0.95rem;color:{TEXT_COLOR};line-height:1.45;margin-top:0.35rem;'>
                    {highlight_metrics_html(body)}
                </div>
            </div>
            """).strip()
        )
    st.markdown(
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:0.9rem;'>"
        + "".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_placeholder_section(
    number: int,
    title: str,
    subtitle: str,
    items: list[tuple[str, str]],
    accent: str = "#cc0000",
) -> None:
    rows = []
    for label, body in items:
        rows.append(
            dedent(f"""
            <div style='border:1px solid {BORDER_COLOR};border-left:4px solid {accent};
                        border-radius:10px;padding:1.05rem 1.1rem;background:#fff;
                        min-height:7.2rem;display:flex;flex-direction:column;'>
                <div style='font-size:0.76rem;font-weight:800;color:{accent};
                            text-transform:uppercase;letter-spacing:0.03em;margin-bottom:0.55rem;'>
                    {escape(label)}
                </div>
                <div style='font-size:0.94rem;color:{TEXT_COLOR};line-height:1.5;'>
                    {highlight_metrics_html(body)}
                </div>
            </div>
            """).strip()
        )

    st.markdown(
        dedent(f"""
        <section style='background:{SURFACE_COLOR};border:1px solid {BORDER_COLOR};border-radius:12px;
                        padding:1.25rem 1.35rem 1.35rem;margin:0 0 1.15rem 0;
                        box-shadow:0 8px 20px rgba(15,23,42,0.035);'>
            <div style='display:flex;align-items:flex-start;margin-bottom:1.15rem;'>
                {_badge_html(number)}
                <div>
                    <div style='font-size:1.18rem;font-weight:800;color:{TEXT_COLOR};line-height:1.2;'>
                        {escape(title)}
                    </div>
                    <div style='font-size:0.92rem;color:{MUTED_TEXT};margin-top:0.24rem;line-height:1.35;'>
                        {escape(subtitle)}
                    </div>
                </div>
            </div>
            <div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;align-items:stretch;'>
                {''.join(rows)}
            </div>
        </section>
        """).strip(),
        unsafe_allow_html=True,
    )


def _render_rank_html_table(table: pd.DataFrame) -> None:
    if table.empty:
        st.info("No CRM/KOL/FB strategies with nonnegative revenue and booking uplift are available for this scope.")
        return

    with st.expander("How score is calculated", expanded=False):
        st.markdown(
            """
            `Score = (Revenue uplift % × 0.60) + (Booking uplift % × 0.40)`

            Campaigns with negative revenue or booking uplift are filtered out before ranking.
            """
        )

    rows = []
    for _, rec in table.head(8).iterrows():
        rank = int(rec.get("rank", 0))
        rank_html = (
            f"<span style='display:inline-flex;align-items:center;justify-content:center;"
            f"width:1.8rem;height:1.8rem;border-radius:999px;"
            f"background:{'#d90000' if rank <= 2 else '#f7f7f2'};"
            f"color:{'#fff' if rank <= 2 else TEXT_COLOR};font-weight:850;'>{rank}</span>"
        )
        rows.append(
            dedent(f"""
            <tr style='border-bottom:1px solid {BORDER_COLOR};'>
                <td style='padding:0.8rem 1rem;'>{rank_html}</td>
                <td style='padding:0.8rem 1rem;font-weight:{850 if rank <= 2 else 650};'>{escape(str(rec.get("strategy_name", "-")))}</td>
                <td style='padding:0.8rem 1rem;'>{fmt_int(rec.get("count"))}</td>
                <td style='padding:0.8rem 1rem;'>{fmt_pct(rec.get("avg_revenue_uplift_pct"))}</td>
                <td style='padding:0.8rem 1rem;'>{fmt_pct(rec.get("avg_bookings_uplift_pct"))}</td>
                <td style='padding:0.8rem 1rem;'>{fmt_num(rec.get("marketing_strategy_score"), 3)}</td>
            </tr>
            """).strip()
        )

    st.markdown(
        dedent(f"""
        <div style='overflow-x:auto;'>
            <table style='width:100%;border-collapse:collapse;font-size:0.95rem;color:{TEXT_COLOR};'>
                <thead>
                    <tr style='border-bottom:1px solid {BORDER_COLOR};color:#3f3f46;text-align:left;background:#f8f9fa;'>
                        <th style='padding:0.8rem 1rem;'>Rank</th>
                        <th style='padding:0.8rem 1rem;'>Activity</th>
                        <th style='padding:0.8rem 1rem;'>Count</th>
                        <th style='padding:0.8rem 1rem;'>Revenue uplift %</th>
                        <th style='padding:0.8rem 1rem;'>Booking uplift %</th>
                        <th style='padding:0.8rem 1rem;'>Score</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def _latest_restaurant_ga(selected: str, ga_monthly: pd.DataFrame) -> dict:
    if ga_monthly.empty:
        return {}
    ga = ga_monthly.copy()
    if "name_norm" not in ga.columns and "name" in ga.columns:
        ga["name_norm"] = ga["name"].apply(_normalize_name)
    if "year_month" in ga.columns:
        ga["year_month"] = pd.to_datetime(ga["year_month"], errors="coerce")
    selected_rows = ga[ga.get("name_norm", pd.Series(dtype=str)).eq(_normalize_name(selected))].copy()
    if selected_rows.empty:
        return {}
    return selected_rows.sort_values("year_month").iloc[-1].to_dict()


def build_ga_snapshot_cards(
    selected: str,
    ga_monthly: pd.DataFrame,
    assignments: pd.DataFrame,
    cluster_id,
) -> list[dict]:
    if ga_monthly.empty:
        return []

    ga = ga_monthly.copy()
    if "name_norm" not in ga.columns and "name" in ga.columns:
        ga["name_norm"] = ga["name"].apply(_normalize_name)
    if "year_month" in ga.columns:
        ga["year_month"] = pd.to_datetime(ga["year_month"], errors="coerce")

    if "cluster_id" not in ga.columns and not assignments.empty:
        a = assignments.copy()
        if "name_norm" not in a.columns and "name" in a.columns:
            a["name_norm"] = a["name"].apply(_normalize_name)
        merge_cols = [c for c in ["name_norm", "cluster_id"] if c in a.columns]
        if merge_cols:
            ga = ga.merge(a[merge_cols].drop_duplicates("name_norm"), on="name_norm", how="left")

    latest = _latest_restaurant_ga(selected, ga)
    if not latest:
        return []

    latest_month = latest.get("year_month")
    cluster_scope = ga.copy()
    if pd.notna(latest_month) and "year_month" in cluster_scope.columns:
        cluster_scope = cluster_scope[cluster_scope["year_month"].eq(latest_month)].copy()
    if cluster_id is not None and "cluster_id" in cluster_scope.columns:
        cluster_scope = cluster_scope[pd.to_numeric(cluster_scope["cluster_id"], errors="coerce").eq(cluster_id)].copy()

    if cluster_scope.empty:
        cluster_scope = ga.copy()

    for col in ["monthly_bookings", "ga_items_viewed", "ga_add_to_cart_rate", "ga_view_to_purchase_rate"]:
        if col in cluster_scope.columns:
            cluster_scope[col] = pd.to_numeric(cluster_scope[col], errors="coerce")

    benchmarks = {
        "monthly_bookings": cluster_scope["monthly_bookings"].mean() if "monthly_bookings" in cluster_scope.columns else np.nan,
        "ga_items_viewed": cluster_scope["ga_items_viewed"].mean() if "ga_items_viewed" in cluster_scope.columns else np.nan,
        "ga_add_to_cart_rate": cluster_scope["ga_add_to_cart_rate"].mean() if "ga_add_to_cart_rate" in cluster_scope.columns else np.nan,
        "ga_view_to_purchase_rate": cluster_scope["ga_view_to_purchase_rate"].mean() if "ga_view_to_purchase_rate" in cluster_scope.columns else np.nan,
    }

    def _status(metric: str, higher_text: str, lower_text: str) -> tuple[str, str]:
        value = _to_float(latest.get(metric))
        benchmark = _to_float(benchmarks.get(metric))
        if pd.isna(value) or pd.isna(benchmark):
            return ("#6b7280", "Benchmark unavailable")
        if value >= benchmark:
            return ("#3f7f2f", higher_text)
        return ("#cc0000", lower_text)

    traffic_color, traffic_status = _status("ga_items_viewed", "Traffic is healthy", "Traffic is below benchmark")
    bookings_color, bookings_status = _status("monthly_bookings", "Booking volume is healthy", "Booking volume is below benchmark")
    atc_color, atc_status = _status("ga_add_to_cart_rate", "Add-to-cart is above benchmark", "below benchmark")
    vtp_color, vtp_status = _status("ga_view_to_purchase_rate", "Purchase conversion is above benchmark", "below benchmark")

    def _delta(metric: str, pct: bool = False) -> str:
        value = _to_float(latest.get(metric))
        benchmark = _to_float(benchmarks.get(metric))
        if pd.isna(value) or pd.isna(benchmark):
            return ""
        diff = value - benchmark
        if pct:
            return f"{diff * 100:+.1f}pp "
        return "above benchmark" if diff >= 0 else "below benchmark"

    return [
        {
            "label": "Items viewed / month",
            "value": fmt_int(latest.get("ga_items_viewed")),
            "benchmark": f"Cluster avg: {fmt_int(benchmarks.get('ga_items_viewed'))} {_delta('ga_items_viewed')}",
            "status": traffic_status,
            "color": traffic_color,
        },
        {
            "label": "Booking volume / month",
            "value": fmt_int(latest.get("monthly_bookings")),
            "benchmark": f"Cluster avg: {fmt_int(benchmarks.get('monthly_bookings'))} {_delta('monthly_bookings')}",
            "status": bookings_status,
            "color": bookings_color,
        },
        {
            "label": "Add-to-cart rate",
            "value": fmt_pct(latest.get("ga_add_to_cart_rate")),
            "benchmark": f"Cluster avg: {fmt_pct(benchmarks.get('ga_add_to_cart_rate'))}",
            "status": f"{_delta('ga_add_to_cart_rate', pct=True)}{atc_status}".strip(),
            "color": atc_color,
        },
        {
            "label": "View-to-purchase rate",
            "value": fmt_pct(latest.get("ga_view_to_purchase_rate")),
            "benchmark": f"Cluster avg: {fmt_pct(benchmarks.get('ga_view_to_purchase_rate'))}",
            "status": f"{_delta('ga_view_to_purchase_rate', pct=True)}{vtp_status}".strip(),
            "color": vtp_color,
        },
    ]


def render_ga_snapshot_cards(cards: list[dict]) -> None:
    if not cards:
        st.info("No restaurant GA snapshot available.")
        return

    st.markdown(_ga_snapshot_cards_html(cards), unsafe_allow_html=True)


def _ga_snapshot_cards_html(cards: list[dict]) -> str:
    card_html = []
    for card in cards:
        border = card["color"] if card["color"] == "#cc0000" else BORDER_COLOR
        card_html.append(
            dedent(f"""
            <div style='background:#fff;border:1px solid {border};border-radius:10px;
                        padding:1rem 1.15rem;min-height:8rem;'>
                <div style='font-size:0.95rem;color:#3f3f46;font-weight:650;'>{escape(card["label"])}</div>
                <div style='font-size:1.55rem;color:{card["color"]};font-weight:850;line-height:1.2;margin-top:0.35rem;'>
                    {escape(card["value"])}
                </div>
                <div style='font-size:0.92rem;color:#3f3f46;margin-top:0.2rem;'>{escape(card["benchmark"])}</div>
                <div style='font-size:0.92rem;color:{card["color"]};font-weight:800;margin-top:0.15rem;'>
                    {escape(card["status"])}
                </div>
            </div>
            """).strip()
        )

    return (
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:0.9rem;'>"
        + "".join(card_html)
        + "</div>"
    )


def render_ga_snapshot_section(number: int, title: str, subtitle: str, cards: list[dict]) -> None:
    body = (
        _ga_snapshot_cards_html(cards)
        if cards
        else f"<div style='color:{MUTED_TEXT};font-size:0.95rem;'>No restaurant GA snapshot available.</div>"
    )
    st.markdown(
        dedent(f"""
        <section style='background:{SURFACE_COLOR};border:1px solid {BORDER_COLOR};border-radius:12px;
                        padding:1.25rem 1.35rem 1.35rem;margin:0 0 1.15rem 0;
                        box-shadow:0 8px 20px rgba(15,23,42,0.035);'>
            <div style='display:flex;align-items:flex-start;margin-bottom:1.15rem;'>
                {_badge_html(number)}
                <div>
                    <div style='font-size:1.18rem;font-weight:800;color:{TEXT_COLOR};line-height:1.2;'>
                        {escape(title)}
                    </div>
                    <div style='font-size:0.92rem;color:{MUTED_TEXT};margin-top:0.24rem;line-height:1.35;'>
                        {escape(subtitle)}
                    </div>
                </div>
            </div>
            {body}
        </section>
        """).strip(),
        unsafe_allow_html=True,
    )


def render_peer_recommender_section(
    selected: str,
    selected_restaurant_row: pd.DataFrame,
    assignments: pd.DataFrame,
    outcomes_df: pd.DataFrame,
) -> None:
    with st.container(border=True):
        st.markdown(
            dedent(f"""
            <div style='display:flex;align-items:flex-start;margin-bottom:1rem;'>
                {_badge_html(5)}
                <div>
                    <div style='font-size:1.18rem;font-weight:800;color:{TEXT_COLOR};line-height:1.2;'>
                        Peer recommender
                    </div>
                    <div style='font-size:0.92rem;color:{MUTED_TEXT};margin-top:0.24rem;line-height:1.35;'>
                        Recommendations are dynamically generated by learning from high-performing peers with identical business models and digital funnel behaviors.
                    </div>
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

        peer_k = st.slider("Number of similar peers to analyze", 3, 20, 4, key="peer_k_slider")
        st.caption(
            "Peers ranked by distance, with priority given to high-momentum segments. "
            "Distance calculated using UMAP coordinates, GMV per GA view, and cluster membership."
        )

        if selected_restaurant_row.empty:
            st.info("No restaurant profile available for peer recommendations.")
            return
        if outcomes_df.empty:
            st.info("No marketing outcomes data available to generate recommendations.")
            return
        if assignments.empty:
            st.info("No cluster assignment data available to generate recommendations.")
            return
        if "restaurant_name" not in outcomes_df.columns:
            st.info("No restaurant name column available to generate recommendations.")
            return

        selected_profile = selected_restaurant_row.copy()
        o = outcomes_df.copy()
        o["_name_norm"] = o["restaurant_name"].apply(_normalize_name)
        restaurants_with_campaigns = set(o["_name_norm"].dropna().unique())

        peer_assignments = assignments.copy()
        if "name_norm" not in peer_assignments.columns and "name" in peer_assignments.columns:
            peer_assignments["name_norm"] = peer_assignments["name"].apply(_normalize_name)
        if "name_norm" not in peer_assignments.columns:
            st.info("Cluster assignments need a normalized restaurant name column.")
            return
        if "latest_segment" not in peer_assignments.columns:
            st.info("Cluster assignments need a latest segment column.")
            return
        if "cluster_id" not in peer_assignments.columns:
            peer_assignments["cluster_id"] = np.nan

        active_rest_norm = _normalize_name(selected)
        HIGH_MOMENTUM = {"Rising Stars", "Established Players"}

        high_momentum_peers = peer_assignments[
            (peer_assignments["name_norm"] != active_rest_norm)
            & (peer_assignments["latest_segment"].isin(HIGH_MOMENTUM))
            & (peer_assignments["name_norm"].isin(restaurants_with_campaigns))
        ].copy()
        high_momentum_peers["momentum_bonus"] = 1.0

        fallback_peers = peer_assignments[
            (peer_assignments["name_norm"] != active_rest_norm)
            & (~peer_assignments["latest_segment"].isin(HIGH_MOMENTUM))
            & (peer_assignments["name_norm"].isin(restaurants_with_campaigns))
        ].copy()
        fallback_peers["momentum_bonus"] = 0.0

        if len(high_momentum_peers) >= peer_k:
            peer_df = high_momentum_peers.copy()
        elif len(high_momentum_peers) > 0:
            slots_needed = peer_k - len(high_momentum_peers)
            peer_df = pd.concat(
                [high_momentum_peers, fallback_peers.head(slots_needed)],
                ignore_index=True,
            )
            st.caption(
                f"Only {len(high_momentum_peers)} high-performing peer(s) found. "
                f"{slots_needed} additional peer(s) included by proximity."
            )
        else:
            peer_df = fallback_peers.copy()
            st.caption(
                "No Rising Stars or Established Players with campaign data found. "
                "Showing closest available peers by distance."
            )

        if peer_df.empty:
            st.warning("No peers with campaign data found for this restaurant.")
            top_peers = pd.DataFrame()
        else:
            distance_features = ["x", "y", "avg_gmv_perview"]
            for col in distance_features:
                if col not in peer_df.columns:
                    peer_df[col] = 0
                if col not in selected_profile.columns:
                    selected_profile[col] = 0

            valid_features = [
                col for col in distance_features
                if peer_df[col].fillna(0).std() > 1e-9
            ]
            if not valid_features:
                valid_features = ["avg_monthly_gmv"]

            scaler = StandardScaler()
            pool_scaled = scaler.fit_transform(peer_df[valid_features].fillna(0))
            active_scaled = scaler.transform(selected_profile[valid_features].fillna(0).values)
            base_distances = euclidean_distances(pool_scaled, active_scaled).flatten()

            active_cluster = selected_profile.iloc[0].get("cluster_id", -1)
            same_cluster = (peer_df["cluster_id"] == active_cluster).astype(float)

            peer_df["distance"] = base_distances - (same_cluster * 0.5) - (peer_df["momentum_bonus"] * 0.5)
            peer_df["distance"] = np.maximum(peer_df["distance"], 0)

            top_peers = peer_df.sort_values("distance").head(peer_k).copy()
            top_peers["sim_w"] = 1 / (top_peers["distance"] + 1e-6)

            peer_dist = top_peers[["name_norm", "sim_w", "distance"]].copy()
            peer_campaigns = o[o["_name_norm"].isin(set(peer_dist["name_norm"]))].copy()
            peer_campaigns = peer_campaigns.merge(peer_dist, left_on="_name_norm", right_on="name_norm", how="left")
            peer_campaigns["revenue_uplift_pct"] = pd.to_numeric(peer_campaigns["revenue_uplift_pct"], errors="coerce")
            peer_campaigns["bookings_uplift_pct"] = pd.to_numeric(peer_campaigns["bookings_uplift_pct"], errors="coerce")
            peer_campaigns = peer_campaigns[
                peer_campaigns["revenue_uplift_pct"].notna()
                & peer_campaigns["bookings_uplift_pct"].notna()
                & peer_campaigns["revenue_uplift_pct"].ge(0)
                & peer_campaigns["bookings_uplift_pct"].ge(0)
            ].copy()

            rec = (
                peer_campaigns.groupby(["strategy_name", "channel"], dropna=False)
                .agg(
                    campaigns=("activity_id", "nunique"),
                    peers_using=("restaurant_name", "nunique"),
                    med_rev_uplift=("revenue_uplift_pct", "median"),
                    med_booking_uplift=("bookings_uplift_pct", "median"),
                    med_roi=("roi", "median"),
                    total_incremental=("incremental_revenue_thb", "sum"),
                    total_sim_weight=("sim_w", "sum"),
                )
                .reset_index()
            )

            min_peers = 2
            rec["is_reliable"] = rec["peers_using"] >= min_peers
            rec["score"] = (
                rec["med_rev_uplift"].fillna(0) * 100
                + rec["med_booking_uplift"].fillna(0) * 60
                + rec["med_roi"].fillna(0) * 50
                + np.log1p(rec["total_sim_weight"]) * 20
                + rec["is_reliable"].astype(int) * 30
            )
            rec = rec.sort_values(["is_reliable", "score"], ascending=[False, False])

            diverse_recs, seen_stages = [], set()
            for _, row in rec.iterrows():
                stage = row.get("funnel_stage", row.get("channel", "Unknown"))
                if stage not in seen_stages or len(diverse_recs) == 0:
                    diverse_recs.append(row)
                    seen_stages.add(stage)
                if len(diverse_recs) >= 3:
                    break

            if len(diverse_recs) < 3:
                remaining = rec[~rec.index.isin([r.name for r in diverse_recs])]
                for _, r in remaining.head(3 - len(diverse_recs)).iterrows():
                    diverse_recs.append(r)

            top_recs = pd.DataFrame(diverse_recs)
            if top_recs.empty:
                top_recs = pd.DataFrame(
                    columns=[
                        "strategy_name",
                        "channel",
                        "med_rev_uplift",
                        "med_booking_uplift",
                        "peers_using",
                        "campaigns",
                    ]
                )

            st.markdown("#### Top Strategies Recommended by Peers")
            st.caption(
                "Percentages show median uplift from similar restaurants' campaign outcomes: "
                "revenue uplift is the GMV change, booking uplift is the booking-count change. "
                "Strategies with negative revenue or booking uplift are excluded."
            )
            with st.expander("How score is calculated", expanded=False):
                st.markdown(
                    """
                    `Peer score = median revenue uplift × 100 + median booking uplift × 60 + median ROI × 50 + similarity weight + reliability bonus`

                    Similarity weight favors closer peer restaurants. Reliability bonus is added when at least two peers used the same strategy.
                    Negative revenue or booking uplift campaigns are removed before this score is calculated.
                    """
                )
            displayable_recs = top_recs[top_recs["med_rev_uplift"].notna()].head(3)
            st.session_state["peer_recs"] = displayable_recs.copy()

            if displayable_recs.empty:
                st.info("No peer-validated strategies with nonnegative revenue and booking uplift. Try increasing peers.")
            else:
                rec_cards = []
                for _, r in displayable_recs.iterrows():
                    uplift = pd.to_numeric(pd.Series([r.get("med_rev_uplift")]), errors="coerce").iloc[0]
                    booking_uplift = pd.to_numeric(pd.Series([r.get("med_booking_uplift")]), errors="coerce").iloc[0]
                    roi_val = pd.to_numeric(pd.Series([r.get("med_roi")]), errors="coerce").iloc[0]
                    # peers_using = int(r.get("peers_using", 0) or 0)
                    # campaigns = int(r.get("campaigns", 0) or 0)
                    total_sim_weight = _to_float(r.get("total_sim_weight"))
                    color = "#3f7f2f" if pd.notna(uplift) and uplift >= 0 else "#cc0000"
                    rec_cards.append(
                        {
                            "label": f"{r.get('strategy_name', 'Unknown')} · {r.get('channel', 'Unknown')}",
                            "value": fmt_pct(uplift),
                            "booking_value": fmt_pct(booking_uplift),
                            # "meta": f"{peers_using} peer{'s' if peers_using != 1 else ''} · {campaigns} campaign{'s' if campaigns != 1 else ''}",
                            "color": color,
                        }
                    )

                st.markdown(
                    "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1.15rem;margin-bottom:1rem;'>"
                    + "".join(
                        dedent(f"""
                        <div style='background:#fff;border:1px solid {card['color']};border-radius:10px;
                                    padding:1rem 1.15rem;min-height:9.25rem;'>
                            <div style='font-size:0.95rem;color:#3f3f46;font-weight:650;'>{escape(card['label'])}</div>
                            <div style='font-size:1.55rem;color:{card['color']};font-weight:850;line-height:1.2;margin-top:0.35rem;'>
                                {escape(card['value'])}
                            </div>
                            <div style='font-size:0.78rem;color:{MUTED_TEXT};margin-top:0.15rem;'>median revenue uplift</div>
                            <div style='font-size:0.88rem;color:#3f3f46;margin-top:0.45rem;'>
                                Booking uplift: <strong>{escape(card['booking_value'])}</strong>
                            </div>
                        </div>
                        """).strip()
                        for card in rec_cards
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

            if not o.empty and "_name_norm" in o.columns:
                peer_names_norm = set(top_peers["name_norm"].dropna())
                o_peers = o[o["_name_norm"].isin(peer_names_norm)].copy()

                if not o_peers.empty:
                    best_ch = (
                        o_peers.groupby(["_name_norm", "channel"])["revenue_uplift_pct"]
                        .median()
                        .reset_index()
                        .sort_values("revenue_uplift_pct", ascending=False)
                        .drop_duplicates("_name_norm")[["_name_norm", "channel"]]
                        .rename(columns={"channel": "best_channel"})
                    )
                    peer_stats = (
                        o_peers.groupby("_name_norm")
                        .agg(
                            n_campaigns=("activity_id", "nunique"),
                            avg_roi=("roi", "median"),
                            revenue_uplift_pct=("revenue_uplift_pct", lambda x : (x * 100).median()),
                            booking_uplift_pct=("bookings_uplift_pct", lambda x : (x * 100).median()),
                            n_positive_lift=("revenue_uplift_pct", lambda x: (x > 0).sum()),
                            n_total=("revenue_uplift_pct", "count"),
                        )
                        .reset_index()
                        .fillna(0)
                    )

                    peer_stats["lift_reliability"] = (
                        peer_stats["n_positive_lift"] / peer_stats["n_total"]
                    ).replace([np.inf, -np.inf, pd.NA], 0)
                    peer_stats = peer_stats.merge(best_ch, on="_name_norm", how="left")
                    top_peers = top_peers.merge(peer_stats, left_on="name_norm", right_on="_name_norm", how="inner")

            with st.expander("🔍 View Raw Peer Evidence (Metadata & Scale)"):
                display_peers = top_peers.copy()

                if "n_campaigns" in display_peers.columns and "avg_incremental_rev" in display_peers.columns:
                    display_peers["is_reliable_peer"] = display_peers["n_campaigns"].fillna(0) > 1
                    if "distance" in display_peers.columns:
                        display_peers = display_peers.sort_values(
                            by=["is_reliable_peer", "distance"],
                            ascending=[False, True],
                        )
                    else:
                        display_peers = display_peers.sort_values(
                            by=["is_reliable_peer"],
                            ascending=[False, False],
                        )

                display_cols = [
                    "name", "best_channel", "n_campaigns", "lift_reliability",
                    "revenue_uplift_pct", "booking_uplift_pct", "latest_segment",
                ]
                existing_cols = [c for c in display_cols if c in display_peers.columns]

                renamed = display_peers[existing_cols].rename(columns={
                    "name": "Peer Restaurant",
                    "best_channel": "Best Channel",
                    "n_campaigns": "Campaigns Run",
                    "lift_reliability": "Lift Reliability",
                    "revenue_uplift_pct": "Median Revenue Uplift",
                    "booking_uplift_pct": "Median Booking Uplift",
                    "latest_segment": "Momentum",
                })

                st.dataframe(
                    renamed.style.format({
                        "Lift Reliability": lambda x: f"{x:.1%}" if pd.notna(x) else "-",
                        "Campaigns Run": lambda x: f"{int(x)}" if pd.notna(x) else "-",
                        "Revenue Uplift": lambda x: f"{x:.1%}" if pd.notna(x) else "-",
                        "Booking Uplift": lambda x: f"{x:.1%}" if pd.notna(x) else "-",
                    }),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    f"Similarity computed based similarity distance calculated during clustering earlier. "
                    "Best Channel and ROI sourced from historical campaign outcomes."
                )

            active_momentum = selected_profile.iloc[0].get("latest_segment", "Unknown")
            segment_advice = {
                "Rising Stars": ("Scale what's working before momentum plateaus. Double down on your best-performing channel.", "🟢"),
                "Emerging Opportunities": ("Conversion efficiency is the priority — get more bookings from existing traffic before increasing spend.", "🟡"),
                "Established Players": ("Focus on retention and upsell. Acquisition costs are high; loyal guests have higher LTV.", "🔵"),
                "Needs Attention": ("Diagnose the funnel before spending on campaigns. Traffic may not be the problem.", "🔴"),
            }.get(active_momentum, ("Insufficient momentum data to generate directional advice.", "⚪"))

            grounded_advice, segment_icon = segment_advice
            top_strat = top_recs.iloc[0] if not top_recs.empty else None
            has_data = top_strat is not None and pd.notna(top_strat.get("med_rev_uplift"))
            reliable_peers = int(top_recs["is_reliable"].sum()) if not top_recs.empty and "is_reliable" in top_recs.columns else 0
            dominant_channel = top_recs["channel"].value_counts().index[0] if not top_recs.empty else "Unknown"

            grounded_lines = [
                f"**{segment_icon} Segment:** {active_momentum} — {grounded_advice}",
                "",
            ]

            if top_strat is not None:
                uplift_str = f"{top_strat['med_rev_uplift']:.1%}" if has_data else "no uplift data recorded"
                grounded_lines += [
                    f"**📌 Priority Campaign:** `{top_strat['strategy_name']}` via **{top_strat['channel']}** "
                    f"— {uplift_str} median uplift across {int(top_strat.get('peers_using', 0))} peer(s).",
                    "",
                    f"**📡 Dominant Channel:** {dominant_channel} appears in {int(top_recs['channel'].value_counts().iloc[0])} of "
                    f"{len(top_recs)} recommended strategies for this peer group.",
                    "",
                    f"**✅ Reliability:** {reliable_peers}/{len(top_recs)} strategies are backed by ≥2 peers with recorded outcomes.",
                ]
            else:
                grounded_lines.append("_No peer-validated strategies available. Try increasing the number of peers._")

            # grounded_brief = "\n".join(grounded_lines)

            # st.markdown("---")
            # st.markdown("#### 📋 Strategy Brief")
            # st.caption("Deterministic summary based on peer data — always available regardless of AI status.")
            # with st.container(border=True):
            #     st.markdown(grounded_brief)

            # st.download_button(
            #     label="⬇️ Download Strategy Brief",
            #     data="PEER STRATEGY BRIEF\n%s\n\n%s" % (
            #         selected_profile.iloc[0].get("name", "Unknown"),
            #         grounded_brief.replace("**", "").replace("`", ""),
            #     ),
            #     file_name="strategy_brief_%s.txt" % selected_profile.iloc[0].get("name", "Unknown").replace(" ", "_"),
            #     mime="text/plain",
            # )


def _filter_marketing_for_scope(
    outcomes_df: pd.DataFrame,
    cluster_id,
    selected_segment: str | None,
    scope: str,
) -> pd.DataFrame:
    if outcomes_df.empty:
        return pd.DataFrame()

    df = outcomes_df.copy()
    if scope in {"Cluster", "Segment"}:
        if cluster_id is None or "cluster_id" not in df.columns:
            return pd.DataFrame()
        df = df[pd.to_numeric(df["cluster_id"], errors="coerce").eq(cluster_id)].copy()
    if scope == "Segment":
        if not selected_segment or "latest_segment" not in df.columns:
            return pd.DataFrame()
        df = df[df["latest_segment"].astype(str).eq(str(selected_segment))].copy()
    return df


def _filter_marketing_for_restaurant(outcomes_df: pd.DataFrame, selected: str) -> pd.DataFrame:
    if outcomes_df.empty:
        return pd.DataFrame()
    name_col = "restaurant_name" if "restaurant_name" in outcomes_df.columns else "name"
    if name_col not in outcomes_df.columns:
        return pd.DataFrame()

    df = outcomes_df.copy()
    return df[df[name_col].fillna("").astype(str).apply(_normalize_name).eq(_normalize_name(selected))].copy()


# =============================================================================
# Scope builders
# =============================================================================

def get_selected_context(selected: str, priority_df: pd.DataFrame, momentum_df: pd.DataFrame, assignments: pd.DataFrame):
    row = get_restaurant_priority_row(priority_df, selected)
    hist = get_restaurant_history(momentum_df, selected)
    segment = factual_segment(row, hist)

    selected_norm = _normalize_name(selected)
    assignment_row = pd.DataFrame()
    if not assignments.empty:
        assignments = assignments.copy()
        if "name_norm" not in assignments.columns and "name" in assignments.columns:
            assignments["name_norm"] = assignments["name"].apply(_normalize_name)
        assignment_row = assignments[assignments["name_norm"] == selected_norm].head(1)

    cluster_id = None
    cluster_label = None
    if len(assignment_row):
        cluster_id_raw = pd.to_numeric(pd.Series([assignment_row.iloc[0].get("cluster_id")]), errors="coerce").iloc[0]
        if pd.notna(cluster_id_raw):
            cluster_id = int(cluster_id_raw)
        cluster_label = assignment_row.iloc[0].get("cluster_label")

    return row, hist, segment, cluster_id, cluster_label, assignment_row


def filter_ga_scopes(ga_rankings: dict[str, pd.DataFrame], cluster_id, segment: str | None):
    cluster_ga = ga_rankings.get("cluster", pd.DataFrame()).copy()
    if cluster_id is not None and not cluster_ga.empty and "cluster_id" in cluster_ga.columns:
        cluster_ga = cluster_ga[pd.to_numeric(cluster_ga["cluster_id"], errors="coerce") == cluster_id].copy()
    else:
        cluster_ga = pd.DataFrame()

    segment_ga = ga_rankings.get("segment", pd.DataFrame()).copy()
    if segment and not segment_ga.empty and "latest_segment" in segment_ga.columns:
        segment_ga = segment_ga[segment_ga["latest_segment"].astype(str) == str(segment)].copy()
    else:
        segment_ga = pd.DataFrame()

    global_ga = ga_rankings.get("global", pd.DataFrame()).copy()
    return cluster_ga, segment_ga, global_ga


def filter_marketing_scopes(outcomes_df: pd.DataFrame, cluster_id, segment: str | None):
    if outcomes_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = outcomes_df.copy()

    cluster_df = df.copy()
    if cluster_id is not None and "cluster_id" in cluster_df.columns:
        cluster_df = cluster_df[pd.to_numeric(cluster_df["cluster_id"], errors="coerce") == cluster_id].copy()
    else:
        cluster_df = pd.DataFrame()

    segment_df = df.copy()
    if segment and "latest_segment" in segment_df.columns:
        segment_df = segment_df[segment_df["latest_segment"].astype(str) == str(segment)].copy()
    else:
        segment_df = pd.DataFrame()

    global_df = df.copy()
    return cluster_df, segment_df, global_df

def enrich_outcomes_with_rank_stats(
    outcomes_df: pd.DataFrame,
    rank_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges strategy-level aggregates from build_marketing_rank_table() back onto
    the row-level outcomes_df, then calculates lift_reliability per strategy.

    Adds three columns to outcomes_df:
    - avg_revenue_uplift_pct  : mean revenue uplift for that strategy (from rank table)
    - avg_bookings_uplift_pct : mean bookings uplift for that strategy (from rank table)
    - lift_reliability        : fraction of campaigns for that strategy with positive revenue uplift
    """
    if outcomes_df.empty or rank_table.empty:
        return outcomes_df

    df = outcomes_df.copy()

    # Determine which column in outcomes_df maps to strategy_name in the rank table
    strategy_col = "strategy_family" if "strategy_family" in df.columns else "strategy_name"

    if strategy_col not in df.columns or "strategy_name" not in rank_table.columns:
        return df

    # Step 1: bring in avg_revenue_uplift_pct and avg_bookings_uplift_pct from rank table
    cols_to_merge = [
        c for c in ["strategy_name", "avg_revenue_uplift_pct", "avg_bookings_uplift_pct"]
        if c in rank_table.columns
    ]
    df = df.merge(
        rank_table[cols_to_merge],
        left_on=strategy_col,
        right_on="strategy_name",
        how="left",
        suffixes=("", "_ranked"),
    )
    # Drop the duplicate strategy_name column if strategy_col != "strategy_name"
    if strategy_col != "strategy_name" and "strategy_name_ranked" in df.columns:
        df = df.drop(columns=["strategy_name_ranked"])

    # Step 2: calculate lift_reliability per strategy
    # = fraction of campaigns where revenue_uplift_pct > 0
    if "revenue_uplift_pct" in df.columns:
        df["revenue_uplift_pct_num"] = pd.to_numeric(df["revenue_uplift_pct"], errors="coerce")

        reliability = (
            df.groupby(strategy_col)["revenue_uplift_pct_num"]
            .agg(
                _n_positive=lambda x: (x > 0).sum(),
                _n_total=lambda x: x.notna().sum(),
            )
            .reset_index()
        )
        reliability["lift_reliability"] = (
            reliability["_n_positive"] / reliability["_n_total"].replace(0, pd.NA)
        )
        reliability = reliability[[strategy_col, "lift_reliability"]]

        df = df.merge(reliability, on=strategy_col, how="left")
        df = df.drop(columns=["revenue_uplift_pct_num"])

    return df

# Keep the session state clean when the user selects a new restaurant.
def on_restaurant_change():
    """Callback to reset dependent state when the main selection changes."""
    # When a new restaurant is selected, we must clear the old AI narrative.
    ai_key = f"ai_strategy_{st.session_state.get('strategy_restaurant_selector')}"
    if ai_key in st.session_state:
        del st.session_state[ai_key]
    
    # Update the session state with the new selection from the dropdown
    st.session_state["strategy_restaurant"] = st.session_state.strategy_restaurant_selector


# =============================================================================
# Package details frontend
# =============================================================================

def render_package_details() -> None:
    """Show exact package capabilities above the AI narrative."""
    st.markdown("### Package Details")

    with st.expander("BASIC PACKAGE", expanded=False):
        st.markdown("""
**Key Capabilities:**
- Revenue Guarantee (30K+ THB)
- Revenue Guarantee (90 Day)
- Send Blogger to Review x1 (20k followers)
- Send Blogger to Review x2 (30k followers)
- Boost post THB 2,000 Baht
- Pop-up Banner: Individual
- Photoshooting
- Guaranteed in Restaurants list home page
- HH Facebook Post : Individual post
- Line@ Broadcasts : Individual post
- Push Notification
""")

    with st.expander("STANDARD PACKAGE", expanded=False):
        st.markdown("""
**Key Capabilities:**
- (+) Push Notification
- (+) Web Footer (2 days)
- (+) Tiktok VDO or Instagram Reels VDO
""")

    with st.expander("PREMIUM PACKAGE", expanded=False):
        st.markdown("""
**Key Capabilities:**
- (+) Guaranteed in Restaurants Promotion Banner (1 week)
- (+) Blog: Advertorial: Individual
""")


def render_ai_diagnosis_explainer() -> None:
    with st.expander("How AI Strategy Diagnosis Is Generated", expanded=False):
        st.markdown(
            dedent(f"""
            <div style='font-size:0.95rem;color:{TEXT_COLOR};line-height:1.6;'>
                <p><b>What the AI does:</b> {highlight_metrics_html(
                    "When you click Generate AI Narrative, the dashboard calls Cohere command-a-03-2025 to turn the selected restaurant's data into two structured outputs: Key issues Identified and Recommended strategy. The response is required to come back as JSON, which is then rendered into the diagnosis cards."
                )}</p>
                <p style='margin-top:0.85rem;'><b>Data passed into the AI prompt:</b></p>
                <ul style='margin-top:0.35rem;'>
                    {highlighted_bullets([
                        "Selected restaurant identity: restaurant name, cluster ID, cluster label, latest segment, Priority Score, and Priority Tier.",
                        "Restaurant funnel snapshot: latest available Google Ads metrics such as items viewed, GMV/GA view, Add-to-cart rate, View-to-purchase rate, and revenue per view.",
                        "Restaurant booking / momentum history: the raw momentum dataframe is passed in so the model can refer to booking trajectory and performance signals.",
                        "Google Ads evidence: the Google Ads ranking object generated from cluster, segment, and global scopes is passed in so the model can connect funnel gaps to campaign-type evidence.",
                        "Package rules: Basic, Standard, and Premium package capabilities are embedded directly in the prompt so the model can match package choice to the funnel stage being addressed.",
                    ])}
                </ul>
                <p style='margin-top:0.85rem;'><b>Guardrails used in the prompt:</b> {highlight_metrics_html(
                    "The model is instructed to use only the supplied data, avoid inventing metrics, mark missing values as insufficient data, rank issues by business impact, and connect every recommendation back to the evidence provided."
                )}</p>
                <p style='margin-top:0.85rem;'><b>Important limitation:</b> {highlight_metrics_html(
                    "The CRM / KOL / FB strategy ranking tables below remain the source of truth for channel-performance evidence. In the current implementation, those ranking tables are shown to users in the dashboard but are not directly passed into the AI diagnosis call."
                )}</p>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

# =============================================================================
# Main page
# =============================================================================
def _add_city_to_marketing_outcomes(outcomes_df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    if outcomes_df.empty or assignments.empty:
        return outcomes_df.copy()

    df = outcomes_df.copy()
    a = assignments.copy()

    if "name_norm" not in a.columns and "name" in a.columns:
        a["name_norm"] = a["name"].apply(_normalize_name)

    if "city" not in a.columns:
        return df

    city_lookup = (
        a[["name_norm", "city"]]
        .dropna(subset=["name_norm"])
        .drop_duplicates("name_norm")
        .rename(columns={"city": "assignment_city"})
    )

    name_col = "restaurant_name" if "restaurant_name" in df.columns else "name"
    if name_col not in df.columns:
        return df

    df["name_norm"] = df[name_col].fillna("").astype(str).apply(_normalize_name)
    df = df.merge(city_lookup, on="name_norm", how="left")

    if "city" in df.columns:
        df["city"] = df["city"].where(
            df["city"].notna()
            & df["city"].astype(str).str.strip().ne("")
            & df["city"].astype(str).str.lower().ne("nan"),
            df["assignment_city"],
        )
    else:
        df["city"] = df["assignment_city"]

    df = df.drop(columns=["assignment_city"], errors="ignore")
    df["city"] = df["city"].fillna("Unknown")
    return df

def _filter_to_city(df: pd.DataFrame, city: str | None) -> pd.DataFrame:
    if df.empty or not city or city == "Not available in data":
        return df.copy()
    if "city" not in df.columns:
        return df.copy()

    return df[
        df["city"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq(str(city).strip())
    ].copy()
    
def render():
    priority_df = load_priority()
    momentum_df = load_momentum()
    assignments = load_cluster_assignments()
    ga_monthly_df = load_ga_restaurant_monthly()
    ga_campaign_monthly_df = load_ga_campaign_type_monthly()
    # Single load — outcomes_df is the single source of truth for all uplift values.
    # marketing_outcomes and outcomes_df both reference this; enrichment happens once below.
    outcomes_df = load_cluster_strategy_outcomes()
    # Ensure assignments have normalized name column for consistent merging and lookups
    if "name_norm" not in assignments.columns and "name" in assignments.columns:
        assignments["name_norm"] = assignments["name"].apply(_normalize_name)

    st.markdown("## Grounded Strategy Brief")
    
    st.markdown(
        f"<p style='color:{MUTED_TEXT};margin-top:-0.5rem;'>"
        "Ranks Google Analytics and marketing strategies at cluster, segment, and global level."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if priority_df.empty:
        st.warning("No priority data found. Run the data pipeline first.")
        return


    # all_names = priority_df.sort_values("priority_score", ascending=False)["name"].dropna().astype(str).tolist()

    # navbar_selected = str(st.session_state.get("selected_restaurant", "All") or "All")
    # previous_selected = st.session_state.get("strategy_restaurant", None)
    # if navbar_selected != "All" and navbar_selected in all_names:
    #     selected = navbar_selected
    # elif previous_selected in all_names:
    #     selected = previous_selected
    # else:
    #     selected = all_names[0]
    # st.session_state["strategy_restaurant"] = selected
    
    
    # Use clustered restaurant universe instead of priority_df only
    if not assignments.empty and "name" in assignments.columns:
        restaurant_universe = assignments.copy()

        # Keep only restaurants with a valid cluster
        if "cluster_id" in restaurant_universe.columns:
            restaurant_universe = restaurant_universe[
                pd.to_numeric(restaurant_universe["cluster_id"], errors="coerce").notna()
            ].copy()

        if "cluster_label" in restaurant_universe.columns:
            restaurant_universe = restaurant_universe[
                ~restaurant_universe["cluster_label"]
                .astype(str)
                .str.contains("Unclustered", case=False, na=False)
            ].copy()

        all_names = (
            restaurant_universe["name"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
    else:
        all_names = (
            priority_df["name"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

    navbar_selected = str(st.session_state.get("selected_restaurant", "All") or "All")
    previous_selected = st.session_state.get("strategy_restaurant", None)
    
    if navbar_selected != "All" and navbar_selected in all_names:
        selected = navbar_selected
    elif previous_selected in all_names:
        selected = previous_selected
    elif all_names:
        selected = all_names[0]

    # Ensure the global session state is always in sync with the dropdown's value
    st.session_state["strategy_restaurant"] = selected

    row, hist, segment, cluster_id, cluster_label, selected_restaurant_row = get_selected_context(selected, priority_df, momentum_df, assignments)
    # ensure distance features and GMV context are available when the peer recommender runs
    if not selected_restaurant_row.empty and not priority_df.empty:
        perf_cols = ["name", "monthly_revenue", "monthly_bookings", "avg_roi",
                    "latest_segment", "avg_lift_per_day"]
        perf_cols_existing = [c for c in perf_cols if c in priority_df.columns]
        priority_row = priority_df[priority_df["name"] == selected][perf_cols_existing].head(1)
        if not priority_row.empty:
            for col in perf_cols_existing:
                if col not in selected_restaurant_row.columns:
                    selected_restaurant_row[col] = priority_row.iloc[0][col]

    cluster_text = f"Cluster {cluster_id}: {cluster_label}" if cluster_id is not None else "Cluster not available"
    segment_text = segment if segment else "Segment not available"

    city_text = "City not available"
    if not assignments.empty and "name" in assignments.columns:
        assignment_city_lookup = assignments.copy()

        if "name_norm" not in assignment_city_lookup.columns:
            assignment_city_lookup["name_norm"] = assignment_city_lookup["name"].apply(_normalize_name)

        selected_assignment = assignment_city_lookup[
            assignment_city_lookup["name_norm"].eq(_normalize_name(selected))
        ].head(1)

        if not selected_assignment.empty and "city" in selected_assignment.columns:
            city_text = display_value(selected_assignment.iloc[0].get("city"))
            
            
    score = pd.to_numeric(row.get("priority_score", 0), errors="coerce")
    
    if pd.isna(score):
        score = 0
        
    st.markdown(
        f"""
        <div style='background:{SURFACE_COLOR};border:1px solid {BORDER_COLOR};
                    border-left:4px solid #cc0000;border-radius:8px;padding:1rem 1.4rem;'>
            <div style='display:flex;justify-content:space-between;gap:1rem;align-items:center;'>
                <div>
                    <div style='font-size:1.25rem;color:{TEXT_COLOR};font-weight:700;'>{selected}</div>
                    <div style='font-size:0.78rem;color:{MUTED_TEXT};margin-top:4px;'>{cluster_text}</div>
                    <div style='font-size:0.78rem;color:{MUTED_TEXT};margin-top:2px;'>Segment: {segment_text}</div>
                    <div style='font-size:0.78rem;color:{MUTED_TEXT};margin-top:2px;'>City: {city_text}</div>
                </div>
                <div style='text-align:right;'>
                    <div style='font-size:1.75rem;color:#cc0000;font-weight:700;'>{score:.0f}</div>
                    <div style='font-size:0.7rem;color:{MUTED_TEXT};'>PRIORITY SCORE</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # # -------------------------------------------------------------------------
    # # Section 1: Google Analytics metrics
    # # -------------------------------------------------------------------------
    # st.markdown("## Section 1: Google Analytics Metrics")
    # formula_card(
    #     "GA Strategy Ranking Formula",
    #     ["Google Ads Strategy Score = (GMV/GA × 0.40) + (Add to Cart × 0.30) + (View to Purchase × 0.30)"],
    # )

    # ga_rankings = build_ga_scope_rankings(ga_monthly_df, ga_campaign_monthly_df, assignments)
    # ga_cluster_df, ga_segment_df, ga_global_df = filter_ga_scopes(ga_rankings, cluster_id, segment)
    # ga_cluster_table = build_ga_rank_table(ga_cluster_df)
    # ga_segment_table = build_ga_rank_table(ga_segment_df)
    # ga_global_table = build_ga_rank_table(ga_global_df)

    # ga_tabs = st.tabs(["Cluster Level", "Segment Level", "Global Level"])
    # with ga_tabs[0]:
    #     display_ga_rank_table("Cluster Level Ranking", ga_cluster_table)
    # with ga_tabs[1]:
    #     display_ga_rank_table("Segment Level Ranking", ga_segment_table)
    # with ga_tabs[2]:
    #     display_ga_rank_table("Global Level Ranking", ga_global_table)

    # st.markdown("---")

    # # -------------------------------------------------------------------------
    # # Section 2: CRM/KOL/FB metrics
    # # -------------------------------------------------------------------------
    # st.markdown("## Section 2: CRM / KOL / FB Metrics")
    # formula_card(
    #     "Marketing Strategy Ranking Formula",
    #     ["Marketing Strategy Score = (Revenue Uplift × 0.60) + (Booking Uplift × 0.40)"],
    # )

    # marketing_outcomes = load_cluster_strategy_outcomes()
    # cluster_rank_df, segment_rank_df, global_rank_df = filter_marketing_scopes(marketing_outcomes, cluster_id, segment)
    # m_cluster_table = build_marketing_rank_table(cluster_rank_df)
    # m_segment_table = build_marketing_rank_table(segment_rank_df)
    # m_global_table = build_marketing_rank_table(global_rank_df)

    # marketing_tabs = st.tabs(["Cluster Level", "Segment Level", "Global Level"])
    # with marketing_tabs[0]:
    #     display_marketing_rank_table("Cluster Level Ranking", m_cluster_table)
    # with marketing_tabs[1]:
    #     display_marketing_rank_table("Segment Level Ranking", m_segment_table)
    # with marketing_tabs[2]:
    #     display_marketing_rank_table("Global Level Ranking", m_global_table)

    # -------------------------------------------------------------------------
    # Package details + AI Narrative
    # -------------------------------------------------------------------------
    # st.markdown("---")
    # render_package_details()

    # -------------------------------------------------------------------------
    # Strategy diagnosis boxes
    # -------------------------------------------------------------------------
    ga_rankings = build_ga_scope_rankings(ga_monthly_df, ga_campaign_monthly_df, assignments)
    # ga_cluster_df, ga_segment_df, ga_global_df = filter_ga_scopes(ga_rankings, cluster_id, segment)
    # ga_cluster_table = build_ga_rank_table(ga_cluster_df)
    # ga_segment_table = build_ga_rank_table(ga_segment_df)
    # ga_global_table = build_ga_rank_table(ga_global_df)

    marketing_outcomes = _add_city_to_marketing_outcomes(
        load_cluster_strategy_outcomes(),
        assignments,
    )

    cluster_rank_df, segment_rank_df, global_rank_df = filter_marketing_scopes(
        marketing_outcomes,
        cluster_id,
        segment,
    )

    m_global_table = build_marketing_rank_table(global_rank_df)

    restaurant_rank_df = _filter_marketing_for_restaurant(marketing_outcomes, selected)
    m_restaurant_table = build_marketing_rank_table(restaurant_rank_df)
    # Enrich once using global rank table so every strategy gets coverage.
    # outcomes_df now carries avg_revenue_uplift_pct, avg_bookings_uplift_pct, lift_reliability.
    outcomes_df = enrich_outcomes_with_rank_stats(outcomes_df, m_global_table)
    marketing_outcomes = outcomes_df  # alias for downstream segment-filter UI references

    st.markdown("## Strategy Diagnosis")
    render_ai_diagnosis_explainer()
    st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

    ## generate issue and strat

    ai_key = "ai_strategy_%s" % selected
    a_col, b_col = st.columns([2, 1])
    with a_col:
        generate_ai = st.button("Generate AI Narrative", key="gen_ai_%s" % selected, width="stretch")
    with b_col:
        if st.button("Clear AI", key="clr_ai_%s" % selected, width="stretch"):
            st.session_state[ai_key] = None

    if generate_ai:
        with st.spinner("Generating AI narrative..."):
            try:
                st.session_state[ai_key] = call_cohere(selected=selected,
                    row=row,
                    hist=hist,
                    segment=segment,
                    cluster_id=cluster_id,
                    cluster_label=cluster_label,
                    ga_rankings=ga_rankings,
                    momentum_df=momentum_df,
                    peerdata=st.session_state["peer_recs"]
                    )
            except Exception as e:
                st.error("AI generation failed: %s" % e)
                st.session_state[ai_key] = None

    if st.session_state.get(ai_key):
        response = extract_json(st.session_state[ai_key])

        # st.markdown(st.session_state[ai_key])
        # st.download_button(
        #     label="Download AI Narrative",
        #     data="AI STRATEGY NARRATIVE\n%s\n\n%s" % (selected, st.session_state[ai_key]),
        #     file_name="ai_strategy_%s.txt" % selected.replace(" ", "_"),
        #     mime="text/plain",
        #     width="stretch",
        # )

       
        issues = response.get("issues", [])

        render_placeholder_section(
            1,
            "Key issues Identified",
            "An analysis on the restaurant's metrics against their cluster",
            [
                (f"Issue {issue.get('issue_no', i+1)}", issue.get("description", ""))
                for i, issue in enumerate(issues)
            ],
        )

        strategy = response.get("strategy", "")

        render_placeholder_section(
            2,
            "Recommended strategy",
            "Personalised action plan based on restaurant's issues identified",
            [
                (f"{strat.get('title', i+1)}", strat.get("description", ""))
                for i, strat in enumerate(strategy)
            ],
            accent="#111827",
        )
    
    st.markdown("---")

    with st.container(border=True):
        _strategy_box_header(
            3,
            "CRM / KOL / FB strategy ranking",
            "What has driven revenue and booking uplift for similar restaurants.",
        )

        available_segments = []
        if not marketing_outcomes.empty and "latest_segment" in marketing_outcomes.columns:
            available_segments = (
                marketing_outcomes["latest_segment"]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )
        cluster_segments = available_segments
        if not marketing_outcomes.empty and cluster_id is not None and "cluster_id" in marketing_outcomes.columns:
            cluster_segment_rows = marketing_outcomes[
                pd.to_numeric(marketing_outcomes["cluster_id"], errors="coerce").eq(cluster_id)
            ].copy()
            if "latest_segment" in cluster_segment_rows.columns:
                cluster_segments = (
                    cluster_segment_rows["latest_segment"]
                    .dropna()
                    .astype(str)
                    .sort_values()
                    .unique()
                    .tolist()
                ) or available_segments

        default_segment = segment if segment in cluster_segments else (cluster_segments[0] if cluster_segments else None)

        rank_tabs = st.tabs(["Restaurant", "Cluster", "Segment", "Global"])

    with rank_tabs[0]:
        st.caption(f"{selected}'s own CRM / KOL / FB activity outcomes")
        _render_rank_html_table(m_restaurant_table)

    with rank_tabs[1]:
        use_city_cluster = st.checkbox(
            f"Only show {city_text} restaurants in this cluster",
            value=city_text not in {"City not available", "Not available in data"},
            disabled=city_text in {"City not available", "Not available in data"},
            key=f"cluster_city_filter_{selected}",
        )

        scoped_cluster = cluster_rank_df.copy()
        if use_city_cluster:
            scoped_cluster = _filter_to_city(scoped_cluster, city_text)

        st.caption(
            f"{cluster_text}"
            + (f" | City: {city_text}" if use_city_cluster else " | All cities")
        )

        _render_rank_html_table(build_marketing_rank_table(scoped_cluster))

    with rank_tabs[2]:
        selector_col, toggle_col = st.columns([0.42, 0.58])

        with selector_col:
            scope_segment = st.selectbox(
                "Segment filter",
                cluster_segments if cluster_segments else ["No segment data"],
                index=cluster_segments.index(default_segment) if default_segment in cluster_segments else 0,
                disabled=not cluster_segments,
                key=f"marketing_segment_filter_{selected}",
            )

        with toggle_col:
            st.markdown("<div style='height:1.72rem;'></div>", unsafe_allow_html=True)
            use_city_segment = st.checkbox(
                f"Only show {city_text} restaurants",
                value=city_text not in {"City not available", "Not available in data"},
                disabled=city_text in {"City not available", "Not available in data"},
                key=f"segment_city_filter_{selected}",
            )

        scoped_marketing = _filter_marketing_for_scope(
            marketing_outcomes,
            cluster_id,
            scope_segment if cluster_segments else None,
            "Segment",
        )

        if use_city_segment:
            scoped_marketing = _filter_to_city(scoped_marketing, city_text)

        st.caption(
            f"{cluster_text} | Segment: {scope_segment}"
            + (f" | City: {city_text}" if use_city_segment else " | All cities")
        )

        _render_rank_html_table(build_marketing_rank_table(scoped_marketing))

    with rank_tabs[3]:
        st.caption("All restaurants")
        _render_rank_html_table(m_global_table)

    render_ga_snapshot_section(
        4,
        "Restaurant Google Ads snapshot",
        "This restaurant's own funnel metrics vs cluster benchmark.",
        build_ga_snapshot_cards(selected, ga_monthly_df, assignments, cluster_id),
    )

    render_peer_recommender_section(selected, selected_restaurant_row, assignments, outcomes_df)
    
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
                
                # grounded_advice, segment_icon = segment_advice

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

                # # ── OPTIONAL AI NARRATIVE (button-triggered, borrowed from strategy.py) ──────
                # st.markdown("---")
                # st.markdown("#### 🤖 Optional AI Narrative")
                # st.caption("Uses Gemini to synthesise peer campaign evidence into a narrative. The Strategy Brief above is the source of truth.")

                # ai_key = "ai_playbook_%s" % selected_restaurant_row.iloc[0].get("name", "")

                # gen_col, clr_col = st.columns([3, 1])
                # with gen_col:
                #     generate_ai = st.button("✨ Generate AI Narrative", key="gen_ai_peer", use_container_width=True)
                # with clr_col:
                #     if st.button("🗑️ Clear", key="clr_ai_peer", use_container_width=True):
                #         st.session_state[ai_key] = None

                # if generate_ai:
                #     prompt = build_recommendation_prompt(
                #         restaurant_name=selected_restaurant_row.iloc[0].get("name", "Unknown"),
                #         momentum=active_momentum,
                #         theme=selected_restaurant_row.iloc[0].get("theme", "Unknown"),
                #         active_gmv=active_gmv,
                #         active_conv=selected_restaurant_row.iloc[0].get("avg_view_to_purchase", 0.0) or 0.0,
                #         top_recs=top_recs,
                #         top_peers=display_peers,   # enriched version with best_channel, reliability etc.
                #     )
                #     _phash = hashlib.sha256(prompt.encode()).hexdigest()
                #     with st.spinner("Gemini is analysing peer strategies..."):
                #         try:
                #             st.session_state[ai_key] = generate_ai_playbook(prompt, _prompt_hash=_phash)
                #         except Exception as e:
                #             st.error("AI generation failed: %s" % e)
                #             st.session_state[ai_key] = None

                # if st.session_state.get(ai_key):
                #     with st.container(border=True):
                #         st.markdown(st.session_state[ai_key])
                #     st.download_button(
                #         label="⬇️ Download AI Narrative",
                #         data="AI STRATEGY NARRATIVE\n%s\n\n%s" % (
                #             selected_restaurant_row.iloc[0].get("name", "Unknown"),
                #             st.session_state[ai_key]
                #         ),
                #         file_name="ai_narrative_%s.txt" % selected_restaurant_row.iloc[0].get("name", "Unknown").replace(" ", "_"),
                #         mime="text/plain",
                #     )

    # st.markdown("---")
    # st.markdown("## Optional AI Narrative")
    # st.caption("Uses `GEMINI_API_KEY` if configured. The ranking tables above remain the source of truth.")

    # ai_key = "ai_strategy_%s" % selected
    # a_col, b_col = st.columns([2, 1])
    # with a_col:
    #     generate_ai = st.button("Generate AI Narrative", key="gen_ai_%s" % selected, width="stretch")
    # with b_col:
    #     if st.button("Clear AI", key="clr_ai_%s" % selected, width="stretch"):
    #         st.session_state[ai_key] = None

    # if generate_ai:
    #     with st.spinner("Generating AI narrative..."):
    #         try:
    #             prompt = build_ai_prompt(
    #                 selected=selected,
    #                 row=row,
    #                 hist=hist,
    #                 segment=segment,
    #                 cluster_id=cluster_id,
    #                 cluster_label=cluster_label,
    #                 ga_rankings=ga_rankings,
    #                 momentum_df=momentum_df
    #             )
    #             st.session_state[ai_key] = call_cohere(prompt)
    #         except Exception as e:
    #             st.error("AI generation failed: %s" % e)
    #             st.session_state[ai_key] = None


    # if st.session_state.get(ai_key):
    #     st.markdown(st.session_state[ai_key])
    #     st.download_button(
    #         label="Download AI Narrative",
    #         data="AI STRATEGY NARRATIVE\n%s\n\n%s" % (selected, st.session_state[ai_key]),
    #         file_name="ai_strategy_%s.txt" % selected.replace(" ", "_"),
    #         mime="text/plain",
    #         width="stretch",
    #     )


if __name__ == "__main__":
    render()
