from __future__ import annotations

import os
import re
from textwrap import dedent
from html import escape
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
import cohere

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


# =============================================================================
# Formatting helpers
# =============================================================================

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
    Build GA campaign type ranking inputs for cluster, segment, and global scopes.

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
    GA Strategy Score = 0.40 x GMV/GA + 0.30 x Add to Cart + 0.30 x View to Purchase.
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
        st.info("No GA campaign strategy data available for this scope.")
        return

    out = table.copy().rename(
        columns={
            "rank": "Rank",
            "strategy": "Activity",
            "gmv_per_ga": "GMV/GA",
            "add_to_cart": "Add to Cart",
            "view_to_purchase": "View to Purchase",
            "ga_strategy_score": "GA Strategy Score",
        }
    )
    out["GMV/GA"] = out["GMV/GA"].apply(fmt_thb)
    out["Add to Cart"] = out["Add to Cart"].apply(fmt_pct)
    out["View to Purchase"] = out["View to Purchase"].apply(fmt_pct)
    out["GA Strategy Score"] = out["GA Strategy Score"].apply(lambda v: fmt_num(v, 3))

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
    if table.empty:
        st.info("No CRM/KOL/FB strategy outcome data available for this scope.")
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
            "avg_revenue_uplift_pct": "Revenue Uplift",
            "avg_bookings_uplift_pct": "Booking Uplift",
            "marketing_strategy_score": "Strategy Score",
        }
    )
    out["Revenue Uplift"] = out["Revenue Uplift"].apply(fmt_pct)
    out["Booking Uplift"] = out["Booking Uplift"].apply(fmt_pct)
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
                    {escape(body)}
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
                    {escape(body)}
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
        st.info("No CRM/KOL/FB strategy outcome data available for this scope.")
        return

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
                        <th style='padding:0.8rem 1rem;'>Revenue uplift</th>
                        <th style='padding:0.8rem 1rem;'>Booking uplift</th>
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

    for col in ["ga_items_viewed", "ga_add_to_cart_rate", "ga_view_to_purchase_rate"]:
        if col in cluster_scope.columns:
            cluster_scope[col] = pd.to_numeric(cluster_scope[col], errors="coerce")

    benchmarks = {
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


def render_peer_recommender_section() -> None:
    st.markdown(
        dedent(f"""
        <section style='background:{SURFACE_COLOR};border:1px solid {BORDER_COLOR};border-radius:12px;
                        padding:1.25rem 1.35rem 1.35rem;margin:0 0 1.15rem 0;
                        box-shadow:0 8px 20px rgba(15,23,42,0.035);'>
            <div style='display:flex;align-items:flex-start;margin-bottom:1.15rem;'>
                {_badge_html(5)}
                <div>
                    <div style='font-size:1.18rem;font-weight:800;color:{TEXT_COLOR};line-height:1.2;'>
                        Peer recommender
                    </div>
                    <div style='font-size:0.92rem;color:{MUTED_TEXT};margin-top:0.24rem;line-height:1.35;'>
                        Reserved for peer-based recommendations.
                    </div>
                </div>
            </div>
            <div style='background:#fff;border:1px dashed {BORDER_COLOR};border-radius:10px;
                        padding:1.05rem 1.1rem;color:{TEXT_COLOR};font-size:0.95rem;line-height:1.5;'>
                Peer recommender placeholder. This box will later show matched peer restaurants,
                the winning peer strategy, and evidence-backed recommendation text.
            </div>
        </section>
        """).strip(),
        unsafe_allow_html=True,
    )


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

    return row, hist, segment, cluster_id, cluster_label


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


# =============================================================================
# AI narrative helpers
# =============================================================================

def _table_for_prompt(table: pd.DataFrame, table_type: str, top_n: int = 5) -> str:
    if table is None or table.empty:
        return "- insufficient data"

    lines = []
    for _, rec in table.head(top_n).iterrows():
        if table_type == "ga":
            lines.append(
                "- Rank {rank}: {name} | GMV/GA={gmv} | Add-to-cart={atc} | View-to-purchase={v2p} | GA score={score}".format(
                    rank=int(rec.get("rank", 0)),
                    name=rec.get("strategy", "-"),
                    gmv=fmt_thb(rec.get("gmv_per_ga")),
                    atc=fmt_pct(rec.get("add_to_cart")),
                    v2p=fmt_pct(rec.get("view_to_purchase")),
                    score=fmt_num(rec.get("ga_strategy_score"), 3),
                )
            )
        else:
            lines.append(
                "- Rank {rank}: {name} | count={count} | revenue uplift={rev} | booking uplift={book} | score={score}".format(
                    rank=int(rec.get("rank", 0)),
                    name=rec.get("strategy_name", "-"),
                    count=fmt_int(rec.get("count")),
                    rev=fmt_pct(rec.get("avg_revenue_uplift_pct")),
                    book=fmt_pct(rec.get("avg_bookings_uplift_pct")),
                    score=fmt_num(rec.get("marketing_strategy_score"), 3),
                )
            )
    return "\n".join(lines) if lines else "- insufficient data"


def _restaurant_ga_snapshot(row: dict, hist: pd.DataFrame) -> str:
    source = row.copy()
    if hist is not None and len(hist):
        latest = hist.sort_values("year_month").iloc[-1].to_dict()
        source.update({k: v for k, v in latest.items() if k not in source or pd.isna(source.get(k))})
    return "\n".join(
        [
            f"- Items viewed: {fmt_int(source.get('ga_items_viewed'))}",
            f"- GMV/GA view: {fmt_thb(source.get('gmv_per_ga_view'))}",
            f"- Add-to-cart rate: {fmt_pct(source.get('ga_add_to_cart_rate'))}",
            f"- View-to-purchase rate: {fmt_pct(source.get('ga_view_to_purchase_rate'))}",
            f"- GA revenue/view: {fmt_thb(source.get('ga_revenue_per_view'))}",
        ]
    )


def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text.strip()

def call_cohere(prompt: str) -> str:
    try:
        api_key = os.getenv("COHERE_API_KEY")

        co = cohere.ClientV2(api_key) 

        # Analyze the text without RAG
        response = co.chat(
            model="command-a-03-2025",
            messages=[
                {
                    "role": "user", 
                    "content": f"{prompt}"
                }
            ]
        )
        if response and response.message.content[0].text:
            return response.message.content[0].text
        return "_(No response generated)_"
        
    except Exception as e:
        return f"_(Playbook generation unavailable: {e})_"
    
def build_ai_prompt(
    selected: str,
    row: dict,
    hist: pd.DataFrame,
    segment: str | None,
    cluster_id,
    cluster_label,
    ga_cluster_table: pd.DataFrame,
    ga_segment_table: pd.DataFrame,
    ga_global_table: pd.DataFrame,
    m_cluster_table: pd.DataFrame,
    m_segment_table: pd.DataFrame,
    m_global_table: pd.DataFrame,
) -> str:
    priority_score = fmt_num(pd.to_numeric(pd.Series([row.get("priority_score")]), errors="coerce").iloc[0], 1)
    preferred_channel = display_value(row.get("recommended_channel"))
    priority_tier = display_value(row.get("priority_tier"))

    return f"""
You are a senior marketing strategist advising a restaurant owner.

IMPORTANT RULES:
- Use ONLY the data provided below.
- DO NOT invent any numbers or assumptions.
- You MUST produce an answer; never return empty.
- If data is missing, say "insufficient data".
- Always mention the restaurant name at least once.
- Always reference REAL metrics from the GA and CRM/KOL/FB tables.
- Be specific to THIS restaurant; no generic advice.

STYLE RULES:
- Use bullet points only; no long paragraphs.
- Max 1-2 lines per point.
- Use simple language.
- Only metric names can be technical.

---------------------
RESTAURANT CONTEXT
---------------------
Restaurant: {selected}
Cluster: {cluster_id} - {display_value(cluster_label)}
Segment: {display_value(segment)}
Priority Score: {priority_score}
Priority Tier: {priority_tier}
Preferred Channel: {preferred_channel}

---------------------
RESTAURANT GA DIAGNOSTIC
---------------------
{_restaurant_ga_snapshot(row, hist)}

---------------------
GA STRATEGY RANKINGS
Formula: GA Strategy Score = (GMV/GA x 0.40) + (Add to Cart x 0.30) + (View to Purchase x 0.30)
Note: GA Count = estimated sessions allocated to the selected scope using scope GA-view share by month. It is not reused platform-wide sessions.
---------------------
Cluster Level:
{_table_for_prompt(ga_cluster_table, 'ga')}

Segment Level:
{_table_for_prompt(ga_segment_table, 'ga')}

Global Level:
{_table_for_prompt(ga_global_table, 'ga')}

---------------------
CRM / KOL / FB STRATEGY RANKINGS
Formula: Marketing Strategy Score = (Average Revenue Uplift x 0.60) + (Average Booking Uplift x 0.40)
---------------------
Cluster Level:
{_table_for_prompt(m_cluster_table, 'marketing')}

Segment Level:
{_table_for_prompt(m_segment_table, 'marketing')}

Global Level:
{_table_for_prompt(m_global_table, 'marketing')}

---------------------
AVAILABLE MARKETING PACKAGES (WITH CAPABILITIES)
---------------------

BASIC PACKAGE (Awareness Starter – Low Cost, Entry Level)
Purpose:
- Drive initial awareness and light engagement
- Suitable for low traffic restaurants

Key Capabilities:
- Revenue Guarantee (30K+ THB)
- Revenue Guarantee (90 Day)
- Send Blogger to Review x1 (20k followers)
- Send Blogger to Review x2 (30k followers)
- Boost post THB 2,000 Baht
- Pop-up Banner: Individual
- Photoshooting
- Guaranteed in Restaurants list home page
- HH Facebook Post : Individual post
- Line@ Broadcasts :  Individual post
- Push Notification

Limitations:
- No strong video content
- Limited reach and weak conversion tools

Use When:
- Traffic is low
- Need visibility, not deep conversion

STANDARD PACKAGE (Growth & Conversion – Balanced)
Purpose:
- Improve customer intent and conversion
- Best for mid-funnel problems (low add-to-cart, weak engagement)

Key Capabilities:
- Revenue Guarantee (40K+ THB)
- Revenue Guarantee (120 Day)
- Send Blogger to Review x1 (50k followers)
- Send Blogger to Review x2 (30k followers)
- Boost post THB 3,000 Baht
- Pop-up Banner: Individual
- Guaranteed in Restaurants list home page
- Photoshooting
- HH Facebook Post : Individual post
- Line@ Broadcasts :  Individual post
- Push Notification
- Web Footer (2 days)
- Tiktok VDO or Instagram Reels VDO

Strengths:
- Combines awareness + conversion tools
- Strong for improving menu appeal and intent

Use When:
- Traffic exists but conversion is weak
- Add-to-cart or engagement is low

PREMIUM PACKAGE (High Impact – Scale + Conversion)
Purpose:
- Maximise reach AND conversion at scale
- Suitable for high-priority restaurants

Key Capabilities:
- Revenue Guarantee (100K+ THB)
- Revenue Guarantee (180 Day)
- Send Blogger to Review x1 (50k followers)
- Send Blogger to Review x2 (50k followers)
- Boost post THB 5,000 Baht
- Pop-up Banner: Individual
- Guaranteed in Restaurants list home page
- Photoshooting
- HH Facebook Post : Individual post
- Line@ Broadcasts :  Individual post
- Push Notification
- Web Footer (2 days)
- Tiktok VDO or Instagram Reels VDO
- Guaranteed in Restaurants Promotion Banner (1 week)
- Blog: Advertorial: Individual

Strengths:
- Highest visibility + strongest conversion ecosystem
- Combines brand + performance marketing

Limitations:
- Highest cost → must justify ROI

Use When:
- High priority tier
- Need scale OR strong brand push
- Conversion AND reach both need improvement

---------------------
TASK
---------------------
Give a decisive, business-focused recommendation for THIS restaurant.

FORMAT STRICTLY:

## 1. Key Issue
- State the MAIN problem using GA and/or marketing ranking evidence.
- Include the metric, benchmark/context, and business pain.

## 2. Recommended Package
- Package: Basic / Standard / Premium
- Explain why this package fixes the exact funnel problem.
- Mention specific package features.
- Explain why cheaper option fails.
- Explain why more expensive option is unnecessary, unless Premium is chosen.

Decision rules:
- Mid-funnel issue, e.g. add-to-cart low → Standard.
- Lower-funnel issue, e.g. view-to-purchase low → Standard or Premium.
- Traffic issue → Basic or Standard.
- Only choose Premium if high priority tier AND scale or strong brand push is needed.

## 3. GA Support Strategy
- Choose up to 2 GA activities from the GA rankings.
- Explain what each improves: traffic, add-to-cart, view-to-purchase, or revenue per view.
- Explain what the GA metrics suggest about traffic quality, add-to-cart intent, view-to-purchase conversion, or revenue per view.

## 4. CRM / KOL / FB Priority Actions
- Choose EXACTLY 2 strategies from the CRM/KOL/FB rankings.
- For each: strategy name, concrete campaign idea, and what it improves.

## 5. Execution Plan
- Blogger Strategy:
- Content:
- Messaging:
- Paid + CRM:
- Funnel Impact:

## 6. Why This Will Work
- Reference GA data.
- Reference strategy evidence.
- Connect to outcome.

## 7. 30-Day Plan
- Launch:
- Optimise:
- Scale:

## 8. Expected Impact
- Translate into more orders, better conversion, or higher revenue per visitor.
- Do not promise exact results.
"""


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

# =============================================================================
# Main page
# =============================================================================

def render():
    priority_df = load_priority()
    momentum_df = load_momentum()
    assignments = load_cluster_assignments()
    ga_monthly_df = load_ga_restaurant_monthly()
    ga_campaign_monthly_df = load_ga_campaign_type_monthly()

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

    preselected = st.session_state.get("strategy_restaurant", None)
    all_names = priority_df.sort_values("priority_score", ascending=False)["name"].dropna().astype(str).tolist()
    default_idx = all_names.index(preselected) if preselected in all_names else 0

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        selected = st.selectbox(
            "Restaurant",
            all_names,
            index=default_idx,
            format_func=lambda n: "#%d  %s" % (all_names.index(n) + 1, n),
        )
        st.session_state["strategy_restaurant"] = selected

    row, hist, segment, cluster_id, cluster_label = get_selected_context(selected, priority_df, momentum_df, assignments)

    with col_info:
        cluster_text = f"Cluster {cluster_id}: {cluster_label}" if cluster_id is not None else "Cluster not available"
        segment_text = segment if segment else "Segment not available"
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
    #     ["GA Strategy Score = (GMV/GA × 0.40) + (Add to Cart × 0.30) + (View to Purchase × 0.30)"],
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
    ga_cluster_df, ga_segment_df, ga_global_df = filter_ga_scopes(ga_rankings, cluster_id, segment)
    ga_cluster_table = build_ga_rank_table(ga_cluster_df)
    ga_segment_table = build_ga_rank_table(ga_segment_df)
    ga_global_table = build_ga_rank_table(ga_global_df)

    marketing_outcomes = load_cluster_strategy_outcomes()
    cluster_rank_df, segment_rank_df, global_rank_df = filter_marketing_scopes(marketing_outcomes, cluster_id, segment)
    m_cluster_table = build_marketing_rank_table(cluster_rank_df)
    m_segment_table = build_marketing_rank_table(segment_rank_df)
    m_global_table = build_marketing_rank_table(global_rank_df)
    restaurant_rank_df = _filter_marketing_for_restaurant(marketing_outcomes, selected)
    m_restaurant_table = build_marketing_rank_table(restaurant_rank_df)

    st.markdown("## Strategy Diagnosis")
    st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

    render_placeholder_section(
        1,
        "Key issue 1, 2, 3",
        "Hardcoded problem statements for revenue, booking, and GA funnel gaps.",
        [
            (
                "Key issue 1",
                "Revenue and booking uplift problem placeholder. Replace this with logic that picks the main growth pain from priority, bookings, and GMV signals.",
            ),
            (
                "Key issue 2",
                "GA funnel problem placeholder. Use this for low item views, weak add-to-cart intent, or poor view-to-purchase conversion once rules are added.",
            ),
            (
                "Key issue 3",
                "Audience and channel fit placeholder. Use cluster and segment evidence to explain why similar restaurants need stronger CRM, KOL, or FB support.",
            ),
        ],
    )

    render_placeholder_section(
        2,
        "Recommended strategy 1, 2, 3",
        "Hardcoded action plan mapped to each issue.",
        [
            (
                "Strategy 1",
                "Prioritise a conversion-focused package with visible offers, CRM reminders, and booking-oriented messaging.",
            ),
            (
                "Strategy 2",
                "Improve funnel intent with stronger menu presentation, clearer value anchors, and retargeting for high-intent visitors.",
            ),
            (
                "Strategy 3",
                "Use peer-proven CRM/KOL/FB activities from the ranking below to reinforce the restaurant's best-fit growth channel.",
            ),
        ],
        accent="#111827",
    )

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
            st.caption(cluster_text)
            _render_rank_html_table(m_cluster_table)
        with rank_tabs[2]:
            st.caption(f"{cluster_text} | Segment-level ranking within selected cluster")
            selector_col, _ = st.columns([0.42, 0.58])
            with selector_col:
                scope_segment = st.selectbox(
                    "Segment filter",
                    cluster_segments if cluster_segments else ["No segment data"],
                    index=cluster_segments.index(default_segment) if default_segment in cluster_segments else 0,
                    disabled=not cluster_segments,
                    key=f"marketing_segment_filter_{selected}",
                )
            scoped_marketing = _filter_marketing_for_scope(
                marketing_outcomes,
                cluster_id,
                scope_segment if cluster_segments else None,
                "Segment",
            )
            _render_rank_html_table(build_marketing_rank_table(scoped_marketing))
        with rank_tabs[3]:
            st.caption("All restaurants")
            _render_rank_html_table(m_global_table)
    st.markdown("<div style='height:1.15rem;'></div>", unsafe_allow_html=True)

    render_ga_snapshot_section(
        4,
        "Restaurant GA snapshot",
        "This restaurant's own funnel metrics vs cluster benchmark.",
        build_ga_snapshot_cards(selected, ga_monthly_df, assignments, cluster_id),
    )

    render_peer_recommender_section()

    st.markdown("---")
    st.markdown("## Optional AI Narrative")
    st.caption("Uses `GEMINI_API_KEY` if configured. The ranking tables above remain the source of truth.")

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
                prompt = build_ai_prompt(
                    selected=selected,
                    row=row,
                    hist=hist,
                    segment=segment,
                    cluster_id=cluster_id,
                    cluster_label=cluster_label,
                    ga_cluster_table=ga_cluster_table,
                    ga_segment_table=ga_segment_table,
                    ga_global_table=ga_global_table,
                    m_cluster_table=m_cluster_table,
                    m_segment_table=m_segment_table,
                    m_global_table=m_global_table,
                )
                st.session_state[ai_key] = call_cohere(prompt)
            except Exception as e:
                st.error("AI generation failed: %s" % e)
                st.session_state[ai_key] = None

    if st.session_state.get(ai_key):
        st.markdown(st.session_state[ai_key])
        st.download_button(
            label="Download AI Narrative",
            data="AI STRATEGY NARRATIVE\n%s\n\n%s" % (selected, st.session_state[ai_key]),
            file_name="ai_strategy_%s.txt" % selected.replace(" ", "_"),
            mime="text/plain",
            width="stretch",
        )


if __name__ == "__main__":
    render()
