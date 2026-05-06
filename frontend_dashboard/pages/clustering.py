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
import streamlit.components.v1 as components
from data.loader import (
    load_cluster_assignments,
    load_ga_restaurant_monthly,
    load_ga_campaign_outreach_raw,
    load_cluster_strategy_outcomes,
)
from theme import BASE_LAYOUT, CHART_THEME, MUTED_TEXT

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

def _num_or_zero(value) -> float:
    try:
        numeric = pd.to_numeric(value, errors="coerce")
    except Exception:
        numeric = np.nan
    if pd.isna(numeric):
        return 0.0
    return float(numeric)

def _fmt_pct_value(value) -> str:
    return f"{_num_or_zero(value):.1%}"

def _fmt_thb_value(value) -> str:
    return f"THB {_num_or_zero(value):,.0f}"


def _clear_strategy_filter() -> None:
    st.session_state["selected_strategy_family"] = "All"


def _clear_ga_campaign_type_filter() -> None:
    st.session_state["selected_ga_campaign_type"] = "All"


def _get_plotly_selected_x(event) -> str | None:
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection", {})

    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points", [])
    if not points:
        return None

    point = points[0]
    value = getattr(point, "x", None)
    if value is None and isinstance(point, dict):
        value = point.get("x")
    if value is None:
        return None
    value = str(value).strip()
    return value or None

def _prepare_scope_ga_monthly(
    ga_restaurant_monthly: pd.DataFrame,
    assignments: pd.DataFrame,
    scope_name_norms: set[str],
) -> pd.DataFrame:
    if ga_restaurant_monthly.empty or assignments.empty or not scope_name_norms:
        return pd.DataFrame()

    ga = ga_restaurant_monthly.copy()
    if "name_norm" not in ga.columns and "name" in ga.columns:
        ga["name_norm"] = ga["name"].apply(_normalize_name)
    if "name_norm" not in ga.columns or "year_month" not in ga.columns:
        return pd.DataFrame()

    ga["name_norm"] = ga["name_norm"].fillna("").astype(str)
    ga["year_month"] = pd.to_datetime(ga["year_month"], errors="coerce")
    ga = ga.dropna(subset=["year_month"])
    if ga.empty:
        return pd.DataFrame()

    ref_cols = [c for c in ["name_norm", "cluster_id", "cluster_label", "latest_segment"] if c in assignments.columns]
    if ref_cols and "name_norm" in ref_cols:
        cluster_ref = assignments[ref_cols].drop_duplicates("name_norm")
        ga = ga.merge(cluster_ref, on="name_norm", how="inner", suffixes=("", "_assignment"))
        for col in ["cluster_id", "cluster_label", "latest_segment"]:
            assignment_col = f"{col}_assignment"
            if assignment_col in ga.columns:
                ga[col] = ga[col].where(ga[col].notna(), ga[assignment_col]) if col in ga.columns else ga[assignment_col]
                ga = ga.drop(columns=[assignment_col])

    numeric_cols = [
        "monthly_gmv",
        "ga_items_viewed",
        "ga_items_added_to_cart",
        "ga_items_purchased",
    ]
    for col in numeric_cols:
        if col not in ga.columns:
            ga[col] = 0
        ga[col] = pd.to_numeric(ga[col], errors="coerce").fillna(0)

    month_total_ga_views = (
        ga.groupby("year_month", as_index=False)["ga_items_viewed"]
        .sum()
        .rename(columns={"ga_items_viewed": "all_ga_items_viewed"})
    )
    scope_ga = ga[ga["name_norm"].isin(scope_name_norms)].copy()
    if scope_ga.empty:
        return pd.DataFrame()

    scope_monthly = (
        scope_ga.groupby("year_month", as_index=False)
        .agg(
            total_monthly_gmv=("monthly_gmv", "sum"),
            total_ga_items_viewed=("ga_items_viewed", "sum"),
            total_ga_items_added_to_cart=("ga_items_added_to_cart", "sum"),
            total_ga_items_purchased=("ga_items_purchased", "sum"),
            restaurants_with_ga_data=("name_norm", "nunique"),
        )
        .merge(month_total_ga_views, on="year_month", how="left")
    )
    scope_monthly["scope_ga_view_share"] = (
        pd.to_numeric(scope_monthly["total_ga_items_viewed"], errors="coerce")
        / pd.to_numeric(scope_monthly["all_ga_items_viewed"], errors="coerce").replace(0, np.nan)
    ).fillna(0)
    scope_monthly["gmv_per_ga_view"] = (
        scope_monthly["total_monthly_gmv"]
        / scope_monthly["total_ga_items_viewed"].replace(0, np.nan)
    )
    scope_monthly["ga_add_to_cart_rate"] = (
        scope_monthly["total_ga_items_added_to_cart"]
        / scope_monthly["total_ga_items_viewed"].replace(0, np.nan)
    )
    scope_monthly["ga_view_to_purchase_rate"] = (
        scope_monthly["total_ga_items_purchased"]
        / scope_monthly["total_ga_items_viewed"].replace(0, np.nan)
    )
    return scope_monthly.sort_values("year_month").reset_index(drop=True)

def _prepare_scope_ga_restaurant_rows(
    ga_restaurant_monthly: pd.DataFrame,
    assignments: pd.DataFrame,
    scope_name_norms: set[str],
) -> pd.DataFrame:
    if ga_restaurant_monthly.empty or assignments.empty or not scope_name_norms:
        return pd.DataFrame()

    ga = ga_restaurant_monthly.copy()
    if "name_norm" not in ga.columns and "name" in ga.columns:
        ga["name_norm"] = ga["name"].apply(_normalize_name)
    required_cols = {"name_norm", "name", "year_month", "ga_items_viewed"}
    if not required_cols.issubset(ga.columns):
        return pd.DataFrame()

    ga["name_norm"] = ga["name_norm"].fillna("").astype(str)
    ga = ga[ga["name_norm"].isin(scope_name_norms)].copy()
    if ga.empty:
        return pd.DataFrame()

    ga["year_month"] = pd.to_datetime(ga["year_month"], errors="coerce")
    ga["ga_items_viewed"] = pd.to_numeric(ga["ga_items_viewed"], errors="coerce").fillna(0)
    ga = ga.dropna(subset=["year_month"])
    if ga.empty:
        return pd.DataFrame()

    ref_cols = [c for c in ["name_norm", "cluster_id", "cluster_label", "latest_segment"] if c in assignments.columns]
    if ref_cols and "name_norm" in ref_cols:
        cluster_ref = assignments[ref_cols].drop_duplicates("name_norm")
        ga = ga.merge(cluster_ref, on="name_norm", how="left", suffixes=("", "_assignment"))
        for col in ["cluster_id", "cluster_label", "latest_segment"]:
            assignment_col = f"{col}_assignment"
            if assignment_col in ga.columns:
                ga[col] = ga[col].where(ga[col].notna(), ga[assignment_col]) if col in ga.columns else ga[assignment_col]
                ga = ga.drop(columns=[assignment_col])

    for col in ["monthly_gmv", "ga_items_added_to_cart", "ga_items_purchased"]:
        if col not in ga.columns:
            ga[col] = 0
        ga[col] = pd.to_numeric(ga[col], errors="coerce").fillna(0)

    out = (
        ga.groupby(
            [
                c
                for c in ["name_norm", "name", "cluster_id", "cluster_label", "latest_segment", "year_month"]
                if c in ga.columns
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            ga_items_viewed=("ga_items_viewed", "sum"),
            monthly_gmv=("monthly_gmv", "sum"),
            ga_items_added_to_cart=("ga_items_added_to_cart", "sum"),
            ga_items_purchased=("ga_items_purchased", "sum"),
        )
        .sort_values(["year_month", "cluster_id", "name"])
        .reset_index(drop=True)
    )
    out["gmv_per_ga_view"] = out["monthly_gmv"] / out["ga_items_viewed"].replace(0, np.nan)
    out["ga_add_to_cart_rate"] = out["ga_items_added_to_cart"] / out["ga_items_viewed"].replace(0, np.nan)
    out["ga_view_to_purchase_rate"] = out["ga_items_purchased"] / out["ga_items_viewed"].replace(0, np.nan)
    return out

def _aggregate_strategy_scope(df: pd.DataFrame, scope_label: str, min_sample_size: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "scope_label",
                "strategy_name",
                "activities",
                "restaurants",
                "avg_revenue_uplift_pct",
                "avg_bookings_uplift_pct",
                "meets_sample_guardrail",
                "ranking_score",
            ]
        )

    scoped = df.copy()
    strategy_family = scoped.get("strategy_family", pd.Series("", index=scoped.index)).fillna("").astype(str)
    fallback_strategy = scoped.get("strategy_name", pd.Series("Unknown", index=scoped.index)).fillna("Unknown").astype(str)
    scoped["strategy_name"] = strategy_family.where(strategy_family.str.strip().ne(""), fallback_strategy)
    scoped["restaurant_name"] = scoped.get("restaurant_name", pd.Series("Unknown", index=scoped.index)).fillna("Unknown")
    scoped["activity_count"] = 1
    scoped["bookings_uplift_pct"] = pd.to_numeric(scoped.get("bookings_uplift_pct"), errors="coerce")
    scoped["revenue_uplift_pct"] = pd.to_numeric(scoped.get("revenue_uplift_pct"), errors="coerce")
    scoped["roi"] = pd.to_numeric(scoped.get("roi"), errors="coerce")
    scoped["incremental_revenue_thb"] = pd.to_numeric(scoped.get("incremental_revenue_thb"), errors="coerce")

    activity_col = "activity_id" if "activity_id" in scoped.columns else "activity_count"
    activity_agg = pd.Series.nunique if activity_col == "activity_id" else "sum"
    agg = (
        scoped.groupby("strategy_name", dropna=False)
        .agg(
            activities=(activity_col, activity_agg),
            restaurants=("restaurant_name", pd.Series.nunique),
            avg_revenue_uplift_pct=("revenue_uplift_pct", "mean"),
            avg_bookings_uplift_pct=("bookings_uplift_pct", "mean"),
            avg_roi=("roi", "mean"),
            success_rate=(
                "incremental_revenue_thb",
                lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean()) if len(s) else np.nan,
            ),
        )
        .reset_index()
    )
    revenue_component = agg["avg_revenue_uplift_pct"].fillna(0)
    bookings_component = agg["avg_bookings_uplift_pct"].fillna(0)
    roi_component = agg["avg_roi"].fillna(0)
    success_component = agg["success_rate"].fillna(0)
    agg["ranking_score"] = revenue_component * 100 + bookings_component * 30 + roi_component * 10 + success_component * 5
    agg["meets_sample_guardrail"] = agg["activities"] >= int(min_sample_size)
    agg["scope_label"] = scope_label
    return agg.sort_values(["meets_sample_guardrail", "ranking_score"], ascending=[False, False]).reset_index(drop=True)

def _prepare_ga_quality_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    display_cols = [
        c
        for c in [
            "year_month",
            "latest_segment",
            "cluster_label",
            "name",
            "ga_items_viewed",
            "gmv_per_ga_view",
            "ga_add_to_cart_rate",
            "ga_view_to_purchase_rate",
        ]
        if c in df.columns
    ]
    out = df[display_cols].copy().rename(
        columns={
            "year_month": "Month",
            "latest_segment": "Segment",
            "cluster_label": "Cluster",
            "name": "Restaurant",
            "ga_items_viewed": "GA Item Views",
            "gmv_per_ga_view": "GMV / GA View",
            "ga_add_to_cart_rate": "Add to Cart Rate",
            "ga_view_to_purchase_rate": "View to Purchase Rate",
        }
    )
    if "Month" in out.columns:
        out["Month"] = pd.to_datetime(out["Month"], errors="coerce").dt.strftime("%Y-%m")
    for col in ["Segment", "Cluster", "Restaurant"]:
        if col in out.columns:
            out[col] = out[col].fillna("Unknown").astype(str)
    if "GA Item Views" in out.columns:
        out["GA Item Views"] = pd.to_numeric(out["GA Item Views"], errors="coerce").round(0)
    if "GMV / GA View" in out.columns:
        out["GMV / GA View"] = out["GMV / GA View"].apply(_fmt_thb)
    for col in ["Add to Cart Rate", "View to Purchase Rate"]:
        if col in out.columns:
            out[col] = out[col].apply(_fmt_pct)
    return out


def _prepare_restaurant_ga_view_contribution(
    restaurant_rows: pd.DataFrame,
    max_restaurants: int = 15,
) -> pd.DataFrame:
    if restaurant_rows.empty:
        return pd.DataFrame()

    required_cols = {"year_month", "name", "monthly_gmv", "ga_items_viewed"}
    if not required_cols.issubset(restaurant_rows.columns):
        return pd.DataFrame()

    rows = restaurant_rows.copy()
    rows["year_month"] = pd.to_datetime(rows["year_month"], errors="coerce")
    rows["name"] = rows["name"].fillna("Unknown").astype(str)
    rows["monthly_gmv"] = pd.to_numeric(rows["monthly_gmv"], errors="coerce").fillna(0)
    rows["ga_items_viewed"] = pd.to_numeric(rows["ga_items_viewed"], errors="coerce").fillna(0)
    rows = rows.dropna(subset=["year_month"])
    if rows.empty:
        return pd.DataFrame()

    top_names = (
        rows.groupby("name")["monthly_gmv"]
        .sum()
        .sort_values(ascending=False)
        .head(max_restaurants)
        .index
    )
    rows["restaurant_group"] = np.where(rows["name"].isin(top_names), rows["name"], "Other restaurants")

    contribution = (
        rows.groupby(["year_month", "restaurant_group"], as_index=False)
        .agg(
            monthly_gmv=("monthly_gmv", "sum"),
            ga_items_viewed=("ga_items_viewed", "sum"),
        )
        .sort_values(["year_month", "monthly_gmv"], ascending=[True, False])
    )
    contribution["restaurant_gmv_per_ga_view"] = (
        contribution["monthly_gmv"] / contribution["ga_items_viewed"].replace(0, np.nan)
    )
    month_views = contribution.groupby("year_month")["ga_items_viewed"].transform("sum").replace(0, np.nan)
    contribution["ga_view_share"] = contribution["ga_items_viewed"] / month_views
    contribution["month_label"] = contribution["year_month"].dt.strftime("%Y-%m")
    return contribution.reset_index(drop=True)


def _prepare_campaign_start_markers(strategy_rows: pd.DataFrame, month_labels: list[str]) -> pd.DataFrame:
    if strategy_rows.empty or "applied_date" not in strategy_rows.columns or not month_labels:
        return pd.DataFrame()

    markers = strategy_rows.copy()
    markers["applied_date"] = pd.to_datetime(markers["applied_date"], errors="coerce")
    markers = markers.dropna(subset=["applied_date"])
    if markers.empty:
        return pd.DataFrame()

    if "strategy_family" not in markers.columns:
        markers["strategy_family"] = markers.get("strategy_name", "Unknown Strategy")
    markers["strategy_family"] = markers["strategy_family"].fillna("Unknown Strategy").astype(str)
    markers["campaign_name"] = markers.get("campaign_name", pd.Series("Unknown Campaign", index=markers.index))
    markers["campaign_name"] = markers["campaign_name"].fillna("Unknown Campaign").astype(str)
    markers["month_label"] = markers["applied_date"].dt.strftime("%Y-%m")
    markers = markers[markers["month_label"].isin(month_labels)].copy()
    if markers.empty:
        return pd.DataFrame()

    if "activity_id" in markers.columns:
        markers = markers.drop_duplicates(["activity_id", "month_label", "strategy_family"])
    else:
        markers = markers.drop_duplicates(["applied_date", "campaign_name", "strategy_family"])

    def _join_examples(group: pd.DataFrame) -> str:
        examples = (
            group.sort_values("applied_date")
            .assign(example=lambda df: df["applied_date"].dt.strftime("%Y-%m-%d") + " | " + df["campaign_name"])
            ["example"]
            .head(6)
            .tolist()
        )
        more = len(group) - len(examples)
        suffix = f"<br>+ {more} more" if more > 0 else ""
        return "<br>".join(html.escape(item) for item in examples) + suffix

    out = (
        markers.groupby(["month_label", "strategy_family"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "campaign_count": len(g),
                    "examples": _join_examples(g),
                }
            )
        )
        .reset_index(drop=True)
    )
    out["month_strategy_rank"] = out.groupby("month_label")["campaign_count"].rank(method="first", ascending=False) - 1
    return out.sort_values(["month_label", "month_strategy_rank"]).reset_index(drop=True)


# Prep campaign breakdown data for display, including renaming and formatting
def _prepare_campaign_breakdown_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    priority_cols = [
        "year_month",
        "googleAdsCampaignType",
        "campaign_name",
        "campaign_id",
        "sessions",
        "session_share",
    ]
    display_cols = [c for c in priority_cols if c in df.columns]
    display_cols += [c for c in df.columns if c not in display_cols]

    rename_map = {
        "year_month": "Year Month",
        "yearMonth": "Year Month",
        "googleAdsCampaignType": "GA Campaign Type",
        "campaign_name": "Campaign Name",
        "campaignName": "Campaign Name",
        "campaign_id": "Campaign ID",
        "campaignId": "Campaign ID",
        "sessions": "Sessions",
        "session_share": "Session Share",
    }
    out = df[display_cols].copy().rename(columns=rename_map)
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
            "revenue_before",
            "revenue_after",
            "revenue_uplift_pct",
            "activity_id",
        ]
        if c in df.columns
    ]
    out = df[display_cols].copy().rename(
        columns={
            "applied_date": "Start Date",
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
            "revenue_before": "Actual Revenue Before",
            "revenue_after": "Actual Revenue After",
            "revenue_uplift_pct": "Actual Revenue Uplift %",
            "activity_id": "Activity ID",
        }
    )
    if "Start Date" in out.columns:
        out["Start Date"] = pd.to_datetime(out["Start Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["Bookings Before", "Bookings After"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(0)
    if "Bookings Uplift %" in out.columns:
        out["Bookings Uplift %"] = out["Bookings Uplift %"].apply(_fmt_pct)
    for col in ["Actual Revenue Before", "Actual Revenue After"]:
        if col in out.columns:
            out[col] = out[col].apply(_fmt_thb)
    if "Actual Revenue Uplift %" in out.columns:
        out["Actual Revenue Uplift %"] = out["Actual Revenue Uplift %"].apply(_fmt_pct)
    return out

def _render_filterable_table(
    df: pd.DataFrame,
    key: str,
    height: int = 320,
    initial_sort_column: str | None = None,
    initial_sort_direction: str = "desc",
) -> None:
    if df.empty:
        st.info("No rows match the current filter.")
        return

    table_id = "filterable_" + hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    columns = df.columns.tolist()
    initial_sort_index = columns.index(initial_sort_column) if initial_sort_column in columns else -1
    safe_initial_direction = "asc" if str(initial_sort_direction).lower() == "asc" else "desc"
    header_cells = "".join(
        (
            "<th>"
            f"<button class='sort-button' type='button' onclick='sort_{table_id}({idx})' "
            f"title='Sort {html.escape(str(col))}'>"
            f"<span>{html.escape(str(col))}</span>"
            f"<span class='sort-indicator' data-sort-indicator='{idx}'></span>"
            "</button>"
            "</th>"
        )
        for idx, col in enumerate(columns)
    )
    filter_cells = "".join(
        (
            "<th>"
            f"<input class='filter-input' type='text' placeholder='Filter {html.escape(str(col))}' "
            f"onkeyup=\"filter_{table_id}()\" />"
            "</th>"
        )
        for col in columns
    )
    body_rows = []
    for _, row in df.fillna("").astype(str).iterrows():
        cells = "".join(f"<td>{html.escape(str(row[col]))}</td>" for col in columns)
        body_rows.append(f"<tr>{cells}</tr>")

    table_html = f"""
    <style>
    .filter-wrap {{
        height: {height}px;
        overflow: auto;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background: #ffffff;
    }}
    #{table_id} {{
        width: 100%;
        border-collapse: collapse;
        font-family: "DM Sans", Arial, sans-serif;
        font-size: 13px;
        color: #111827;
    }}
    #{table_id} th {{
        position: sticky;
        top: 0;
        z-index: 2;
        background: #f8fafc;
        border-bottom: 1px solid #d1d5db;
        padding: 8px 10px;
        text-align: left;
        white-space: nowrap;
    }}
    #{table_id} .sort-button {{
        width: 100%;
        border: 0;
        background: transparent;
        color: inherit;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        font: inherit;
        font-weight: 700;
        padding: 0;
        text-align: left;
    }}
    #{table_id} .sort-indicator {{
        color: #cc0000;
        font-size: 10px;
        min-width: 28px;
        text-align: right;
    }}
    #{table_id} thead tr.filters th {{
        top: 35px;
        background: #ffffff;
        z-index: 1;
        padding: 6px 8px;
    }}
    #{table_id} td {{
        border-bottom: 1px solid #eef2f7;
        padding: 8px 10px;
        vertical-align: top;
        white-space: nowrap;
    }}
    #{table_id} tbody tr:hover {{
        background: #fff7f7;
    }}
    #{table_id} .filter-input {{
        width: 100%;
        box-sizing: border-box;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 5px 7px;
        font-size: 12px;
    }}
    .filter-count {{
        margin: 6px 0 8px;
        color: #4b5563;
        font-family: "DM Sans", Arial, sans-serif;
        font-size: 12px;
    }}
    </style>
    <div id="{table_id}_count" class="filter-count">Showing {len(df):,} of {len(df):,} rows</div>
    <div class="filter-wrap">
        <table id="{table_id}">
            <thead>
                <tr>{header_cells}</tr>
                <tr class="filters">{filter_cells}</tr>
            </thead>
            <tbody>
                {''.join(body_rows)}
            </tbody>
        </table>
    </div>
    <script>
    const sortState_{table_id} = {{
        column: {initial_sort_index},
        direction: "{safe_initial_direction}"
    }};

    function parseSortValue_{table_id}(text) {{
        const raw = text.trim();
        if (!raw || raw === "-") {{
            return {{ type: "empty", value: null }};
        }}

        if (/^\\d{{4}}-\\d{{2}}(-\\d{{2}})?$/.test(raw)) {{
            return {{ type: "date", value: Date.parse(raw) }};
        }}

        const numericText = raw
            .replace(/THB/gi, "")
            .replace(/,/g, "")
            .replace(/%/g, "")
            .replace(/x$/gi, "")
            .replace(/^\\+/, "")
            .trim();

        if (/^-?\\d+(\\.\\d+)?$/.test(numericText)) {{
            return {{ type: "number", value: Number(numericText) }};
        }}

        return {{ type: "text", value: raw.toLowerCase() }};
    }}

    function compareValues_{table_id}(a, b) {{
        if (a.type === "empty" && b.type === "empty") return 0;
        if (a.type === "empty") return 1;
        if (b.type === "empty") return -1;

        if ((a.type === "number" || a.type === "date") && a.type === b.type) {{
            return a.value - b.value;
        }}

        return String(a.value).localeCompare(String(b.value), undefined, {{
            numeric: true,
            sensitivity: "base"
        }});
    }}

    function updateSortIndicators_{table_id}() {{
        const table = document.getElementById("{table_id}");
        const indicators = table.querySelectorAll("[data-sort-indicator]");
        indicators.forEach((indicator) => {{
            const colIndex = Number(indicator.dataset.sortIndicator);
            indicator.innerText = colIndex === sortState_{table_id}.column
                ? sortState_{table_id}.direction.toUpperCase()
                : "";
        }});
    }}

    function sort_{table_id}(columnIndex) {{
        const table = document.getElementById("{table_id}");
        const tbody = table.tBodies[0];
        const rows = Array.from(tbody.querySelectorAll("tr"));

        if (sortState_{table_id}.column === columnIndex) {{
            sortState_{table_id}.direction = sortState_{table_id}.direction === "asc" ? "desc" : "asc";
        }} else {{
            sortState_{table_id}.column = columnIndex;
            sortState_{table_id}.direction = "asc";
        }}

        rows.sort((rowA, rowB) => {{
            const valueA = parseSortValue_{table_id}(rowA.cells[columnIndex].innerText);
            const valueB = parseSortValue_{table_id}(rowB.cells[columnIndex].innerText);
            if (valueA.type === "empty" || valueB.type === "empty") {{
                return compareValues_{table_id}(valueA, valueB);
            }}
            const result = compareValues_{table_id}(valueA, valueB);
            return sortState_{table_id}.direction === "asc" ? result : -result;
        }});

        rows.forEach((row) => tbody.appendChild(row));
        updateSortIndicators_{table_id}();
    }}

    function filter_{table_id}() {{
        const table = document.getElementById("{table_id}");
        const inputs = table.querySelectorAll("thead tr.filters input");
        const rows = table.querySelectorAll("tbody tr");
        let shown = 0;
        rows.forEach((row) => {{
            const cells = row.querySelectorAll("td");
            let visible = true;
            inputs.forEach((input, index) => {{
                const query = input.value.toLowerCase().trim();
                if (query && !cells[index].innerText.toLowerCase().includes(query)) {{
                    visible = false;
                }}
            }});
            row.style.display = visible ? "" : "none";
            if (visible) shown += 1;
        }});
          document.getElementById("{table_id}_count").innerText = `Showing ${{shown.toLocaleString()}} of {len(df):,} rows`;
      }}

      updateSortIndicators_{table_id}();
      </script>
      """
    components.html(table_html, height=height + 70, scrolling=False)


def _render_scope_bar(
    scope_label: str,
    restaurant_count: int,
    restaurant_filter: str,
    segment_filter: str,
    cluster_filter: str,
) -> None:
    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:0.75rem;
            flex-wrap:wrap;
            border:1px solid #e5e7eb;
            border-radius:8px;
            padding:0.55rem 0.75rem;
            background:#fafafa;
            margin:0.2rem 0 1rem;
        ">
            <div style="display:flex; align-items:center; gap:0.55rem; min-width:0; flex-wrap:wrap;">
                <span style="font-size:0.72rem; color:#6b7280; font-weight:800; text-transform:uppercase;">Scope</span>
                <span style="font-size:0.92rem; color:#111827; font-weight:800;">{html.escape(scope_label)}</span>
            </div>
            <div style="display:flex; gap:0.45rem; flex-wrap:wrap;">
                <span style="font-size:0.76rem; color:#374151; background:#ffffff; border:1px solid #e5e7eb; border-radius:999px; padding:0.18rem 0.55rem;">
                    {restaurant_count:,} restaurants
                </span>
                <span style="font-size:0.76rem; color:#374151; background:#ffffff; border:1px solid #e5e7eb; border-radius:999px; padding:0.18rem 0.55rem;">
                    Restaurant: {html.escape(restaurant_filter)}
                </span>
                <span style="font-size:0.76rem; color:#374151; background:#ffffff; border:1px solid #e5e7eb; border-radius:999px; padding:0.18rem 0.55rem;">
                    Segment: {html.escape(segment_filter)}
                </span>
                <span style="font-size:0.76rem; color:#374151; background:#ffffff; border:1px solid #e5e7eb; border-radius:999px; padding:0.18rem 0.55rem;">
                    Cluster: {html.escape(cluster_filter)}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render():
    assignments = load_cluster_assignments() # Load from clustering_results
    ga_campaign_raw = load_ga_campaign_outreach_raw()
    ga_restaurant_monthly = load_ga_restaurant_monthly()
    outcomes_df = load_cluster_strategy_outcomes()
    if "selected_strategy_family" not in st.session_state:
        st.session_state["selected_strategy_family"] = "All"
    if "selected_ga_campaign_type" not in st.session_state:
        st.session_state["selected_ga_campaign_type"] = "All"

    st.markdown("## Clustering Explorer")
    st.markdown(
        f"<p style='color:{MUTED_TEXT}; margin-top:-0.5rem;'>"
        "Compare strategy outcomes and Google Ads campaign effectiveness for the current navbar scope."
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
    cluster_label_map = dict(zip(cluster_rows["cluster_id"], cluster_rows["cluster_label"])) # Map cluster_id to cluster_label for display in selection box, handle missing labels

    if not cluster_options: # No valid clusters after processing, show warning and exit
        st.warning("No valid clusters available in clustering data.")
        return

    selected_restaurant_filter = str(st.session_state.get("selected_restaurant", "All") or "All").strip()
    selected_segment_filter = str(st.session_state.get("selected_segment", "All") or "All").strip()
    selected_cluster_filter = st.session_state.get("selected_cluster", "All")
    if not selected_restaurant_filter or selected_restaurant_filter.lower() in {"none", "nan"}:
        selected_restaurant_filter = "All"
    if not selected_segment_filter or selected_segment_filter.lower() in {"none", "nan"}:
        selected_segment_filter = "All"
    selected_cluster_value = None
    if str(selected_cluster_filter).strip() not in {"", "All", "All Clusters", "None", "nan"}:
        selected_cluster_numeric = pd.to_numeric(pd.Series([selected_cluster_filter]), errors="coerce").iloc[0]
        if pd.notna(selected_cluster_numeric):
            selected_cluster_value = int(selected_cluster_numeric)
    selected_cluster_label = (
        "All"
        if selected_cluster_value is None
        else f"Cluster {selected_cluster_value}: {cluster_label_map.get(selected_cluster_value, 'Unknown')}"
    )

    restaurant_selected = selected_restaurant_filter != "All"
    segment_selected = selected_segment_filter != "All"
    cluster_selected = selected_cluster_value is not None

    segment_assignments = assignments.copy()
    if cluster_selected:
        segment_assignments = segment_assignments[segment_assignments["cluster_id"] == selected_cluster_value].copy()
    if segment_selected and "latest_segment" in segment_assignments.columns:
        segment_assignments = segment_assignments[
            segment_assignments["latest_segment"].astype(str).eq(selected_segment_filter)
        ].copy()
    elif segment_selected:
        segment_assignments = segment_assignments.iloc[0:0].copy()

    restaurant_lookup = pd.DataFrame()
    if restaurant_selected:
        restaurant_lookup = assignments[
            assignments["name_norm"] == _normalize_name(selected_restaurant_filter)
        ].head(1)
        if restaurant_lookup.empty:
            st.warning(
                f"Navbar restaurant filter '{selected_restaurant_filter}' was not found in the clustering data. "
                "Showing the broader navbar segment scope instead."
            )
            restaurant_selected = False
            selected_restaurant_filter = "All"

    if restaurant_selected and not restaurant_lookup.empty:
        active_restaurant = str(restaurant_lookup.iloc[0].get("name", selected_restaurant_filter))
        active_cluster_raw = pd.to_numeric(
            pd.Series([restaurant_lookup.iloc[0].get("cluster_id")]),
            errors="coerce",
        ).iloc[0]
        restaurant_cluster = int(active_cluster_raw) if pd.notna(active_cluster_raw) else all_clusters_option
        active_cluster = selected_cluster_value if cluster_selected else restaurant_cluster
        cluster_df = (
            assignments[assignments["cluster_id"] == int(active_cluster)].copy()
            if active_cluster != all_clusters_option
            else assignments.copy()
        )
        if segment_selected and "latest_segment" in cluster_df.columns:
            cluster_df = cluster_df[cluster_df["latest_segment"].astype(str).eq(selected_segment_filter)].copy()
    else:
        active_cluster = selected_cluster_value if cluster_selected else all_clusters_option
        cluster_df = segment_assignments.copy()
        active_restaurant = cluster_df.iloc[0]["name"] if len(cluster_df) else assignments.iloc[0]["name"]

    cluster_df = cluster_df.sort_values("name").copy()
    active_rest_norm = _normalize_name(active_restaurant)
    selected_restaurant_row = assignments[assignments["name_norm"] == active_rest_norm].head(1)
    selected_cluster_ids = (
        pd.to_numeric(cluster_df.get("cluster_id", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    def _filter_by_selected_cluster(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "cluster_id" in out.columns:
            if active_cluster == all_clusters_option:
                if segment_selected and "latest_segment" not in out.columns:
                    out = out[pd.to_numeric(out["cluster_id"], errors="coerce").isin(selected_cluster_ids)].copy()
            else:
                out = out[pd.to_numeric(out["cluster_id"], errors="coerce") == int(active_cluster)].copy()
                if segment_selected and "latest_segment" not in out.columns and not selected_cluster_ids:
                    out = out.iloc[0:0].copy()
        if segment_selected and "latest_segment" in out.columns:
            out = out[out["latest_segment"].astype(str).eq(selected_segment_filter)].copy()
        return out

    scope_parts = []
    if cluster_selected:
        scope_parts.append(selected_cluster_label)
    if restaurant_selected:
        if cluster_selected:
            scope_parts.append(f"Restaurant: {active_restaurant}")
        else:
            cluster_name = cluster_label_map.get(active_cluster, "Unknown")
            scope_parts.append(f"{active_restaurant} / Cluster {active_cluster}: {cluster_name}")
    if segment_selected:
        scope_parts.append(f"Segment: {selected_segment_filter}")
    scope_label = " | ".join(scope_parts) if scope_parts else all_clusters_option

    min_sample_size = 3
    _render_scope_bar(
        scope_label,
        len(cluster_df),
        selected_restaurant_filter,
        selected_segment_filter,
        selected_cluster_label,
    )
    if restaurant_selected and active_rest_norm not in set(cluster_df.get("name_norm", pd.Series(dtype=str))):
        st.warning(
            "The selected restaurant is outside the current cluster/segment scope, so restaurant-specific rows may be empty."
        )

    st.markdown("### Marketing Strategy Effectiveness")
    with st.container():
        strategy_scope_outcomes = _filter_by_selected_cluster(outcomes_df)
        if restaurant_selected and "restaurant_name" in strategy_scope_outcomes.columns:
            strategy_scope_outcomes = strategy_scope_outcomes[
                strategy_scope_outcomes["restaurant_name"].apply(_normalize_name).eq(active_rest_norm)
            ].copy()
        cluster_rank = _aggregate_strategy_scope(
            strategy_scope_outcomes,
            scope_label=scope_label,
            min_sample_size=int(min_sample_size),
        )

        if cluster_rank.empty:
            st.info("No strategy outcomes available for this navbar filter scope.")
        else:
            st.caption(
                "Revenue Uplift % compares actual revenue after the campaign window against the matching before window; "
                "Bookings Uplift % uses the same before/after window."
            )

            if active_cluster == all_clusters_option and segment_selected:
                best_recs = (
                    cluster_rank[cluster_rank["meets_sample_guardrail"]]
                    .sort_values("ranking_score", ascending=False)
                    .head(3)
                )
                if best_recs.empty:
                    best_recs = cluster_rank.sort_values("ranking_score", ascending=False).head(3)
            elif active_cluster == all_clusters_option:
                best_recs = (
                    cluster_rank[cluster_rank["meets_sample_guardrail"]]
                    .sort_values("ranking_score", ascending=False)
                    .head(3)
                )
                if best_recs.empty:
                    best_recs = cluster_rank.sort_values("ranking_score", ascending=False).head(3)
            else:
                best_recs = (
                    cluster_rank[cluster_rank["meets_sample_guardrail"]]
                    .sort_values("ranking_score", ascending=False)
                    .head(3)
                )
                if best_recs.empty:
                    best_recs = cluster_rank.sort_values("ranking_score", ascending=False).head(3)
            if best_recs.empty:
                st.warning("No strategies meet the sample-size guardrail for this navbar filter scope.")
            else:
                st.caption(
                    "Top strategy cards show average Revenue Uplift % and Bookings Uplift % from matched campaign windows."
                )
                top_cols = st.columns(min(3, len(best_recs)))
                for idx, (_, rec) in enumerate(best_recs.iterrows()):
                    with top_cols[idx]:
                        strategy_value = str(rec.get("strategy_name", "Unknown Strategy"))
                        strategy_name = html.escape(strategy_value)
                        revenue_uplift = _fmt_pct_value(rec.get("avg_revenue_uplift_pct"))
                        bookings_uplift = _fmt_pct_value(rec.get("avg_bookings_uplift_pct"))
                        activities = rec.get("activities")
                        restaurants = rec.get("restaurants")
                        activity_text = f"{int(_num_or_zero(activities)):,}"
                        restaurant_text = f"{int(_num_or_zero(restaurants)):,}"
                        st.markdown(
                            f"""
                            <div style="
                                border:1px solid #e5e7eb;
                                border-radius:8px;
                                padding:0.85rem 0.9rem;
                                background:#ffffff;
                                min-height:150px;
                            ">
                                <div style="font-size:0.78rem; color:#6b7280; font-weight:700;">#{idx + 1} Strategy</div>
                                <div style="
                                    margin-top:0.15rem;
                                    font-size:0.95rem;
                                    color:#111827;
                                    font-weight:800;
                                    line-height:1.25;
                                    min-height:2.4rem;
                                ">{strategy_name}</div>
                                <div style="
                                    display:grid;
                                    grid-template-columns:1fr 1fr;
                                    gap:0.65rem;
                                    margin-top:0.85rem;
                                ">
                                    <div>
                                        <div style="font-size:0.72rem; color:#6b7280; font-weight:700;">Revenue Uplift %</div>
                                        <div style="font-size:1.35rem; color:#cc0000; font-weight:850;">{revenue_uplift}</div>
                                    </div>
                                    <div>
                                        <div style="font-size:0.72rem; color:#6b7280; font-weight:700;">Bookings Uplift %</div>
                                        <div style="font-size:1.35rem; color:#111827; font-weight:850;">{bookings_uplift}</div>
                                    </div>
                                </div>
                                <div style="margin-top:0.75rem; font-size:0.75rem; color:#6b7280;">
                                    {activity_text} activities across {restaurant_text} restaurants
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            f"Filter: {strategy_value}",
                            key=f"strategy_card_filter_{idx}_{hashlib.md5(strategy_value.encode('utf-8')).hexdigest()[:8]}",
                            width="stretch",
                        ):
                            st.session_state["selected_strategy_family"] = strategy_value
                st.markdown("<div style='height:0.95rem;'></div>", unsafe_allow_html=True)

            rank_display_source = cluster_rank.sort_values(
                ["meets_sample_guardrail", "ranking_score"],
                ascending=[False, False],
            )
            scope_col = "scope_label" if "scope_label" in rank_display_source.columns else "cluster_label"
            rank_display = rank_display_source[
                [
                    scope_col,
                    "strategy_name",
                    "activities",
                    "restaurants",
                    "avg_revenue_uplift_pct",
                    "avg_bookings_uplift_pct",
                ]
            ].copy()

            rank_display = rank_display.rename(
                columns={
                    scope_col: "Scope",
                    "strategy_name": "Strategy",
                    "activities": "Activities",
                    "restaurants": "Restaurants",
                    "avg_revenue_uplift_pct": "Avg Revenue Uplift %",
                    "avg_bookings_uplift_pct": "Avg Bookings Uplift %",
                }
            )

            for col in ["Avg Revenue Uplift %", "Avg Bookings Uplift %"]:
                rank_display[col] = rank_display[col].apply(_fmt_pct)

            st.dataframe(rank_display, hide_index=True, width="stretch", height=280)

            st.markdown("#### Raw Campaign Strategy Breakdown")
            raw_strategy_campaigns = strategy_scope_outcomes.copy()

            if raw_strategy_campaigns.empty:
                st.info("No raw campaign rows available for this strategy scope.")
            else:
                if "strategy_family" not in raw_strategy_campaigns.columns:
                    raw_strategy_campaigns["strategy_family"] = "Unknown"
                raw_strategy_campaigns["strategy_family"] = raw_strategy_campaigns["strategy_family"].fillna("Unknown")
                available_strategies = sorted(raw_strategy_campaigns["strategy_family"].dropna().astype(str).unique())
                selected_strategy_family = str(st.session_state.get("selected_strategy_family", "All") or "All")
                if selected_strategy_family != "All" and selected_strategy_family not in available_strategies:
                    selected_strategy_family = "All"
                    st.session_state["selected_strategy_family"] = "All"

                strategy_filter_cols = st.columns([2, 1])
                with strategy_filter_cols[0]:
                    strategy_options = ["All"] + available_strategies
                    selected_strategy_family = st.selectbox(
                        "Assigned Strategy",
                        strategy_options,
                        index=strategy_options.index(selected_strategy_family)
                        if selected_strategy_family in strategy_options else 0,
                        key="selected_strategy_family",
                    )
                with strategy_filter_cols[1]:
                    st.markdown("<div style='height:1.72rem;'></div>", unsafe_allow_html=True)
                    st.button("Clear strategy filter", on_click=_clear_strategy_filter, width="stretch")

                if selected_strategy_family != "All":
                    raw_strategy_campaigns = raw_strategy_campaigns[
                        raw_strategy_campaigns["strategy_family"].astype(str).eq(selected_strategy_family)
                    ].copy()

                strategy_mix = (
                    raw_strategy_campaigns.assign(
                        bookings_uplift_pct=pd.to_numeric(raw_strategy_campaigns.get("bookings_uplift_pct"), errors="coerce"),
                        revenue_uplift_pct=pd.to_numeric(raw_strategy_campaigns.get("revenue_uplift_pct"), errors="coerce"),
                    )
                    .groupby("strategy_family", as_index=False)
                    .agg(
                        activities=("activity_id", "nunique"),
                        avg_bookings_uplift_pct=("bookings_uplift_pct", "mean"),
                        avg_revenue_uplift_pct=("revenue_uplift_pct", "mean"),
                    )
                    .sort_values("activities", ascending=False)
                )
                strategy_mix["activity_share"] = np.where(
                    strategy_mix["activities"].sum() > 0,
                    strategy_mix["activities"] / strategy_mix["activities"].sum(),
                    np.nan,
                )

                if not strategy_mix.empty:
                    strategy_custom = np.stack(
                        [
                            strategy_mix["avg_bookings_uplift_pct"].apply(_fmt_pct).to_numpy(),
                            strategy_mix["avg_revenue_uplift_pct"].apply(_fmt_pct).to_numpy(),
                        ],
                        axis=-1,
                    )
                    fig_strategy_mix = go.Figure(
                        go.Bar(
                            x=strategy_mix["strategy_family"],
                            y=strategy_mix["activities"],
                            marker_color="#cc0000",
                            text=strategy_mix["activity_share"].apply(lambda v: "-" if pd.isna(v) else f"{v:.0%}"),
                            textposition="outside",
                            customdata=strategy_custom,
                            hovertemplate=(
                                "%{x}<br>"
                                "Activities: %{y:,.0f}<br>"
                                "Share: %{text}<br>"
                                "Avg Bookings Uplift: %{customdata[0]}<br>"
                                "Avg Revenue Uplift: %{customdata[1]}<extra></extra>"
                            ),
                        )
                    )
                    fig_strategy_mix.update_layout(
                        **BASE_LAYOUT,
                        height=260,
                        showlegend=False,
                        xaxis=dict(**CHART_THEME["xaxis"], title="Assigned Strategy", tickangle=-20),
                        yaxis=dict(**CHART_THEME["yaxis"], title="Activities"),
                    )
                    st.plotly_chart(fig_strategy_mix, width="stretch")

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
                raw_strategy_campaigns["_booking_uplift_sort"] = pd.to_numeric(
                    raw_strategy_campaigns.get("bookings_uplift_pct"),
                    errors="coerce",
                )
                raw_strategy_campaigns = (
                    raw_strategy_campaigns.sort_values(
                        ["_booking_uplift_sort", "applied_date", "channel", "campaign_name"],
                        ascending=[False, False, True, True],
                        na_position="last",
                    )
                    .drop(columns=["_booking_uplift_sort"])
                    .reset_index(drop=True)
                )
                st.caption(
                    f"{len(raw_strategy_campaigns):,} campaign rows mapped to "
                    f"{raw_strategy_campaigns['strategy_family'].nunique():,} assigned strategies. "
                    "Sorted by Bookings Uplift % from highest to lowest; click column headers to toggle ASC/DESC "
                    "and use the filter boxes inside the table headers."
                )
                strategy_campaign_display = _prepare_strategy_campaign_display(raw_strategy_campaigns)
                _render_filterable_table(
                    strategy_campaign_display,
                    key=f"raw_strategy_campaign_table_{str(active_cluster)}_{active_rest_norm}_{selected_segment_filter}",
                    height=360,
                    initial_sort_column="Bookings Uplift %",
                    initial_sort_direction="desc",
                )

    st.markdown("---")
    st.markdown("### GA Campaign Effectiveness")
    ga_scope_assignments = cluster_df.copy()
    if restaurant_selected:
        ga_scope_assignments = ga_scope_assignments[ga_scope_assignments["name_norm"] == active_rest_norm].copy()
    scope_name_norms = set(ga_scope_assignments.get("name_norm", pd.Series(dtype=str)).dropna().astype(str))
    scope_ga_monthly = _prepare_scope_ga_monthly(ga_restaurant_monthly, assignments, scope_name_norms)
    scope_ga_rows = _prepare_scope_ga_restaurant_rows(ga_restaurant_monthly, assignments, scope_name_norms)

    if scope_ga_monthly.empty:
        st.info("No GA campaign effectiveness data available for this scope.")
    else:
        st.caption(
            "Observed GA quality for the current navbar scope. GMV / GA View, Add to Cart Rate, "
            "and View to Purchase Rate come from restaurant-level GA item activity and GMV; no sessions "
            "are allocated in this section."
        )

        total_ga_views = pd.to_numeric(scope_ga_monthly["total_ga_items_viewed"], errors="coerce").fillna(0).sum()
        total_gmv = pd.to_numeric(scope_ga_monthly["total_monthly_gmv"], errors="coerce").fillna(0).sum()
        total_added_to_cart = pd.to_numeric(scope_ga_monthly["total_ga_items_added_to_cart"], errors="coerce").fillna(0).sum()
        total_purchased = pd.to_numeric(scope_ga_monthly["total_ga_items_purchased"], errors="coerce").fillna(0).sum()
        summary_metrics = [
            ("GA Item Views", f"{total_ga_views:,.0f}", "#111827"),
            ("GMV / GA View", _fmt_thb(total_gmv / total_ga_views) if total_ga_views > 0 else "-", "#cc0000"),
            ("Add to Cart Rate", _fmt_pct(total_added_to_cart / total_ga_views) if total_ga_views > 0 else "-", "#2563eb"),
            ("View to Purchase Rate", _fmt_pct(total_purchased / total_ga_views) if total_ga_views > 0 else "-", "#047857"),
        ]
        metric_cols = st.columns(4)
        for idx, (label, value, color) in enumerate(summary_metrics):
            with metric_cols[idx]:
                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #e5e7eb;
                        border-radius:8px;
                        padding:0.85rem 0.9rem;
                        background:#ffffff;
                        min-height:96px;
                    ">
                        <div style="font-size:0.74rem; color:#6b7280; font-weight:750;">{html.escape(label)}</div>
                        <div style="
                            margin-top:0.35rem;
                            font-size:1.45rem;
                            line-height:1.15;
                            color:{color};
                            font-weight:850;
                        ">{html.escape(value)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)

        chart_df = scope_ga_monthly.copy()
        chart_df["month_label"] = pd.to_datetime(chart_df["year_month"], errors="coerce").dt.strftime("%Y-%m")
        fig_ga_quality = go.Figure()
        restaurant_contribution = _prepare_restaurant_ga_view_contribution(scope_ga_rows)
        if restaurant_contribution.empty:
            fig_ga_quality.add_trace(
                go.Bar(
                    x=chart_df["month_label"],
                    y=chart_df["total_ga_items_viewed"],
                    name="GA Item Views",
                    marker_color="#cc0000",
                    hovertemplate="%{x}<br>GA Item Views: %{y:,.0f}<extra></extra>",
                )
            )
        else:
            restaurant_names = (
                restaurant_contribution.groupby("restaurant_group")["monthly_gmv"]
                .sum()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
            palette = px.colors.qualitative.Safe + px.colors.qualitative.Vivid + px.colors.qualitative.Set3
            for idx, restaurant_name in enumerate(restaurant_names):
                rest_df = restaurant_contribution[
                    restaurant_contribution["restaurant_group"].eq(restaurant_name)
                ].copy()
                rest_df = chart_df[["month_label"]].merge(rest_df, on="month_label", how="left")
                customdata = np.stack(
                    [
                        rest_df["restaurant_gmv_per_ga_view"],
                        rest_df["monthly_gmv"],
                        rest_df["ga_view_share"],
                    ],
                    axis=-1,
                )
                fig_ga_quality.add_trace(
                    go.Bar(
                        x=rest_df["month_label"],
                        y=rest_df["ga_items_viewed"].fillna(0),
                        name=restaurant_name,
                        marker_color="#9ca3af" if restaurant_name == "Other restaurants" else palette[idx % len(palette)],
                        customdata=customdata,
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "%{x}<br>"
                            "GA item views: %{y:,.0f}<br>"
                            "Restaurant GMV / GA View: THB %{customdata[0]:,.0f}<br>"
                            "GMV: THB %{customdata[1]:,.0f}<br>"
                            "GA view share: %{customdata[2]:.1%}<extra></extra>"
                        ),
                    )
                )
        fig_ga_quality.add_trace(
            go.Scatter(
                x=chart_df["month_label"],
                y=chart_df["gmv_per_ga_view"],
                name="GMV / GA View",
                mode="lines+markers",
                line=dict(color="#cc0000", width=3),
                marker=dict(size=7),
                yaxis="y2",
                hovertemplate="%{x}<br>GMV / GA View: THB %{y:,.0f}<extra></extra>",
            )
        )
        fig_ga_quality.add_trace(
            go.Scatter(
                x=chart_df["month_label"],
                y=chart_df["ga_add_to_cart_rate"],
                name="Add to Cart Rate",
                mode="lines+markers",
                line=dict(color="#2563eb", width=3, dash="dot"),
                marker=dict(size=7),
                yaxis="y3",
                hovertemplate="%{x}<br>Add to Cart Rate: %{y:.1%}<extra></extra>",
            )
        )
        campaign_markers = _prepare_campaign_start_markers(
            strategy_scope_outcomes,
            chart_df["month_label"].dropna().astype(str).tolist(),
        )
        if not campaign_markers.empty:
            marker_positions = chart_df[["month_label", "total_ga_items_viewed"]].copy()
            marker_positions["total_ga_items_viewed"] = pd.to_numeric(
                marker_positions["total_ga_items_viewed"], errors="coerce"
            ).fillna(0)
            campaign_markers = campaign_markers.merge(marker_positions, on="month_label", how="left")
            campaign_markers["marker_y"] = campaign_markers["total_ga_items_viewed"] * (
                1.08 + campaign_markers["month_strategy_rank"] * 0.08
            )
            campaign_markers["marker_y"] = campaign_markers["marker_y"].where(
                campaign_markers["marker_y"].gt(0),
                1 + campaign_markers["month_strategy_rank"],
            )
            marker_palette = px.colors.qualitative.Bold + px.colors.qualitative.Safe + px.colors.qualitative.Vivid
            strategy_color_map = {
                strategy: marker_palette[idx % len(marker_palette)]
                for idx, strategy in enumerate(sorted(campaign_markers["strategy_family"].unique()))
            }
            marker_customdata = np.stack(
                [
                    campaign_markers["strategy_family"],
                    campaign_markers["campaign_count"],
                    campaign_markers["examples"],
                ],
                axis=-1,
            )
            fig_ga_quality.add_trace(
                go.Scatter(
                    x=campaign_markers["month_label"],
                    y=campaign_markers["marker_y"],
                    name="Marketing starts",
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=13,
                        color=campaign_markers["strategy_family"].map(strategy_color_map),
                        line=dict(color="#111827", width=1),
                    ),
                    customdata=marker_customdata,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "%{x}<br>"
                        "Campaign starts: %{customdata[1]}<br>"
                        "%{customdata[2]}<extra></extra>"
                    ),
                )
            )
        fig_ga_quality.update_layout(
            **BASE_LAYOUT,
            height=340,
            barmode="stack",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(**CHART_THEME["xaxis"], title="Month"),
            yaxis=dict(**CHART_THEME["yaxis"], title="GA Item Views"),
            yaxis2=dict(
                title="GMV / GA View",
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
            ),
            yaxis3=dict(
                title="Add to Cart Rate",
                overlaying="y",
                side="right",
                anchor="free",
                position=0.94,
                tickformat=".0%",
                showgrid=False,
                zeroline=False,
            ),
        )
        st.plotly_chart(fig_ga_quality, width="stretch")
        st.caption("Triangle markers show marketing strategy campaign starts in that month; hover to see campaign dates and names.")

        ga_quality_display = _prepare_ga_quality_display(scope_ga_rows)
        st.caption("Restaurant-month GA quality rows for the current navbar scope.")
        _render_filterable_table(
            ga_quality_display,
            key=f"ga_quality_table_{str(active_cluster)}_{active_rest_norm}_{selected_segment_filter}",
            height=380,
            initial_sort_column="Month",
            initial_sort_direction="desc",
        )

    st.markdown("#### Raw GA Campaign Breakdown")
    st.caption(
        "Source: `data/marketing/googleAPI/campaigns_outreach.parquet`. "
        "Rows are raw GA campaign sessions by campaign and month. These are not allocated to clusters "
        "or restaurants."
    )

    if ga_campaign_raw.empty:
        st.info("No raw GA campaign outreach rows found.")
    else:
        campaign_scope = ga_campaign_raw.copy()
        if not scope_ga_monthly.empty:
            scope_months = set(pd.to_datetime(scope_ga_monthly["year_month"], errors="coerce").dropna())
            campaign_scope = campaign_scope[pd.to_datetime(campaign_scope["year_month"], errors="coerce").isin(scope_months)].copy()

        if campaign_scope.empty:
            st.info("No campaign outreach rows match the available month window.")
        else:
            campaign_scope["sessions"] = pd.to_numeric(campaign_scope["sessions"], errors="coerce").fillna(0)
            total_sessions = campaign_scope["sessions"].sum()
            st.caption(f"Raw GA campaign rows total {total_sessions:,.0f} sessions in the visible month window.")
            campaign_scope["session_share"] = np.where(
                total_sessions > 0,
                campaign_scope["sessions"] / total_sessions,
                np.nan,
            )

            mix_df = (
                campaign_scope.groupby("googleAdsCampaignType", as_index=False)["sessions"]
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
            campaign_mix_event = st.plotly_chart(
                fig_campaign_mix,
                width="stretch",
                key=f"raw_ga_campaign_mix_{str(active_cluster)}_{active_rest_norm}_{selected_segment_filter}",
                on_select="rerun",
                selection_mode="points",
            )
            selected_campaign_type_from_chart = _get_plotly_selected_x(campaign_mix_event)
            if selected_campaign_type_from_chart:
                st.session_state["selected_ga_campaign_type"] = selected_campaign_type_from_chart

            available_campaign_types = sorted(
                campaign_scope["googleAdsCampaignType"].fillna("Unknown").astype(str).unique()
            )
            selected_ga_campaign_type = str(st.session_state.get("selected_ga_campaign_type", "All") or "All")
            if selected_ga_campaign_type != "All" and selected_ga_campaign_type not in available_campaign_types:
                selected_ga_campaign_type = "All"
                st.session_state["selected_ga_campaign_type"] = "All"

            ga_filter_cols = st.columns([2, 1])
            with ga_filter_cols[0]:
                st.caption(
                    "Click a bar to filter the raw table by GA Campaign Type."
                    + (
                        f" Active filter: {selected_ga_campaign_type}."
                        if selected_ga_campaign_type != "All"
                        else ""
                    )
                )
            with ga_filter_cols[1]:
                st.button("Clear GA type filter", on_click=_clear_ga_campaign_type_filter, width="stretch")

            filtered_campaign_scope = campaign_scope.copy()
            if selected_ga_campaign_type != "All":
                filtered_campaign_scope = filtered_campaign_scope[
                    filtered_campaign_scope["googleAdsCampaignType"]
                    .fillna("Unknown")
                    .astype(str)
                    .eq(selected_ga_campaign_type)
                ].copy()

            campaign_display = _prepare_campaign_breakdown_display(
                filtered_campaign_scope.sort_values(["sessions", "campaign_name"], ascending=[False, True])
            )
            st.caption("Click column headers to toggle ASC/DESC and use the filter boxes inside the table headers.")
            _render_filterable_table(
                campaign_display,
                key=(
                    f"raw_ga_campaign_table_{str(active_cluster)}_{active_rest_norm}_"
                    f"{selected_segment_filter}_{selected_ga_campaign_type}"
                ),
                height=320,
                initial_sort_column="Sessions",
                initial_sort_direction="desc",
            )
