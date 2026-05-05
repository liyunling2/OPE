# -*- coding: utf-8 -*-
"""
pages/clustering.py
Cluster exploration dashboard with cross-highlighting and strategy effectiveness ranking.
"""

from __future__ import annotations

import html, re, hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from data.loader import (
    load_cluster_assignments,
    load_ga_restaurant_monthly,
    load_ga_campaign_outreach_raw,
    load_ga_campaign_type_monthly,
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

def _min_max_norm(series: pd.Series, neutral: float = 0.5) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(neutral, index=series.index, dtype=float)
    span = valid.max() - valid.min()
    if span == 0 or pd.isna(span):
        return pd.Series(neutral, index=series.index, dtype=float)
    return (numeric - valid.min()) / span

def _empty_ga_effectiveness() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "scope_label",
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
    )

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

def _build_scope_ga_effectiveness(
    scope_monthly: pd.DataFrame,
    campaign_type_monthly: pd.DataFrame,
    scope_label: str,
) -> pd.DataFrame:
    if scope_monthly.empty or campaign_type_monthly.empty:
        return _empty_ga_effectiveness()

    campaign_monthly = campaign_type_monthly.copy()
    campaign_monthly["year_month"] = pd.to_datetime(campaign_monthly["year_month"], errors="coerce")
    campaign_monthly["total_sessions"] = pd.to_numeric(campaign_monthly.get("total_sessions"), errors="coerce").fillna(0)
    campaign_monthly["active_campaigns"] = pd.to_numeric(campaign_monthly.get("active_campaigns"), errors="coerce").fillna(0)
    campaign_monthly = campaign_monthly.dropna(subset=["year_month"])
    if campaign_monthly.empty:
        return _empty_ga_effectiveness()

    merged = scope_monthly.merge(campaign_monthly, on="year_month", how="inner")
    if merged.empty:
        return _empty_ga_effectiveness()

    merged["total_sessions"] = (
        pd.to_numeric(merged["total_sessions"], errors="coerce").fillna(0)
        * pd.to_numeric(merged["scope_ga_view_share"], errors="coerce").fillna(0)
    )
    merged["active_campaigns"] = (
        pd.to_numeric(merged["active_campaigns"], errors="coerce").fillna(0)
        * pd.to_numeric(merged["scope_ga_view_share"], errors="coerce").fillna(0)
    )

    rows = []
    for campaign_type, grp in merged.groupby("googleAdsCampaignType", dropna=False):
        weights = pd.to_numeric(grp["total_sessions"], errors="coerce").fillna(0)
        valid_weight = weights.gt(0)

        def _weighted_metric_mean(col: str) -> float:
            metric = pd.to_numeric(grp[col], errors="coerce")
            valid = valid_weight & metric.notna()
            if valid.any():
                return float(np.average(metric[valid], weights=weights[valid]))
            fallback = metric.mean()
            return float(fallback) if pd.notna(fallback) else np.nan

        corr = np.nan
        if grp["year_month"].nunique() >= 3:
            corr = grp["total_sessions"].corr(grp["gmv_per_ga_view"])

        rows.append(
            {
                "scope_label": scope_label,
                "googleAdsCampaignType": campaign_type,
                "active_months": int(grp["year_month"].nunique()),
                "total_sessions": float(grp["total_sessions"].sum()),
                "active_campaigns": float(grp["active_campaigns"].sum()),
                "session_weighted_gmv_per_ga_view": _weighted_metric_mean("gmv_per_ga_view"),
                "session_weighted_add_to_cart_rate": _weighted_metric_mean("ga_add_to_cart_rate"),
                "session_weighted_view_to_purchase_rate": _weighted_metric_mean("ga_view_to_purchase_rate"),
                "sessions_to_gmv_correlation": float(corr) if pd.notna(corr) else np.nan,
            }
        )

    effect_df = pd.DataFrame(rows)
    if effect_df.empty:
        return _empty_ga_effectiveness()

    effect_df["gmv_norm"] = _min_max_norm(effect_df["session_weighted_gmv_per_ga_view"], neutral=0.5)
    effect_df["atc_norm"] = _min_max_norm(effect_df["session_weighted_add_to_cart_rate"], neutral=0.5)
    corr_norm = effect_df["sessions_to_gmv_correlation"].clip(-1, 1)
    corr_norm = (corr_norm.fillna(0) + 1) / 2
    effect_df["ga_effectiveness_score"] = (
        0.55 * effect_df["gmv_norm"]
        + 0.25 * effect_df["atc_norm"]
        + 0.20 * corr_norm
    ) * 100
    effect_df = effect_df.drop(columns=["gmv_norm", "atc_norm"])
    return effect_df.sort_values(
        ["ga_effectiveness_score", "session_weighted_gmv_per_ga_view"],
        ascending=[False, False],
    ).reset_index(drop=True)

def _allocate_campaign_breakdown_to_scope(
    ga_campaign_raw: pd.DataFrame,
    scope_monthly: pd.DataFrame,
) -> pd.DataFrame:
    if ga_campaign_raw.empty or scope_monthly.empty:
        return pd.DataFrame()

    allocation = scope_monthly[["year_month", "scope_ga_view_share"]].copy()
    allocation["year_month"] = pd.to_datetime(allocation["year_month"], errors="coerce")
    allocation["scope_ga_view_share"] = pd.to_numeric(allocation["scope_ga_view_share"], errors="coerce").fillna(0)
    allocation = allocation.dropna(subset=["year_month"])
    if allocation.empty:
        return pd.DataFrame()

    campaign_scope = ga_campaign_raw.copy()
    campaign_scope["year_month"] = pd.to_datetime(campaign_scope["year_month"], errors="coerce")
    campaign_scope["platform_sessions"] = pd.to_numeric(campaign_scope.get("sessions"), errors="coerce").fillna(0)
    campaign_scope = campaign_scope.dropna(subset=["year_month"])
    campaign_scope = campaign_scope.merge(allocation, on="year_month", how="inner")
    if campaign_scope.empty:
        return campaign_scope

    campaign_scope["allocation_share"] = campaign_scope["scope_ga_view_share"]
    campaign_scope["sessions"] = campaign_scope["platform_sessions"] * campaign_scope["allocation_share"]
    return campaign_scope.drop(columns=["scope_ga_view_share"]).reset_index(drop=True)

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
        "platform_sessions",
        "allocation_share",
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
        "sessions": "Allocated Sessions",
        "platform_sessions": "Platform Sessions",
        "allocation_share": "Scope GA View Share",
        "session_share": "Allocated Session Share",
    }
    out = df[display_cols].copy().rename(columns=rename_map)
    if "Year Month" in out.columns:
        out["Year Month"] = pd.to_datetime(out["Year Month"], errors="coerce").dt.strftime("%Y-%m")
    for col in ["Allocated Sessions", "Platform Sessions"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(0)
    for col in ["Scope GA View Share", "Allocated Session Share"]:
        if col in out.columns:
            out[col] = out[col].apply(_fmt_pct)
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
    ga_campaign_type_monthly = load_ga_campaign_type_monthly()
    ga_restaurant_monthly = load_ga_restaurant_monthly()
    outcomes_df = load_cluster_strategy_outcomes()

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
                        strategy_name = html.escape(str(rec.get("strategy_name", "Unknown Strategy")))
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
                strategy_mix = (
                    raw_strategy_campaigns.assign(
                        strategy_family=raw_strategy_campaigns["strategy_family"].fillna("Unknown"),
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
    scope_ga = _build_scope_ga_effectiveness(scope_ga_monthly, ga_campaign_type_monthly, scope_label)

    if scope_ga.empty:
        st.info("No GA campaign effectiveness data available for this scope.")
    else:
        st.caption(
            "Campaign effectiveness is inferred from month-level overlap between cluster GMV per GA view "
            "and the platform Google Ads campaign mix. Sessions are allocated to the current navbar scope by "
            "that scope's monthly share of GA item views. The raw GA breakdown below uses the same allocation, "
            "so campaign-row totals reconcile with this section."
        )

        ga_card_source = scope_ga.copy()
        top_ga = ga_card_source.sort_values("ga_effectiveness_score", ascending=False).head(3)
        st.caption("Top GA campaign cards show GMV / GA View and Add to Cart Rate.")
        ga_cols = st.columns(min(3, len(top_ga)))
        for idx, (_, rec) in enumerate(top_ga.iterrows()):
            with ga_cols[idx]:
                campaign_type = html.escape(str(rec.get("googleAdsCampaignType", "Unknown Campaign Type")))
                gmv_per_ga = _fmt_thb_value(rec.get("session_weighted_gmv_per_ga_view"))
                atc_rate = _fmt_pct_value(rec.get("session_weighted_add_to_cart_rate"))
                sessions = rec.get("total_sessions")
                session_text = "0" if pd.isna(sessions) else f"{int(_num_or_zero(sessions)):,}"
                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #e5e7eb;
                        border-radius:8px;
                        padding:0.85rem 0.9rem;
                        background:#ffffff;
                        min-height:150px;
                    ">
                        <div style="font-size:0.78rem; color:#6b7280; font-weight:700;">#{idx + 1} GA Campaign Type</div>
                        <div style="
                            margin-top:0.15rem;
                            font-size:0.95rem;
                            color:#111827;
                            font-weight:800;
                            line-height:1.25;
                            min-height:2.4rem;
                        ">{campaign_type}</div>
                        <div style="
                            display:grid;
                            grid-template-columns:1fr 1fr;
                            gap:0.65rem;
                            margin-top:0.85rem;
                        ">
                            <div>
                                <div style="font-size:0.72rem; color:#6b7280; font-weight:700;">GMV / GA View</div>
                                <div style="font-size:1.35rem; color:#cc0000; font-weight:850;">{gmv_per_ga}</div>
                            </div>
                            <div>
                                <div style="font-size:0.72rem; color:#6b7280; font-weight:700;">Add to Cart Rate</div>
                                <div style="font-size:1.35rem; color:#111827; font-weight:850;">{atc_rate}</div>
                            </div>
                        </div>
                        <div style="margin-top:0.75rem; font-size:0.75rem; color:#6b7280;">
                            <span style="font-weight:800; color:#374151;">{session_text}</span> allocated sessions
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='height:0.95rem;'></div>", unsafe_allow_html=True)

        ga_display_source = scope_ga.sort_values("ga_effectiveness_score", ascending=False)
        ga_display_cols = [
            c
            for c in [
                "scope_label",
                "googleAdsCampaignType",
                "total_sessions",
                "session_weighted_gmv_per_ga_view",
                "session_weighted_add_to_cart_rate",
                "session_weighted_view_to_purchase_rate",
            ]
            if c in scope_ga.columns
        ]
        ga_display = ga_display_source[ga_display_cols].copy().rename(
            columns={
                "scope_label": "Scope",
                "googleAdsCampaignType": "GA Campaign Type",
                "total_sessions": "Allocated Sessions",
                "session_weighted_gmv_per_ga_view": "GMV / GA View",
                "session_weighted_add_to_cart_rate": "Add to Cart Rate",
                "session_weighted_view_to_purchase_rate": "View to Purchase Rate",
            }
        )
        if "Scope" in ga_display.columns:
            ga_display = ga_display.drop(columns=["Scope"])

        if "Allocated Sessions" in ga_display.columns:
            ga_display["Allocated Sessions"] = pd.to_numeric(ga_display["Allocated Sessions"], errors="coerce").round(0)
        if "GMV / GA View" in ga_display.columns:
            ga_display["GMV / GA View"] = ga_display["GMV / GA View"].apply(_fmt_thb)
        for col in ["Add to Cart Rate", "View to Purchase Rate"]:
            if col in ga_display.columns:
                ga_display[col] = ga_display[col].apply(_fmt_pct)

        st.dataframe(
            ga_display,
            hide_index=True,
            width="stretch",
            height=280,
        )

    st.markdown("#### Raw GA Campaign Breakdown")
    st.caption(
        "Source: `data/marketing/googleAPI/campaigns_outreach.parquet`. "
        "Rows are raw Google Ads campaigns, but session counts are allocated to the current navbar scope using "
        "the same monthly GA item-view share as GA Campaign Effectiveness."
    )

    if ga_campaign_raw.empty:
        st.info("No raw GA campaign outreach rows found.")
    else:
        campaign_scope = _allocate_campaign_breakdown_to_scope(ga_campaign_raw, scope_ga_monthly)

        if campaign_scope.empty:
            st.info("No campaign outreach rows match the available month window for this navbar scope.")
        else:
            campaign_scope["sessions"] = pd.to_numeric(campaign_scope["sessions"], errors="coerce").fillna(0)
            total_sessions = campaign_scope["sessions"].sum()
            effectiveness_total_sessions = (
                pd.to_numeric(scope_ga.get("total_sessions"), errors="coerce").fillna(0).sum()
                if not scope_ga.empty
                else 0
            )
            st.caption(
                f"Allocated campaign rows total {total_sessions:,.0f} sessions; "
                f"GA Campaign Effectiveness total {effectiveness_total_sessions:,.0f} sessions."
            )
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
                    hovertemplate="%{x}<br>Allocated Sessions: %{y:,.0f}<br>Share: %{text}<extra></extra>",
                )
            )
            fig_campaign_mix.update_layout(
                **BASE_LAYOUT,
                height=260,
                showlegend=False,
                xaxis=dict(**CHART_THEME["xaxis"], title="GA Campaign Type", tickangle=-20),
                yaxis=dict(**CHART_THEME["yaxis"], title="Allocated Sessions"),
            )
            st.plotly_chart(fig_campaign_mix, width="stretch")

            campaign_display = _prepare_campaign_breakdown_display(
                campaign_scope.sort_values(["sessions", "campaign_name"], ascending=[False, True])
            )
            st.caption("Click column headers to toggle ASC/DESC and use the filter boxes inside the table headers.")
            _render_filterable_table(
                campaign_display,
                key=f"raw_ga_campaign_table_{str(active_cluster)}_{active_rest_norm}_{selected_segment_filter}",
                height=320,
                initial_sort_column="Allocated Sessions",
                initial_sort_direction="desc",
            )
