"""
data/loader.py
--------------
Reads pipeline outputs for the dashboard.
Falls back to realistic sample data if files are not yet present,
so the dashboard remains runnable during development.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

EDA_DIR = BASE_DIR / "_1_eda"
EDA_OUTPUT_DIR = EDA_DIR / "data_output"

MOMENTUM_DIR = BASE_DIR / "_2_feature_engineering+momentum"
MOMENTUM_OUTPUT_DIR = MOMENTUM_DIR / "data_output"

MARKETING_DIR = BASE_DIR / "_3_marketing"
MARKETING_OUTPUT_DIR = MARKETING_DIR / "data_output"

PRIORITY_DIR = BASE_DIR / "_4_final_outputs"
PRIORITY_OUTPUT_DIR = PRIORITY_DIR / "data_output"

GA_DIR = BASE_DIR / "_5_GA_data"
GA_OUTPUT_DIR = GA_DIR / "data_output"

CLUSTERING_DIR = BASE_DIR / "clustering"
CLUSTERING_OUTPUT_DIR = CLUSTERING_DIR / "data_output"

MOMENTUM_PATH = MOMENTUM_OUTPUT_DIR / "restaurants_agg_performance.parquet"
MOMENTUM_GA_PATH = GA_OUTPUT_DIR / "combined_restaurant_ga.parquet"
MOMENTUM_LABELS_PATH = MOMENTUM_OUTPUT_DIR / "priority_latest_momentum_labels.parquet"
MARKETING_PATH = MARKETING_OUTPUT_DIR / "activity_performance_with_roi.csv"
PRIORITY_PATH = PRIORITY_OUTPUT_DIR / "priority_list.csv"
GA_OUTREACH_PATH = BASE_DIR / "data" / "marketing" / "googleAPI" / "campaigns_outreach.parquet"
GA_GMV_VIEW_PATH = GA_OUTPUT_DIR / "gmv" / "gmv_view.parquet"
MOMENTUM_VALID_BOOKINGS_PATH = MOMENTUM_OUTPUT_DIR / "bookings_cleaned.parquet"
MOMENTUM_BOOKINGS_EXPORT_PATH = Path(__file__).resolve().parent / "momentum" / "restaurant_bookings_history.parquet"

CLUSTER_RESULTS_PATH = CLUSTERING_OUTPUT_DIR / "clustering_results.csv"
REVIEWS_PATH = CLUSTERING_OUTPUT_DIR / "reviews.csv"

SEGMENT_COLORS = {
    "Rising Stars": "#2ecc71",
    "Emerging Opportunities": "#3498db",
    "Established Players": "#9b59b6",
    "Needs Attention": "#e74c3c",
}

PRIORITY_GROWTH_WEIGHT = 0.75
PRIORITY_GA_WEIGHT = 0.25


def _normalize_name(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "monthly_revenue": "monthly_gmv",
        "avg_revenue_per_booking": "avg_gmv_per_booking",
        "revenue_growth_mom": "gmv_growth_mom",
        "revenue_growth_yoy": "gmv_growth_yoy",
        "revenue_growth_rolling": "gmv_growth_rolling",
        "revenue_growth_mom_rolling": "gmv_growth_mom_rolling",
        "revenue_growth_yoy_rolling": "gmv_growth_yoy_rolling",
    }
    applicable = {
        old: new
        for old, new in rename_map.items()
        if old in df.columns and new not in df.columns
    }
    if not applicable:
        return df
    return df.rename(columns=applicable)


def _normalize_combined_ga_export(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "monthly_gmv_x": "monthly_gmv",
        "monthly_bookings_x": "monthly_bookings",
        "itemsViewed": "ga_items_viewed",
        "itemsAddedToCart": "ga_items_added_to_cart",
        "itemsPurchased": "ga_items_purchased",
        "itemRevenue": "ga_item_revenue",
        "gmv_per_view": "gmv_per_ga_view",
        "view_to_purchase_rate": "ga_view_to_purchase_rate",
        "revenue_per_view": "ga_revenue_per_view",
    }
    applicable = {
        old: new
        for old, new in rename_map.items()
        if old in out.columns and new not in out.columns
    }
    if applicable:
        out = out.rename(columns=applicable)

    legacy_ga_cols = {"ga_items_viewed", "ga_items_added_to_cart", "ga_items_purchased", "ga_item_revenue"}
    if not legacy_ga_cols.intersection(out.columns):
        return out

    if "year_month" in out.columns:
        out["year_month"] = pd.to_datetime(out["year_month"], errors="coerce")

    drop_cols = [c for c in ["monthly_gmv_y", "monthly_bookings_y", "yearMonth", "itemId", "itemName"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    group_cols = [c for c in ["restaurant_id", "name", "year_month"] if c in out.columns]
    if len(group_cols) < 2:
        return out

    ga_sum_cols = [
        c
        for c in ["ga_items_viewed", "ga_items_added_to_cart", "ga_items_purchased", "ga_item_revenue"]
        if c in out.columns
    ]
    rate_cols = [c for c in ["gmv_per_ga_view", "ga_view_to_purchase_rate", "ga_revenue_per_view"] if c in out.columns]
    passthrough_cols = [c for c in out.columns if c not in group_cols + ga_sum_cols + rate_cols]
    if not out.duplicated(group_cols, keep=False).any():
        return out

    agg_map: dict[str, str] = {col: "sum" for col in ga_sum_cols}
    agg_map.update({col: "first" for col in passthrough_cols})
    out = out.groupby(group_cols, as_index=False).agg(agg_map)
    return out


def _to_numeric_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    series = df[col]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce")


def _safe_divide(numerator, denominator) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num / den.replace(0, np.nan)


def _min_max_norm(series: pd.Series, neutral: float = 0.5) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(neutral, index=numeric.index, dtype="float64")
    span = valid.max() - valid.min()
    if pd.isna(span) or span == 0:
        return pd.Series(neutral, index=numeric.index, dtype="float64")
    normalized = (numeric - valid.min()) / span
    return normalized.fillna(neutral)


def _add_ga_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_combined_ga_export(_normalize_metric_columns(df.copy()))

    if "year_month" in out.columns:
        out["year_month"] = pd.to_datetime(out["year_month"], errors="coerce")

    if "name" in out.columns:
        out["name_norm"] = _normalize_name(out["name"])

    ga_numeric_cols = [
        "monthly_gmv",
        "monthly_bookings",
        "ga_items_viewed",
        "ga_items_added_to_cart",
        "ga_items_purchased",
        "ga_item_revenue",
        "ga_view_to_purchase_rate",
        "ga_revenue_per_view",
    ]
    for col in ga_numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "ga_items_viewed" not in out.columns:
        return out

    if "gmv_per_ga_view" in out.columns:
        out["gmv_per_ga_view"] = pd.to_numeric(out["gmv_per_ga_view"], errors="coerce")
    else:
        out["gmv_per_ga_view"] = _safe_divide(out.get("monthly_gmv"), out["ga_items_viewed"])

    out["bookings_per_ga_view"] = _safe_divide(out.get("monthly_bookings"), out["ga_items_viewed"])
    out["ga_add_to_cart_rate"] = _safe_divide(out.get("ga_items_added_to_cart"), out["ga_items_viewed"])

    if "ga_view_to_purchase_rate" in out.columns:
        out["ga_view_to_purchase_rate"] = pd.to_numeric(out["ga_view_to_purchase_rate"], errors="coerce")
    else:
        out["ga_view_to_purchase_rate"] = _safe_divide(out.get("ga_items_purchased"), out["ga_items_viewed"])

    out["ga_purchase_to_cart_rate"] = _safe_divide(out.get("ga_items_purchased"), out.get("ga_items_added_to_cart"))

    if "ga_revenue_per_view" in out.columns:
        out["ga_revenue_per_view"] = pd.to_numeric(out["ga_revenue_per_view"], errors="coerce")
    else:
        out["ga_revenue_per_view"] = _safe_divide(out.get("ga_item_revenue"), out["ga_items_viewed"])

    out["gmv_per_view"] = out["gmv_per_ga_view"]
    out["has_ga_data"] = out["ga_items_viewed"].fillna(0).gt(0)
    return out


def _latest_snapshot(df: pd.DataFrame, key_col: str = "name") -> pd.DataFrame:
    if df.empty or key_col not in df.columns or "year_month" not in df.columns:
        return df.copy()
    return df.sort_values("year_month").groupby(key_col, as_index=False).last()


# Sample data generator for development mode
def _make_sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n_restaurants = 80
    months = pd.date_range("2024-08-01", "2026-01-01", freq="MS")
    names = [f"Restaurant {chr(65 + i % 26)}{i // 26 if i >= 26 else ''}" for i in range(n_restaurants)]
    segments = rng.choice(
        ["Rising Stars", "Emerging Opportunities", "Established Players", "Needs Attention"],
        size=n_restaurants,
        p=[0.15, 0.20, 0.25, 0.40],
    )
    locations = rng.choice(["Bangkok", "Chiang Mai", "Phuket", "Pattaya", "Hua Hin"], size=n_restaurants)
    cuisines = rng.choice(["Thai", "Japanese", "Italian", "Chinese", "International", "Fusion"], size=n_restaurants)

    rows = []
    for i, name in enumerate(names):
        base = rng.integers(5, 60)
        for month in months:
            trend = 1 + rng.normal(0.02, 0.08)
            season = 1 + 0.15 * np.sin(2 * np.pi * month.month / 12)
            bookings = max(0, int(base * trend * season + rng.normal(0, 3)))
            rows.append(
                {
                    "restaurant_id": 1000 + i,
                    "name": name,
                    "year_month": month,
                    "monthly_bookings": bookings,
                    "monthly_gmv": bookings * rng.uniform(300, 1200),
                    "avg_gmv_per_booking": rng.uniform(300, 1200),
                    "avg_guests": rng.uniform(2.0, 4.5),
                    "active_days": rng.integers(10, 28),
                    "location": locations[i],
                    "cuisine": cuisines[i],
                    "booking_growth_rolling": rng.uniform(-0.3, 0.8),
                    "gmv_growth_rolling": rng.uniform(-0.3, 0.8),
                    "booking_growth_yoy": rng.uniform(-0.2, 0.6),
                    "gmv_growth_yoy": rng.uniform(-0.2, 0.6),
                    "growth_signal_used": rng.choice(["YoY", "MoM"], p=[0.6, 0.4]),
                    "score_perf": rng.uniform(0, 1),
                    "score_growth": rng.uniform(0, 1),
                    "delta_growth_book": rng.uniform(-0.2, 0.2),
                    "in_analysis_window": month >= pd.Timestamp("2025-01-01"),
                }
            )

    momentum_df = pd.DataFrame(rows)

    priority_rows = []
    tiers = [
        "Activate - proven marketing response",
        "Activate - untapped, no prior spend",
        "Activate - review channel strategy",
    ]
    channels = ["FB", "KOL", "CRM"]
    for i in range(40):
        name = names[i]
        has_mkt = bool(rng.random() > 0.35)
        priority_rows.append(
            {
                "rank": i + 1,
                "restaurant_id": 1000 + i,
                "name": name,
                "priority_score": round(100 - i * 2.3 + rng.uniform(-3, 3), 1),
                "priority_tier": rng.choice(tiers, p=[0.45, 0.35, 0.20]),
                "recommended_channel": rng.choice(channels),
                "latest_segment": segments[i],
                "growth_months": int(rng.integers(2, 4)),
                "is_stable_growth": True,
                "score_perf": rng.uniform(0.3, 1.0),
                "score_growth": rng.uniform(0.3, 1.0),
                "monthly_bookings": int(rng.integers(10, 150)),
                "monthly_gmv": rng.uniform(5000, 200000),
                "booking_growth_rolling": rng.uniform(0.05, 0.80),
                "gmv_growth_rolling": rng.uniform(0.05, 0.80),
                "booking_growth_yoy": rng.uniform(0.03, 0.60),
                "growth_signal_used": rng.choice(["YoY", "MoM"], p=[0.65, 0.35]),
                "delta_growth_book": rng.uniform(-0.1, 0.3),
                "has_marketing": has_mkt,
                "n_campaigns": int(rng.integers(1, 8)) if has_mkt else 0,
                "avg_lift_per_day": float(rng.uniform(0.1, 3.5)) if has_mkt else None,
                "total_incremental_rev": float(rng.uniform(1000, 50000)) if has_mkt else None,
                "avg_roi": float(rng.uniform(-0.1, 2.5)) if has_mkt else None,
                "n_positive_lift": int(rng.integers(1, 5)) if has_mkt else 0,
                "n_negative_lift": int(rng.integers(0, 2)) if has_mkt else 0,
                "channels_used": rng.choice(["FB", "KOL", "CRM", "FB+KOL", "CRM+FB"]) if has_mkt else None,
                "best_channel": rng.choice(channels) if has_mkt else None,
                "pct_yoy_baseline": float(rng.uniform(0, 1)) if has_mkt else None,
                "location": locations[i],
                "cuisine": cuisines[i],
            }
        )

    priority_df = pd.DataFrame(priority_rows)
    return momentum_df, priority_df


@st.cache_data(ttl=300)
def load_momentum() -> pd.DataFrame:
    source_path = MOMENTUM_GA_PATH if MOMENTUM_GA_PATH.exists() else MOMENTUM_PATH
    if source_path.exists():
        df = _add_ga_derived_columns(pd.read_parquet(source_path))
        if "year_month" in df.columns:
            df["year_month"] = pd.to_datetime(df["year_month"])
        if "latest_segment" not in df.columns and "name" in df.columns:
            seg_df = load_momentum_segments()
            if not seg_df.empty and "name_norm" in seg_df.columns and "latest_segment" in seg_df.columns:
                df["name_norm"] = _normalize_name(df["name"])
                seg_join = seg_df[["name_norm", "latest_segment"]].drop_duplicates("name_norm")
                df = df.merge(seg_join, on="name_norm", how="left")
                df = df.drop(columns=["name_norm"])
        return df
    st.toast("Using sample momentum data. Run momentum_seasonality.ipynb to load real data.", icon="⚠️")
    momentum_df, _ = _make_sample_data()
    return _add_ga_derived_columns(momentum_df)


@st.cache_data(ttl=300)
def load_ga_restaurant_monthly() -> pd.DataFrame:
    momentum_df = load_momentum()
    if momentum_df.empty or "gmv_per_ga_view" not in momentum_df.columns:
        return pd.DataFrame(
            columns=[
                "restaurant_id",
                "name",
                "name_norm",
                "year_month",
                "ga_items_viewed",
                "ga_items_added_to_cart",
                "ga_items_purchased",
                "ga_item_revenue",
                "gmv_per_ga_view",
                "gmv_per_view",
                "bookings_per_ga_view",
                "ga_add_to_cart_rate",
                "ga_view_to_purchase_rate",
                "ga_purchase_to_cart_rate",
                "ga_revenue_per_view",
                "has_ga_data",
            ]
        )

    if "name_norm" not in momentum_df.columns and "name" in momentum_df.columns:
        momentum_df = momentum_df.copy()
        momentum_df["name_norm"] = _normalize_name(momentum_df["name"])

    keep_cols = [
        c
        for c in [
            "restaurant_id",
            "name",
            "name_norm",
            "year_month",
            "monthly_gmv",
            "monthly_bookings",
            "ga_items_viewed",
            "ga_items_added_to_cart",
            "ga_items_purchased",
            "ga_item_revenue",
            "gmv_per_ga_view",
            "gmv_per_view",
            "bookings_per_ga_view",
            "ga_add_to_cart_rate",
            "ga_view_to_purchase_rate",
            "ga_purchase_to_cart_rate",
            "ga_revenue_per_view",
            "has_ga_data",
        ]
        if c in momentum_df.columns
    ]
    ga_df = momentum_df[keep_cols].copy()
    if "ga_items_viewed" in ga_df.columns:
        ga_df = ga_df[ga_df["ga_items_viewed"].fillna(0) > 0]
    return ga_df.reset_index(drop=True)


@st.cache_data(ttl=300)
def load_raw_gmv_view_monthly() -> pd.DataFrame:
    if not GA_GMV_VIEW_PATH.exists():
        return pd.DataFrame(
            columns=[
                "restaurant_id",
                "name",
                "name_norm",
                "year_month",
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
        )

    out = pd.read_parquet(GA_GMV_VIEW_PATH).copy()
    rename_map = {
        "gmv_per_view": "gmv_per_ga_view",
        "bookings_per_view": "bookings_per_ga_view",
        "view_to_purchase_rate": "ga_view_to_purchase_rate",
        "purchase_to_cart_rate": "ga_purchase_to_cart_rate",
        "revenue_per_view": "ga_revenue_per_view",
    }
    applicable = {
        old: new
        for old, new in rename_map.items()
        if old in out.columns and new not in out.columns
    }
    if applicable:
        out = out.rename(columns=applicable)

    if "year_month" in out.columns:
        out["year_month"] = pd.to_datetime(out["year_month"], errors="coerce")
    if "restaurant_id" in out.columns:
        out["restaurant_id"] = pd.to_numeric(out["restaurant_id"], errors="coerce").astype("Int64")
    if "name" in out.columns and "name_norm" not in out.columns:
        out["name_norm"] = _normalize_name(out["name"])

    numeric_cols = [
        c
        for c in [
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
        if c in out.columns
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    keep_cols = [
        c
        for c in [
            "restaurant_id",
            "name",
            "name_norm",
            "year_month",
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
        if c in out.columns
    ]
    out = out[keep_cols].copy()

    if "ga_items_viewed" in out.columns:
        out = out[out["ga_items_viewed"].fillna(0) > 0]

    return out.sort_values(["name", "year_month"]).reset_index(drop=True)


@st.cache_data(ttl=300)
def load_ga_campaign_outreach_raw() -> pd.DataFrame:
    empty_cols = [
        "year_month",
        "campaign_id",
        "campaign_name",
        "googleAdsCampaignType",
        "sessions",
    ]
    if not GA_OUTREACH_PATH.exists():
        return pd.DataFrame(columns=empty_cols)

    outreach_df = pd.read_parquet(GA_OUTREACH_PATH)
    if outreach_df.empty:
        return pd.DataFrame(columns=empty_cols)

    out = outreach_df.copy()
    out = out[
        out["campaignId"].notna()
        & out["campaignId"].astype(str).ne("(not set)")
    ]
    out["googleAdsCampaignType"] = out["googleAdsCampaignType"].fillna("Unknown").astype(str).str.strip()
    out = out[
        out["googleAdsCampaignType"].ne("")
        & out["googleAdsCampaignType"].ne("(not set)")
    ]
    out["campaignName"] = out["campaignName"].fillna("Unknown Campaign").astype(str).str.strip()
    out["year_month"] = pd.to_datetime(out["yearMonth"].astype(str), format="%Y%m", errors="coerce")
    out["sessions"] = pd.to_numeric(
        out["sessions"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
    )
    out = out.dropna(subset=["year_month", "sessions"])
    if out.empty:
        return pd.DataFrame(columns=empty_cols)

    return (
        out.groupby(
            ["year_month", "campaignId", "campaignName", "googleAdsCampaignType"],
            as_index=False,
        )["sessions"]
        .sum()
        .rename(
            columns={
                "campaignId": "campaign_id",
                "campaignName": "campaign_name",
            }
        )
        .sort_values(["year_month", "sessions"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _merge_latest_ga_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "name" not in df.columns:
        return df.copy()

    ga_latest = _latest_snapshot(load_ga_restaurant_monthly())
    if ga_latest.empty:
        return df.copy()

    ga_latest = ga_latest.copy()
    if "name_norm" not in ga_latest.columns and "name" in ga_latest.columns:
        ga_latest["name_norm"] = _normalize_name(ga_latest["name"])
    ga_latest["ga_data_month"] = ga_latest.get("year_month")
    ga_cols = [
        c
        for c in [
            "gmv_per_ga_view",
            "gmv_per_view",
            "bookings_per_ga_view",
            "ga_add_to_cart_rate",
            "ga_view_to_purchase_rate",
            "ga_purchase_to_cart_rate",
            "ga_revenue_per_view",
            "ga_items_viewed",
            "ga_items_added_to_cart",
            "ga_items_purchased",
            "ga_item_revenue",
            "ga_data_month",
            "has_ga_data",
        ]
        if c in ga_latest.columns
    ]

    out = df.copy().drop(columns=ga_cols, errors="ignore")
    out["name_norm"] = _normalize_name(out["name"])

    if "restaurant_id" in out.columns and "restaurant_id" in ga_latest.columns:
        out["restaurant_id"] = pd.to_numeric(out["restaurant_id"], errors="coerce").astype("Int64")
        ga_latest["restaurant_id"] = pd.to_numeric(ga_latest["restaurant_id"], errors="coerce").astype("Int64")
        ga_by_id = ga_latest[["restaurant_id"] + ga_cols].dropna(subset=["restaurant_id"]).drop_duplicates("restaurant_id")
        out = out.merge(ga_by_id, on="restaurant_id", how="left")

    ga_by_name = ga_latest[["name_norm"] + ga_cols].drop_duplicates("name_norm")
    rename_map = {col: f"{col}_by_name" for col in ga_cols}
    out = out.merge(ga_by_name.rename(columns=rename_map), on="name_norm", how="left")

    for col in ga_cols:
        fallback_col = f"{col}_by_name"
        if col in out.columns:
            out[col] = out[col].where(out[col].notna(), out[fallback_col])
        else:
            out[col] = out[fallback_col]
        out = out.drop(columns=[fallback_col])

    return out


def score_priority_with_ga(df: pd.DataFrame) -> pd.DataFrame:
    out = _merge_latest_ga_metrics(df)
    out = _normalize_metric_columns(out)

    if "priority_score_saved" not in out.columns:
        out["priority_score_saved"] = _to_numeric_series(out, "priority_score")
    if "rank_saved" not in out.columns and "rank" in out.columns:
        out["rank_saved"] = _to_numeric_series(out, "rank")

    out["score_growth_norm"] = _min_max_norm(_to_numeric_series(out, "score_growth"), neutral=0.5)
    out["delta_growth_norm"] = _min_max_norm(_to_numeric_series(out, "delta_growth_book"), neutral=0.5)
    out["growth_component"] = 0.60 * out["score_growth_norm"] + 0.40 * out["delta_growth_norm"]
    out["gmv_per_ga_view_norm"] = _min_max_norm(_to_numeric_series(out, "gmv_per_ga_view"), neutral=0.5)
    out["ga_component"] = out["gmv_per_ga_view_norm"]

    out["priority_raw_recomputed"] = (
        PRIORITY_GROWTH_WEIGHT * out["growth_component"]
        + PRIORITY_GA_WEIGHT * out["ga_component"]
    )
    out["priority_score_recomputed"] = _min_max_norm(out["priority_raw_recomputed"], neutral=0.5) * 100
    out["priority_score"] = out["priority_score_recomputed"].round(2)

    ga_lead = out["ga_component"] - out["growth_component"]
    growth_lead = out["growth_component"] - out["ga_component"]
    out["priority_reason"] = np.select(
        [
            ga_lead >= 0.12,
            growth_lead >= 0.12,
            _to_numeric_series(out, "ga_add_to_cart_rate").fillna(0).ge(0.08),
        ],
        [
            "High GMV per GA view",
            "Strong growth momentum",
            "Strong GA conversion intent",
        ],
        default="Balanced growth + GA efficiency",
    )

    if len(out):
        out["rank"] = out["priority_score"].rank(method="first", ascending=False).astype(int)
    return out


@st.cache_data(ttl=300)
def load_priority() -> pd.DataFrame:
    if PRIORITY_PATH.exists():
        return score_priority_with_ga(pd.read_csv(PRIORITY_PATH))
    _, priority_df = _make_sample_data()
    return score_priority_with_ga(priority_df)


@st.cache_data(ttl=300)
def load_momentum_raw_bookings() -> pd.DataFrame:
    bookings_df = None
    for source_path in (MOMENTUM_BOOKINGS_EXPORT_PATH, MOMENTUM_VALID_BOOKINGS_PATH):
        if source_path.exists():
            bookings_df = pd.read_parquet(source_path)
            break

    if bookings_df is None:
        return pd.DataFrame(
            columns=[
                "booking_id",
                "restaurant_id",
                "restaurant_name",
                "booking_date",
                "created_at",
                "start_time",
                "end_time",
                "channel",
                "medium",
                "adults",
                "kids",
                "total_guests",
                "revenue_thb",
                "revenue_dollars",
                "arrived",
                "no_show",
                "refund",
                "adjusted",
            ]
        )

    rename_map = {
        "id": "booking_id",
        "name": "restaurant_name",
        "adult": "adults",
        "child": "kids",
        "party_size": "total_guests",
        "revenue": "revenue_thb",
    }
    bookings_df = bookings_df.rename(columns=rename_map)

    if "channel_name" in bookings_df.columns:
        channel_labels = bookings_df["channel_name"].astype("string").str.strip()
        if "channel" in bookings_df.columns:
            raw_channel = bookings_df["channel"].astype("string").str.strip()
            bookings_df["channel"] = channel_labels.where(channel_labels.notna() & channel_labels.ne(""), raw_channel)
        else:
            bookings_df["channel"] = channel_labels

    if "restaurant_id" in bookings_df.columns:
        bookings_df["restaurant_id"] = pd.to_numeric(bookings_df["restaurant_id"], errors="coerce").astype("Int64")

    if "restaurant_name" not in bookings_df.columns and "restaurant_id" in bookings_df.columns:
        rest_ref = (
            load_momentum()[["restaurant_id", "name"]]
            .dropna(subset=["restaurant_id"])
            .drop_duplicates("restaurant_id")
            .rename(columns={"name": "restaurant_name"})
        )
        bookings_df = bookings_df.merge(rest_ref, on="restaurant_id", how="left")

    if "restaurant_name" not in bookings_df.columns:
        bookings_df["restaurant_name"] = "Unknown"

    has_adults = "adults" in bookings_df.columns
    has_kids = "kids" in bookings_df.columns

    if has_adults:
        bookings_df["adults"] = pd.to_numeric(bookings_df["adults"], errors="coerce").astype("Int64")
    if has_kids:
        bookings_df["kids"] = pd.to_numeric(bookings_df["kids"], errors="coerce").astype("Int64")

    if "total_guests" in bookings_df.columns:
        bookings_df["total_guests"] = pd.to_numeric(bookings_df["total_guests"], errors="coerce").astype("Int64")
    elif has_adults or has_kids:
        adult_vals = pd.to_numeric(bookings_df["adults"], errors="coerce") if has_adults else 0
        kid_vals = pd.to_numeric(bookings_df["kids"], errors="coerce") if has_kids else 0
        bookings_df["total_guests"] = (adult_vals.fillna(0) + kid_vals.fillna(0)).astype("Int64")

    for dt_col in ["booking_date", "created_at"]:
        if dt_col in bookings_df.columns:
            bookings_df[dt_col] = pd.to_datetime(bookings_df[dt_col], errors="coerce")
        else:
            bookings_df[dt_col] = pd.NaT

    if "revenue_thb" in bookings_df.columns:
        bookings_df["revenue_thb"] = pd.to_numeric(bookings_df["revenue_thb"], errors="coerce")
    else:
        bookings_df["revenue_thb"] = pd.NA

    if "revenue_dollars" in bookings_df.columns:
        bookings_df["revenue_dollars"] = pd.to_numeric(bookings_df["revenue_dollars"], errors="coerce")
    else:
        bookings_df["revenue_dollars"] = pd.NA

    if "refund" not in bookings_df.columns and "refund_guarantee_status" in bookings_df.columns:
        refund_status = bookings_df["refund_guarantee_status"].astype("string").str.strip()
        bookings_df["refund"] = refund_status.where(refund_status.notna() & refund_status.ne("pending"), pd.NA)

    for bool_col in ["arrived", "no_show", "refund", "adjusted"]:
        if bool_col not in bookings_df.columns:
            bookings_df[bool_col] = pd.NA

    keep_cols = [
        c
        for c in [
            "booking_id",
            "restaurant_id",
            "restaurant_name",
            "booking_date",
            "created_at",
            "start_time",
            "end_time",
            "channel",
            "medium",
            "adults",
            "kids",
            "total_guests",
            "revenue_thb",
            "revenue_dollars",
            "arrived",
            "no_show",
            "refund",
            "adjusted",
        ]
        if c in bookings_df.columns
    ]

    bookings_df = bookings_df[keep_cols].copy()
    bookings_df = bookings_df.sort_values(["booking_date", "created_at"], ascending=[False, False], na_position="last")
    return bookings_df.reset_index(drop=True)


def get_restaurant_booking_history(
    bookings_df: pd.DataFrame,
    restaurant_name: str,
    restaurant_id: int | None = None,
) -> pd.DataFrame:
    if bookings_df.empty:
        return bookings_df.copy()

    out = bookings_df.copy()
    if restaurant_id is not None and "restaurant_id" in out.columns:
        out = out[out["restaurant_id"] == int(restaurant_id)]

    if out.empty and "restaurant_name" in bookings_df.columns:
        target = _normalize_name(pd.Series([restaurant_name])).iloc[0]
        name_norm = _normalize_name(bookings_df["restaurant_name"])
        out = bookings_df[name_norm == target]

    if "booking_date" in out.columns:
        out = out.sort_values(["booking_date", "created_at"], ascending=[False, False], na_position="last")
    return out.reset_index(drop=True)


@st.cache_data(ttl=300)
def load_momentum_segments() -> pd.DataFrame:
    if MOMENTUM_LABELS_PATH.exists():
        seg_df = _normalize_metric_columns(pd.read_parquet(MOMENTUM_LABELS_PATH))
        if "segment" in seg_df.columns and "latest_segment" not in seg_df.columns:
            seg_df = seg_df.rename(columns={"segment": "latest_segment"})
    else:
        priority_df = load_priority()
        cols = [c for c in ["name", "latest_segment"] if c in priority_df.columns]
        seg_df = priority_df[cols].copy() if cols else pd.DataFrame(columns=["name", "latest_segment"])

    if "name" not in seg_df.columns:
        return pd.DataFrame(columns=["name", "latest_segment", "name_norm"])

    keep_cols = [
        c
        for c in ["restaurant_id", "name", "latest_segment", "score_perf", "score_growth", "year_month"]
        if c in seg_df.columns
    ]
    seg_df = seg_df[keep_cols].copy()
    if "year_month" in seg_df.columns:
        seg_df["year_month"] = pd.to_datetime(seg_df["year_month"], errors="coerce")
        seg_df = seg_df.sort_values("year_month").drop_duplicates("name", keep="last")
    else:
        seg_df = seg_df.drop_duplicates("name")
    seg_df["name_norm"] = _normalize_name(seg_df["name"])
    return seg_df


@st.cache_data(ttl=300)
def load_cluster_assignments() -> pd.DataFrame:
    df = pd.DataFrame()
    if CLUSTER_RESULTS_PATH.exists():
        df = pd.read_csv(CLUSTER_RESULTS_PATH)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "restaurant_id",
                "name",
                "cluster_id",
                "cluster_label",
                "x",
                "y",
                "cluster_confidence",
                "latest_segment",
            ]
        )

    if "restaurant name" in df.columns and "name" in df.columns:
        right_name = df["name"]
        left_name = df["restaurant name"]
        if isinstance(right_name, pd.DataFrame):
            right_name = right_name.iloc[:, 0]
        if isinstance(left_name, pd.DataFrame):
            left_name = left_name.iloc[:, 0]
        df["name"] = right_name.where(right_name.notna(), left_name)
        df = df.drop(columns=["restaurant name"])

    rename_map = {
        "restaurant name": "name",
        "cluster": "cluster_id",
        "hybrid_cluster": "cluster_id",
        "Primary Theme": "cluster_label",
        "theme": "cluster_label",
        "UMAP Component 1 (Semantic Similarity Axis)": "x",
        "UMAP Component 2 (Theme Variation Axis)": "y",
    }
    df = df.rename(columns=rename_map)

    if "cluster_id" not in df.columns:
        df["cluster_id"] = 0
    if "cluster_label" not in df.columns:
        df["cluster_label"] = df["cluster_id"].apply(lambda v: f"Cluster {v}")
    if "cluster_confidence" not in df.columns:
        df["cluster_confidence"] = np.nan
    if "name" not in df.columns and "restaurant" in df.columns:
        df = df.rename(columns={"restaurant": "name"})

    if "x" not in df.columns:
        df["x"] = np.arange(len(df), dtype=float)
    if "y" not in df.columns:
        df["y"] = df["cluster_id"].astype(float)

    if "restaurant_id" in df.columns:
        df["restaurant_id"] = pd.to_numeric(df["restaurant_id"], errors="coerce").astype("Int64")
    else:
        df["restaurant_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    df["name"] = df["name"].astype(str).str.strip()
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").fillna(-1).astype(int)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["name_norm"] = _normalize_name(df["name"])

    seg_df = load_momentum_segments()[["name_norm", "latest_segment"]].drop_duplicates("name_norm")
    df = df.drop(columns=["latest_segment"], errors="ignore").merge(seg_df, on="name_norm", how="left")

    momentum_df = load_momentum()
    latest_momentum = momentum_df.sort_values("year_month").groupby("name", as_index=False).last()
    latest_momentum["name_norm"] = _normalize_name(latest_momentum["name"])
    if "restaurant_id" in latest_momentum.columns:
        latest_momentum["restaurant_id"] = pd.to_numeric(latest_momentum["restaurant_id"], errors="coerce").astype("Int64")
    m_cols = [
        c
        for c in [
            "name_norm",
            "restaurant_id",
            "monthly_bookings",
            "monthly_gmv",
            "score_perf",
            "score_growth",
            "growth_signal_used",
            "gmv_per_ga_view",
            "gmv_per_view",
            "bookings_per_ga_view",
            "ga_add_to_cart_rate",
            "ga_view_to_purchase_rate",
            "ga_purchase_to_cart_rate",
            "ga_revenue_per_view",
            "ga_items_viewed",
            "ga_items_added_to_cart",
            "ga_items_purchased",
            "ga_item_revenue",
            "has_ga_data",
        ]
        if c in latest_momentum.columns
    ]
    if m_cols:
        latest_join = latest_momentum[m_cols].copy()
        if "restaurant_id" in latest_join.columns:
            latest_join = latest_join.rename(columns={"restaurant_id": "restaurant_id_momentum"})
        df = df.merge(latest_join, on="name_norm", how="left")
        if "restaurant_id_momentum" in df.columns:
            if "restaurant_id" in df.columns:
                df["restaurant_id"] = df["restaurant_id"].where(df["restaurant_id"].notna(), df["restaurant_id_momentum"])
            else:
                df["restaurant_id"] = df["restaurant_id_momentum"]
            df = df.drop(columns=["restaurant_id_momentum"])

    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").fillna(-1).astype(int)
    df["cluster_label"] = df["cluster_label"].fillna(df["cluster_id"].apply(lambda v: f"Cluster {v}"))
    df.loc[df["cluster_id"] == -1, "cluster_label"] = "Unclustered - no clustering text"

    keep = [
        c
        for c in [
            "restaurant_id",
            "name",
            "name_norm",
            "cluster_id",
            "cluster_label",
            "x",
            "y",
            "cluster_confidence",
            "latest_segment",
            "monthly_bookings",
            "monthly_gmv",
            "score_perf",
            "score_growth",
            "growth_signal_used",
            "gmv_per_ga_view",
            "gmv_per_view",
            "bookings_per_ga_view",
            "ga_add_to_cart_rate",
            "ga_view_to_purchase_rate",
            "ga_purchase_to_cart_rate",
            "ga_revenue_per_view",
            "ga_items_viewed",
            "ga_items_added_to_cart",
            "ga_items_purchased",
            "ga_item_revenue",
            "has_ga_data",
        ]
        if c in df.columns
    ]
    return df[keep].drop_duplicates("name_norm").sort_values(["cluster_id", "name"]).reset_index(drop=True)


def _tokenize_text(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
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
        "thai",
        "bangkok",
    }
    return [t for t in tokens if t not in stopwords]


def _is_nonempty_text(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    text = str(value).strip()
    return text != "" and text.lower() not in {"nan", "none", "null"}


def _normalize_text_snippet(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip(" -\n\t")


def _chunk_sentences(sentences: list[str], target_chars: int = 320, max_chars: int = 480) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        clean_sentence = _normalize_text_snippet(sentence)
        if not clean_sentence:
            continue
        proposed_len = current_len + len(clean_sentence) + (1 if current else 0)
        if current and proposed_len > target_chars:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = [clean_sentence]
            current_len = len(clean_sentence)
        else:
            current.append(clean_sentence)
            current_len = proposed_len

        if current_len >= max_chars:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = []
            current_len = 0

    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _split_text_blob(text: str, target_chars: int = 320, max_chars: int = 480) -> list[str]:
    if not _is_nonempty_text(text):
        return []

    raw = str(text).replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()

    primary_parts = [
        _normalize_text_snippet(part)
        for part in re.split(
            r"\n{2,}|(?<=[.!?])\s{2,}(?=[A-Z0-9'\"(])|\s+-\s*(?=[A-Z0-9'\"(])",
            raw,
        )
        if _normalize_text_snippet(part)
    ]

    if len(primary_parts) <= 1:
        sentence_parts = [
            _normalize_text_snippet(part)
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9'\"(])", raw)
            if _normalize_text_snippet(part)
        ]
        primary_parts = _chunk_sentences(sentence_parts, target_chars=target_chars, max_chars=max_chars)

    chunks: list[str] = []
    for part in primary_parts:
        if len(part) <= max_chars:
            chunks.append(part)
            continue

        sentence_parts = [
            _normalize_text_snippet(sentence)
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9'\"(])", part)
            if _normalize_text_snippet(sentence)
        ]
        if sentence_parts:
            chunks.extend(_chunk_sentences(sentence_parts, target_chars=target_chars, max_chars=max_chars))
        else:
            for start in range(0, len(part), max_chars):
                chunks.append(part[start : start + max_chars].strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        normalized_chunk = _normalize_text_snippet(chunk)
        if not normalized_chunk or normalized_chunk in seen:
            continue
        seen.add(normalized_chunk)
        deduped.append(normalized_chunk)
    return deduped


def _expand_cluster_text_corpus(text_df: pd.DataFrame) -> pd.DataFrame:
    if text_df.empty:
        return pd.DataFrame(
            columns=["name", "name_norm", "text_id", "text_source", "raw_text", "clean_text", "cluster_id", "year_month"]
        )

    source_columns = [
        ("review_text", "Review"),
        ("google_text", "Google"),
        ("kol_text", "KOL"),
    ]
    records: list[dict] = []
    next_text_id = 1

    for row in text_df.to_dict(orient="records"):
        row_sources = [
            (source_label, row.get(source_col))
            for source_col, source_label in source_columns
            if source_col in text_df.columns and _is_nonempty_text(row.get(source_col))
        ]
        if not row_sources and _is_nonempty_text(row.get("raw_text")):
            row_sources = [("Combined", row.get("raw_text"))]

        for source_label, source_text in row_sources:
            for snippet in _split_text_blob(source_text):
                records.append(
                    {
                        "name": row.get("name"),
                        "name_norm": row.get("name_norm"),
                        "text_id": next_text_id,
                        "text_source": source_label,
                        "raw_text": snippet,
                        "clean_text": snippet.lower(),
                        "cluster_id": row.get("cluster_id"),
                        "year_month": row.get("year_month"),
                    }
                )
                next_text_id += 1

    if not records:
        return pd.DataFrame(
            columns=["name", "name_norm", "text_id", "text_source", "raw_text", "clean_text", "cluster_id", "year_month"]
        )
    return pd.DataFrame(records)


@st.cache_data(ttl=300)
def load_cluster_text_corpus() -> pd.DataFrame:
    text_df = pd.DataFrame()
    if REVIEWS_PATH.exists():
        text_df = pd.read_csv(REVIEWS_PATH)

    if text_df.empty:
        return pd.DataFrame(columns=["name", "text_id", "raw_text", "clean_text", "cluster_id", "year_month"]) 

    rename_map = {
        "restaurant name": "name",
        "text": "raw_text",
        "cluster": "cluster_id",
    }
    text_df = text_df.rename(columns=rename_map)

    if "name" not in text_df.columns:
        return pd.DataFrame(columns=["name", "text_id", "raw_text", "clean_text", "cluster_id", "year_month"])

    if "raw_text" not in text_df.columns:
        text_df["raw_text"] = ""
    if "clean_text" not in text_df.columns:
        text_df["clean_text"] = text_df["raw_text"].fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    if "cluster_id" not in text_df.columns:
        text_df["cluster_id"] = pd.NA
    if "text_id" not in text_df.columns:
        text_df["text_id"] = np.arange(1, len(text_df) + 1)
    if "year_month" in text_df.columns:
        text_df["year_month"] = pd.to_datetime(text_df["year_month"], errors="coerce")
    else:
        text_df["year_month"] = pd.NaT

    text_df["name"] = text_df["name"].astype(str).str.strip()
    text_df["name_norm"] = _normalize_name(text_df["name"])
    text_df["cluster_id"] = pd.to_numeric(text_df["cluster_id"], errors="coerce")

    assignments = load_cluster_assignments()[["name_norm", "cluster_id"]].drop_duplicates("name_norm")
    text_df = text_df.drop(columns=["cluster_id"], errors="ignore").merge(assignments, on="name_norm", how="left")
    text_df = _expand_cluster_text_corpus(text_df)

    keep_cols = ["name", "name_norm", "text_id", "text_source", "raw_text", "clean_text", "cluster_id", "year_month"]
    return text_df[keep_cols]


@st.cache_data(ttl=300)
def load_cluster_keywords() -> pd.DataFrame:
    text_df = load_cluster_text_corpus()
    if not text_df.empty:
        rows = []
        for cluster_id, cdf in text_df.groupby("cluster_id", dropna=True):
            token_counts: dict[str, int] = {}
            for text in cdf["clean_text"].fillna(""):
                for token in _tokenize_text(text):
                    token_counts[token] = token_counts.get(token, 0) + 1
            if not token_counts:
                continue
            top_terms = sorted(token_counts.items(), key=lambda kv: kv[1], reverse=True)[:40]
            for rank, (token, weight) in enumerate(top_terms, start=1):
                rows.append({"cluster_id": int(cluster_id), "keyword": token, "weight": float(weight), "rank": rank})
        if rows:
            return pd.DataFrame(rows)

    return pd.DataFrame(columns=["cluster_id", "keyword", "weight", "rank"])


@st.cache_data(ttl=300)
def load_ga_campaign_type_monthly() -> pd.DataFrame:
    if not GA_OUTREACH_PATH.exists():
        return pd.DataFrame(columns=["year_month", "googleAdsCampaignType", "total_sessions", "active_campaigns"])

    outreach_df = pd.read_parquet(GA_OUTREACH_PATH)
    if outreach_df.empty:
        return pd.DataFrame(columns=["year_month", "googleAdsCampaignType", "total_sessions", "active_campaigns"])

    outreach_df = outreach_df.copy()
    outreach_df = outreach_df[
        outreach_df["campaignId"].notna()
        & outreach_df["campaignId"].astype(str).ne("(not set)")
    ]
    outreach_df["googleAdsCampaignType"] = outreach_df["googleAdsCampaignType"].fillna("Unknown").astype(str).str.strip()
    outreach_df = outreach_df[
        outreach_df["googleAdsCampaignType"].ne("")
        & outreach_df["googleAdsCampaignType"].ne("(not set)")
    ]
    outreach_df["year_month"] = pd.to_datetime(outreach_df["yearMonth"].astype(str), format="%Y%m", errors="coerce")
    outreach_df["sessions"] = pd.to_numeric(
        outreach_df["sessions"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
    )
    outreach_df = outreach_df.dropna(subset=["year_month", "sessions"])

    return (
        outreach_df.groupby(["year_month", "googleAdsCampaignType"], as_index=False)
        .agg(
            total_sessions=("sessions", "sum"),
            active_campaigns=("campaignId", "nunique"),
        )
        .sort_values(["year_month", "googleAdsCampaignType"])
        .reset_index(drop=True)
    )


@st.cache_data(ttl=300)
def load_cluster_ga_campaign_effectiveness() -> pd.DataFrame:
    ga_monthly = load_ga_restaurant_monthly()
    cluster_assignments = load_cluster_assignments()
    campaign_monthly = load_ga_campaign_type_monthly()

    if ga_monthly.empty or cluster_assignments.empty or campaign_monthly.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
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
        )

    cluster_ref = cluster_assignments[["name_norm", "cluster_id", "cluster_label"]].drop_duplicates("name_norm")
    cluster_monthly = (
        ga_monthly.merge(cluster_ref, on="name_norm", how="inner")
        .groupby(["cluster_id", "cluster_label", "year_month"], as_index=False)
        .agg(
            total_monthly_gmv=("monthly_gmv", "sum"),
            total_ga_items_viewed=("ga_items_viewed", "sum"),
            total_ga_items_added_to_cart=("ga_items_added_to_cart", "sum"),
            total_ga_items_purchased=("ga_items_purchased", "sum"),
            restaurants_with_ga_data=("name_norm", "nunique"),
        )
    )
    cluster_monthly["gmv_per_ga_view"] = _safe_divide(
        cluster_monthly["total_monthly_gmv"],
        cluster_monthly["total_ga_items_viewed"],
    )
    cluster_monthly["ga_add_to_cart_rate"] = _safe_divide(
        cluster_monthly["total_ga_items_added_to_cart"],
        cluster_monthly["total_ga_items_viewed"],
    )
    cluster_monthly["ga_view_to_purchase_rate"] = _safe_divide(
        cluster_monthly["total_ga_items_purchased"],
        cluster_monthly["total_ga_items_viewed"],
    )

    merged = cluster_monthly.merge(campaign_monthly, on="year_month", how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
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
        )

    rows = []
    for (cluster_id, cluster_label, campaign_type), grp in merged.groupby(
        ["cluster_id", "cluster_label", "googleAdsCampaignType"],
        dropna=False,
    ):
        weights = pd.to_numeric(grp["total_sessions"], errors="coerce").fillna(0)
        valid_weight = weights.gt(0)

        def _weighted_metric_mean(col: str) -> float:
            metric = pd.to_numeric(grp[col], errors="coerce")
            valid = valid_weight & metric.notna()
            if valid.any():
                return float(np.average(metric[valid], weights=weights[valid]))
            fallback = metric.mean()
            return float(fallback) if pd.notna(fallback) else np.nan

        gmv_weighted = _weighted_metric_mean("gmv_per_ga_view")
        atc_weighted = _weighted_metric_mean("ga_add_to_cart_rate")
        purchase_weighted = _weighted_metric_mean("ga_view_to_purchase_rate")

        corr = np.nan
        if grp["year_month"].nunique() >= 3:
            corr = grp["total_sessions"].corr(grp["gmv_per_ga_view"])

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": cluster_label,
                "googleAdsCampaignType": campaign_type,
                "active_months": int(grp["year_month"].nunique()),
                "total_sessions": float(grp["total_sessions"].sum()),
                "active_campaigns": int(grp["active_campaigns"].sum()),
                "session_weighted_gmv_per_ga_view": float(gmv_weighted) if pd.notna(gmv_weighted) else np.nan,
                "session_weighted_add_to_cart_rate": float(atc_weighted) if pd.notna(atc_weighted) else np.nan,
                "session_weighted_view_to_purchase_rate": float(purchase_weighted) if pd.notna(purchase_weighted) else np.nan,
                "sessions_to_gmv_correlation": float(corr) if pd.notna(corr) else np.nan,
            }
        )

    effect_df = pd.DataFrame(rows)
    if effect_df.empty:
        return effect_df

    effect_df["gmv_norm"] = effect_df.groupby("cluster_id")["session_weighted_gmv_per_ga_view"].transform(
        lambda s: _min_max_norm(s, neutral=0.5)
    )
    effect_df["atc_norm"] = effect_df.groupby("cluster_id")["session_weighted_add_to_cart_rate"].transform(
        lambda s: _min_max_norm(s, neutral=0.5)
    )
    corr_norm = effect_df["sessions_to_gmv_correlation"].clip(-1, 1)
    corr_norm = (corr_norm.fillna(0) + 1) / 2
    effect_df["ga_effectiveness_score"] = (
        0.55 * effect_df["gmv_norm"]
        + 0.25 * effect_df["atc_norm"]
        + 0.20 * corr_norm
    ) * 100
    effect_df = effect_df.drop(columns=["gmv_norm", "atc_norm"])

    return effect_df.sort_values(
        ["cluster_id", "ga_effectiveness_score", "session_weighted_gmv_per_ga_view"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def _series_or_default(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col in df.columns:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s.fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype="object")


def _build_strategy_name(df: pd.DataFrame) -> pd.Series:
    channel = _series_or_default(df, "channel", "Unknown").str.upper().str.strip()

    crm_name = _series_or_default(df, "crm_topic", "").str.strip()
    crm_fallback = _series_or_default(df, "crm_campaign_name", "").str.strip()
    fb_name = _series_or_default(df, "fb_campaign", "").str.strip()
    kol_user = _series_or_default(df, "kol_username", "").str.strip()

    strategy_name = np.where(
        channel.eq("CRM"),
        np.where(crm_name.ne(""), "CRM | " + crm_name, np.where(crm_fallback.ne(""), "CRM | " + crm_fallback, "CRM | campaign")),
        np.where(
            channel.eq("FB"),
            np.where(fb_name.ne(""), "FB | " + fb_name, "FB | campaign"),
            np.where(
                channel.eq("KOL"),
                np.where(kol_user.ne(""), "KOL | @" + kol_user, "KOL | creator"),
                channel + " | campaign",
            ),
        ),
    )
    return pd.Series(strategy_name, index=df.index).str.replace(r"\s+", " ", regex=True).str.strip()


def _build_strategy_family(df: pd.DataFrame) -> pd.Series:
    channel = _series_or_default(df, "channel", "Unknown").str.upper().str.strip()
    raw_strategy = _series_or_default(df, "strategy_name", "")
    fallback_strategy = _build_strategy_name(df)
    raw_strategy = raw_strategy.where(raw_strategy.str.strip().ne(""), fallback_strategy)

    text_blob = (
        raw_strategy
        + " "
        + _series_or_default(df, "crm_topic", "")
        + " "
        + _series_or_default(df, "crm_campaign_name", "")
        + " "
        + _series_or_default(df, "fb_campaign", "")
        + " "
        + _series_or_default(df, "kol_username", "")
    ).str.lower()

    families = pd.Series(["Other | Campaign"] * len(df), index=df.index, dtype="object")

    promo_mask = text_blob.str.contains(r"promo|discount|voucher|coupon|deal|sale|flash|bundle|off", regex=True)
    loyalty_mask = text_blob.str.contains(r"loyal|member|vip|retention|repeat|reward|point", regex=True)
    reactivate_mask = text_blob.str.contains(r"reactivat|winback|inactive|lapsed|dormant|comeback|churn", regex=True)
    seasonal_mask = text_blob.str.contains(r"season|festival|holiday|songkran|new year|christmas|valentine|ramadan|lunar", regex=True)
    retarget_mask = text_blob.str.contains(r"retarget|remarket|re-engage|engaged audience|visited", regex=True)
    prospect_mask = text_blob.str.contains(r"prospecting|lookalike|acquisition|awareness|reach|traffic|new customer", regex=True)
    influencer_mask = text_blob.str.contains(r"influencer|creator|kol|review|ugc|collab|partnership|tasting", regex=True)

    crm_mask = channel.eq("CRM")
    fb_mask = channel.eq("FB")
    kol_mask = channel.eq("KOL")

    families.loc[crm_mask & reactivate_mask] = "CRM | Reactivation"
    families.loc[crm_mask & loyalty_mask & ~reactivate_mask] = "CRM | Loyalty & Retention"
    families.loc[crm_mask & promo_mask & ~reactivate_mask & ~loyalty_mask] = "CRM | Promotional Blast"
    families.loc[crm_mask & seasonal_mask & ~reactivate_mask & ~loyalty_mask & ~promo_mask] = "CRM | Seasonal Campaign"
    families.loc[crm_mask & families.eq("Other | Campaign")] = "CRM | Lifecycle Nurture"

    families.loc[fb_mask & retarget_mask] = "FB | Retargeting"
    families.loc[fb_mask & promo_mask & ~retarget_mask] = "FB | Conversion Offer"
    families.loc[fb_mask & prospect_mask & ~retarget_mask & ~promo_mask] = "FB | Prospecting & Awareness"
    families.loc[fb_mask & families.eq("Other | Campaign")] = "FB | Performance Campaign"

    families.loc[kol_mask & influencer_mask] = "KOL | Creator Collaboration"
    families.loc[kol_mask & promo_mask & ~influencer_mask] = "KOL | Creator Promo Push"
    families.loc[kol_mask & families.eq("Other | Campaign")] = "KOL | Influencer Partnership"

    unmapped = families.eq("Other | Campaign")
    families.loc[unmapped] = channel[unmapped].replace("", "Other").fillna("Other") + " | Campaign"
    return families.str.replace(r"\s+", " ", regex=True).str.strip()


@st.cache_data(ttl=300)
def load_cluster_strategy_outcomes() -> pd.DataFrame:
    marketing_df = _read_table(MARKETING_PATH)
    assignments = load_cluster_assignments()
    empty_outcomes = pd.DataFrame(
        columns=[
            "cluster_id",
            "cluster_label",
            "strategy_name",
            "strategy_family",
            "restaurant_name",
            "restaurant_id",
            "latest_segment",
            "channel",
            "applied_date",
            "bookings_before",
            "bookings_after",
            "bookings_uplift_pct",
            "revenue_before",
            "revenue_after",
            "revenue_uplift_pct",
            "incremental_revenue_thb",
            "roi",
            "sample_size",
            "activity_id",
        ]
    )

    if marketing_df.empty or assignments.empty:
        return empty_outcomes

    df = marketing_df.copy()
    df["restaurant_id"] = pd.to_numeric(df.get("restaurant_id"), errors="coerce").astype("Int64")
    if "restaurant_name" in df.columns:
        df["restaurant_name_norm"] = _normalize_name(df["restaurant_name"])

    assignments_join = assignments[
        ["restaurant_id", "name", "name_norm", "cluster_id", "cluster_label", "latest_segment"]
    ].drop_duplicates("name_norm")

    assignments_by_id = (
        assignments_join[
            ["restaurant_id", "name", "cluster_id", "cluster_label", "latest_segment"]
        ]
        .dropna(subset=["restaurant_id"])
        .drop_duplicates("restaurant_id")
    )
    merged = df.merge(assignments_by_id, on="restaurant_id", how="left", suffixes=("", "_cluster"))
    if {"name_norm", "restaurant_name_norm"}.issubset(merged.columns):
        fallback = assignments_join.rename(
            columns={
                "restaurant_id": "restaurant_id_cluster",
                "name": "name_cluster",
                "cluster_id": "cluster_id_cluster",
                "cluster_label": "cluster_label_cluster",
                "latest_segment": "latest_segment_cluster",
            }
        )
        merged = merged.merge(
            fallback[
                [
                    "name_norm",
                    "restaurant_id_cluster",
                    "name_cluster",
                    "cluster_id_cluster",
                    "cluster_label_cluster",
                    "latest_segment_cluster",
                ]
            ],
            left_on="restaurant_name_norm",
            right_on="name_norm",
            how="left",
        )
        for col in ["restaurant_id", "name", "cluster_id", "cluster_label", "latest_segment"]:
            fallback_col = f"{col}_cluster"
            if fallback_col in merged.columns:
                if col in merged.columns:
                    merged[col] = merged[col].where(merged[col].notna(), merged[fallback_col])
                else:
                    merged[col] = merged[fallback_col]
                merged = merged.drop(columns=[fallback_col])
        merged = merged.drop(columns=["name_norm"], errors="ignore")
    if merged.empty:
        return empty_outcomes

    merged["strategy_name"] = _build_strategy_name(merged)
    merged["strategy_family"] = _build_strategy_family(merged)

    merged["bookings_before"] = pd.to_numeric(merged.get("bookings_baseline"), errors="coerce")
    merged["bookings_after"] = pd.to_numeric(merged.get("bookings_during"), errors="coerce")
    merged["incremental_revenue_thb"] = pd.to_numeric(merged.get("incremental_revenue_thb"), errors="coerce")
    merged["revenue_after"] = pd.to_numeric(merged.get("total_campaign_revenue"), errors="coerce")
    merged["roi"] = pd.to_numeric(merged.get("roi"), errors="coerce")

    merged["bookings_uplift_pct"] = np.where(
        merged["bookings_before"] > 0,
        (merged["bookings_after"] - merged["bookings_before"]) / merged["bookings_before"],
        np.nan,
    )

    merged["revenue_before"] = merged["revenue_after"] - merged["incremental_revenue_thb"]
    merged["revenue_uplift_pct"] = np.where(
        merged["revenue_before"] > 0,
        merged["incremental_revenue_thb"] / merged["revenue_before"],
        np.nan,
    )

    merged["applied_date"] = pd.to_datetime(merged.get("activity_start"), errors="coerce")
    merged["sample_size"] = 1

    out_df = merged.rename(columns={"name": "restaurant_name"})

    rename_pairs = [
        ("cluster", "cluster_id"),
        ("name", "restaurant_name"),
        ("applied_period", "applied_date"),
        ("bookings_baseline", "bookings_before"),
        ("bookings_during", "bookings_after"),
        ("total_campaign_revenue", "revenue_after"),
    ]
    for src, dst in rename_pairs:
        if src in out_df.columns and dst not in out_df.columns:
            out_df = out_df.rename(columns={src: dst})
        elif src in out_df.columns and dst in out_df.columns:
            out_df = out_df.drop(columns=[src])

    if "strategy_name" not in out_df.columns:
        out_df["strategy_name"] = _build_strategy_name(out_df)
    if "strategy_family" not in out_df.columns:
        out_df["strategy_family"] = _build_strategy_family(out_df)
    else:
        raw_family = _series_or_default(out_df, "strategy_family", "")
        out_df["strategy_family"] = raw_family.where(raw_family.str.strip().ne(""), _build_strategy_family(out_df))

    numeric_cols = [
        "cluster_id",
        "bookings_before",
        "bookings_after",
        "bookings_uplift_pct",
        "revenue_before",
        "revenue_after",
        "revenue_uplift_pct",
        "incremental_revenue_thb",
        "roi",
        "sample_size",
    ]
    for col in numeric_cols:
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce")

    if "sample_size" not in out_df.columns:
        out_df["sample_size"] = 1
    if "restaurant_name" not in out_df.columns:
        out_df["restaurant_name"] = "Unknown"
    if "channel" not in out_df.columns:
        out_df["channel"] = "Unknown"
    if "cluster_label" not in out_df.columns:
        out_df["cluster_label"] = out_df["cluster_id"].apply(lambda v: f"Cluster {int(v)}" if pd.notna(v) else "Unknown")
    if "latest_segment" not in out_df.columns:
        out_df["latest_segment"] = "Unknown"
    else:
        out_df["latest_segment"] = out_df["latest_segment"].fillna("Unknown").replace("", "Unknown")

    # Guard against unstable uplift ratios from tiny/zero baselines.
    # Example: revenue_before ~= 0 with positive incremental revenue would explode to absurd % values.
    if {"revenue_after", "incremental_revenue_thb"}.issubset(out_df.columns):
        calc_revenue_before = out_df["revenue_after"] - out_df["incremental_revenue_thb"]
        if "revenue_before" not in out_df.columns:
            out_df["revenue_before"] = calc_revenue_before
        else:
            out_df["revenue_before"] = out_df["revenue_before"].where(out_df["revenue_before"].notna(), calc_revenue_before)

    if {"revenue_before", "incremental_revenue_thb"}.issubset(out_df.columns):
        min_revenue_baseline_thb = 100.0
        valid_revenue_baseline = out_df["revenue_before"] >= min_revenue_baseline_thb
        out_df["revenue_uplift_pct"] = np.where(
            valid_revenue_baseline,
            out_df["incremental_revenue_thb"] / out_df["revenue_before"],
            np.nan,
        )
        out_df["revenue_uplift_pct"] = pd.to_numeric(out_df["revenue_uplift_pct"], errors="coerce")
        out_df["revenue_uplift_pct"] = out_df["revenue_uplift_pct"].where(np.isfinite(out_df["revenue_uplift_pct"]), np.nan)
        out_df["revenue_uplift_pct"] = out_df["revenue_uplift_pct"].where(
            out_df["revenue_uplift_pct"].between(-0.99, 20.0),
            np.nan,
        )

    if {"bookings_before", "bookings_after"}.issubset(out_df.columns):
        min_booking_baseline = 5.0
        valid_booking_baseline = out_df["bookings_before"] >= min_booking_baseline
        out_df["bookings_uplift_pct"] = np.where(
            valid_booking_baseline,
            (out_df["bookings_after"] - out_df["bookings_before"]) / out_df["bookings_before"],
            np.nan,
        )
        out_df["bookings_uplift_pct"] = pd.to_numeric(out_df["bookings_uplift_pct"], errors="coerce")
        out_df["bookings_uplift_pct"] = out_df["bookings_uplift_pct"].where(np.isfinite(out_df["bookings_uplift_pct"]), np.nan)
        out_df["bookings_uplift_pct"] = out_df["bookings_uplift_pct"].where(
            out_df["bookings_uplift_pct"].between(-0.99, 20.0),
            np.nan,
        )

    out_df["applied_date"] = pd.to_datetime(out_df.get("applied_date"), errors="coerce")

    keep_cols = [
        c
        for c in [
            "cluster_id",
            "cluster_label",
            "strategy_name",
            "strategy_family",
            "restaurant_name",
            "restaurant_id",
            "latest_segment",
            "channel",
            "applied_date",
            "bookings_before",
            "bookings_after",
            "bookings_uplift_pct",
            "revenue_before",
            "revenue_after",
            "revenue_uplift_pct",
            "incremental_revenue_thb",
            "roi",
            "sample_size",
            "activity_id",
        ]
        if c in out_df.columns
    ]
    out_df = out_df[keep_cols].dropna(subset=["cluster_id"]).reset_index(drop=True)
    if out_df.empty:
        return empty_outcomes
    family = out_df["strategy_family"].fillna(out_df["strategy_name"])
    out_df["strategy_family"] = family.where(family.astype(str).str.strip().ne(""), out_df["strategy_name"])
    return out_df


def _nan_quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(np.nanquantile(values.to_numpy(), q))


def _empty_rankings() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "scope_type",
            "scope_key",
            "scope_label",
            "cluster_id",
            "cluster_label",
            "latest_segment",
            "strategy_name",
            "strategy_examples",
            "activities",
            "restaurants",
            "avg_incremental_revenue_thb",
            "total_incremental_revenue_thb",
            "avg_revenue_uplift_pct",
            "avg_bookings_uplift_pct",
            "avg_roi",
            "revenue_uplift_p25",
            "revenue_uplift_p75",
            "success_rate",
            "ranking_score",
            "context_adjusted_score",
            "confidence_score",
            "confidence_level",
            "meets_sample_guardrail",
            "data_quality_note",
            "strategy_rank",
        ]
    )


def _aggregate_strategy_rankings(
    outcomes_df: pd.DataFrame,
    scope_type: str = "cluster",
    min_sample_size: int = 3,
) -> pd.DataFrame:
    if outcomes_df.empty:
        return _empty_rankings()

    df = outcomes_df.copy()
    strategy_family = _series_or_default(df, "strategy_family", "")
    df["strategy_name"] = strategy_family.where(strategy_family.str.strip().ne(""), _series_or_default(df, "strategy_name", "Unknown"))
    df["restaurant_name"] = _series_or_default(df, "restaurant_name", "Unknown")
    df["sample_size"] = pd.to_numeric(df.get("sample_size"), errors="coerce").fillna(1.0)
    df["bookings_uplift_pct"] = pd.to_numeric(df.get("bookings_uplift_pct"), errors="coerce")
    df["revenue_uplift_pct"] = pd.to_numeric(df.get("revenue_uplift_pct"), errors="coerce")
    df["incremental_revenue_thb"] = pd.to_numeric(df.get("incremental_revenue_thb"), errors="coerce")
    df["roi"] = pd.to_numeric(df.get("roi"), errors="coerce")
    df["cluster_id"] = pd.to_numeric(df.get("cluster_id"), errors="coerce")
    df["cluster_label"] = _series_or_default(df, "cluster_label", "Unknown")
    df["latest_segment"] = _series_or_default(df, "latest_segment", "Unknown").replace("", "Unknown")

    scope_type = str(scope_type).lower().strip()
    if scope_type == "cluster":
        scope_group_cols = ["cluster_id", "cluster_label"]
    elif scope_type == "segment":
        scope_group_cols = ["latest_segment"]
    elif scope_type == "global":
        scope_group_cols = []
    else:
        raise ValueError(f"Unknown scope_type: {scope_type}")

    group_cols = scope_group_cols + ["strategy_name"]
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            activities=("sample_size", "sum"),
            restaurants=("restaurant_name", pd.Series.nunique),
            avg_incremental_revenue_thb=("incremental_revenue_thb", "mean"),
            total_incremental_revenue_thb=("incremental_revenue_thb", "sum"),
            avg_revenue_uplift_pct=("revenue_uplift_pct", "mean"),
            avg_bookings_uplift_pct=("bookings_uplift_pct", "mean"),
            avg_roi=("roi", "mean"),
            revenue_uplift_p25=("revenue_uplift_pct", lambda s: _nan_quantile(s, 0.25)),
            revenue_uplift_p75=("revenue_uplift_pct", lambda s: _nan_quantile(s, 0.75)),
            success_rate=("incremental_revenue_thb", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean()) if len(s) else np.nan),
        )
        .reset_index()
    )

    example_map = (
        df.groupby(group_cols, dropna=False)["strategy_name"]
        .agg(lambda s: " | ".join(pd.Series(s).dropna().astype(str).value_counts().head(3).index.tolist()))
        .reset_index(name="strategy_examples")
    )
    agg = agg.merge(example_map, on=group_cols, how="left")

    if scope_type == "cluster":
        agg["scope_type"] = "cluster"
        agg["scope_key"] = pd.to_numeric(agg["cluster_id"], errors="coerce").fillna(-1).astype(int)
        agg["scope_label"] = agg["cluster_label"].fillna(agg["scope_key"].map(lambda v: f"Cluster {v}"))
    elif scope_type == "segment":
        agg["scope_type"] = "segment"
        agg["scope_key"] = agg["latest_segment"].fillna("Unknown").replace("", "Unknown")
        agg["scope_label"] = "Segment: " + agg["scope_key"]
        agg["cluster_id"] = np.nan
        agg["cluster_label"] = np.nan
    else:
        agg["scope_type"] = "global"
        agg["scope_key"] = "ALL"
        agg["scope_label"] = "Global"
        agg["cluster_id"] = np.nan
        agg["cluster_label"] = np.nan
        agg["latest_segment"] = "All Segments"

    rank_partition_cols = ["scope_key"]
    partition_max = agg.groupby(rank_partition_cols)["avg_incremental_revenue_thb"].transform(lambda s: s.abs().max())
    rev_fallback = np.where((partition_max > 0) & partition_max.notna(), agg["avg_incremental_revenue_thb"] / partition_max, 0)
    revenue_component = agg["avg_revenue_uplift_pct"].where(agg["avg_revenue_uplift_pct"].notna(), rev_fallback)
    bookings_component = agg["avg_bookings_uplift_pct"].fillna(0)
    roi_component = agg["avg_roi"].fillna(0)
    success_component = agg["success_rate"].fillna(0)

    agg["ranking_score"] = (
        revenue_component.fillna(0) * 100
        + bookings_component * 30
        + roi_component * 10
        + success_component * 5
    )
    agg["context_adjusted_score"] = agg["ranking_score"]

    agg["meets_sample_guardrail"] = agg["activities"] >= int(min_sample_size)
    metric_coverage = (agg[["avg_revenue_uplift_pct", "avg_bookings_uplift_pct", "avg_roi"]].notna().sum(axis=1) / 3.0)
    activity_scale = np.clip(agg["activities"] / max(float(min_sample_size) * 4.0, 1.0), 0, 1)
    restaurant_scale = np.clip(agg["restaurants"] / 6.0, 0, 1)
    confidence_score = (0.40 * activity_scale + 0.20 * restaurant_scale + 0.25 * success_component.clip(0, 1) + 0.15 * metric_coverage) * 100
    agg["confidence_score"] = confidence_score.round(1)
    agg["confidence_level"] = np.where(
        agg["confidence_score"] >= 75,
        "High",
        np.where(agg["confidence_score"] >= 55, "Medium", "Low"),
    )

    agg["data_quality_note"] = np.where(
        ~agg["meets_sample_guardrail"],
        f"Low sample size (<{int(min_sample_size)} activities)",
        np.where(metric_coverage < 0.67, "Partial metric coverage", agg["confidence_level"] + " confidence evidence"),
    )

    sort_cols = rank_partition_cols + ["meets_sample_guardrail", "ranking_score"]
    agg = agg.sort_values(sort_cols, ascending=[True, False, False])
    agg["strategy_rank"] = np.nan
    eligible_mask = agg["meets_sample_guardrail"]
    if eligible_mask.any():
        agg.loc[eligible_mask, "strategy_rank"] = (
            agg[eligible_mask].groupby(rank_partition_cols)["ranking_score"].rank(method="first", ascending=False)
        )

    return agg.reset_index(drop=True)


@st.cache_data(ttl=300)
def load_strategy_rankings(scope_type: str = "cluster", min_sample_size: int = 3) -> pd.DataFrame:
    outcomes_df = load_cluster_strategy_outcomes()
    return _aggregate_strategy_rankings(outcomes_df, scope_type=scope_type, min_sample_size=min_sample_size)


@st.cache_data(ttl=300)
def load_cluster_strategy_rankings(min_sample_size: int = 3) -> pd.DataFrame:
    return load_strategy_rankings(scope_type="cluster", min_sample_size=min_sample_size)


def _select_scope_recommendations(
    rankings_df: pd.DataFrame,
    scope_key,
    top_n: int = 5,
    include_low_sample_if_needed: bool = True,
) -> pd.DataFrame:
    if rankings_df.empty:
        return pd.DataFrame()

    recs = rankings_df[rankings_df["scope_key"] == scope_key].copy()
    if recs.empty:
        return recs

    eligible = recs[recs["meets_sample_guardrail"]].sort_values("ranking_score", ascending=False)
    if len(eligible):
        return eligible.head(top_n)
    if include_low_sample_if_needed:
        return recs.sort_values("ranking_score", ascending=False).head(top_n)
    return pd.DataFrame()


def get_cluster_strategy_recommendations(cluster_id: int, top_n: int = 5, min_sample_size: int = 3) -> pd.DataFrame:
    rankings = load_cluster_strategy_rankings(min_sample_size=min_sample_size)
    if rankings.empty:
        return _empty_rankings()

    recs = _select_scope_recommendations(rankings, int(cluster_id), top_n=top_n, include_low_sample_if_needed=True)
    if recs.empty:
        return recs
    recs["recommendation_scope"] = "cluster"
    return recs


def _contextual_strategy_boost(strategy_name: str, segment: str, priority_tier: str, preferred_channel: str) -> float:
    text = str(strategy_name).lower()
    segment_l = str(segment).lower()
    tier_l = str(priority_tier).lower()
    preferred_channel_l = str(preferred_channel).lower().strip()

    boost = 0.0
    if "needs attention" in segment_l:
        if any(k in text for k in ["reactivation", "retarget", "retention", "loyalty"]):
            boost += 10
        if "awareness" in text:
            boost -= 2
    elif "established players" in segment_l:
        if any(k in text for k in ["loyalty", "retention", "crm"]):
            boost += 8
    elif "emerging opportunities" in segment_l:
        if any(k in text for k in ["conversion", "promo", "retarget", "prospecting"]):
            boost += 7
    elif "rising stars" in segment_l:
        if any(k in text for k in ["creator", "influencer", "awareness", "prospecting"]):
            boost += 8

    if "proven" in tier_l and any(k in text for k in ["conversion", "retarget", "loyalty"]):
        boost += 3
    if "untapped" in tier_l and any(k in text for k in ["prospecting", "creator", "awareness"]):
        boost += 3
    if "review channel strategy" in tier_l and any(k in text for k in ["performance", "retarget", "crm"]):
        boost += 2

    if preferred_channel_l and text.startswith(preferred_channel_l):
        boost += 2
    return float(boost)


def recommend_strategies_for_restaurant(name: str, top_n: int = 3, min_sample_size: int = 3) -> pd.DataFrame:
    assignments = load_cluster_assignments()
    if assignments.empty:
        return _empty_rankings()

    name_norm = _normalize_name(pd.Series([name])).iloc[0]
    row = assignments[assignments["name_norm"] == name_norm].head(1)
    if row.empty:
        return _empty_rankings()

    cluster_id_raw = pd.to_numeric(row.iloc[0].get("cluster_id"), errors="coerce")
    cluster_id = int(cluster_id_raw) if pd.notna(cluster_id_raw) else -1
    cluster_label = row.iloc[0].get("cluster_label", f"Cluster {cluster_id}")
    segment = row.iloc[0].get("latest_segment", "Unknown")

    priority_df = load_priority().copy()
    if "name" in priority_df.columns:
        priority_df["name_norm"] = _normalize_name(priority_df["name"])
        p_row = priority_df[priority_df["name_norm"] == name_norm].head(1)
    else:
        p_row = pd.DataFrame()

    priority_tier = p_row.iloc[0].get("priority_tier", "") if len(p_row) else ""
    preferred_channel = p_row.iloc[0].get("recommended_channel", "") if len(p_row) else ""

    frames: list[pd.DataFrame] = []

    cluster_rank = load_strategy_rankings(scope_type="cluster", min_sample_size=min_sample_size)
    cluster_recs = _select_scope_recommendations(cluster_rank, cluster_id, top_n=top_n, include_low_sample_if_needed=True)
    if len(cluster_recs):
        cluster_recs = cluster_recs.copy()
        cluster_recs["recommendation_scope"] = "cluster"
        frames.append(cluster_recs)

    segment_rank = load_strategy_rankings(scope_type="segment", min_sample_size=min_sample_size)
    segment_key = str(segment) if pd.notna(segment) and str(segment).strip() else "Unknown"
    segment_recs = _select_scope_recommendations(segment_rank, segment_key, top_n=max(top_n, 5), include_low_sample_if_needed=True)
    if len(segment_recs):
        segment_recs = segment_recs.copy()
        segment_recs["recommendation_scope"] = "segment"
        frames.append(segment_recs)

    global_rank = load_strategy_rankings(scope_type="global", min_sample_size=min_sample_size)
    global_recs = _select_scope_recommendations(global_rank, "ALL", top_n=max(top_n, 5), include_low_sample_if_needed=True)
    if len(global_recs):
        global_recs = global_recs.copy()
        global_recs["recommendation_scope"] = "global"
        frames.append(global_recs)

    if not frames:
        return _empty_rankings()

    recs = pd.concat(frames, ignore_index=True)
    recs["scope_weight"] = recs["recommendation_scope"].map({"cluster": 3, "segment": 2, "global": 1}).fillna(0)
    recs["context_boost"] = recs["strategy_name"].apply(
        lambda s: _contextual_strategy_boost(s, segment=segment, priority_tier=priority_tier, preferred_channel=preferred_channel)
    )
    recs["context_adjusted_score"] = recs["ranking_score"] + recs["context_boost"] + recs["scope_weight"]
    recs = recs.sort_values(
        ["context_adjusted_score", "scope_weight", "meets_sample_guardrail", "confidence_score"],
        ascending=[False, False, False, False],
    )

    recs = recs.drop_duplicates(subset=["strategy_name"], keep="first").head(top_n).copy()
    if recs.empty:
        return _empty_rankings()

    scope_note = {
        "cluster": "Cluster history",
        "segment": "Momentum segment history",
        "global": "Global platform history",
    }
    recs["recommendation_reason"] = recs["recommendation_scope"].map(scope_note).fillna("Historical evidence")
    recs.insert(0, "restaurant_name", row.iloc[0].get("name", name))
    recs.insert(1, "restaurant_segment", segment if pd.notna(segment) else "Unknown")
    recs.insert(2, "restaurant_cluster_id", cluster_id)
    recs.insert(3, "restaurant_cluster_label", cluster_label)
    recs.insert(4, "restaurant_priority_tier", priority_tier if priority_tier else "Unknown")
    recs.insert(5, "restaurant_preferred_channel", preferred_channel if preferred_channel else "Unknown")
    return recs.reset_index(drop=True)


def get_restaurant_history(momentum_df: pd.DataFrame, name: str) -> pd.DataFrame:
    return momentum_df[momentum_df["name"] == name].sort_values("year_month").copy()


def get_restaurant_priority_row(priority_df: pd.DataFrame, name: str) -> dict:
    rows = priority_df[priority_df["name"] == name]
    if len(rows):
        return rows.iloc[0].to_dict()
    return {}
