"""
data/loader.py
--------------
Reads pipeline outputs from the notebook working directory.
Falls back to realistic sample data if files are not yet present
so the dashboard is always runnable during development.
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ── Point these at wherever your notebooks save their outputs ─────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent.parent  # adjust if needed
MOMENTUM_PATH  = BASE_DIR / "_2_feature_engineering+momentum" / "start" / "restaurants_agg_performance.parquet"
MARKETING_PATH = BASE_DIR / "_3_marketing" / "activity_performance_with_roi.csv"
PRIORITY_PATH  = BASE_DIR / "_4_final_outputs" / "priority_list.csv"

TIER_COLORS = {
    "🔴 Activate — proven marketing response" : "#e74c3c",
    "🟠 Activate — untapped, no prior spend"  : "#e67e22",
    "🟡 Activate — review channel strategy"   : "#f1c40f",
}

SEGMENT_COLORS = {
    "Rising Stars"           : "#2ecc71",
    "Emerging Opportunities" : "#3498db",
    "Established Players"    : "#9b59b6",
    "Needs Attention"        : "#e74c3c",
}


# ── Sample data generator (dev / demo mode) ───────────────────────────────────
def _make_sample_data():
    rng = np.random.default_rng(42)
    n_restaurants = 80
    months = pd.date_range("2024-08-01", "2026-01-01", freq="MS")
    names = [
        f"Restaurant {chr(65 + i % 26)}{i // 26 if i >= 26 else ''}" 
        for i in range(n_restaurants)
    ]
    segments = rng.choice(
        ["Rising Stars", "Emerging Opportunities", "Established Players", "Needs Attention"],
        size=n_restaurants, p=[0.15, 0.20, 0.25, 0.40]
    )
    locations = rng.choice(
        ["Bangkok", "Chiang Mai", "Phuket", "Pattaya", "Hua Hin"],
        size=n_restaurants
    )
    cuisines = rng.choice(
        ["Thai", "Japanese", "Italian", "Chinese", "International", "Fusion"],
        size=n_restaurants
    )

    rows = []
    for i, name in enumerate(names):
        base = rng.integers(5, 60)
        for m in months:
            trend  = 1 + rng.normal(0.02, 0.08)
            season = 1 + 0.15 * np.sin(2 * np.pi * m.month / 12)
            bk     = max(0, int(base * trend * season + rng.normal(0, 3)))
            rows.append({
                "restaurant_id"          : 1000 + i,
                "name"                   : name,
                "year_month"             : m,
                "monthly_bookings"       : bk,
                "monthly_revenue"        : bk * rng.uniform(300, 1200),
                "avg_revenue_per_booking": rng.uniform(300, 1200),
                "avg_guests"             : rng.uniform(2.0, 4.5),
                "active_days"            : rng.integers(10, 28),
                "location"               : locations[i],
                "cuisine"                : cuisines[i],
                "booking_growth_rolling" : rng.uniform(-0.3, 0.8),
                "revenue_growth_rolling" : rng.uniform(-0.3, 0.8),
                "booking_growth_yoy"     : rng.uniform(-0.2, 0.6),
                "revenue_growth_yoy"     : rng.uniform(-0.2, 0.6),
                "growth_signal_used"     : rng.choice(["YoY", "MoM"], p=[0.6, 0.4]),
                "score_perf"             : rng.uniform(0, 1),
                "score_growth"           : rng.uniform(0, 1),
                "delta_growth_book"      : rng.uniform(-0.2, 0.2),
                "in_analysis_window"     : m >= pd.Timestamp("2025-01-01"),
            })

    momentum_df = pd.DataFrame(rows)

    # Priority list — only stable growth restaurants
    priority_rows = []
    tiers = [
        "🔴 Activate — proven marketing response",
        "🟠 Activate — untapped, no prior spend",
        "🟡 Activate — review channel strategy",
    ]
    channels = ["FB", "KOL", "CRM"]
    for i in range(40):
        name = names[i]
        has_mkt = rng.random() > 0.35
        priority_rows.append({
            "rank"                   : i + 1,
            "restaurant_id"          : 1000 + i,
            "name"                   : name,
            "priority_score"         : round(100 - i * 2.3 + rng.uniform(-3, 3), 1),
            "priority_tier"          : rng.choice(tiers, p=[0.45, 0.35, 0.20]),
            "recommended_channel"    : rng.choice(channels),
            "latest_segment"         : segments[i],
            "growth_months"          : rng.integers(2, 4),
            "is_stable_growth"       : True,
            "score_perf"             : rng.uniform(0.3, 1.0),
            "score_growth"           : rng.uniform(0.3, 1.0),
            "monthly_bookings"       : rng.integers(10, 150),
            "monthly_revenue"        : rng.uniform(5000, 200000),
            "booking_growth_rolling" : rng.uniform(0.05, 0.80),
            "revenue_growth_rolling" : rng.uniform(0.05, 0.80),
            "booking_growth_yoy"     : rng.uniform(0.03, 0.60),
            "growth_signal_used"     : rng.choice(["YoY", "MoM"], p=[0.65, 0.35]),
            "delta_growth_book"      : rng.uniform(-0.1, 0.3),
            "has_marketing"          : has_mkt,
            "n_campaigns"            : rng.integers(1, 8) if has_mkt else 0,
            "avg_lift_per_day"       : rng.uniform(0.1, 3.5) if has_mkt else None,
            "total_incremental_rev"  : rng.uniform(1000, 50000) if has_mkt else None,
            "avg_roi"                : rng.uniform(-0.1, 2.5) if has_mkt else None,
            "n_positive_lift"        : rng.integers(1, 5) if has_mkt else 0,
            "n_negative_lift"        : rng.integers(0, 2) if has_mkt else 0,
            "channels_used"          : rng.choice(["FB", "KOL", "CRM", "FB+KOL", "CRM+FB"]) if has_mkt else None,
            "best_channel"           : rng.choice(channels) if has_mkt else None,
            "pct_yoy_baseline"       : rng.uniform(0, 1) if has_mkt else None,
            "location"               : locations[i],
            "cuisine"                : cuisines[i],
        })

    priority_df = pd.DataFrame(priority_rows)
    return momentum_df, priority_df


# ── Public loaders ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)  # re-read every 5 minutes
def load_momentum() -> pd.DataFrame:
    if MOMENTUM_PATH.exists():
        df = pd.read_parquet(MOMENTUM_PATH)
        df["year_month"] = pd.to_datetime(df["year_month"])
        return df
    st.toast("⚠️ Using sample data — run momentum_seasonality.ipynb to load real data", icon="⚠️")
    momentum_df, _ = _make_sample_data()
    return momentum_df


@st.cache_data(ttl=300)
def load_priority() -> pd.DataFrame:
    if PRIORITY_PATH.exists():
        df = pd.read_csv(PRIORITY_PATH)
        return df
    _, priority_df = _make_sample_data()
    return priority_df


@st.cache_data(ttl=300)
def load_marketing() -> pd.DataFrame:
    if MARKETING_PATH.exists():
        return pd.read_csv(MARKETING_PATH)
    return pd.DataFrame()


def get_restaurant_history(momentum_df: pd.DataFrame, name: str) -> pd.DataFrame:
    return (
        momentum_df[momentum_df["name"] == name]
        .sort_values("year_month")
        .copy()
    )


def get_restaurant_priority_row(priority_df: pd.DataFrame, name: str) -> dict:
    rows = priority_df[priority_df["name"] == name]
    if len(rows):
        return rows.iloc[0].to_dict()
    return {}


def is_demo_mode() -> bool:
    return not PRIORITY_PATH.exists()
