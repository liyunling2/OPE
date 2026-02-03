# ============================================================
# PHASE 1 — Momentum Engineering (append after main.ipynb)
# Assumes you already have: valid_bookings_df
# Required columns in valid_bookings_df:
#   - restaurant_id
#   - booking_date (datetime)
#   - id (booking id)
#   - revenue_dollars
# Optional:
#   - total_guests (we’ll create if missing)
# ============================================================

import numpy as np
import pandas as pd
import os 

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
valid_bookings_df = pd.read_parquet(BASE_DIR / "valid_bookings.parquet")


# ---------------------------
# 0) Guardrails / required cols
# ---------------------------
required_cols = ["restaurant_id", "booking_date", "id", "revenue_dollars"]
missing = [c for c in required_cols if c not in valid_bookings_df.columns]
if missing:
    raise ValueError(f"valid_bookings_df missing required columns: {missing}")

# Ensure datetime
valid_bookings_df["booking_date"] = pd.to_datetime(valid_bookings_df["booking_date"], errors="coerce")
valid_bookings_df = valid_bookings_df.dropna(subset=["booking_date", "restaurant_id"])

# Create total_guests if missing
if "total_guests" not in valid_bookings_df.columns:
    if "adult" in valid_bookings_df.columns and "kids" in valid_bookings_df.columns:
        valid_bookings_df["total_guests"] = (valid_bookings_df["adult"].fillna(0) + valid_bookings_df["kids"].fillna(0))
    else:
        valid_bookings_df["total_guests"] = np.nan  # ok if you don't have guest counts


# ============================================================
# 1) Build MONTHLY restaurants_agg (1 row = restaurant x month)
# ============================================================

# Create month key
valid_bookings_df["year_month"] = valid_bookings_df["booking_date"].dt.to_period("M").dt.to_timestamp()

# Filter out advance bookings beyond Dec 2025
cutoff_month = pd.Timestamp("2025-12-01")
valid_bookings_df = valid_bookings_df[valid_bookings_df["year_month"] <= cutoff_month].copy()

restaurants_agg = (
    valid_bookings_df
    .groupby(["restaurant_id", "year_month"], as_index=False)
    .agg(
        monthly_bookings=("id", "count"),
        monthly_revenue=("revenue_dollars", "sum"),
        avg_revenue_per_booking=("revenue_dollars", "mean"),
        avg_guests=("total_guests", "mean"),
        active_days=("booking_date", lambda x: x.dt.date.nunique()),
    )
)

# Basic cleaning
restaurants_agg["monthly_bookings"] = restaurants_agg["monthly_bookings"].fillna(0).astype(int)
restaurants_agg["monthly_revenue"] = restaurants_agg["monthly_revenue"].fillna(0.0)
restaurants_agg["avg_revenue_per_booking"] = restaurants_agg["avg_revenue_per_booking"].replace([np.inf, -np.inf], np.nan)
restaurants_agg["avg_revenue_per_booking"] = restaurants_agg["avg_revenue_per_booking"].fillna(0.0)

# Sort for time-series ops
restaurants_agg = restaurants_agg.sort_values(["restaurant_id", "year_month"]).reset_index(drop=True)


# ============================================================
# 2) Minimum history filter (stability)
# ============================================================
MIN_MONTHS = 3  # bump to 6 if you have enough history
hist = restaurants_agg.groupby("restaurant_id")["year_month"].nunique()
keep_ids = hist[hist >= 0].index # keep_ids = hist[hist >= MIN_MONTHS].index # 
restaurants_agg = restaurants_agg[restaurants_agg["restaurant_id"].isin(keep_ids)].copy()


# ============================================================
# 3) Winsorise/clamp extreme monthly values (pre-growth)
# ============================================================
def winsorise_series(s: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)

for col in ["monthly_bookings", "monthly_revenue", "avg_revenue_per_booking"]:
    restaurants_agg[col] = winsorise_series(restaurants_agg[col])


# ============================================================
# 4) Growth + rolling growth (momentum "velocity")
# ============================================================
restaurants_agg["booking_growth_pct"] = (
    restaurants_agg.groupby("restaurant_id")["monthly_bookings"]
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
)

restaurants_agg["revenue_growth_pct"] = (
    restaurants_agg.groupby("restaurant_id")["monthly_revenue"]
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
)

# Rolling mean growth (smooth noise)
ROLL = 3
restaurants_agg["booking_growth_rolling"] = (
    restaurants_agg.groupby("restaurant_id")["booking_growth_pct"]
    .rolling(ROLL, min_periods=ROLL)
    .mean()
    .reset_index(level=0, drop=True)
)

restaurants_agg["revenue_growth_rolling"] = (
    restaurants_agg.groupby("restaurant_id")["revenue_growth_pct"]
    .rolling(ROLL, min_periods=ROLL)
    .mean()
    .reset_index(level=0, drop=True)
)

# Fill remaining NaNs for scoring (keep original growth cols if you want diagnostics)
restaurants_agg["booking_growth_rolling"] = restaurants_agg["booking_growth_rolling"].fillna(0.0)
restaurants_agg["revenue_growth_rolling"] = restaurants_agg["revenue_growth_rolling"].fillna(0.0)


# ============================================================
# 5) Standardise into comparable scores (percentile ranks)
#    - Performance score: "scale"
#    - Growth score: "trajectory"
# ============================================================
def pct_rank(s: pd.Series) -> pd.Series:
    # percentile rank in [0,1]
    return s.rank(pct=True, method="average")

# Performance components
restaurants_agg["perf_bookings_rank"] = pct_rank(restaurants_agg["monthly_bookings"])
restaurants_agg["perf_spend_rank"] = pct_rank(restaurants_agg["avg_revenue_per_booking"])
restaurants_agg["performance_score"] = (restaurants_agg["perf_bookings_rank"] + restaurants_agg["perf_spend_rank"]) / 2

# Growth components (rolling)
restaurants_agg["growth_bookings_rank"] = pct_rank(restaurants_agg["booking_growth_rolling"])
restaurants_agg["growth_revenue_rank"] = pct_rank(restaurants_agg["revenue_growth_rolling"])
restaurants_agg["growth_score"] = (restaurants_agg["growth_bookings_rank"] + restaurants_agg["growth_revenue_rank"]) / 2


# ============================================================
# 6) Composite Momentum Score
# ============================================================
ALPHA = 0.5  # 0.5 = equal weight, easy to justify
restaurants_agg["momentum_score"] = ALPHA * restaurants_agg["performance_score"] + (1 - ALPHA) * restaurants_agg["growth_score"]


# ============================================================
# 7) Momentum Segmentation (4 quadrants)
#    Use median cutoffs (robust + interpretable)
# ============================================================
perf_cut = restaurants_agg["performance_score"].median()
grow_cut = restaurants_agg["growth_score"].median()

restaurants_agg["momentum_segment"] = np.select(
    [
        (restaurants_agg["performance_score"] >= perf_cut) & (restaurants_agg["growth_score"] >= grow_cut),
        (restaurants_agg["performance_score"] <  perf_cut) & (restaurants_agg["growth_score"] >= grow_cut),
        (restaurants_agg["performance_score"] >= perf_cut) & (restaurants_agg["growth_score"] <  grow_cut),
        (restaurants_agg["performance_score"] <  perf_cut) & (restaurants_agg["growth_score"] <  grow_cut),
    ],
    [
        "Rising Stars",
        "Emerging Opportunities",
        "Established Players",
        "Needs Attention",
    ],
    default="Unclassified"
)


# ============================================================
# 8) Output: Latest-month prioritised restaurant list
#    (this is what you typically plot / hand to marketing)
# ============================================================
latest_month = restaurants_agg["year_month"].max()

priority_latest = (
    restaurants_agg[restaurants_agg["year_month"] == latest_month]
    .sort_values("momentum_score", ascending=False)
    .reset_index(drop=True)
)

# Quick sanity checks
print("Latest month:", latest_month.date())
print(priority_latest["momentum_segment"].value_counts(dropna=False))
print("\nTop 10 by momentum_score:")
print(priority_latest[["restaurant_id", "momentum_score", "momentum_segment",
                      "monthly_bookings", "monthly_revenue", "avg_revenue_per_booking",
                      "booking_growth_rolling", "revenue_growth_rolling"]].head(10))

# priority_latest is your Phase 1 final ranked list (for Appendix L / prioritisation tables)

# ============================================================
# 9) Output: Latest-month with momentum labels
# ============================================================
output_path = BASE_DIR / "priority_latest_momentum_labels.csv"
priority_latest.to_csv(output_path, index=False)
print("Saved latest-month momentum labels to:", output_path)
