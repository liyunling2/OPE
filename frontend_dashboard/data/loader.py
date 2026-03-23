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

MOMENTUM_PATH = BASE_DIR / "_2_feature_engineering+momentum" / "start" / "restaurants_agg_performance.parquet"
MOMENTUM_LABELS_PATH = BASE_DIR / "_2_feature_engineering+momentum" / "start" / "priority_latest_momentum_labels.parquet"
MARKETING_PATH = BASE_DIR / "_3_marketing" / "activity_performance_with_roi.csv"
PRIORITY_PATH = BASE_DIR / "_4_final_outputs" / "priority_list.csv"
MOMENTUM_VALID_BOOKINGS_PATH = BASE_DIR / "_2_feature_engineering+momentum" / "start" / "bookings_cleaned.parquet"
MOMENTUM_VALID_BOOKINGS_ENRICHED_PATH = BASE_DIR / "_2_feature_engineering+momentum" / "start" / "bookings_cleaned.parquet"
MOMENTUM_EXPORT_DIR = Path(__file__).resolve().parent / "momentum"
MOMENTUM_BOOKINGS_EXPORT_PATH = MOMENTUM_EXPORT_DIR / "restaurant_bookings_history.parquet"

CLUSTERING_DIR = BASE_DIR / "clustering"
CLUSTER_EXPORT_DIR = Path(__file__).resolve().parent / "clustering"

CLUSTER_ASSIGN_EXPORT_PATH = CLUSTER_EXPORT_DIR / "restaurant_cluster_assignments.parquet"
CLUSTER_KEYWORDS_EXPORT_PATH = CLUSTER_EXPORT_DIR / "cluster_keywords.parquet"
CLUSTER_TEXT_EXPORT_PATH = CLUSTER_EXPORT_DIR / "restaurant_text_corpus.parquet"
CLUSTER_STRATEGY_EXPORT_PATH = CLUSTER_EXPORT_DIR / "cluster_strategy_outcomes.parquet"

CLUSTER_RESULTS_PATH = CLUSTERING_DIR / "clustering_results.csv"
VIZ_DF_PATH = CLUSTERING_DIR / "viz_df.csv"
MULTI_THEME_PATH = CLUSTERING_DIR / "restaurants_with_multi_themes.csv"
THEME_DETAILS_PATH = CLUSTERING_DIR / "restaurant_theme_details.csv"
REVIEWS_PATH = CLUSTERING_DIR / "reviews.csv"

TIER_COLORS = {
    "Activate - proven marketing response": "#e74c3c",
    "Activate - untapped, no prior spend": "#e67e22",
    "Activate - review channel strategy": "#f1c40f",
}

SEGMENT_COLORS = {
    "Rising Stars": "#2ecc71",
    "Emerging Opportunities": "#3498db",
    "Established Players": "#9b59b6",
    "Needs Attention": "#e74c3c",
}


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
                    "monthly_revenue": bookings * rng.uniform(300, 1200),
                    "avg_revenue_per_booking": rng.uniform(300, 1200),
                    "avg_guests": rng.uniform(2.0, 4.5),
                    "active_days": rng.integers(10, 28),
                    "location": locations[i],
                    "cuisine": cuisines[i],
                    "booking_growth_rolling": rng.uniform(-0.3, 0.8),
                    "revenue_growth_rolling": rng.uniform(-0.3, 0.8),
                    "booking_growth_yoy": rng.uniform(-0.2, 0.6),
                    "revenue_growth_yoy": rng.uniform(-0.2, 0.6),
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
                "monthly_revenue": rng.uniform(5000, 200000),
                "booking_growth_rolling": rng.uniform(0.05, 0.80),
                "revenue_growth_rolling": rng.uniform(0.05, 0.80),
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
    if MOMENTUM_PATH.exists():
        df = pd.read_parquet(MOMENTUM_PATH)
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
    return momentum_df


@st.cache_data(ttl=300)
def load_priority() -> pd.DataFrame:
    if PRIORITY_PATH.exists():
        return pd.read_csv(PRIORITY_PATH)
    _, priority_df = _make_sample_data()
    return priority_df


@st.cache_data(ttl=300)
def load_marketing() -> pd.DataFrame:
    if MARKETING_PATH.exists():
        return pd.read_csv(MARKETING_PATH)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_momentum_raw_bookings() -> pd.DataFrame:
    if MOMENTUM_BOOKINGS_EXPORT_PATH.exists():
        bookings_df = pd.read_parquet(MOMENTUM_BOOKINGS_EXPORT_PATH)
    elif MOMENTUM_VALID_BOOKINGS_ENRICHED_PATH.exists():
        bookings_df = pd.read_parquet(MOMENTUM_VALID_BOOKINGS_ENRICHED_PATH)
    elif MOMENTUM_VALID_BOOKINGS_PATH.exists():
        bookings_df = pd.read_parquet(MOMENTUM_VALID_BOOKINGS_PATH)
    else:
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
    }
    bookings_df = bookings_df.rename(columns=rename_map)

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

    if "adults" not in bookings_df.columns:
        bookings_df["adults"] = 0
    if "kids" not in bookings_df.columns:
        bookings_df["kids"] = 0

    bookings_df["adults"] = pd.to_numeric(bookings_df["adults"], errors="coerce").fillna(0).astype(int)
    bookings_df["kids"] = pd.to_numeric(bookings_df["kids"], errors="coerce").fillna(0).astype(int)

    if "total_guests" not in bookings_df.columns:
        bookings_df["total_guests"] = bookings_df["adults"] + bookings_df["kids"]
    else:
        bookings_df["total_guests"] = pd.to_numeric(bookings_df["total_guests"], errors="coerce").fillna(
            bookings_df["adults"] + bookings_df["kids"]
        )

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
        seg_df = pd.read_parquet(MOMENTUM_LABELS_PATH)
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
    df = _read_table(CLUSTER_ASSIGN_EXPORT_PATH)

    if df.empty and CLUSTER_RESULTS_PATH.exists():
        df = pd.read_csv(CLUSTER_RESULTS_PATH)

    if df.empty and VIZ_DF_PATH.exists():
        df = pd.read_csv(VIZ_DF_PATH)

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
    m_cols = [
        c
        for c in ["name_norm", "monthly_bookings", "monthly_revenue", "score_perf", "score_growth", "growth_signal_used"]
        if c in latest_momentum.columns
    ]
    if m_cols:
        df = df.merge(latest_momentum[m_cols], on="name_norm", how="left")

    # Ensure every restaurant from the momentum universe has a cluster category.
    # Restaurants with no text/embedding assignment are explicitly categorized as "Unclustered".
    momentum_ref_cols = [
        c
        for c in [
            "restaurant_id",
            "name",
            "name_norm",
            "monthly_bookings",
            "monthly_revenue",
            "score_perf",
            "score_growth",
            "growth_signal_used",
        ]
        if c in latest_momentum.columns or c == "name_norm"
    ]
    momentum_ref = latest_momentum[momentum_ref_cols].copy()
    if "restaurant_id" in momentum_ref.columns:
        momentum_ref["restaurant_id"] = pd.to_numeric(momentum_ref["restaurant_id"], errors="coerce").astype("Int64")

    seg_ref = load_momentum_segments()[["name_norm", "latest_segment"]].drop_duplicates("name_norm")
    momentum_ref = momentum_ref.merge(seg_ref, on="name_norm", how="left")

    # if "restaurant_id" in df.columns and "restaurant_id" in momentum_ref.columns and df["restaurant_id"].notna().any():
    #     existing_ids = set(pd.to_numeric(df["restaurant_id"], errors="coerce").dropna().astype(int))
    #     missing_mask = ~pd.to_numeric(momentum_ref["restaurant_id"], errors="coerce").fillna(-1).astype(int).isin(existing_ids)
    #     missing = momentum_ref[missing_mask].copy()
    # else:
    #     missing = momentum_ref[~momentum_ref["name_norm"].isin(set(df["name_norm"]))].copy()

    # if len(missing):
    #     x_series = pd.to_numeric(df.get("x"), errors="coerce")
    #     y_series = pd.to_numeric(df.get("y"), errors="coerce")
    #     anchor_x = float(x_series.min()) - 1.6 if x_series.notna().any() else -1.6
    #     anchor_y = float(y_series.min()) - 1.6 if y_series.notna().any() else -1.6

    #     n_missing = len(missing)
    #     angles = np.linspace(0, 2 * np.pi, n_missing, endpoint=False)
    #     radius = np.linspace(0.08, 0.28, n_missing)

    #     missing["cluster_id"] = -1
    #     missing["cluster_label"] = "Unclustered - no clustering text"
    #     missing["cluster_confidence"] = 0.0
    #     missing["x"] = anchor_x + radius * np.cos(angles)
    #     missing["y"] = anchor_y + radius * np.sin(angles)

    #     if "latest_segment" not in missing.columns:
    #         missing["latest_segment"] = np.nan

    #     common_cols = sorted(set(df.columns).union(set(missing.columns)))
    #     df = df.reindex(columns=common_cols)
    #     missing = missing.reindex(columns=common_cols)
    #     df = pd.concat([df, missing], ignore_index=True)

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
            "monthly_revenue",
            "score_perf",
            "score_growth",
            "growth_signal_used",
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


@st.cache_data(ttl=300)
def load_cluster_text_corpus() -> pd.DataFrame:
    text_df = _read_table(CLUSTER_TEXT_EXPORT_PATH)

    if text_df.empty and REVIEWS_PATH.exists():
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

    keep_cols = ["name", "name_norm", "text_id", "raw_text", "clean_text", "cluster_id", "year_month"]
    return text_df[keep_cols]


@st.cache_data(ttl=300)
def load_cluster_keywords() -> pd.DataFrame:
    key_df = _read_table(CLUSTER_KEYWORDS_EXPORT_PATH)

    if not key_df.empty:
        rename_map = {"cluster": "cluster_id"}
        key_df = key_df.rename(columns=rename_map)
        if "rank" not in key_df.columns:
            key_df["rank"] = key_df.groupby("cluster_id").cumcount() + 1
        return key_df[[c for c in ["cluster_id", "keyword", "weight", "rank"] if c in key_df.columns]]

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

    if THEME_DETAILS_PATH.exists():
        theme_df = pd.read_csv(THEME_DETAILS_PATH).rename(columns={"cluster": "cluster_id", "theme": "keyword", "percentage": "weight"})
        theme_df["rank"] = theme_df.groupby("cluster_id").cumcount() + 1
        return theme_df[[c for c in ["cluster_id", "keyword", "weight", "rank"] if c in theme_df.columns]]

    return pd.DataFrame(columns=["cluster_id", "keyword", "weight", "rank"])


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
    out_df = _read_table(CLUSTER_STRATEGY_EXPORT_PATH)

    if out_df.empty:
        marketing_df = load_marketing()
        assignments = load_cluster_assignments()

        if marketing_df.empty or assignments.empty:
            return pd.DataFrame(
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

        df = marketing_df.copy()
        df["restaurant_id"] = pd.to_numeric(df.get("restaurant_id"), errors="coerce").astype("Int64")

        assignments_join = assignments[
            ["restaurant_id", "name", "cluster_id", "cluster_label", "latest_segment"]
        ].dropna(subset=["restaurant_id"])
        assignments_join = assignments_join.drop_duplicates("restaurant_id")

        merged = df.merge(assignments_join, on="restaurant_id", how="inner", suffixes=("", "_cluster"))
        if merged.empty:
            return pd.DataFrame()

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


def is_demo_mode() -> bool:
    return not PRIORITY_PATH.exists()
