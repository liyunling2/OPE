# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data.loader import load_cluster_assignments, load_priority, load_momentum, load_momentum_segments
from theme import BASE_LAYOUT, AXIS, CHART_THEME, GRID_COLOR, BORDER_COLOR, MUTED_TEXT, SURFACE_COLOR, TEXT_COLOR


SEG_COLOR_LIST = {
    "Rising Stars": "#2ecc71",
    "Emerging Opportunities": "#3b82f6",
    "Established Players": "#9b59b6",
    "Needs Attention": "#e74c3c",
}
def layout(height=300, **kwargs):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, family="DM Sans"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=height,
    )
    base.update(kwargs)
    return base

BOOL_TRUE_VALUES = {"true", "1", "yes", "y", "t"}
BOOL_FALSE_VALUES = {"false", "0", "no", "n", "f", ""}

def fmt_pct(val):
    if val is None or (isinstance(val, float) and pd.isna(val)): return "-"
    return "%.1f%%" % (val * 100)

def fmt_thb(val):
    if val is None or pd.isna(val): return "-"
    return f"THB {val:,.2f}" if abs(val) < 1000 else f"THB {val:,.0f}"

def normalize_name_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)

def fmt_cluster(row: pd.Series) -> str:
    cluster_id = pd.to_numeric(pd.Series([row.get("cluster_id")]), errors="coerce").iloc[0]
    cluster_label = row.get("cluster_label", None)
    has_label = pd.notna(cluster_label) and str(cluster_label).strip() not in {"", "nan", "None"}
    if pd.notna(cluster_id) and has_label:
        return f"Cluster {int(cluster_id)} - {cluster_label}"
    if pd.notna(cluster_id):
        return f"Cluster {int(cluster_id)}"
    if has_label:
        return str(cluster_label)
    return "Unclustered"

def get_tier(tier):
    t = str(tier).lower()
    if "proven"   in t: return "red",    "Proven",   "#e74c3c"
    if "untapped" in t: return "orange", "Untapped", "#e67e22"
    if "review"   in t: return "yellow", "Review",   "#f1c40f"
    return "gray", str(tier), "#7c82a0"


def filter_untapped_priority(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "priority_tier" not in df.columns:
        return df
    return df[df["priority_tier"].fillna("").astype(str).str.contains("untapped", case=False, na=False)].copy()


def sync_navbar_restaurant_from_ranked_list(row: pd.Series) -> None:
    restaurant_name = str(row.get("Restaurant", "")).strip()
    if not restaurant_name or restaurant_name.lower() in {"nan", "none"}:
        return

    if st.session_state.get("selected_restaurant") == restaurant_name:
        return

    row_segment = str(row.get("Segment", "")).strip()
    if (
        st.session_state.get("selected_segment", "All") != "All"
        and row_segment
        and row_segment != st.session_state.get("selected_segment")
    ):
        st.session_state["selected_segment"] = "All"

    row_cluster = str(row.get("Cluster", "")).strip()
    selected_cluster = st.session_state.get("selected_cluster", "All")
    if selected_cluster != "All":
        try:
            cluster_matches = row_cluster.startswith(f"Cluster {int(selected_cluster)}")
        except (TypeError, ValueError):
            cluster_matches = False
        if not cluster_matches:
            st.session_state["selected_cluster"] = "All"

    st.session_state["selected_restaurant"] = restaurant_name
    st.rerun()


def filter_priority_for_navbar(
    priority_df: pd.DataFrame,
    selected_restaurant: str,
    selected_segment: str,
    selected_cluster,
) -> pd.DataFrame:
    scoped = priority_df.copy()

    if selected_segment != "All" and "latest_segment" in scoped.columns:
        scoped = scoped[scoped["latest_segment"].astype(str).eq(str(selected_segment))].copy()

    if selected_cluster != "All" and "cluster_id" in scoped.columns:
        cluster_value = pd.to_numeric(pd.Series([selected_cluster]), errors="coerce").iloc[0]
        if pd.notna(cluster_value):
            scoped = scoped[pd.to_numeric(scoped["cluster_id"], errors="coerce").eq(int(cluster_value))].copy()

    if selected_restaurant != "All" and "name" in scoped.columns:
        scoped = scoped[scoped["name"].astype(str).eq(str(selected_restaurant))].copy()

    return scoped


def coerce_bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)

    series = df[column]
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean").fillna(default).astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(float(default)).astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {value: True for value in BOOL_TRUE_VALUES}
        | {value: False for value in BOOL_FALSE_VALUES}
    )
    return mapped.astype("boolean").fillna(default).astype(bool)


def build_priority_universe(priority_df: pd.DataFrame) -> pd.DataFrame:
    momentum_df = load_momentum()
    if momentum_df.empty:
        out = priority_df.copy()
        out["is_in_priority_list"] = True
        return out

    latest = (
        momentum_df.sort_values("year_month")
        .groupby("name", as_index=False)
        .last()
    )

    seg_df = load_momentum_segments()
    seg_cols = [c for c in ["name", "restaurant_id", "latest_segment"] if c in seg_df.columns]
    if seg_cols:
        latest = latest.merge(seg_df[seg_cols].drop_duplicates("name"), on="name", how="left", suffixes=("", "_seg"))
        if "restaurant_id_seg" in latest.columns:
            latest["restaurant_id"] = latest["restaurant_id"].where(latest["restaurant_id"].notna(), latest["restaurant_id_seg"])
            latest = latest.drop(columns=["restaurant_id_seg"])
        if "latest_segment_seg" in latest.columns:
            if "latest_segment" in latest.columns:
                latest["latest_segment"] = latest["latest_segment"].where(latest["latest_segment"].notna(), latest["latest_segment_seg"])
            else:
                latest["latest_segment"] = latest["latest_segment_seg"]
            latest = latest.drop(columns=["latest_segment_seg"])

    base_cols = [
        c
        for c in [
            "name",
            "restaurant_id",
            "latest_segment",
            "score_perf",
            "score_growth",
            "monthly_bookings",
            "monthly_gmv",
            "booking_growth_rolling",
            "booking_growth_mom_rolling",
            "booking_growth_yoy_rolling",
            "gmv_growth_rolling",
            "booking_growth_yoy",
            "gmv_growth_yoy",
            "growth_signal_used",
            "delta_growth_book",
            "delta_growth_rev",
            "gmv_per_ga_view",
            "bookings_per_ga_view",
            "ga_add_to_cart_rate",
            "ga_view_to_purchase_rate",
            "ga_purchase_to_cart_rate",
            "ga_revenue_per_view",
            "ga_items_viewed",
            "has_ga_data",
            "months_of_history",
            "has_full_year",
            "is_seasonal",
        ]
        if c in latest.columns
    ]
    base = latest[base_cols].copy()

    existing = priority_df.copy()
    existing["is_in_priority_list"] = True
    base["is_in_priority_list"] = False

    combined = existing.merge(base, on="restaurant_id", how="outer", suffixes=("_base", ""))

    coalesce_cols = [
        'restaurant_id',
        'segment', 'growth_months', 'months_observed', 'is_stable_growth', 'has_marketing',
        # Tier & final score
        'priority_score', 'priority_tier', 'recommended_channel',
        # Level-1 scores
        'growth_component', 'booking_demand', 'ga4_demand',
        'internal_mkt_campaign_responsiveness', 'campaign_responsiveness',
        # Subscores — growth
        'score_growth', 'score_growth_norm', 'delta_growth_norm',
        # Subscores — booking
        'monthly_bookings', 'monthly_gmv',
        'booking_volume_norm', 'revenue_per_booking_norm', 'booking_consistency_norm',
        # Subscores — GA4 demand
        'listing_pull_raw', 'months_cnt', 'listing_pull_norm',
        'view_to_cart_rate', 'cart_to_purchase_rate', 'funnel_depth_raw', 'funnel_depth_norm',
        # Subscores — internal marketing
        'n_campaigns', 'avg_lift_per_day', 'avg_roi',
        'n_positive_lift', 'n_negative_lift', 'channels_used', 'best_channel',
        'lift_norm', 'roi_norm',
        # Subscores — GA campaign responsiveness
        'avg_gmv_per_view', 'n_months_gmv', 'ga_campaign_responsiveness',
        # Risk flags
        'risk_thin_ga_demand_data', 'risk_inverse_attribution_rs',
        'risk_thin_gmv_data', 'risk_low_gmv_per_view',
        # prev dataset
        "is_seasonal",
        "booking_growth_rolling",
        "booking_growth_mom_rolling",
        "booking_growth_yoy_rolling",
        'score_growth_mom', 'score_growth_yoy',
    ]
    
    for col in coalesce_cols:
        bcol = f"{col}_base"
        if bcol in combined.columns:
            if col in combined.columns:
                combined[col] = combined[col].where(combined[col].notna(), combined[bcol])
            else:
                combined[col] = combined[bcol]
            combined = combined.drop(columns=[bcol])

        if "is_in_priority_list_base" in combined.columns:
            combined["is_in_priority_list"] = combined["is_in_priority_list_base"].fillna(False).astype(bool)
            combined = combined.drop(columns=["is_in_priority_list_base"])
        else:
            combined["is_in_priority_list"] = combined["is_in_priority_list"].fillna(False).astype(bool)

    combined["priority_tier"]       = combined["priority_tier"].fillna("Monitor - outside stable-growth priority universe")
    combined["recommended_channel"] = combined["recommended_channel"].where(combined["recommended_channel"].notna(), pd.NA)
    combined["priority_score"]      = pd.to_numeric(combined.get("priority_score"), errors="coerce")
    combined["has_marketing"]       = np.where(combined["has_marketing"].isna(), False, combined["has_marketing"]).astype(bool) \
                                      if "has_marketing" in combined.columns else False
    combined["n_campaigns"]         = pd.to_numeric(combined.get("n_campaigns"), errors="coerce").fillna(0).astype(int)

    cluster_df = load_cluster_assignments()
    cluster_cols = [c for c in ["restaurant_id", "cluster_id", "cluster_label", "latest_segment"] if c in cluster_df.columns]
    if not cluster_df.empty and "restaurant_id" in combined.columns and "restaurant_id" in cluster_cols:
        cluster_ref = (
            cluster_df[cluster_cols]
            .dropna(subset=["restaurant_id"])
            .drop_duplicates("restaurant_id")
        )
        combined = combined.merge(cluster_ref, on="restaurant_id", how="left", suffixes=("", "_cluster"))

        for col in ["cluster_id", "cluster_label", "latest_segment"]:
            cluster_col = f"{col}_cluster"
            if cluster_col in combined.columns:
                if col in combined.columns:
                    combined[col] = combined[col].where(combined[col].notna(), combined[cluster_col])
                else:
                    combined[col] = combined[cluster_col]
                combined = combined.drop(columns=[cluster_col])

    if not cluster_df.empty and "name" in combined.columns and "name" in cluster_df.columns:
        name_ref_cols = [c for c in ["name", "cluster_id", "cluster_label", "latest_segment"] if c in cluster_df.columns]
        name_ref = cluster_df[name_ref_cols].copy()
        name_ref["name_norm_join"] = normalize_name_series(name_ref["name"])
        name_ref = (
            name_ref.drop(columns=["name"])
            .drop_duplicates("name_norm_join")
            .rename(
                columns={
                    "cluster_id": "cluster_id_name_match",
                    "cluster_label": "cluster_label_name_match",
                    "latest_segment": "latest_segment_name_match",
                }
            )
        )
        combined["name_norm_join"] = normalize_name_series(combined["name"])
        combined = combined.merge(name_ref, on="name_norm_join", how="left")

        for col in ["cluster_id", "cluster_label", "latest_segment"]:
            name_match_col = f"{col}_name_match"
            if name_match_col in combined.columns:
                if col in combined.columns:
                    combined[col] = combined[col].where(combined[col].notna(), combined[name_match_col])
                else:
                    combined[col] = combined[name_match_col]
                combined = combined.drop(columns=[name_match_col])
        combined = combined.drop(columns=["name_norm_join"])

    if "latest_segment" in combined.columns and "segment" in combined.columns:
        combined["latest_segment"] = combined["latest_segment"].where(combined["latest_segment"].notna(), combined["segment"])
        combined["segment"] = combined["segment"].where(combined["segment"].notna(), combined["latest_segment"])
    elif "segment" in combined.columns:
        combined["latest_segment"] = combined["segment"]
    elif "latest_segment" in combined.columns:
        combined["segment"] = combined["latest_segment"]

    return combined


def render():
    if "priority_untapped_only" not in st.session_state:
        st.session_state["priority_untapped_only"] = False

    selected_restaurant = st.session_state["selected_restaurant"]
    selected_segment = st.session_state["selected_segment"]
    selected_cluster = st.session_state.get("selected_cluster", "All")
    base_priority_df = load_priority()
    priority_df = build_priority_universe(base_priority_df)
    ranked_priority_df = filter_priority_for_navbar(
        priority_df,
        selected_restaurant,
        selected_segment,
        selected_cluster,
    )
    
    st.markdown("## Prioritised Ranked List")
    st.markdown('List is derived by ranking highest GMV/GA View, highest priority score and lowest number of marketing efforts')
    quick_cols = st.columns([1, 1, 4])
    with quick_cols[0]:
        if st.button("Untapped only", width="stretch", type="primary" if st.session_state["priority_untapped_only"] else "secondary"):
            st.session_state["priority_untapped_only"] = True
    with quick_cols[1]:
        if st.button("Show all tiers", width="stretch", disabled=not st.session_state["priority_untapped_only"]):
            st.session_state["priority_untapped_only"] = False
    if st.session_state["priority_untapped_only"]:
        ranked_priority_df = filter_untapped_priority(ranked_priority_df)
        st.caption("Quick filter active: showing Untapped restaurants only.")

    cols = []
    for idx, row in ranked_priority_df.iterrows():
        rank     = idx + 1
        name     = row["name"]
        score    = row["priority_score"]
        badge_color, label, hex_color = get_tier(row.get("priority_tier", "-"))
        channel  = row.get("recommended_channel", pd.NA)
        if pd.isna(channel) or str(channel).strip() in {"", "-", "Unknown", "unknown", "N/A", "n/a", "nan"}:
            channel = "No channel assigned"
        growth   = row.get("booking_growth_rolling", None)
        n_camp   = int(row.get("n_campaigns", 0)) if pd.notna(row.get("n_campaigns")) else 0
        lift     = row.get("avg_lift_per_day", None)
        gmv = row["avg_gmv_per_view"]
        segment = row.get("latest_segment", row.get("segment", "Unknown"))
        if pd.isna(segment) or str(segment).strip() in {"", "nan", "None"}:
            segment = "Unknown"
        
        
        cols.append({
                "Restaurant": name,
                "Segment": segment,
                "Cluster": fmt_cluster(row),
                "Priority Score": f"{score:.0f}",
                "GMV/GA View": f"{gmv:.0f}",
                "No. of Campaigns": n_camp,
                "Tier": label,
            })
   
    display_df = pd.DataFrame(cols)
    if display_df.empty:
        st.info("No restaurants match the current navbar filters.")
        st.markdown("---")
        st.markdown("## Priority Scoring")
    else:
        display_df["Priority Score"] = pd.to_numeric(display_df["Priority Score"], errors="coerce")
        display_df["GMV/GA View"] = pd.to_numeric(display_df["GMV/GA View"], errors="coerce")
        display_df["No. of Campaigns"] = pd.to_numeric(display_df["No. of Campaigns"], errors="coerce")
        sorted_display_df = display_df.sort_values(
            by=['GMV/GA View','Priority Score','No. of Campaigns'],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        ranked_list_event = st.dataframe(
            sorted_display_df,
            width="stretch",
            height=300,
            hide_index=True,
            key="priority_ranked_list_table",
            on_select="rerun",
            selection_mode="single-row",
        )
        selection = getattr(ranked_list_event, "selection", None)
        if selection is None and isinstance(ranked_list_event, dict):
            selection = ranked_list_event.get("selection", {})
        selected_rows = getattr(selection, "rows", None)
        if selected_rows is None and isinstance(selection, dict):
            selected_rows = selection.get("rows", [])
        selected_rows = selected_rows or []
        if selected_rows:
            sync_navbar_restaurant_from_ranked_list(sorted_display_df.iloc[selected_rows[0]])

        st.markdown("---")

        st.markdown("## Priority Scoring")

    with st.expander("__How Priority Score Is Calculated__", expanded=False):
        st.markdown("""
        ### Priority Score Calculation

        ##### 1. Segment Stability Filter
        Restaurants are narrowed down to those who are **consistently in a growth segment** over the last 3 months. All others are deprioritised before scoring begins.

        ##### 2. Priority Score
        Overview of score composition
        | Component | With Mkt History | No Mkt History | No GA records & Mkt History |
        |---|---|---|---|
        | `growth_component` | 0.30 | 0.40 |  0.50 |
        | `booking_demand` | 0.30 | 0.40 | 0.50 |
        | `ga4_demand` | 0.20 | 0.15 | 0 |
        | `campaign_responsiveness` | 0.20 | 0.05 | 0 |
        
        > ⓘ For restaurants with no marketing history, campaign_responsiveness consists of gmv per ga view metric only
                    
        """)

        with st.expander("__Sub score Calculations__", expanded=False):
            st.markdown("""
            **Growth Component**\n
            `growth_component = (score_growth_norm × 0.60) + (delta_growth_norm × 0.40)`

            **Booking Demand**\n
            `
            booking_demand = (booking_volume_norm × 0.40) + (revenue_per_booking_norm × 0.30) + (booking_consistency_norm  × 0.30)
            `

            **GA4 Demand**\n
            `ga4_demand = (listing_pull_norm × 0.50) + (funnel_depth_norm × 0.50)`\n
            - `listing_pull = avg(itemsViewed) across all months`
            - `funnel_depth = view_to_cart_rate × 0.40 + cart_to_purchase_rate × 0.60`\n
                └─ `view_to_cart_rate = itemsAddedToCart / itemsViewed`\n
                └─ `cart_to_purchase_rate = itemsPurchased   / itemsAddedToCart`

            **Campaign Responsiveness**\n
            `
            campaign_responsiveness = (internal_mkt_campaign_responsiveness × 0.60) + (ga_campaign_responsiveness × 0.40)
            `
            - `internal_mkt_campaign_responsiveness = lift_norm × 0.60 + roi_norm × 0.40`
            - `ga_campaign_responsiveness = norm(avg_gmv_per_view)`
        """)

        st.markdown("""
             ##### 3. Risks Adjustments (Dynamic Weight Reduction)
            After scoring, weights are reduced if data quality or signal reliability is poor.

            | # | Condition | Interpretation | Penalty |
            |---|---|---|---|
            | R1 | < 3 consecutive months of GA4 data | High campaign reliance; organic demand unestablished | `ga4_demand × −20%` |
            | R2 | `booking_demand > median` AND `ga4_demand < median` | Booking & GA4 signals move in opposite directions | `ga4_demand × −10%` |
            | R3 | < 3 consecutive months of GA campaign data | Too few GA4 observations for reliable responsiveness | `ga_campaign_responsiveness × −20%` |
            | R4 | avg GMV per view < 1 | Campaigns correlate with worse funnel performance | `ga_campaign_responsiveness × −10%` |

            > ⓘ Risks are **additive** — a restaurant can trigger multiple penalties simultaneously.
        """
        )
    
    if selected_restaurant != "All":
        selected_breakdown = priority_df[priority_df["name"] == selected_restaurant]
        with st.expander(f"__View {selected_restaurant}'s Individual Subscores__", expanded=False):
            if selected_breakdown.empty:
                st.info("No priority subscore row is available for the selected restaurant.")
            else:
                brow = selected_breakdown.iloc[0]
                st.metric("Priority Score", f"{brow.get('priority_score', 0):.2f}")

                ### subscores display
                b1, b2, b3 = st.columns(3)
                b4, b5, b6 = st.columns(3)
                b1.metric("Growth Component", f"{brow.get('growth_component', 0):.4f}")
                b2.metric("Booking Demand", f"{brow.get('booking_demand', 0):.4f}")
                b3.metric("GA4 Demand", f"{brow.get('ga4_demand', 0):.4f}")
                b4.metric("Campaign Responsiveness", f"{brow.get('campaign_responsiveness', 0):.4f}")
                b5.metric("Internal Campaign Responsiveness", f"{brow.get('internal_mkt_campaign_responsiveness', 0):.4f}")
                b6.metric("GA Campaign Responsiveness", f"{brow.get('avg_gmv_per_view', 0):.4f}")

                breakdown_df = pd.DataFrame(
                    [
                        ("score_growth_norm", brow.get("score_growth_norm", np.nan)),
                        ("delta_growth_norm", brow.get("delta_growth_norm", np.nan)),
                        ("booking_volume_norm", brow.get("booking_volume_norm", np.nan)),
                        ("revenue_per_booking_norm", brow.get("revenue_per_booking_norm", np.nan)),
                        ("booking_consistency_norm", brow.get("booking_consistency_norm", np.nan)),
                        ("listing_pull_norm", brow.get("listing_pull_norm", np.nan)),
                        ("funnel_depth_norm", brow.get("funnel_depth_norm", np.nan)),
                        ("view_to_cart_rate", brow.get("view_to_cart_rate", np.nan)),
                        ("cart_to_purchase_rate", brow.get("cart_to_purchase_rate", np.nan)),
                    ],
                    columns=["Figure Used In Calculation", "Value"],
                )

                c1,c2 = st.columns([2, 1])
                c1.dataframe(breakdown_df, width="stretch", hide_index=True)
                with c2:
                    st.markdown("#### ⚠️ Risk Flags")
                    RISKS = [
                        ("risk_thin_ga_demand_data", "Thin GA4 Demand data",    "ga4_demand −20%"),
                        ("risk_inverse_attribution_rs","Inverse signal of Bookings data & GA demand", "ga4_demand −10%"),
                        ("risk_thin_gmv_data","Thin GMV/view data",          "ga_campaign_responsiveness −20%"),
                        ("risk_low_gmv_per_view","Low GMV/view metric",       "ga_campaign_responsiveness −10%"),
                    ]

                    active_risks = [(label, desc) for col, label, desc in RISKS if brow.get(col) == True]

                    if not active_risks:
                        st.success("No risks identified in data.")
                    else:
                        for label, desc in active_risks:
                            st.markdown(
                                f"- :red-badge[{label}] :gray-badge[`{desc}`]"
                            )
                    


    df = filter_priority_for_navbar(
        priority_df,
        selected_restaurant,
        selected_segment,
        selected_cluster,
    )
    if st.session_state["priority_untapped_only"]:
        df = filter_untapped_priority(df)
    st.markdown(f"### _Top Restaurants based on Priority Scores_")
    st.caption("🟡 Amber bars = Seasonal flag — strong MoM but weak YoY. Timing-sensitive activation.")

    f1, f3, f4, f5 = st.columns([1, 1, 1, 1])
    with f1:
        t_opts = ["All"] + priority_df["priority_tier"].dropna().unique().tolist()
        t_filt = st.selectbox("Tier", t_opts)
    with f3:
        sig_filt = st.selectbox("Signal", ["All", "YoY", "MoM"])
    with f4:
        seas_filt = st.selectbox("Seasonal", ["All", "Seasonal only", "Non-seasonal only"])
    with f5:
        min_sc = st.slider("Min Score", 0, 100, 0)

    if t_filt != "All": df = df[df["priority_tier"] == t_filt]
    if sig_filt != "All" and "growth_signal_used" in df.columns:
        df = df[df["growth_signal_used"] == sig_filt]
    if seas_filt == "Seasonal only" and "is_seasonal" in df.columns:
        df = df[df["is_seasonal"].fillna(False) == True]
    if seas_filt == "Non-seasonal only" and "is_seasonal" in df.columns:
        df = df[df["is_seasonal"].fillna(False) == False]

    df = df[df["priority_score"] >= min_sc].sort_values("priority_score", ascending=False).reset_index(drop=True)

    top_n = min(10, len(df))
    tc    = df.head(top_n)
    st.caption(f"_Showing {top_n} restaurants_")

    st.caption("Showing %d of %d" % (len(df), len(priority_df)))
    if len(df) == 0:
        st.info("No restaurants match filters.")
        return
    # Seasonal restaurants get amber bars; others use tier colour
    _bar_colors = []
    for _, _row in tc.iterrows():
        if "is_seasonal" in _row and bool(_row.get("is_seasonal", False)):
            _bar_colors.append("#f0a500")
        else:
            _bar_colors.append(get_tier(_row.get("priority_tier",""))[2])

    fig_rank = go.Figure(go.Bar(
        x=tc["priority_score"], y=tc["name"], orientation="h",
        marker_color=_bar_colors, marker_opacity=0.85,
        text=["%.0f" % s for s in tc["priority_score"]], textposition="inside",
        textfont=dict(color="#fff", size=13),
    ))

    fig_rank.update_layout(**layout(max(180, top_n * 36),
        xaxis=dict(**AXIS, title="Priority Score", range=[0, 105], tickfont=dict(size=13)),
        yaxis=dict(**AXIS, autorange="reversed", tickfont=dict(size=13))))

    st.plotly_chart(fig_rank, width="stretch")


    ## show quadrant
    st.markdown("---")
    st.markdown("## Momentum Quadrants")
    momentum_df = load_momentum()
    latest_all = momentum_df.sort_values("year_month").groupby("name", as_index=False).last()
    seg_col = "latest_segment" if "latest_segment" in latest_all.columns else "segment"

    has_segments = seg_col in latest_all.columns

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("#### Segment Distribution")
        if has_segments:
            seg_counts = latest_all[seg_col].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            colors = [SEG_COLOR_LIST.get(s, MUTED_TEXT) for s in seg_counts["segment"]]
            pull_vals = [
                0.1 if selected_segment is not None and seg == selected_segment else 0
                for seg in seg_counts["segment"]
            ]
            fig_donut = go.Figure(
                go.Pie(
                    labels=seg_counts["segment"],
                    values=seg_counts["count"],
                    hole=0.55,
                    pull=pull_vals,
                    marker=dict(colors=colors, line=dict(color="#0f1117", width=3)),
                    textinfo="percent+label",
                    textfont=dict(size=11),
                    hovertemplate="<b>%{label}</b><br>%{value} restaurants (%{percent})<extra></extra>",
                )
            )
            center_text = (
                f"<b>{latest_all['name'].nunique()}</b><br>"
                "<span style='font-size:10px'>restaurants</span>"
            )
            if selected_segment is not None:
                center_text += f"<br><span style='font-size:10px;color:#cc0000'>Selected segment: {selected_segment}</span>"
            fig_donut.update_layout(
                **CHART_THEME,
                height=300,
                showlegend=False,
                annotations=[
                    dict(
                        text=center_text,
                        x=0.5,
                        y=0.5,
                        font_size=14,
                        showarrow=False,
                        font_color=TEXT_COLOR,
                    )
                ],
            )
            st.plotly_chart(fig_donut, width="stretch")
        else:
            st.info("Run notebooks to see segment distribution.")

    with col_right:
        st.markdown("#### Performance vs Growth (Strategic Matrix)")
        scatter_df = latest_all.copy()
        if "location" in scatter_df.columns:
            scatter_df["location_plot"] = scatter_df["location"].fillna("Unknown")
        else:
            scatter_df["location_plot"] = "Unknown"

        fig_scatter = go.Figure()
        if has_segments:
            for seg, color in SEG_COLOR_LIST.items():
                sub = scatter_df[scatter_df[seg_col] == seg]
                if len(sub):
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=sub["score_perf"],
                            y=sub["score_growth"],
                            mode="markers",
                            name=seg,
                            marker=dict(
                                color=color,
                                size=8,
                                opacity=0.75,
                                line=dict(color="#0f1117", width=1),
                            ),
                            hovertemplate="<b>%{customdata[0]}</b><br>Location: %{customdata[1]}<br>Perf: %{x:.2f} | Growth: %{y:.2f}<extra></extra>",
                            customdata=sub[["name", "location_plot"]].to_numpy(),
                        )
                    )
        else:
            fig_scatter.add_trace(
                go.Scatter(
                    x=scatter_df["score_perf"],
                    y=scatter_df["score_growth"],
                    mode="markers",
                    marker=dict(color="#3b82f6", size=7, opacity=0.7),
                    hovertemplate="<b>%{customdata[0]}</b><br>Location: %{customdata[1]}<br>Perf: %{x:.2f} | Growth: %{y:.2f}<extra></extra>",
                    customdata=scatter_df[["name", "location_plot"]].to_numpy(),
                )
            )

        # if not selected_row.empty:
        #     hi = selected_row.iloc[0]
        #     if pd.notna(hi.get("score_perf")) and pd.notna(hi.get("score_growth")):
        #         fig_scatter.add_trace(
        #             go.Scatter(
        #                 x=[hi["score_perf"]],
        #                 y=[hi["score_growth"]],
        #                 mode="markers+text",
        #                 name="Selected Restaurant",
        #                 marker=dict(color="#cc0000", size=16, symbol="diamond", line=dict(color="#111827", width=2)),
        #                 text=[selected_restaurant],
        #                 textposition="top center",
        #                 hovertemplate=(
        #                     f"<b>{selected_restaurant}</b><br>"
        #                     f"Location: {hi.get('location_plot', 'Unknown')}<br>"
        #                     "Perf: %{x:.2f} | Growth: %{y:.2f}<extra></extra>"
        #                 ),
        #                 showlegend=True,
        #             )
        #         )

        p75_perf = scatter_df["score_perf"].quantile(0.75)
        p75_growth = scatter_df["score_growth"].quantile(0.75)
        fig_scatter.add_vline(x=p75_perf, line_dash="dash", line_color=GRID_COLOR, line_width=1)
        fig_scatter.add_hline(y=p75_growth, line_dash="dash", line_color=GRID_COLOR, line_width=1)

        fig_scatter.update_layout(
            **BASE_LAYOUT,
            height=300,
            xaxis=dict(**CHART_THEME["xaxis"], title="Performance Score"),
            yaxis=dict(**CHART_THEME["yaxis"], title="Growth Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font_size=11),
        )
        st.plotly_chart(fig_scatter, width="stretch")
        # if not selected_row.empty:
        #     perf_val = pd.to_numeric(pd.Series([hi.get("score_perf")]), errors="coerce").iloc[0]
        #     growth_val = pd.to_numeric(pd.Series([hi.get("score_growth")]), errors="coerce").iloc[0]
        #     st.caption(
        #         f"Selected: {selected_restaurant} | Segment: {hi.get(seg_col, 'Unknown')} | "
        #         f"Location: {hi.get('location_plot', 'Unknown')} | "
        #         f"Strategic matrix position: Perf {perf_val:.2f} / Growth {growth_val:.2f}"
        #         if pd.notna(perf_val) and pd.notna(growth_val)
        #         else f"Strategic matrix position: Perf {'-' if pd.isna(perf_val) else f'{perf_val:.2f}'} / "
        #              f"Growth {'-' if pd.isna(growth_val) else f'{growth_val:.2f}'}"
        #     )
