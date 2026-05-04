# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data.loader import load_priority, load_momentum, load_momentum_segments
from theme import AXIS, BORDER_COLOR, MUTED_TEXT, SURFACE_COLOR, TEXT_COLOR


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

def get_tier(tier):
    t = str(tier).lower()
    if "proven"   in t: return "red",    "Proven",   "#e74c3c"
    if "untapped" in t: return "orange", "Untapped", "#e67e22"
    if "review"   in t: return "yellow", "Review",   "#f1c40f"
    return "gray", str(tier), "#7c82a0"


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
        'score_growth_mom', 'score_growth_yoy'

    ]
    
    for col in coalesce_cols:
        bcol = f"{col}_base"
        if bcol in combined.columns:
            if col in combined.columns:
                combined[col] = combined[col].where(combined[col].notna(), combined[bcol])
            else:
                combined[col] = combined[bcol]
            combined = combined.drop(columns=[bcol])

    combined["is_in_priority_list"] = np.where(
        combined["is_in_priority_list"].isna(), False, combined["is_in_priority_list"]
    ).astype(bool)
    combined["priority_tier"]       = combined["priority_tier"].fillna("Monitor - outside stable-growth priority universe")
    combined["recommended_channel"] = combined["recommended_channel"].where(combined["recommended_channel"].notna(), pd.NA)
    combined["priority_score"]      = pd.to_numeric(combined.get("priority_score"), errors="coerce")
    combined["has_marketing"]       = np.where(combined["has_marketing"].isna(), False, combined["has_marketing"]).astype(bool) \
                                      if "has_marketing" in combined.columns else False
    combined["n_campaigns"]         = pd.to_numeric(combined.get("n_campaigns"), errors="coerce").fillna(0).astype(int)

    return combined


def render():
    base_priority_df = load_priority()
    priority_df = build_priority_universe(base_priority_df)
    st.markdown("## Priority List")
    st.markdown(
        f"<p style='color:{MUTED_TEXT};margin-top:-0.5rem;'>Stable-growth restaurants ranked by composite priority score.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if len(priority_df) == 0:
        st.warning("No priority data. Run priority_scoring_seasonality.ipynb first.")
        return

    is_seasonal_series = coerce_bool_series(priority_df, "is_seasonal", default=False)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("In Priority List", len(priority_df))
    def ct(kw): return sum(kw in str(t).lower() for t in priority_df["priority_tier"])
    k2.metric("Proven",   str(ct("proven")))
    k3.metric("Untapped", str(ct("untapped")))
    k4.metric("Review",   str(ct("review")))
    seasonal_n = int(is_seasonal_series.sum())
    k5.metric("\U0001F30A Seasonal", str(seasonal_n))
    k6.metric("Avg GMV / GA View", fmt_thb(priority_df["avg_gmv_per_view"].mean()))
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        f"Total restaurants shown: {len(priority_df):,}. "
        "Restaurants outside the stable-growth priority universe are included as 'Monitor'."
    )

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
            | R2 | `avg_lift_per_day > 0` AND `ga_campaign_responsiveness < 0` | Booking & GA4 signals move in opposite directions | `ga4_demand × −10%` |
            | R3 | < 3 consecutive months of GA campaign data | Too few GA4 observations for reliable responsiveness | `ga_campaign_responsiveness × −20%` |
            | R4 | avg GMV per view < 1 | Campaigns correlate with worse funnel performance | `ga_campaign_responsiveness × −10%` |

            > ⓘ Risks are **additive** — a restaurant can trigger multiple penalties simultaneously.
        """
        )
    
    with st.expander("__View Restaurant's Individual Subscores__", expanded=False):
        breakdown_names = priority_df.sort_values("priority_score", ascending=False)["name"].tolist()
        breakdown_name = st.selectbox("Restaurant breakdown", breakdown_names, key="priority_breakdown_name")
        brow = priority_df[priority_df["name"] == breakdown_name].iloc[0]
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
                    
    f1, f2, f3, f4, f5, f6 = st.columns([1, 1, 1, 1, 1, 1.4])
    with f1:
        t_opts = ["All"] + priority_df["priority_tier"].dropna().unique().tolist()
        t_filt = st.selectbox("Tier", t_opts)
    with f2:
        has_seg = "segment" in priority_df.columns
        s_opts  = ["All"] + (priority_df["segment"].dropna().unique().tolist() if has_seg else [])
        s_filt  = st.selectbox("Segment", s_opts)
    with f3:
        sig_filt = st.selectbox("Signal", ["All", "YoY", "MoM"])
    with f4:
        seas_filt = st.selectbox("Seasonal", ["All", "Seasonal only", "Non-seasonal only"])
    with f5:
        min_sc = st.slider("Min Score", 0, 100, 0)
    with f6:
        name_search = st.text_input("Search Restaurant", placeholder="Type restaurant name", key="priority_name_search")

    df = priority_df.copy()
    if t_filt != "All": df = df[df["priority_tier"] == t_filt]
    if s_filt != "All" and has_seg: df = df[df["latest_segment"] == s_filt]
    if sig_filt != "All" and "growth_signal_used" in df.columns:
        df = df[df["growth_signal_used"] == sig_filt]
    if seas_filt == "Seasonal only" and "is_seasonal" in df.columns:
        df = df[df["is_seasonal"].fillna(False) == True]
    if seas_filt == "Non-seasonal only" and "is_seasonal" in df.columns:
        df = df[df["is_seasonal"].fillna(False) == False]
    if str(name_search).strip():
        df = df[df["name"].str.contains(name_search.strip(), case=False, na=False)]
    df = df[df["priority_score"] >= min_sc].sort_values("priority_score", ascending=False).reset_index(drop=True)

    st.caption("Showing %d of %d" % (len(df), len(priority_df)))
    if len(df) == 0:
        st.info("No restaurants match filters.")
        return

    top_n = min(10, len(df))
    tc    = df.head(top_n)
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

    st.markdown("### Top %d Restaurants" % top_n)
    st.caption("🟡 Amber bars = Seasonal flag — strong MoM but weak YoY. Timing-sensitive activation.")

    fig_rank.update_layout(**layout(max(180, top_n * 36),
        xaxis=dict(**AXIS, title="Priority Score", range=[0, 105], tickfont=dict(size=13)),
        yaxis=dict(**AXIS, autorange="reversed", tickfont=dict(size=13))))

    st.plotly_chart(fig_rank, width="stretch")

    st.markdown("---")

        
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Score Distribution by Tier")
        fig_box = go.Figure()
        for kw, color, label in [("proven","#e74c3c","Proven"),("untapped","#e67e22","Untapped"),("review","#f1c40f","Review")]:
            mask = priority_df["priority_tier"].apply(lambda t: kw in str(t).lower())
            sub  = priority_df[mask]["priority_score"].dropna()
            if len(sub):
                fig_box.add_trace(go.Box(y=sub, name=label, marker_color=color, line_color=color))
        fig_box.update_layout(**layout(280, showlegend=False,
            xaxis=dict(**AXIS), yaxis=dict(**AXIS)))
        st.plotly_chart(fig_box, width="stretch")

    with col2:
        # has_mom_sc = "score_growth_mom" in priority_df.columns
        # has_yoy_sc = "score_growth_yoy" in priority_df.columns
        # if has_mom_sc and has_yoy_sc:
        st.markdown("### Score Distribution by Normalised GMV/View")
        pf = priority_df.copy()

        pf = pf.sort_values("priority_score", ascending=False).reset_index(drop=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=pf['ga_campaign_responsiveness'],
            y=pf["priority_score"],
            mode="markers",
            marker=dict(
                size=8,
                color=pf["priority_score"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Priority Score"),
                line=dict(color="rgba(255,255,255,0.3)", width=1),
            ),
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "Normalised GMV/View: %{x}<br>"
                "Priority Score: %{y:.2f}"
                "<extra></extra>"
            ),
            customdata=pf["name"].astype(str).values,
        ))

        fig.update_layout(
            xaxis_title="Normalised GMV/View",
            yaxis_title="Priority Score",
            template="plotly_dark",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)
           
            #     st.markdown("### MoM vs YoY Score")
            #     pf = priority_df[priority_df["is_in_priority_list"]].copy() if "is_in_priority_list" in priority_df.columns else priority_df.copy()
            #     is_seas = coerce_bool_series(pf, "is_seasonal", default=False)
            #     fig_mv = go.Figure()
            #     not_seas = pf[~is_seas]
            #     if len(not_seas):
            #         fig_mv.add_trace(go.Scatter(
            #             x=not_seas["score_growth_mom"], y=not_seas["score_growth_yoy"],
            #             mode="markers", name="Non-seasonal",
            #             marker=dict(color="#2ecc71", size=8, opacity=0.7, line=dict(color="#0f1117", width=1)),
            #             hovertemplate="<b>%{customdata[0]}</b><br>MoM: %{x:.2f} | YoY: %{y:.2f}<extra></extra>",
            #             customdata = not_seas["name"].astype(str).values.reshape(-1, 1)))
            #     seas_sub = pf[is_seas]
            #     if len(seas_sub):
            #         fig_mv.add_trace(go.Scatter(
            #             x=seas_sub["score_growth_mom"], y=seas_sub["score_growth_yoy"],
            #             mode="markers", name="\U0001F30A Seasonal",
            #             marker=dict(color="#f0a500", size=10, symbol="diamond", opacity=0.85, line=dict(color="#0f1117", width=1)),
            #             hovertemplate="<b>%{customdata[0]}</b><br>MoM: %{x:.2f} | YoY: %{y:.2f}<extra></extra>",
            #             customdata = seas_sub["name"].astype(str).values.reshape(-1, 1)))
            #     if "score_growth_mom" in pf.columns and "score_growth_yoy" in pf.columns:
            #         fig_mv.add_vline(x=pf["score_growth_mom"].median(), line_dash="dash", line_color="#3b82f6", line_width=1)
            #         fig_mv.add_hline(y=pf["score_growth_yoy"].dropna().median(), line_dash="dash", line_color="#f0a500", line_width=1)
            #     fig_mv.update_layout(**layout(280, showlegend=True,
            #         xaxis=dict(**AXIS, title="MoM Score (0-1)"),
            #         yaxis=dict(**AXIS, title="YoY Score (0-1)"),
            #         legend=dict(orientation="h", y=1.05, x=0, font_size=10)))
            #     st.plotly_chart(fig_mv, width="stretch")
            #     st.caption("Top-right = strong on both signals. Bottom-right = strong MoM, weak YoY =Seasonal. Dashed lines = portfolio medians.")
        # else:
        #     st.markdown("### Channel Mix")

        #     # Fallback to channel mix if scores not in data
        #     channel_series = priority_df["recommended_channel"].dropna().astype(str).str.strip()
        #     channel_series = channel_series[~channel_series.str.lower().isin(["", "-", "unknown", "n/a", "nan"])]
        #     ch_counts = channel_series.value_counts()
        #     cmap = {"FB":"#3b82f6","KOL":"#9b59b6","CRM":"#2ecc71"}
        #     if len(ch_counts):
        #         fig_ch = go.Figure(go.Bar(
        #             x=ch_counts.index, y=ch_counts.values,
        #             marker_color=[cmap.get(str(c), MUTED_TEXT) for c in ch_counts.index],
        #             text=ch_counts.values, textposition="outside", textfont=dict(color=TEXT_COLOR)))
        #         fig_ch.update_layout(**layout(280, showlegend=False,
        #             xaxis=dict(**AXIS, title="Channel"), yaxis=dict(**AXIS, title="Restaurants")))
        #         st.plotly_chart(fig_ch, width="stretch")
        #     else:
        #         st.info("Re-run notebooks to see MoM vs YoY score distribution.")


    st.markdown("---")
    st.markdown("### Full Ranked List")
    for idx, row in df.iterrows():
        rank     = idx + 1
        name     = row["name"]
        score    = row["priority_score"]
        badge_color, label, hex_color = get_tier(row.get("priority_tier", "-"))
        channel  = row.get("recommended_channel", pd.NA)
        if pd.isna(channel) or str(channel).strip() in {"", "-", "Unknown", "unknown", "N/A", "n/a", "nan"}:
            channel = "No channel assigned"
        bookings = int(row.get("monthly_bookings", 0))
        growth   = row.get("booking_growth_rolling", None)
        n_camp   = int(row.get("n_campaigns", 0)) if pd.notna(row.get("n_campaigns")) else 0
        lift     = row.get("avg_lift_per_day", None)
        signal   = row.get("growth_signal_used", "-")
        gc       = "normal" if growth is not None and pd.notna(growth) and growth > 0 else "inverse"
        growth_fmt = fmt_pct(growth) if growth is not None and pd.notna(growth) else "-"
        lift_fmt   = ("%.2f" % lift) if lift is not None and pd.notna(lift) else "-"

        with st.container(border=True):
            col_name, col_score = st.columns([5, 1])

            with col_name:
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;'>"
                    f"<span style='font-size:1.1rem;font-weight:700;'>#{rank} {name}</span>"
                    f"<span style='font-size:0.75rem;padding:2px 10px;border-radius:999px;border:1px solid currentColor;color:{hex_color};'>{label}</span>"
                    f"<span style='font-size:0.75rem;padding:2px 10px;border-radius:999px;background:#1a2744;color:#3b82f6;'>{channel}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col_score:
                st.markdown(
                    f"<div style='text-align:right;font-size:1.6rem;font-weight:700;color:#cc0000;'>"
                    f"{score:.0f}<span style='font-size:0.8rem;color:#9ca3c4;'>/100</span></div>",
                    unsafe_allow_html=True
                )

            st.caption(
                f"Bookings: **{bookings:,}**  &nbsp;|&nbsp;  "
                f"Growth: **{growth_fmt} ({signal})**  &nbsp;|&nbsp;  "
                f"Campaigns: **{n_camp}**  &nbsp;|&nbsp;  "
                f"Lift/day: **{lift_fmt}**"
            )

            # MoM / YoY detail row
            _mom = row.get("booking_growth_mom_rolling")
            _yoy = row.get("booking_growth_yoy_rolling")

            if _mom is not None and pd.notna(_mom):
                mom_str = f"MoM (3m): **{fmt_pct(_mom)}**"
                yoy_str = f"YoY (3m): **{fmt_pct(_yoy)}**" if _yoy is not None and pd.notna(_yoy) else "YoY (3m): **N/A**"
                st.caption(f"{mom_str} &nbsp;|&nbsp; {yoy_str}")

            if row.get("is_seasonal", False):
                st.caption("🌊 **Seasonal pattern** — strong recent MoM but YoY below portfolio median. Activate at seasonal peak for best ROI.")
            

            can_generate = bool(row.get("is_in_priority_list", False))
            if st.button("Generate Strategy for %s" % name, key="strat_%d" % rank, disabled=not can_generate):
                st.session_state["strategy_restaurant"] = name
                st.session_state["_nav_to_strategy"] = True
                st.rerun()


    st.caption(
        "`Monitor - outside stable-growth priority universe` means the restaurant is shown for full visibility, "
        "but it did not meet the stable-growth criteria used to build the ranked priority list. "
        "These restaurants are not prioritized for immediate activation and should be monitored until momentum stabilizes."
    )

    if st.session_state.get("_nav_to_strategy"):
        st.session_state["_nav_to_strategy"] = False
        st.info("Go to Strategy Engine in sidebar for: %s" % st.session_state.get("strategy_restaurant",""))







