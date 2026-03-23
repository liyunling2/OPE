# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data.loader import load_priority, load_momentum, load_momentum_segments


def layout(height=300, **kwargs):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaf0", family="DM Sans"), # Switched to dark text
        margin=dict(l=0, r=0, t=30, b=0),
        height=height,
    )
    base.update(kwargs)
    return base

AXIS = dict(gridcolor="#2e3350", showline=False, zeroline=False) # Switched to light grid

# Priority scoring weights from priority_scoring_seasonality.ipynb
W_SCORE_GROWTH = 0.60
W_DELTA_GROWTH = 0.40

def fmt_pct(val):
    if val is None or (isinstance(val, float) and pd.isna(val)): return "-"
    return "%.1f%%" % (val * 100)

def get_tier(tier):
    t = str(tier).lower()
    if "proven"   in t: return "#e74c3c", "Proven"
    if "untapped" in t: return "#e67e22", "Untapped"
    if "review"   in t: return "#f1c40f", "Review"
    return "#7c82a0", str(tier)


def min_max_norm(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    rng = s.max() - s.min()
    if pd.isna(rng) or rng == 0:
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / rng


def add_priority_breakdown_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    growth_series = out["score_growth"] if "score_growth" in out.columns else pd.Series(0, index=out.index)
    out["score_growth_norm"] = min_max_norm(growth_series.fillna(0))
    if "delta_growth_book" in out.columns:
        out["delta_growth_norm"] = min_max_norm(out["delta_growth_book"])
    else:
        out["delta_growth_norm"] = 0.0

    out["growth_component"] = (
        out["score_growth_norm"] * W_SCORE_GROWTH +
        out["delta_growth_norm"] * W_DELTA_GROWTH
    )
    out["priority_raw_recomputed"] = out["growth_component"]
    out["priority_score_recomputed"] = min_max_norm(out["priority_raw_recomputed"]) * 100
    return out


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
            "gmv_growth_rolling",
            "booking_growth_yoy",
            "gmv_growth_yoy",
            "growth_signal_used",
            "delta_growth_book",
            "delta_growth_rev",
            "months_of_history",
            "has_full_year",
        ]
        if c in latest.columns
    ]
    base = latest[base_cols].copy()

    existing = priority_df.copy()
    existing["is_in_priority_list"] = True
    base["is_in_priority_list"] = False

    combined = base.merge(existing, on="name", how="left", suffixes=("_base", ""))
    coalesce_cols = [
        "restaurant_id",
        "latest_segment",
        "score_perf",
        "score_growth",
        "monthly_bookings",
        "monthly_gmv",
        "booking_growth_rolling",
        "gmv_growth_rolling",
        "booking_growth_yoy",
        "gmv_growth_yoy",
        "growth_signal_used",
        "delta_growth_book",
        "delta_growth_rev",
        "has_full_year",
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
        combined["is_in_priority_list"].isna(),
        False,
        combined["is_in_priority_list"],
    ).astype(bool)
    combined["priority_tier"] = combined["priority_tier"].fillna("Monitor - outside stable-growth priority universe")
    combined["recommended_channel"] = combined["recommended_channel"].where(
        combined["recommended_channel"].notna(),
        pd.NA,
    )
    combined["priority_score"] = pd.to_numeric(combined.get("priority_score"), errors="coerce").fillna(0.0)
    if "has_marketing" in combined.columns:
        combined["has_marketing"] = np.where(
            combined["has_marketing"].isna(),
            False,
            combined["has_marketing"],
        ).astype(bool)
    else:
        combined["has_marketing"] = False
    combined["n_campaigns"] = pd.to_numeric(combined.get("n_campaigns"), errors="coerce").fillna(0).astype(int)
    return combined

def render():
    base_priority_df = load_priority()
    priority_df = build_priority_universe(base_priority_df)
    priority_df = add_priority_breakdown_cols(priority_df)
    st.markdown("## Priority List")
    st.markdown("<p style='color:#7c82a0;margin-top:-0.5rem;'>Stable-growth restaurants ranked by composite priority score.</p>", unsafe_allow_html=True)
    st.markdown("---")

    if len(priority_df) == 0:
        st.warning("No priority data. Run priority_scoring_seasonality.ipynb first.")
        return

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("In Priority List", str(int(priority_df["is_in_priority_list"].sum())))
    def ct(kw): return sum(kw in str(t).lower() for t in priority_df["priority_tier"])
    k2.metric("Proven",   str(ct("proven")))
    k3.metric("Untapped", str(ct("untapped")))
    k4.metric("Review",   str(ct("review")))
    seasonal_n = int(priority_df["is_seasonal"].fillna(False).sum()) if "is_seasonal" in priority_df.columns else 0
    k5.metric("\U0001F30A Seasonal", str(seasonal_n))
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        f"Total restaurants shown: {len(priority_df):,}. "
        "Restaurants outside the stable-growth priority universe are included as 'Monitor'."
    )

    with st.expander("How Priority Score Is Calculated", expanded=False):
        st.markdown(
            "**Formula (from `priority_scoring_seasonality.ipynb`)**\n\n"
            "- `growth_component = 0.60 * score_growth_norm + 0.40 * delta_growth_norm`\n"
            "- `priority_raw = growth_component`\n"
            "- Final score: `priority_score = min_max_norm(priority_raw) * 100`"
        )

        breakdown_names = priority_df.sort_values("priority_score", ascending=False)["name"].tolist()
        breakdown_name = st.selectbox("Restaurant breakdown", breakdown_names, key="priority_breakdown_name")
        brow = priority_df[priority_df["name"] == breakdown_name].iloc[0]

        b1, b2, b3 = st.columns(3)
        b1.metric("Priority Score", f"{brow.get('priority_score', 0):.2f}")
        b2.metric("Growth Component", f"{brow.get('growth_component', 0):.4f}")
        b3.metric("Priority Raw", f"{brow.get('priority_raw_recomputed', 0):.4f}")

        breakdown_df = pd.DataFrame(
            [
                ("score_growth", brow.get("score_growth", np.nan)),
                ("delta_growth_book", brow.get("delta_growth_book", np.nan)),
                ("score_growth_norm", brow.get("score_growth_norm", np.nan)),
                ("delta_growth_norm", brow.get("delta_growth_norm", np.nan)),
                ("growth_component", brow.get("growth_component", np.nan)),
                ("priority_raw_recomputed", brow.get("priority_raw_recomputed", np.nan)),
                ("priority_score_recomputed", brow.get("priority_score_recomputed", np.nan)),
                ("priority_score_saved", brow.get("priority_score", np.nan)),
            ],
            columns=["Figure Used In Calculation", "Value"],
        )
        st.dataframe(breakdown_df, width="stretch", hide_index=True, height=430)

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
        st.markdown("### MoM vs YoY Score - Priority Universe")
        has_mom_sc = "score_growth_mom" in priority_df.columns
        has_yoy_sc = "score_growth_yoy" in priority_df.columns
        if has_mom_sc and has_yoy_sc:
            pf = priority_df[priority_df["is_in_priority_list"]].copy() if "is_in_priority_list" in priority_df.columns else priority_df.copy()
            is_seas = pf.get("is_seasonal", pd.Series(False, index=pf.index)).fillna(False)
            fig_mv = go.Figure()
            not_seas = pf[~is_seas]
            if len(not_seas):
                fig_mv.add_trace(go.Scatter(
                    x=not_seas["score_growth_mom"], y=not_seas["score_growth_yoy"],
                    mode="markers", name="Non-seasonal",
                    marker=dict(color="#2ecc71", size=8, opacity=0.7, line=dict(color="#0f1117", width=1)),
                    hovertemplate="<b>%{customdata}</b><br>MoM: %{x:.2f} | YoY: %{y:.2f}<extra></extra>",
                    customdata=not_seas["name"].tolist()))
            seas_sub = pf[is_seas]
            if len(seas_sub):
                fig_mv.add_trace(go.Scatter(
                    x=seas_sub["score_growth_mom"], y=seas_sub["score_growth_yoy"],
                    mode="markers", name="\U0001F30A Seasonal",
                    marker=dict(color="#f0a500", size=10, symbol="diamond", opacity=0.85, line=dict(color="#0f1117", width=1)),
                    hovertemplate="<b>%{customdata}</b><br>MoM: %{x:.2f} | YoY: %{y:.2f}<extra></extra>",
                    customdata=seas_sub["name"].tolist()))
            if "score_growth_mom" in pf.columns and "score_growth_yoy" in pf.columns:
                fig_mv.add_vline(x=pf["score_growth_mom"].median(), line_dash="dash", line_color="#3b82f6", line_width=1)
                fig_mv.add_hline(y=pf["score_growth_yoy"].dropna().median(), line_dash="dash", line_color="#f0a500", line_width=1)
            fig_mv.update_layout(**layout(280, showlegend=True,
                xaxis=dict(**AXIS, title="MoM Score (0-1)"),
                yaxis=dict(**AXIS, title="YoY Score (0-1)"),
                legend=dict(orientation="h", y=1.05, x=0, font_size=10)))
            st.plotly_chart(fig_mv, width="stretch")
            st.caption("Top-right = strong on both signals. Bottom-right = strong MoM, weak YoY = \U0001F30A Seasonal. Dashed lines = portfolio medians.")
        else:
            # Fallback to channel mix if scores not in data
            channel_series = priority_df["recommended_channel"].dropna().astype(str).str.strip()
            channel_series = channel_series[~channel_series.str.lower().isin(["", "-", "unknown", "n/a", "nan"])]
            ch_counts = channel_series.value_counts()
            cmap = {"FB":"#3b82f6","KOL":"#9b59b6","CRM":"#2ecc71"}
            if len(ch_counts):
                fig_ch = go.Figure(go.Bar(
                    x=ch_counts.index, y=ch_counts.values,
                    marker_color=[cmap.get(str(c),"#7c82a0") for c in ch_counts.index],
                    text=ch_counts.values, textposition="outside", textfont=dict(color="#e8eaf0")))
                fig_ch.update_layout(**layout(280, showlegend=False,
                    xaxis=dict(**AXIS, title="Channel"), yaxis=dict(**AXIS, title="Restaurants")))
                st.plotly_chart(fig_ch, width="stretch")
            else:
                st.info("Re-run notebooks to see MoM vs YoY score distribution.")

    st.markdown("---")

    f1, f2, f3, f4, f5, f6 = st.columns([1, 1, 1, 1, 1, 1.4])
    with f1:
        t_opts = ["All"] + priority_df["priority_tier"].dropna().unique().tolist()
        t_filt = st.selectbox("Tier", t_opts)
    with f2:
        has_seg = "latest_segment" in priority_df.columns
        s_opts  = ["All"] + (priority_df["latest_segment"].dropna().unique().tolist() if has_seg else [])
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
        df = df[df["is_seasonal"] == True]
    if seas_filt == "Non-seasonal only" and "is_seasonal" in df.columns:
        df = df[df["is_seasonal"] == False]
    if str(name_search).strip():
        df = df[df["name"].str.contains(name_search.strip(), case=False, na=False)]
    df = df[df["priority_score"] >= min_sc].sort_values("priority_score", ascending=False).reset_index(drop=True)

    st.caption("Showing %d of %d" % (len(df), len(priority_df)))
    if len(df) == 0:
        st.info("No restaurants match filters.")
        return

    top_n = min(25, len(df))
    tc    = df.head(top_n)
    # Seasonal restaurants get amber bars; others use tier colour
    _bar_colors = []
    for _, _row in tc.iterrows():
        if "is_seasonal" in _row and bool(_row.get("is_seasonal", False)):
            _bar_colors.append("#f0a500")
        else:
            _bar_colors.append(get_tier(_row.get("priority_tier",""))[0])
    fig_rank = go.Figure(go.Bar(
        x=tc["priority_score"], y=tc["name"], orientation="h",
        marker_color=_bar_colors, marker_opacity=0.85,
        text=["%.0f" % s for s in tc["priority_score"]], textposition="inside",
        textfont=dict(color="#fff", size=10),
    ))
    st.caption("\U0001F7E1 Amber bars = Seasonal flag - strong MoM but weak YoY. Timing-sensitive activation.")
    fig_rank.update_layout(**layout(max(300, top_n*24),
        xaxis=dict(**AXIS, title="Priority Score", range=[0,105]),
        yaxis=dict(**AXIS, autorange="reversed", tickfont=dict(size=10))))
    st.markdown("### Top %d Restaurants" % top_n)
    st.plotly_chart(fig_rank, width="stretch")

    st.markdown("---")
    st.markdown("### Full Ranked List")

    for idx, row in df.iterrows():
        rank    = idx + 1
        name    = row["name"]
        score   = row["priority_score"]
        color, label = get_tier(row.get("priority_tier","-"))
        channel = row.get("recommended_channel", pd.NA)
        if pd.isna(channel) or str(channel).strip() in {"", "-", "Unknown", "unknown", "N/A", "n/a", "nan"}:
            channel = "No channel assigned"
        bookings = int(row.get("monthly_bookings", 0))
        growth  = row.get("booking_growth_rolling", None)
        n_camp  = int(row.get("n_campaigns", 0)) if pd.notna(row.get("n_campaigns")) else 0
        lift    = row.get("avg_lift_per_day", None)
        signal  = row.get("growth_signal_used", "-")
        gc      = "#2ecc71" if growth is not None and pd.notna(growth) and growth > 0 else "#e74c3c"

        st.markdown(
                "<div style='background:#1e2130;border:1px solid #2e3350;border-left:4px solid {c};border-radius:8px;padding:1.2rem;margin-bottom:1rem;box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>"
                "<div style='display:flex;justify-content:space-between;align-items:flex-start;'>"
                "<div><span style='font-size:1.2rem;font-weight:700;color:#e8eaf0;'>#{r} {n}</span> "
                "<span style='margin-left:8px;font-size:0.75rem;color:{c};border:1px solid {c};padding:2px 8px;border-radius:10px;'>{l}</span> "
                "<span style='color:#3b82f6;font-size:0.72rem;font-weight:700;padding:2px 8px;"
                "background:#f3f4f6;border-radius:6px;'>{ch}</span></div>"
                "<span style='font-size:1.5rem;color:#cc0000;font-weight:700;'>{s:.0f}"
                "<span style='font-size:0.75rem;color:#9ca3c4;'>/100</span></span></div>"
                "<div style='margin-top:0.5rem;font-size:0.79rem;color:#9ca3c4;display:flex;gap:1.5rem;'>"
                "<span>Bookings: <b style='color:#e8eaf0;'>{b}</b></span>"
                "<span>Growth: <b style='color:{gc};'>{g} ({sig})</b></span>"
                "<span>Campaigns: <b style='color:#e8eaf0;'>{nc}</b></span>"
                "<span>Lift/day: <b style='color:#e8eaf0;'>{li}</b></span>"
                "</div></div>".format(
                    c=color, r=rank, n=name, l=label, ch=channel, s=score,
                    b=bookings, g=fmt_pct(growth), sig=signal, gc=gc,
                    nc=n_camp, li=("%.2f" % lift if lift is not None and pd.notna(lift) else "-")
                ),
                unsafe_allow_html=True
            )

        # MoM / YoY detail row
        _mom = row.get("booking_growth_mom_rolling")
        _yoy = row.get("booking_growth_yoy_rolling")
        if _mom is not None and pd.notna(_mom):
            _mc1, _mc2 = st.columns(2)
            _mc1.metric("MoM Growth (3m)", fmt_pct(_mom),
                        help="Month-over-month 3m rolling average")
            if _yoy is not None and pd.notna(_yoy):
                _mc2.metric("YoY Growth (3m)", fmt_pct(_yoy),
                            help="Year-over-year 3m rolling (>=2 of 3 months valid)")
            else:
                _mc2.metric("YoY Growth", "N/A",
                            help="Insufficient prior-year data or <12 months history")
        if row.get("is_seasonal", False):
            st.warning("\U0001F30A **Seasonal pattern** - strong recent MoM but YoY below portfolio median. "
                       "Activate when timing aligns with seasonal peak for best ROI.")

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







