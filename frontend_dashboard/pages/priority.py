# -*- coding: utf-8 -*-
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data.loader import load_priority


def layout(height=300, **kwargs):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#111827", family="DM Sans"), # Switched to dark text
        margin=dict(l=0, r=0, t=30, b=0),
        height=height,
    )
    base.update(kwargs)
    return base

AXIS = dict(gridcolor="#e0e0e0", showline=False, zeroline=False) # Switched to light grid

def fmt_pct(val):
    if val is None or (isinstance(val, float) and pd.isna(val)): return "-"
    return "%.1f%%" % (val * 100)

def get_tier(tier):
    t = str(tier).lower()
    if "proven"   in t: return "#e74c3c", "Proven"
    if "untapped" in t: return "#e67e22", "Untapped"
    if "review"   in t: return "#f1c40f", "Review"
    return "#7c82a0", str(tier)

def render():
    priority_df = load_priority()
    st.markdown("## Priority List")
    st.markdown("<p style='color:#7c82a0;margin-top:-0.5rem;'>Stable-growth restaurants ranked by composite priority score.</p>", unsafe_allow_html=True)
    st.markdown("---")

    if len(priority_df) == 0:
        st.warning("No priority data. Run priority_scoring_seasonality.ipynb first.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("In Priority List", str(len(priority_df)))
    def ct(kw): return sum(kw in str(t).lower() for t in priority_df["priority_tier"])
    k2.metric("Proven",   str(ct("proven")))
    k3.metric("Untapped", str(ct("untapped")))
    k4.metric("Review",   str(ct("review")))
    st.markdown("<br>", unsafe_allow_html=True)

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
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        st.markdown("### Channel Mix")
        ch_counts = priority_df["recommended_channel"].value_counts()
        cmap = {"FB":"#3b82f6","KOL":"#9b59b6","CRM":"#2ecc71"}
        fig_ch = go.Figure(go.Bar(
            x=ch_counts.index, y=ch_counts.values,
            marker_color=[cmap.get(str(c),"#7c82a0") for c in ch_counts.index],
            text=ch_counts.values, textposition="outside", textfont=dict(color="#e8eaf0"),
        ))
        fig_ch.update_layout(**layout(280, showlegend=False,
            xaxis=dict(**AXIS, title="Channel"), yaxis=dict(**AXIS, title="Restaurants")))
        st.plotly_chart(fig_ch, use_container_width=True)

    st.markdown("---")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        t_opts = ["All"] + priority_df["priority_tier"].dropna().unique().tolist()
        t_filt = st.selectbox("Tier", t_opts)
    with f2:
        has_seg = "latest_segment" in priority_df.columns
        s_opts  = ["All"] + (priority_df["latest_segment"].dropna().unique().tolist() if has_seg else [])
        s_filt  = st.selectbox("Segment", s_opts)
    with f3:
        c_opts = ["All"] + priority_df["recommended_channel"].dropna().unique().tolist()
        c_filt = st.selectbox("Channel", c_opts)
    with f4:
        min_sc = st.slider("Min Score", 0, 100, 0)

    df = priority_df.copy()
    if t_filt != "All": df = df[df["priority_tier"] == t_filt]
    if s_filt != "All" and has_seg: df = df[df["latest_segment"] == s_filt]
    if c_filt != "All": df = df[df["recommended_channel"] == c_filt]
    df = df[df["priority_score"] >= min_sc].sort_values("priority_score", ascending=False).reset_index(drop=True)

    st.caption("Showing %d of %d" % (len(df), len(priority_df)))
    if len(df) == 0:
        st.info("No restaurants match filters.")
        return

    top_n = min(25, len(df))
    tc    = df.head(top_n)
    fig_rank = go.Figure(go.Bar(
        x=tc["priority_score"], y=tc["name"], orientation="h",
        marker_color=[get_tier(t)[0] for t in tc["priority_tier"]], marker_opacity=0.85,
        text=["%.0f" % s for s in tc["priority_score"]], textposition="inside",
        textfont=dict(color="#fff", size=10),
    ))
    fig_rank.update_layout(**layout(max(300, top_n*24),
        xaxis=dict(**AXIS, title="Priority Score", range=[0,105]),
        yaxis=dict(**AXIS, autorange="reversed", tickfont=dict(size=10))))
    st.markdown("### Top %d Restaurants" % top_n)
    st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("---")
    st.markdown("### Full Ranked List")

    for idx, row in df.iterrows():
        rank    = idx + 1
        name    = row["name"]
        score   = row["priority_score"]
        color, label = get_tier(row.get("priority_tier","-"))
        channel = row.get("recommended_channel", "-")
        bookings = int(row.get("monthly_bookings", 0))
        growth  = row.get("booking_growth_rolling", None)
        n_camp  = int(row.get("n_campaigns", 0)) if pd.notna(row.get("n_campaigns")) else 0
        lift    = row.get("avg_lift_per_day", None)
        signal  = row.get("growth_signal_used", "-")
        gc      = "#2ecc71" if growth is not None and pd.notna(growth) and growth > 0 else "#e74c3c"

        st.markdown(
                "<div style='background:#ffffff;border:1px solid #e0e0e0;border-left:4px solid {c};border-radius:8px;padding:1.2rem;margin-bottom:1rem;box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>"
                "<div style='display:flex;justify-content:space-between;align-items:flex-start;'>"
                "<div><span style='font-size:1.2rem;font-weight:700;color:#111827;'>#{r} {n}</span> "
                "<span style='margin-left:8px;font-size:0.75rem;color:{c};border:1px solid {c};padding:2px 8px;border-radius:10px;'>{l}</span> "
                "<span style='color:#3b82f6;font-size:0.72rem;font-weight:700;padding:2px 8px;"
                "background:#f3f4f6;border-radius:6px;'>{ch}</span></div>"
                "<span style='font-size:1.5rem;color:#cc0000;font-weight:700;'>{s:.0f}"
                "<span style='font-size:0.75rem;color:#6b7280;'>/100</span></span></div>"
                "<div style='margin-top:0.5rem;font-size:0.79rem;color:#6b7280;display:flex;gap:1.5rem;'>"
                "<span>Bookings: <b style='color:#111827;'>{b}</b></span>"
                "<span>Growth: <b style='color:{gc};'>{g} ({sig})</b></span>"
                "<span>Campaigns: <b style='color:#111827;'>{nc}</b></span>"
                "<span>Lift/day: <b style='color:#111827;'>{li}</b></span>"
                "</div></div>".format(
                    c=color, r=rank, n=name, l=label, ch=channel, s=score,
                    b=bookings, g=fmt_pct(growth), sig=signal, gc=gc,
                    nc=n_camp, li=("%.2f" % lift if lift is not None and pd.notna(lift) else "-")
                ),
                unsafe_allow_html=True
            )

        if st.button("Generate Strategy for %s" % name, key="strat_%d" % rank):
            st.session_state["strategy_restaurant"] = name
            st.session_state["_nav_to_strategy"] = True
            st.rerun()

    if st.session_state.get("_nav_to_strategy"):
        st.session_state["_nav_to_strategy"] = False
        st.info("Go to Strategy Engine in sidebar for: %s" % st.session_state.get("strategy_restaurant",""))
