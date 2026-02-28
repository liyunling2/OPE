# -*- coding: utf-8 -*-
import time
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data.loader import load_priority, load_momentum, get_restaurant_history, get_restaurant_priority_row


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


def fmt_thb(val):
    if pd.isna(val): return "-"
    if val >= 1_000_000: return "%.1fM THB" % (val/1_000_000)
    if val >= 1_000:     return "%.0fK THB" % (val/1_000)
    return "%.0f THB" % val

def fmt_pct(val):
    if val is None or (isinstance(val, float) and pd.isna(val)): return "No data"
    return "%.1f%%" % (val * 100)

def get_tier_color(tier):
    t = str(tier).lower()
    if "proven"   in t: return "#e74c3c"
    if "untapped" in t: return "#e67e22"
    if "review"   in t: return "#f1c40f"
    return "#7c82a0"

def build_prompt(row, hist):
    recent = hist.sort_values("year_month").tail(3)
    trend  = " -> ".join([
        "%s: %d bookings" % (r["year_month"].strftime("%b %Y"), int(r["monthly_bookings"]))
        for _, r in recent.iterrows()
    ]) if len(recent) else "insufficient data"

    has_mkt = row.get("has_marketing", False)
    lift    = row.get("avg_lift_per_day")
    roi     = row.get("avg_roi")
    n_camp  = row.get("n_campaigns", 0)

    if has_mkt and lift is not None and pd.notna(lift):
        lift_ctx = "Avg lift/day: %.2f | ROI: %s | Positive campaigns: %s/%s" % (
            lift, fmt_pct(roi),
            str(int(row.get("n_positive_lift",0))),
            str(int(n_camp) if pd.notna(n_camp) else "?")
        )
    else:
        lift_ctx = "No prior marketing campaigns."

    return """You are a senior restaurant marketing strategist for a Southeast Asian dining platform.
Write a specific, actionable marketing strategy brief based strictly on the data below.

RESTAURANT: {name}
Segment: {seg} | Tier: {tier} | Score: {score:.0f}/100
Growth signal: {signal} | Stable months: {gm}

PERFORMANCE
Bookings: {bk:,} | Revenue: {rev} | Growth: {gr} | YoY: {yoy}
3-month trend: {trend}

MARKETING
{lift_ctx}
Channels used: {ch} | Best channel: {bc} | Recommended: {rec}

Write the strategy in this structure (400-500 words):
## Situation Summary
## Recommended Channel: {rec}
## Campaign Objective
## Tactical Playbook (3-4 specific tactics with rationale)
## Budget Guidance
## Risks""".format(
        name=row.get("name","-"), seg=row.get("latest_segment","-"),
        tier=row.get("priority_tier","-"), score=row.get("priority_score",0),
        signal=row.get("growth_signal_used","-"), gm=int(row.get("growth_months",0)),
        bk=int(row.get("monthly_bookings",0)), rev=fmt_thb(row.get("monthly_revenue")),
        gr=fmt_pct(row.get("booking_growth_rolling")), yoy=fmt_pct(row.get("booking_growth_yoy")),
        trend=trend, lift_ctx=lift_ctx,
        ch=row.get("channels_used","-"), bc=row.get("best_channel","-"),
        rec=row.get("recommended_channel","-"),
    )

def call_claude(prompt):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"Content-Type": "application/json"},
        json={"model":"claude-sonnet-4-20250514","max_tokens":1000,
              "messages":[{"role":"user","content":prompt}]},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError("API error %d: %s" % (r.status_code, r.text))
    data = r.json()
    return "".join(b.get("text","") for b in data.get("content",[]))

def render():
    priority_df = load_priority()
    momentum_df = load_momentum()

    st.markdown("## Strategy Engine")
    st.markdown("<p style='color:#7c82a0;margin-top:-0.5rem;'>Select a restaurant to generate a tailored marketing strategy.</p>", unsafe_allow_html=True)
    st.markdown("---")

    if len(priority_df) == 0:
        st.warning("No priority data. Run priority_scoring_seasonality.ipynb first.")
        return

    preselected = st.session_state.get("strategy_restaurant", None)
    all_names   = priority_df.sort_values("priority_score", ascending=False)["name"].tolist()
    default_idx = all_names.index(preselected) if preselected and preselected in all_names else 0

    col_sel, col_info = st.columns([2,3])
    with col_sel:
        selected = st.selectbox("Restaurant", all_names, index=default_idx,
                                format_func=lambda n: "#%d  %s" % (all_names.index(n)+1, n))
        st.session_state["strategy_restaurant"] = selected

    row  = get_restaurant_priority_row(priority_df, selected)
    hist = get_restaurant_history(momentum_df, selected)

    tier_color = get_tier_color(row.get("priority_tier","-"))
    score      = row.get("priority_score", 0)

    with col_info:
            st.markdown(
                "<div style='background:#ffffff;border:1px solid #e0e0e0;"
                "border-left:4px solid {c};border-radius:8px;padding:1rem 1.4rem;box-shadow:0 1px 2px rgba(0,0,0,0.05);'>"
                "<div style='display:flex;justify-content:space-between;align-items:center;'>"
                "<div><div style='font-size:1.3rem;color:#111827;font-weight:700;'>{n}</div>"
                "<div style='font-size:0.8rem;color:#6b7280;margin-top:4px;'>{tier}</div></div>"
                "<div style='text-align:right;'><div style='font-size:1.8rem;color:#cc0000;font-weight:700;'>{s:.0f}</div>"
                "<div style='font-size:0.7rem;color:#6b7280;'>SCORE</div></div>"
                "</div></div>".format(c=tier_color, n=selected, tier=row.get("priority_tier","-"), s=score),
                unsafe_allow_html=True
            )
    

    if len(hist):
        c1,c2,c3,c4,c5 = st.columns(5)
        lat = hist.sort_values("year_month").iloc[-1]
        c1.metric("Bookings",      "%d" % int(lat.get("monthly_bookings",0)))
        c2.metric("Revenue",       fmt_thb(lat.get("monthly_revenue")))
        c3.metric("Rolling Growth",fmt_pct(lat.get("booking_growth_rolling")))
        c4.metric("YoY Growth",    fmt_pct(lat.get("booking_growth_yoy")))
        c5.metric("Campaigns",     "%d" % (int(row.get("n_campaigns",0)) if pd.notna(row.get("n_campaigns")) else 0))

    st.markdown("---")
    chart_col, strat_col = st.columns([1,1])

    with chart_col:
        st.markdown("### Booking Trend")
        if len(hist):
            hs = hist.sort_values("year_month")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hs["year_month"], y=hs["monthly_bookings"],
                marker_color=tier_color, marker_opacity=0.7,
                hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,} bookings<extra></extra>",
            ))
            if "booking_growth_yoy" in hs.columns:
                yv = hs[hs["booking_growth_yoy"].notna()]
                if len(yv):
                    fig.add_trace(go.Scatter(
                        x=yv["year_month"], y=yv["booking_growth_yoy"],
                        mode="lines", name="YoY Growth",
                        line=dict(color="#f0a500", width=2, dash="dot"),
                        yaxis="y2",
                        hovertemplate="YoY: %{y:.1%}<extra></extra>",
                    ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8eaf0", family="DM Sans"),
                margin=dict(l=0, r=0, t=30, b=0),
                height=260,
                xaxis=dict(gridcolor="#2e3350", showline=False, zeroline=False, tickangle=-30),
                yaxis=dict(gridcolor="#2e3350", showline=False, zeroline=False),
                yaxis2=dict(overlaying="y", side="right", showgrid=False,
                            tickformat=".0%", tickfont=dict(color="#f0a500", size=9)),
                legend=dict(orientation="h", y=1.02, x=0, font_size=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Campaign History")
        if row.get("has_marketing"):
            for k, v in [
                ("Channels Used",      row.get("channels_used","-")),
                ("Best Channel",       row.get("best_channel","-")),
                ("Avg Lift / Day",     "%.2f bookings" % row["avg_lift_per_day"] if pd.notna(row.get("avg_lift_per_day")) else "-"),
                ("Avg ROI",            fmt_pct(row.get("avg_roi"))),
                ("Positive Campaigns", "%d / %s" % (int(row.get("n_positive_lift",0)), str(int(row.get("n_campaigns",0))) if pd.notna(row.get("n_campaigns")) else "?")),
            ]:
                ca, cb = st.columns([2,3])
                ca.markdown("<span style='color:#7c82a0;font-size:0.8rem;'>%s</span>" % k, unsafe_allow_html=True)
                cb.markdown("<span style='color:#e8eaf0;font-size:0.85rem;font-weight:500;'>%s</span>" % v, unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#ffffff;border:1px dashed #e0e0e0;border-radius:10px;padding:3rem;text-align:center;color:#6b7280;'>Click Generate Strategy to produce a data-grounded brief.</div>", unsafe_allow_html=True)

    with strat_col:
        st.markdown("### Marketing Strategy")
        cache_key = "strategy_%s" % selected

        g_col, c_col = st.columns([2,1])
        with g_col:
            generate = st.button("Generate Strategy", key="gen_%s" % selected, use_container_width=True)
        with c_col:
            if st.button("Clear", key="clr_%s" % selected, use_container_width=True):
                st.session_state[cache_key] = None
                st.rerun()

        if generate:
            with st.spinner("Generating..."):
                try:
                    st.session_state[cache_key] = call_claude(build_prompt(row, hist))
                except Exception as e:
                    st.error("API error: %s" % e)
                    st.session_state[cache_key] = None

        if st.session_state.get(cache_key):
            txt = st.session_state[cache_key]
            st.markdown(txt)
            st.download_button(
                label="Download Strategy Brief",
                data="MARKETING STRATEGY\n%s\n\n%s" % (selected, txt),
                file_name="strategy_%s.txt" % selected.replace(" ","_"),
                mime="text/plain", use_container_width=True,
            )
        else:
            st.markdown("<div style='background:#1a1d27;border:1px dashed #2e3350;border-radius:10px;padding:3rem;text-align:center;color:#7c82a0;'>Click Generate Strategy to produce a data-grounded brief.</div>", unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("Batch Generate for Top N Restaurants"):
        batch_n = st.slider("Number", 1, min(10, len(priority_df)), 5)
        if st.button("Run Batch"):
            batch_df = priority_df.sort_values("priority_score", ascending=False).head(batch_n)
            prog = st.progress(0)
            for i, (_, brow) in enumerate(batch_df.iterrows()):
                bname = brow["name"]
                bkey  = "strategy_%s" % bname
                if not st.session_state.get(bkey):
                    try:
                        bhist = get_restaurant_history(momentum_df, bname)
                        st.session_state[bkey] = call_claude(build_prompt(brow.to_dict(), bhist))
                        time.sleep(0.5)
                    except Exception as e:
                        st.session_state[bkey] = "[Error: %s]" % e
                prog.progress((i+1)/batch_n)
            st.success("Batch complete.")
