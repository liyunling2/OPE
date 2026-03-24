# -*- coding: utf-8 -*-
import os
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data.loader import (
    load_priority,
    load_momentum,
    get_restaurant_history,
    get_restaurant_priority_row,
    recommend_strategies_for_restaurant,
)


def layout(height=300, **kwargs):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaf0", family="DM Sans"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=height,
    )
    base.update(kwargs)
    return base

AXIS = dict(gridcolor="#2e3350", showline=False, zeroline=False)


def fmt_thb(val):
    if pd.isna(val): return "-"
    if val >= 1_000_000: return "%.1fM THB" % (val / 1_000_000)
    if val >= 1_000:     return "%.0fK THB" % (val / 1_000)
    return "%.0f THB" % val

def fmt_pct(val):
    if val is None or pd.isna(val): return "No data"
    return "%.1f%%" % (val * 100)

def get_tier_color(tier):
    t = str(tier).lower()
    if "proven"   in t: return "#e74c3c"
    if "untapped" in t: return "#e67e22"
    if "review"   in t: return "#f1c40f"
    return "#7c82a0"


def clean_tier_label(tier: str) -> str:
    """Strip emoji characters and mojibake prefixes from tier label strings.
    priority_list.csv may contain emoji-prefixed tiers (e.g. '🔴 Activate...')
    which render as mojibake (ðŸ"´) when the CSV is written/read with mismatched
    encoding on Windows. We strip everything up to and including the first
    space-preceded word boundary so only the readable text remains.
    """
    import re
    s = str(tier).strip()
    # Remove any leading non-ASCII characters and surrounding whitespace
    s = re.sub(r'^[^\x00-\x7F\s]+\s*', '', s)
    # Also catch common mojibake prefixes like ðŸ"´, ðŸŸ , ðŸŸ¡
    s = re.sub(r'^[Ã°Å¸â€œÂ´\s\x80-\xff]+\s*', '', s)
    return s.strip()

def _playbook_actions(strategy_name: str) -> str:
    text = str(strategy_name).lower()
    if "reactivation" in text:
        return "Target lapsed diners with return incentives, 7/14-day reminder flow, and table-time scarcity messaging."
    if "retarget" in text:
        return "Retarget menu viewers and past engagers with a 5-7 day conversion window and audience-frequency caps."
    if "loyalty" in text or "retention" in text:
        return "Launch member-only value bundles and repeat-visit nudges tied to 30-day revisit behavior."
    if "prospecting" in text or "awareness" in text:
        return "Run new-customer acquisition campaigns by lookalike audiences with strict CAC and first-booking targets."
    if "creator" in text or "influencer" in text or "kol" in text:
        return "Deploy creator-led social proof bursts with limited-time codes and track first-time bookings by creator."
    if "promo" in text or "conversion" in text:
        return "Use time-bound promotional offers with daily pacing controls and stop-loss rules on weak cohorts."
    return "Execute controlled A/B testing with one clear CTA, strict audience split, and weekly budget reallocation."


def build_grounded_brief(row: dict, hist: pd.DataFrame, recs: pd.DataFrame) -> str:
    recent = hist.sort_values("year_month").tail(3)
    score_val = pd.to_numeric(row.get("priority_score"), errors="coerce")
    score_val = 0.0 if pd.isna(score_val) else float(score_val)
    bookings_val = pd.to_numeric(row.get("monthly_bookings"), errors="coerce")
    bookings_val = 0 if pd.isna(bookings_val) else int(bookings_val)
    trend = " -> ".join(
        [
            "%s: %d bookings" % (
                r["year_month"].strftime("%b %Y"),
                int(
                    0
                    if pd.isna(pd.to_numeric(r.get("monthly_bookings"), errors="coerce"))
                    else pd.to_numeric(r.get("monthly_bookings"), errors="coerce")
                ),
            )
            for _, r in recent.iterrows()
            if pd.notna(r.get("year_month"))
        ]
    ) if len(recent) else "insufficient data"

    lines = []
    lines.append("## Situation Summary")
    lines.append(
        "- **Restaurant:** {name}\n- **Segment:** {seg}\n- **Priority Tier:** {tier}\n- **Priority Score:** {score:.1f}/100".format(
            name=row.get("name", "-"),
            seg=row.get("latest_segment", "Unknown"),
            tier=row.get("priority_tier", "Unknown"),
            score=score_val,
        )
    )
    mom_gr = pd.to_numeric(row.get("booking_growth_mom_rolling",
                                     row.get("booking_growth_rolling")), errors="coerce")
    yoy_gr = pd.to_numeric(row.get("booking_growth_yoy_rolling",
                                     row.get("booking_growth_yoy")), errors="coerce")
    signal_used = row.get("growth_signal_used", "-")
    is_seasonal = bool(row.get("is_seasonal", False))
    lines.append(
        "- **Latest:** {bk:,} bookings, {rev}  |  MoM 3m: {mom}  |  YoY 3m: {yoy}  |  Signal: {sig}\n- **Recent trend:** {trend}".format(
            bk=bookings_val,
            rev=fmt_thb(pd.to_numeric(row.get("monthly_gmv"), errors="coerce")),
            mom=fmt_pct(mom_gr),
            yoy=fmt_pct(yoy_gr),
            sig=signal_used,
            trend=trend,
        )
    )
    if is_seasonal:
        lines.append(
            "- **Seasonal flag** — strong MoM but YoY below portfolio median. "
            "Activate at seasonal peak for best ROI."
        )

    lines.append("## Data-Driven Recommended Playbook")
    if recs.empty:
        lines.append("- No robust historical strategy signal is available yet for this restaurant's context.")
    else:
        for idx, (_, rec) in enumerate(recs.head(3).iterrows(), start=1):
            lines.append(
                "{idx}. **{name}** ({scope})\n"
                "- Expected revenue uplift: {rev_uplift}\n"
                "- Expected bookings uplift: {book_uplift}\n"
                "- Avg ROI: {roi}\n"
                "- Evidence: {acts} activities, {rests} restaurants, {conf} confidence\n"
                "- Action: {action}".format(
                    idx=idx,
                    name=rec.get("strategy_name", "-"),
                    scope=str(rec.get("recommendation_scope", "historical")).title(),
                    rev_uplift=fmt_pct(pd.to_numeric(rec.get("avg_revenue_uplift_pct"), errors="coerce")),
                    book_uplift=fmt_pct(pd.to_numeric(rec.get("avg_bookings_uplift_pct"), errors="coerce")),
                    roi="-" if pd.isna(rec.get("avg_roi")) else "%.2f" % rec.get("avg_roi"),
                    acts=int(pd.to_numeric(rec.get("activities"), errors="coerce") or 0),
                    rests=int(pd.to_numeric(rec.get("restaurants"), errors="coerce") or 0),
                    conf=rec.get("confidence_level", "Unknown"),
                    action=_playbook_actions(rec.get("strategy_name", "")),
                )
            )

    lines.append("## Execution Guardrails")
    lines.append("- Set weekly stop-loss on campaigns with negative incremental revenue after sufficient spend.")
    lines.append("- Prioritize strategies with stronger sample size and confidence; treat low-sample tactics as experiments.")
    lines.append("- Re-rank every month as new campaign outcomes are added.")
    return "\n".join(lines)


def build_prompt(row: dict, hist: pd.DataFrame, recs: pd.DataFrame, grounded_brief: str) -> str:
    top_rows = []
    for _, rec in recs.head(5).iterrows():
        top_rows.append(
            "- {name} | scope={scope} | rev_uplift={rev} | booking_uplift={book} | roi={roi} | n={n} | confidence={conf}".format(
                name=rec.get("strategy_name", "-"),
                scope=rec.get("recommendation_scope", "-"),
                rev=fmt_pct(pd.to_numeric(rec.get("avg_revenue_uplift_pct"), errors="coerce")),
                book=fmt_pct(pd.to_numeric(rec.get("avg_bookings_uplift_pct"), errors="coerce")),
                roi="-" if pd.isna(rec.get("avg_roi")) else "%.2f" % rec.get("avg_roi"),
                n=int(pd.to_numeric(rec.get("activities"), errors="coerce") or 0),
                conf=rec.get("confidence_level", "Unknown"),
            )
        )
    top_block = "\n".join(top_rows) if top_rows else "- No recommendation rows available"

    return """You are a senior restaurant marketing strategist.
Use only the provided data. Do not invent metrics.
Improve and tighten the grounded strategy brief below.

Restaurant: {name}
Segment: {seg}
Priority tier: {tier}
Priority score: {score}
Recommended channel (existing model): {channel}

Top ranked strategies:
{top}

Grounded brief draft:
{brief}

Return concise markdown with sections:
## Situation Summary
## Recommended Strategy Mix
## 30-Day Execution Plan
## KPI Targets
## Risks & Caveats""".format(
        name=row.get("name", "-"),
        seg=row.get("latest_segment", "Unknown"),
        tier=row.get("priority_tier", "Unknown"),
        score=row.get("priority_score", "-"),
        channel=row.get("recommended_channel", "-"),
        top=top_block,
        brief=grounded_brief,
    )


def call_claude(prompt: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1200,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=90,
    )
    if r.status_code != 200:
        raise RuntimeError("API error %d: %s" % (r.status_code, r.text))
    data = r.json()
    return "".join(block.get("text", "") for block in data.get("content", []) if isinstance(block, dict))


def render():
    priority_df = load_priority()
    momentum_df = load_momentum()

    st.markdown("## Strategy Engine")
    st.markdown(
        "<p style='color:#9ca3c4;margin-top:-0.5rem;'>"
        "Data-first recommendations are ranked by observed campaign outcomes, "
        "with fallback from cluster to segment to global evidence."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if len(priority_df) == 0:
        st.warning("No priority data. Run priority_scoring_seasonality.ipynb first.")
        return

    preselected = st.session_state.get("strategy_restaurant", None)
    all_names   = priority_df.sort_values("priority_score", ascending=False)["name"].tolist()
    default_idx = all_names.index(preselected) if preselected and preselected in all_names else 0

    col_sel, col_cfg, col_info = st.columns([2, 1.2, 2.8])
    with col_sel:
        selected = st.selectbox(
            "Restaurant", all_names, index=default_idx,
            format_func=lambda n: "#%d  %s" % (all_names.index(n) + 1, n),
        )
        st.session_state["strategy_restaurant"] = selected

    with col_cfg:
        min_sample_size = st.number_input("Min sample", min_value=1, max_value=20, value=3, step=1)
        top_n = st.slider("Top strategies", min_value=1, max_value=6, value=3, step=1)

    row           = get_restaurant_priority_row(priority_df, selected)
    hist          = get_restaurant_history(momentum_df, selected)
    strategy_recs = recommend_strategies_for_restaurant(
        selected, top_n=int(top_n), min_sample_size=int(min_sample_size)
    )

    tier_color = get_tier_color(row.get("priority_tier", "-"))
    score      = row.get("priority_score", 0)

    with col_info:
        st.markdown(
            "<div style='background:#1e2130;border:1px solid #2e3350;"
            "border-left:4px solid {c};border-radius:8px;padding:1rem 1.4rem;"
            "box-shadow:0 1px 2px rgba(0,0,0,0.05);'>"
            "<div style='display:flex;justify-content:space-between;align-items:center;'>"
            "<div>"
            "<div style='font-size:1.3rem;color:#e8eaf0;font-weight:700;'>{n}</div>"
            "<div style='font-size:0.8rem;color:#9ca3c4;margin-top:4px;'>{tier}</div>"
            "<div style='font-size:0.75rem;color:#9ca3c4;margin-top:2px;'>Segment: {seg}</div>"
            "</div>"
            "<div style='text-align:right;'>"
            "<div style='font-size:1.8rem;color:#cc0000;font-weight:700;'>{s:.0f}</div>"
            "<div style='font-size:0.7rem;color:#9ca3c4;'>SCORE</div>"
            "</div>"
            "</div></div>".format(
                c=tier_color,
                n=selected,
                tier=clean_tier_label(row.get("priority_tier", "-")),
                seg=row.get("latest_segment", "Unknown"),
                s=score,
            ),
            unsafe_allow_html=True,
        )

    # ── Seasonal warning — plain st.warning, no HTML, no emoji escape issues ──
    is_seasonal_flag = bool(row.get("is_seasonal", False))
    if is_seasonal_flag:
        st.warning(
            "Seasonal pattern detected — strong recent MoM but YoY is below portfolio median. "
            "Consider timing activation to align with the seasonal peak."
        )

    if len(hist):
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        lat = hist.sort_values("year_month").iloc[-1]
        c1.metric("Bookings",   "%d" % int(lat.get("monthly_bookings", 0)))
        c2.metric("Revenue",    fmt_thb(lat.get("monthly_gmv")))
        c3.metric("MoM Growth", fmt_pct(lat.get("booking_growth_mom_rolling",
                                                  lat.get("booking_growth_rolling"))))
        c4.metric("YoY Growth", fmt_pct(lat.get("booking_growth_yoy_rolling",
                                                  lat.get("booking_growth_yoy"))))
        c5.metric("Signal",     str(row.get("growth_signal_used", "-")))
        c6.metric("Campaigns",  "%d" % (int(row.get("n_campaigns", 0)) if pd.notna(row.get("n_campaigns")) else 0))
        c7.metric("Strategies", "%d ranked" % len(strategy_recs))

    st.markdown("---")
    chart_col, strat_col = st.columns([1.05, 0.95])

    with chart_col:
        st.markdown("### Booking Trend + Growth Signals")
        if len(hist):
            hs = hist.sort_values("year_month")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hs["year_month"], y=hs["monthly_bookings"],
                marker_color=tier_color, marker_opacity=0.7, name="Bookings",
                hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,} bookings<extra></extra>",
            ))
            mom_col = "booking_growth_mom_rolling" if "booking_growth_mom_rolling" in hs.columns else None
            if mom_col:
                fig.add_trace(go.Scatter(
                    x=hs["year_month"], y=hs[mom_col], mode="lines", name="MoM 3m avg",
                    line=dict(color="#3b82f6", width=2), yaxis="y2",
                    hovertemplate="MoM: %{y:.1%}<extra></extra>",
                ))
            if "booking_growth_yoy_rolling" in hs.columns:
                yv = hs[hs["booking_growth_yoy_rolling"].notna()]
                if len(yv):
                    fig.add_trace(go.Scatter(
                        x=yv["year_month"], y=yv["booking_growth_yoy_rolling"],
                        mode="lines", name="YoY 3m avg",
                        line=dict(color="#f0a500", width=2, dash="dot"), yaxis="y2",
                        hovertemplate="YoY: %{y:.1%}<extra></extra>",
                    ))
            elif "booking_growth_yoy" in hs.columns:
                yv = hs[hs["booking_growth_yoy"].notna()]
                if len(yv):
                    fig.add_trace(go.Scatter(
                        x=yv["year_month"], y=yv["booking_growth_yoy"],
                        mode="lines", name="YoY (raw)",
                        line=dict(color="#f0a500", width=2, dash="dot"), yaxis="y2",
                        hovertemplate="YoY: %{y:.1%}<extra></extra>",
                    ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8eaf0", family="DM Sans"),
                margin=dict(l=0, r=0, t=30, b=0), height=260,
                xaxis=dict(gridcolor="#2e3350", showline=False, zeroline=False,
                           tickangle=-30, color="#9ca3c4"),
                yaxis=dict(gridcolor="#2e3350", showline=False, zeroline=False, color="#9ca3c4"),
                yaxis2=dict(
                    overlaying="y", side="right", showgrid=False,
                    tickformat=".0%", tickfont=dict(color="#9ca3c4", size=9),
                    zeroline=True, zerolinecolor="#7c82a0",
                ),
                legend=dict(orientation="h", y=1.02, x=0, font_size=10, bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, width="stretch")

        st.markdown("### Campaign History")
        if row.get("has_marketing"):
            for k, v in [
                ("Channels Used",      row.get("channels_used", "-")),
                ("Best Channel",       row.get("best_channel", "-")),
                ("Avg Lift / Day",     "%.2f bookings" % row["avg_lift_per_day"] if pd.notna(row.get("avg_lift_per_day")) else "-"),
                ("Avg ROI",            fmt_pct(row.get("avg_roi"))),
                ("Positive Campaigns", "%d / %s" % (
                    int(row.get("n_positive_lift", 0)),
                    str(int(row.get("n_campaigns", 0))) if pd.notna(row.get("n_campaigns")) else "?",
                )),
            ]:
                ca, cb = st.columns([2, 3])
                ca.markdown(
                    "<span style='color:#7c82a0;font-size:0.8rem;'>%s</span>" % k,
                    unsafe_allow_html=True,
                )
                cb.markdown(
                    "<span style='color:#e8eaf0;font-size:0.85rem;font-weight:500;'>%s</span>" % v,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='background:#1e2130;border:1px dashed #2e3350;border-radius:10px;"
                "padding:1.5rem;text-align:center;color:#9ca3c4;'>"
                "No campaign history available for this restaurant."
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("### Ranked Strategy Evidence")
        st.caption(
            "Ranking score = 100 x Revenue uplift component + 30 x Bookings uplift + 10 x ROI + 5 x success rate. "
            "If revenue uplift % is missing, normalized incremental revenue is used."
        )
        if len(strategy_recs):
            recs_display = strategy_recs[[
                "strategy_name",
                "recommendation_scope",
                "recommendation_reason",
                "activities",
                "restaurants",
                "confidence_level",
                "avg_revenue_uplift_pct",
                "avg_bookings_uplift_pct",
                "avg_roi",
                "ranking_score",
                "context_adjusted_score",
                "data_quality_note",
            ]].copy()
            recs_display.columns = [
                "Strategy", "Scope", "Rationale", "Activities", "Restaurants",
                "Confidence Level", "Revenue Uplift", "Bookings Uplift",
                "Avg ROI", "Base Score", "Final Score", "Data Quality",
            ]
            recs_display["Revenue Uplift"]  = recs_display["Revenue Uplift"].apply(fmt_pct)
            recs_display["Bookings Uplift"] = recs_display["Bookings Uplift"].apply(fmt_pct)
            recs_display["Avg ROI"]         = recs_display["Avg ROI"].apply(lambda v: "-" if pd.isna(v) else "%.2f" % v)
            recs_display["Scope"]           = recs_display["Scope"].str.title()
            recs_display["Base Score"]      = pd.to_numeric(recs_display["Base Score"], errors="coerce").round(2)
            recs_display["Final Score"]     = pd.to_numeric(recs_display["Final Score"], errors="coerce").round(2)
            st.dataframe(recs_display, width="stretch", hide_index=True, height=260)
        else:
            st.info("No strategy recommendations available for this restaurant yet.")

    with strat_col:
        st.markdown("### Grounded Strategy Brief")
        grounded_key  = "grounded_strategy_%s" % selected
        ai_key        = "ai_strategy_%s" % selected
        grounded_brief = build_grounded_brief(row, hist, strategy_recs)
        st.session_state[grounded_key] = grounded_brief

        st.markdown(st.session_state[grounded_key])
        st.download_button(
            label="Download Grounded Brief",
            data="MARKETING STRATEGY\n%s\n\n%s" % (selected, st.session_state[grounded_key]),
            file_name="grounded_strategy_%s.txt" % selected.replace(" ", "_"),
            mime="text/plain",
            width="stretch",
        )

        st.markdown("---")
        st.markdown("### Optional AI Narrative")
        st.caption("Uses `ANTHROPIC_API_KEY` if configured. Deterministic brief above is the source of truth.")
        a_col, b_col = st.columns([2, 1])
        with a_col:
            generate_ai = st.button("Generate AI Narrative", key="gen_ai_%s" % selected, width="stretch")
        with b_col:
            if st.button("Clear AI", key="clr_ai_%s" % selected, width="stretch"):
                st.session_state[ai_key] = None

        if generate_ai:
            with st.spinner("Generating AI narrative..."):
                try:
                    prompt = build_prompt(row, hist, strategy_recs, grounded_brief)
                    st.session_state[ai_key] = call_claude(prompt)
                except Exception as e:
                    st.error("AI generation failed: %s" % e)
                    st.session_state[ai_key] = None

        if st.session_state.get(ai_key):
            st.markdown(st.session_state[ai_key])
            st.download_button(
                label="Download AI Narrative",
                data="AI STRATEGY NARRATIVE\n%s\n\n%s" % (selected, st.session_state[ai_key]),
                file_name="ai_strategy_%s.txt" % selected.replace(" ", "_"),
                mime="text/plain",
                width="stretch",
            )
