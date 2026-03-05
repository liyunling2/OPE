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
        font=dict(color="#111827", family="DM Sans"), # Switched to dark text
        margin=dict(l=0, r=0, t=30, b=0),
        height=height,
    )
    base.update(kwargs)
    return base

AXIS = dict(gridcolor="#e0e0e0", showline=False, zeroline=False) # Switched to light grid

# Priority scoring weights from priority_scoring_seasonality.ipynb
W_SCORE_GROWTH = 0.60
W_DELTA_GROWTH = 0.40
W_GROWTH_FINAL = 0.60
W_MARKETING_FINAL = 0.40
W_LIFT = 0.60
W_ROI = 0.40

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

    lift_series = out["avg_lift_per_day"] if "avg_lift_per_day" in out.columns else pd.Series(0, index=out.index)
    roi_series = out["avg_roi"] if "avg_roi" in out.columns else pd.Series(0, index=out.index)
    out["lift_norm"] = min_max_norm(lift_series.fillna(0).clip(lower=0))
    out["roi_norm"] = min_max_norm(roi_series.fillna(0).clip(lower=0))

    if "pct_yoy_baseline" in out.columns:
        out["lift_reliability_calc"] = 0.8 + 0.2 * pd.to_numeric(out["pct_yoy_baseline"], errors="coerce").fillna(0)
    else:
        out["lift_reliability_calc"] = 1.0

    out["marketing_component"] = (
        out["lift_norm"] * out["lift_reliability_calc"] * W_LIFT +
        out["roi_norm"] * W_ROI
    )

    out["growth_component_norm"] = min_max_norm(out["growth_component"])
    has_mkt = out["has_marketing"].fillna(False) if "has_marketing" in out.columns else pd.Series(False, index=out.index)
    out["priority_raw_recomputed"] = np.where(
        has_mkt,
        out["growth_component"] * W_GROWTH_FINAL + out["marketing_component"] * W_MARKETING_FINAL,
        out["growth_component_norm"],
    )
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
            "monthly_revenue",
            "booking_growth_rolling",
            "revenue_growth_rolling",
            "booking_growth_yoy",
            "revenue_growth_yoy",
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
        "monthly_revenue",
        "booking_growth_rolling",
        "revenue_growth_rolling",
        "booking_growth_yoy",
        "revenue_growth_yoy",
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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("In Priority List", str(int(priority_df["is_in_priority_list"].sum())))
    def ct(kw): return sum(kw in str(t).lower() for t in priority_df["priority_tier"])
    k2.metric("Proven",   str(ct("proven")))
    k3.metric("Untapped", str(ct("untapped")))
    k4.metric("Review",   str(ct("review")))
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        f"Total restaurants shown: {len(priority_df):,}. "
        "Restaurants outside the stable-growth priority universe are included as 'Monitor'."
    )

    with st.expander("How Priority Score Is Calculated", expanded=False):
        st.markdown(
            "**Formula (from `priority_scoring_seasonality.ipynb`)**\n\n"
            "- `growth_component = 0.60 * score_growth_norm + 0.40 * delta_growth_norm`\n"
            "- `marketing_component = 0.60 * lift_norm * lift_reliability + 0.40 * roi_norm`\n"
            "- If `has_marketing`:\n"
            "  `priority_raw = 0.60 * growth_component + 0.40 * marketing_component`\n"
            "- If no marketing history:\n"
            "  `priority_raw = growth_component_norm`\n"
            "- Final score: `priority_score = min_max_norm(priority_raw) * 100`"
        )

        breakdown_names = priority_df.sort_values("priority_score", ascending=False)["name"].tolist()
        breakdown_name = st.selectbox("Restaurant breakdown", breakdown_names, key="priority_breakdown_name")
        brow = priority_df[priority_df["name"] == breakdown_name].iloc[0]

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Priority Score", f"{brow.get('priority_score', 0):.2f}")
        b2.metric("Growth Component", f"{brow.get('growth_component', 0):.4f}")
        b3.metric("Marketing Component", f"{brow.get('marketing_component', 0):.4f}")
        b4.metric("Has Marketing", "Yes" if bool(brow.get("has_marketing", False)) else "No")

        breakdown_df = pd.DataFrame(
            [
                ("score_growth", brow.get("score_growth", np.nan)),
                ("delta_growth_book", brow.get("delta_growth_book", np.nan)),
                ("score_growth_norm", brow.get("score_growth_norm", np.nan)),
                ("delta_growth_norm", brow.get("delta_growth_norm", np.nan)),
                ("growth_component", brow.get("growth_component", np.nan)),
                ("avg_lift_per_day", brow.get("avg_lift_per_day", np.nan)),
                ("avg_roi", brow.get("avg_roi", np.nan)),
                ("lift_norm", brow.get("lift_norm", np.nan)),
                ("roi_norm", brow.get("roi_norm", np.nan)),
                ("lift_reliability", brow.get("lift_reliability", brow.get("lift_reliability_calc", np.nan))),
                ("marketing_component", brow.get("marketing_component", np.nan)),
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
        st.markdown("### Channel Mix")
        channel_series = priority_df["recommended_channel"].dropna().astype(str).str.strip()
        channel_series = channel_series[
            ~channel_series.str.lower().isin(["", "-", "unknown", "n/a", "nan"])
        ]
        ch_counts = channel_series.value_counts()
        cmap = {"FB":"#3b82f6","KOL":"#9b59b6","CRM":"#2ecc71"}
        if len(ch_counts):
            fig_ch = go.Figure(go.Bar(
                x=ch_counts.index, y=ch_counts.values,
                marker_color=[cmap.get(str(c),"#7c82a0") for c in ch_counts.index],
                text=ch_counts.values, textposition="outside", textfont=dict(color="#e8eaf0"),
            ))
            fig_ch.update_layout(**layout(280, showlegend=False,
                xaxis=dict(**AXIS, title="Channel"), yaxis=dict(**AXIS, title="Restaurants")))
            st.plotly_chart(fig_ch, width="stretch")
        else:
            st.info("No valid channel assignments in the current view.")

    st.markdown("---")

    f1, f2, f3, f4, f5 = st.columns([1, 1, 1, 1, 1.6])
    with f1:
        t_opts = ["All"] + priority_df["priority_tier"].dropna().unique().tolist()
        t_filt = st.selectbox("Tier", t_opts)
    with f2:
        has_seg = "latest_segment" in priority_df.columns
        s_opts  = ["All"] + (priority_df["latest_segment"].dropna().unique().tolist() if has_seg else [])
        s_filt  = st.selectbox("Segment", s_opts)
    with f3:
        channel_opts = priority_df["recommended_channel"].dropna().astype(str).str.strip()
        channel_opts = channel_opts[
            ~channel_opts.str.lower().isin(["", "-", "unknown", "n/a", "nan"])
        ]
        c_opts = ["All"] + sorted(channel_opts.unique().tolist())
        c_filt = st.selectbox("Channel", c_opts)
    with f4:
        min_sc = st.slider("Min Score", 0, 100, 0)
    with f5:
        name_search = st.text_input("Search Restaurant", placeholder="Type restaurant name", key="priority_name_search")

    df = priority_df.copy()
    if t_filt != "All": df = df[df["priority_tier"] == t_filt]
    if s_filt != "All" and has_seg: df = df[df["latest_segment"] == s_filt]
    if c_filt != "All": df = df[df["recommended_channel"] == c_filt]
    if str(name_search).strip():
        df = df[df["name"].str.contains(name_search.strip(), case=False, na=False)]
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
    st.plotly_chart(fig_rank, width="stretch")

    st.markdown("---")
    st.markdown("### Full Ranked List")

    # Highlight if the globally selected restaurant appears in this list
    _sel = st.session_state.get("selected_restaurant")
    if _sel and _sel in df["name"].values:
        _sel_score = df[df["name"] == _sel]["priority_score"].iloc[0]
        _sel_rank  = df[df["name"] == _sel].index[0] + 1
        st.success(
            "Currently selected restaurant **%s** appears at rank #%d "
            "with a priority score of **%.0f / 100**." % (_sel, _sel_rank, _sel_score)
        )
    elif _sel:
        st.warning("Currently selected restaurant **%s** is not in the filtered list." % _sel)


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
