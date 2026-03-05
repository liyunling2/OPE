# -*- coding: utf-8 -*-
"""
pages/overview.py
Page 1 — Restaurant Explorer
Shows restaurant details, current performance metrics, and booking trends.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data.loader import (
    load_momentum,
    load_priority,
    load_momentum_raw_bookings,
    get_restaurant_history,
    get_restaurant_priority_row,
    get_restaurant_booking_history,
    SEGMENT_COLORS,
)

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#111827", family="DM Sans"),
    xaxis=dict(gridcolor="#e0e0e0", showline=False, zeroline=False),
    yaxis=dict(gridcolor="#e0e0e0", showline=False, zeroline=False),
    margin=dict(l=0, r=0, t=30, b=0),
)


def fmt_thb(val):
    if pd.isna(val):
        return "—"
    if val >= 1_000_000:
        return f"฿{val/1_000_000:.1f}M"
    if val >= 1_000:
        return f"฿{val/1_000:.0f}K"
    return f"฿{val:.0f}"


def segment_pill(segment: str) -> str:
    color_map = {
        "Rising Stars"           : "green",
        "Emerging Opportunities" : "blue",
        "Established Players"    : "purple",
        "Needs Attention"        : "red",
    }
    c = color_map.get(segment, "yellow")
    return f'<span class="pill pill-{c}">{segment}</span>'


def render():
    momentum_df  = load_momentum()
    priority_df  = load_priority()
    bookings_raw_df = load_momentum_raw_bookings()

    st.markdown("## Restaurant Explorer")
    st.markdown("<p style='color:#6b7280; margin-top:-0.5rem;'>Drill into any restaurant's performance, growth trajectory, and momentum segment.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])

    all_names = sorted(momentum_df["name"].unique())

    # ── Shared restaurant selection ────────────────────────────────────────────
    # "selected_restaurant" in session_state is the single source of truth
    # that all pages read. Changing it here propagates to Momentum, Strategy, etc.
    DEFAULT_RESTAURANT = "JW Cafe at JW Marriott Hotel Bangkok"

    # Initialise to default on first load if not already set
    if "selected_restaurant" not in st.session_state:
        # Try exact match first, then partial, then fall back to index 0
        if DEFAULT_RESTAURANT in all_names:
            st.session_state["selected_restaurant"] = DEFAULT_RESTAURANT
        else:
            partial = [n for n in all_names if "JW" in n and "Marriott" in n]
            st.session_state["selected_restaurant"] = partial[0] if partial else all_names[0]

    current = st.session_state["selected_restaurant"]
    default_idx = all_names.index(current) if current in all_names else 0

    with col_f1:
        selected = st.selectbox(
            "Select Restaurant",
            all_names,
            index=default_idx,
            key="overview_restaurant_select",
        )
        # Write back to shared state so other pages pick it up
        st.session_state["selected_restaurant"] = selected

    # Segment filter for the summary table at the bottom
    segments_available = sorted(momentum_df["latest_segment"].dropna().unique()) if "latest_segment" in momentum_df.columns else []
    with col_f2:
        seg_filter = st.selectbox(
            "Filter table by segment",
            ["All"] + (list(SEGMENT_COLORS.keys()) if not segments_available else segments_available)
        )
    with col_f3:
        sort_by = st.selectbox("Sort table by", ["monthly_bookings", "monthly_revenue", "score_growth", "score_perf"])

    st.markdown("---")

    # ── Restaurant detail ─────────────────────────────────────────────────────
    hist    = get_restaurant_history(momentum_df, selected)
    p_row   = get_restaurant_priority_row(priority_df, selected)
    latest  = hist.sort_values("year_month").iloc[-1] if len(hist) else {}

    if len(hist) == 0:
        st.warning("No data found for this restaurant.")
        return

    # Header row
    hc1, hc2 = st.columns([3, 1])
    with hc1:
        segment = p_row.get("latest_segment", latest.get("latest_segment", "—"))
        location = latest.get("location", "Bangkok")
        cuisine  = latest.get("cuisine", "—")
        st.markdown(f"""
        <div style='margin-bottom: 0.5rem;'>
            <span style='font-family: "DM Sans", sans-serif; font-size: 1.8rem; font-weight: 700; color: #111827;'>{selected}</span>
            &nbsp;&nbsp;{segment_pill(segment)}
        </div>
        <div style='color: #6b7280; font-size: 0.85rem;'>
            📍 {location} &nbsp;·&nbsp; 🍴 {cuisine}
            &nbsp;·&nbsp; Growth signal: <b style='color:#cc0000;'>{latest.get("growth_signal_used", "—")}</b>
        </div>
        """, unsafe_allow_html=True)
    with hc2:
        priority_score = p_row.get("priority_score", None)
        if priority_score is not None:
            tier = p_row.get("priority_tier", "—")
            st.markdown(f"""
            <div style='text-align:right;'>
                <div style='font-size:0.75rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em;'>Priority Score</div>
                <div style='font-family: "DM Sans", sans-serif; font-weight: 700; font-size: 2rem; color: #cc0000;'>{priority_score:.0f}</div>
                <div style='font-size:0.75rem; color:#6b7280;'>{tier}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI metrics ───────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    bk     = int(latest.get("monthly_bookings", 0))
    rev    = latest.get("monthly_revenue", 0)
    avg_rev = latest.get("avg_revenue_per_booking", 0)
    guests  = latest.get("avg_guests", 0)
    act_days = int(latest.get("active_days", 0))
    bk_growth = latest.get("booking_growth_rolling", 0)

    # compute MoM delta for bookings
    if len(hist) >= 2:
        prev_bk = hist.sort_values("year_month").iloc[-2].get("monthly_bookings", bk)
        bk_delta = int(bk - prev_bk)
    else:
        bk_delta = 0

    m1.metric("Monthly Bookings",      f"{bk:,}",         f"{bk_delta:+,} vs prev month")
    m2.metric("Monthly Revenue",       fmt_thb(rev),       "")
    m3.metric("Avg Rev / Booking",     fmt_thb(avg_rev),   "")
    m4.metric("Avg Guests / Booking",  f"{guests:.1f}",    "")
    m5.metric("Active Days",           f"{act_days}",      "")
    m6.metric("Rolling Growth (3m)",   f"{bk_growth:.1%}", "YoY" if latest.get("growth_signal_used") == "YoY" else "MoM")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📦 Booking Volume", "💰 Revenue", "📈 Growth Rate"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hist["year_month"], y=hist["monthly_bookings"],
            marker_color="#3b82f6", marker_opacity=0.85,
            name="Bookings",
            hovertemplate="<b>%{x|%b %Y}</b><br>Bookings: %{y:,}<extra></extra>"
        ))
        # 3-month rolling average line
        hist_sorted = hist.sort_values("year_month")
        hist_sorted["bk_ma3"] = hist_sorted["monthly_bookings"].rolling(3, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=hist_sorted["year_month"], y=hist_sorted["bk_ma3"],
            mode="lines", line=dict(color="#f0a500", width=2, dash="dot"),
            name="3m avg",
            hovertemplate="3m avg: %{y:.0f}<extra></extra>"
        ))
        fig.update_layout(
            **CHART_THEME,
            height=280,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            barmode="overlay",
        )
        st.plotly_chart(fig, width="stretch")

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hist["year_month"], y=hist["monthly_revenue"],
            fill="tozeroy",
            line=dict(color="#2ecc71", width=2),
            fillcolor="rgba(46,204,113,0.12)",
            name="Revenue (THB)",
            hovertemplate="<b>%{x|%b %Y}</b><br>฿%{y:,.0f}<extra></extra>"
        ))
        fig2.update_layout(**CHART_THEME, height=280)
        st.plotly_chart(fig2, width="stretch")

    with tab3:
        fig3 = go.Figure()
        # YoY growth if available
        if "booking_growth_yoy" in hist.columns:
            yoy = hist[hist["booking_growth_yoy"].notna()]
            fig3.add_trace(go.Scatter(
                x=yoy["year_month"], y=yoy["booking_growth_yoy"],
                mode="lines+markers",
                line=dict(color="#f0a500", width=2),
                name="YoY Growth",
                hovertemplate="<b>%{x|%b %Y}</b><br>YoY: %{y:.1%}<extra></extra>"
            ))
        # rolling growth
        fig3.add_trace(go.Scatter(
            x=hist["year_month"], y=hist["booking_growth_rolling"],
            mode="lines",
            line=dict(color="#3b82f6", width=2, dash="dot"),
            name="3m Rolling (blended)",
            hovertemplate="<b>%{x|%b %Y}</b><br>Rolling: %{y:.1%}<extra></extra>"
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="#7c82a0", line_width=1)
        fig3.update_layout(
            **{k: v for k, v in CHART_THEME.items() if k != "yaxis"},
            height=280,
            yaxis=dict(**CHART_THEME["yaxis"], tickformat=".0%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig3, width="stretch")

    st.markdown("---")

    # ── Portfolio summary table ───────────────────────────────────────────────
    st.markdown("### All Restaurants — Latest Snapshot")

    latest_all = (
        momentum_df.sort_values("year_month")
        .groupby("name", as_index=False)
        .last()
    )

    # Merge priority metadata only (keep segment source from momentum universe).
    if "priority_score" in priority_df.columns:
        merge_cols = ["name", "priority_score", "priority_tier", "recommended_channel"]
        merge_cols = [c for c in merge_cols if c in priority_df.columns]
        latest_all = latest_all.merge(priority_df[merge_cols].drop_duplicates("name"), on="name", how="left")

    # Fallback if momentum table has no segment column.
    if "latest_segment" not in latest_all.columns and "latest_segment" in priority_df.columns:
        latest_all = latest_all.merge(
            priority_df[["name", "latest_segment"]].drop_duplicates("name"),
            on="name",
            how="left",
        )

    # Defensive handling in case an earlier merge produced suffixed segment columns.
    if "latest_segment" not in latest_all.columns and (
        "latest_segment_x" in latest_all.columns or "latest_segment_y" in latest_all.columns
    ):
        seg_x = latest_all["latest_segment_x"] if "latest_segment_x" in latest_all.columns else pd.Series([pd.NA] * len(latest_all))
        seg_y = latest_all["latest_segment_y"] if "latest_segment_y" in latest_all.columns else pd.Series([pd.NA] * len(latest_all))
        latest_all["latest_segment"] = seg_x.where(seg_x.notna(), seg_y)
        latest_all = latest_all.drop(columns=[c for c in ["latest_segment_x", "latest_segment_y"] if c in latest_all.columns])

    if seg_filter != "All" and "latest_segment" in latest_all.columns:
        latest_all = latest_all[latest_all["latest_segment"].fillna("Unknown") == seg_filter]

    if "latest_segment" in latest_all.columns:
        latest_all["latest_segment"] = latest_all["latest_segment"].fillna("Unknown")

    display_cols = [c for c in [
        "name", "latest_segment", "monthly_bookings", "monthly_revenue",
        "avg_revenue_per_booking", "avg_guests", "booking_growth_rolling",
        "growth_signal_used", "priority_score", "recommended_channel"
    ] if c in latest_all.columns]

    display_df = latest_all[display_cols].copy()
    if sort_by in display_df.columns:
        display_df = display_df.sort_values(sort_by, ascending=False)

    # Format
    for col in ["monthly_revenue", "avg_revenue_per_booking"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt_thb)
    for col in ["booking_growth_rolling"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    if "priority_score" in display_df.columns:
        display_df["priority_score"] = display_df["priority_score"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")

    st.dataframe(
        display_df.reset_index(drop=True),
        width="stretch",
        height=400,
    )

    st.markdown("---")
    st.markdown("### Historical Booking Records")

    selected_restaurant_id = pd.to_numeric(pd.Series([latest.get("restaurant_id")]), errors="coerce").iloc[0]
    booking_hist = get_restaurant_booking_history(
        bookings_raw_df,
        restaurant_name=selected,
        restaurant_id=int(selected_restaurant_id) if pd.notna(selected_restaurant_id) else None,
    )

    if booking_hist.empty:
        st.info("No raw booking records found for this restaurant.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw Bookings", f"{len(booking_hist):,}")

        min_dt = booking_hist["booking_date"].min() if "booking_date" in booking_hist.columns else pd.NaT
        max_dt = booking_hist["booking_date"].max() if "booking_date" in booking_hist.columns else pd.NaT
        c2.metric("First Booking", min_dt.strftime("%Y-%m-%d") if pd.notna(min_dt) else "-")
        c3.metric("Latest Booking", max_dt.strftime("%Y-%m-%d") if pd.notna(max_dt) else "-")

        rows_default = int(min(300, len(booking_hist)))
        rows_step = 1 if len(booking_hist) < 50 else 50
        rows_to_show = int(
            st.number_input(
                "Rows to display",
                min_value=1,
                max_value=int(len(booking_hist)),
                value=rows_default,
                step=rows_step,
                key="overview_raw_booking_rows",
            )
        )

        raw_display = booking_hist.head(rows_to_show).copy()
        if "booking_date" in raw_display.columns:
            raw_display["booking_date"] = pd.to_datetime(raw_display["booking_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "created_at" in raw_display.columns:
            raw_display["created_at"] = pd.to_datetime(raw_display["created_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

        if "revenue_thb" in raw_display.columns:
            raw_display["revenue_thb"] = pd.to_numeric(raw_display["revenue_thb"], errors="coerce").apply(
                lambda x: f"THB {x:,.0f}" if pd.notna(x) else "-"
            )
        if "revenue_dollars" in raw_display.columns:
            raw_display["revenue_dollars"] = pd.to_numeric(raw_display["revenue_dollars"], errors="coerce").apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "-"
            )

        display_cols = [
            c
            for c in [
                "booking_id",
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
            if c in raw_display.columns
        ]
        raw_display = raw_display[display_cols].rename(
            columns={
                "booking_id": "Booking ID",
                "booking_date": "Booking Date",
                "created_at": "Created At",
                "start_time": "Start Time",
                "end_time": "End Time",
                "channel": "Channel",
                "medium": "Medium",
                "adults": "Adults",
                "kids": "Kids",
                "total_guests": "Total Guests",
                "revenue_thb": "Revenue (THB)",
                "revenue_dollars": "Revenue (USD)",
                "arrived": "Arrived",
                "no_show": "No Show",
                "refund": "Refund",
                "adjusted": "Adjusted",
            }
        )

        st.dataframe(raw_display, width="stretch", height=360)
