# -*- coding: utf-8 -*-
"""
pages/momentum.py
Page 2 — Momentum Dashboard
Segment distribution, growth heatmap, YoY vs MoM breakdown, stability analysis.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import numpy as np

from data.loader import load_momentum, load_priority, SEGMENT_COLORS

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#111827", family="DM Sans"),
    xaxis=dict(gridcolor="#e0e0e0", showline=False, zeroline=False),
    yaxis=dict(gridcolor="#e0e0e0", showline=False, zeroline=False),
    margin=dict(l=0, r=0, t=30, b=0),
)

SEG_COLOR_LIST = {
    "Rising Stars"           : "#2ecc71",
    "Emerging Opportunities" : "#3b82f6",
    "Established Players"    : "#9b59b6",
    "Needs Attention"        : "#e74c3c",
}


def render():
    momentum_df = load_momentum()
    priority_df = load_priority()

    st.markdown("## Momentum Dashboard")
    st.markdown("<p style='color:#6b7280; margin-top:-0.5rem;'>Segment distribution, growth trajectories, and seasonality-adjusted momentum signals.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Latest snapshot ───────────────────────────────────────────────────────
    latest_all = (
        momentum_df.sort_values("year_month")
        .groupby("name", as_index=False)
        .last()
    )

    # Merge segment from priority if not in momentum
    if "latest_segment" not in latest_all.columns and "latest_segment" in priority_df.columns:
        latest_all = latest_all.merge(
            priority_df[["name", "latest_segment"]].drop_duplicates("name"),
            on="name", how="left"
        )

    seg_col = "latest_segment" if "latest_segment" in latest_all.columns else "segment"
    has_segments = seg_col in latest_all.columns

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Active Restaurants", f"{latest_all['name'].nunique():,}")
    k2.metric("Total Monthly Bookings",   f"{latest_all['monthly_bookings'].sum():,.0f}")
    k3.metric("Total Monthly Revenue",    f"฿{latest_all['monthly_revenue'].sum()/1e6:.1f}M")

    if has_segments:
        rising = (latest_all[seg_col] == "Rising Stars").sum()
        k4.metric("Rising Stars", f"{rising}", f"+ Emerging: {(latest_all[seg_col]=='Emerging Opportunities').sum()}")
    else:
        k4.metric("Avg Growth (3m)", f"{latest_all['booking_growth_rolling'].mean():.1%}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Segment donut + scatter ────────────────────────────────────────
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### Segment Distribution")
        if has_segments:
            seg_counts = latest_all[seg_col].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            colors = [SEG_COLOR_LIST.get(s, "#7c82a0") for s in seg_counts["segment"]]
            fig_donut = go.Figure(go.Pie(
                labels=seg_counts["segment"],
                values=seg_counts["count"],
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#0f1117", width=3)),
                textinfo="percent+label",
                textfont=dict(size=11),
                hovertemplate="<b>%{label}</b><br>%{value} restaurants (%{percent})<extra></extra>",
            ))
            fig_donut.update_layout(
                **CHART_THEME,
                height=300,
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{latest_all['name'].nunique()}</b><br><span style='font-size:10px'>restaurants</span>",
                    x=0.5, y=0.5, font_size=14, showarrow=False, font_color="#e8eaf0"
                )]
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("Run notebooks to see segment distribution.")

    with col_right:
        st.markdown("### Performance vs Growth (Strategic Matrix)")
        scatter_df = latest_all.copy()
        color_vals = scatter_df[seg_col].map(SEG_COLOR_LIST).fillna("#7c82a0") if has_segments else ["#3b82f6"] * len(scatter_df)

        fig_scatter = go.Figure()
        if has_segments:
            for seg, color in SEG_COLOR_LIST.items():
                sub = scatter_df[scatter_df[seg_col] == seg]
                if len(sub):
                    fig_scatter.add_trace(go.Scatter(
                        x=sub["score_perf"],
                        y=sub["score_growth"],
                        mode="markers",
                        name=seg,
                        marker=dict(color=color, size=8, opacity=0.75, line=dict(color="#0f1117", width=1)),
                        hovertemplate="<b>%{customdata}</b><br>Perf: %{x:.2f} | Growth: %{y:.2f}<extra></extra>",
                        customdata=sub["name"],
                    ))
        else:
            fig_scatter.add_trace(go.Scatter(
                x=scatter_df["score_perf"], y=scatter_df["score_growth"],
                mode="markers",
                marker=dict(color="#3b82f6", size=7, opacity=0.7),
                hovertemplate="<b>%{customdata}</b><extra></extra>",
                customdata=scatter_df["name"],
            ))

        # Quadrant lines at 75th percentile
        p75_perf   = scatter_df["score_perf"].quantile(0.75)
        p75_growth = scatter_df["score_growth"].quantile(0.75)
        fig_scatter.add_vline(x=p75_perf,   line_dash="dash", line_color="#2e3350", line_width=1)
        fig_scatter.add_hline(y=p75_growth, line_dash="dash", line_color="#2e3350", line_width=1)

        fig_scatter.update_layout(
            **{k: v for k, v in CHART_THEME.items() if k not in ()},
            height=300,
            xaxis=dict(**CHART_THEME["xaxis"], title="Performance Score"),
            yaxis=dict(**CHART_THEME["yaxis"], title="Growth Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font_size=11),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # ── Row 2: Growth signal breakdown + stability ────────────────────────────
    col2a, col2b = st.columns(2)

    with col2a:
        st.markdown("### Growth Signal Used (YoY vs MoM)")
        if "growth_signal_used" in latest_all.columns:
            sig_counts = latest_all["growth_signal_used"].value_counts()
            fig_sig = go.Figure(go.Bar(
                x=sig_counts.index,
                y=sig_counts.values,
                marker_color=["#f0a500", "#3b82f6"],
                text=sig_counts.values,
                textposition="outside",
                textfont=dict(color="#e8eaf0"),
                hovertemplate="%{x}: %{y} restaurants<extra></extra>",
            ))
            fig_sig.update_layout(
                **{k: v for k, v in CHART_THEME.items() if k not in ()},
                height=260,
                showlegend=False,
                xaxis=dict(**CHART_THEME["xaxis"], title="Signal"),
                yaxis=dict(**CHART_THEME["yaxis"], title="Restaurants"),
            )
            st.plotly_chart(fig_sig, use_container_width=True)
            st.caption("🟡 YoY = seasonality-adjusted (preferred) · 🔵 MoM = fallback for <12 months history")
        else:
            st.info("growth_signal_used column not found — re-run momentum_seasonality.ipynb.")

    with col2b:
        st.markdown("### Segment Stability (last 3 months)")
        if "growth_months" in priority_df.columns:
            stab = priority_df["growth_months"].value_counts().sort_index()
            fig_stab = go.Figure(go.Bar(
                x=[f"{v} month{'s' if v != 1 else ''}" for v in stab.index],
                y=stab.values,
                marker_color=["#e74c3c", "#e67e22", "#2ecc71"],
                text=stab.values,
                textposition="outside",
                textfont=dict(color="#e8eaf0"),
                hovertemplate="In growth segment %{x}: %{y} restaurants<extra></extra>",
            ))
            fig_stab.update_layout(
                **{k: v for k, v in CHART_THEME.items() if k not in ()},
                height=260,
                showlegend=False,
                xaxis=dict(**CHART_THEME["xaxis"], title="Consecutive growth months"),
                yaxis=dict(**CHART_THEME["yaxis"], title="Restaurants"),
            )
            st.plotly_chart(fig_stab, use_container_width=True)
            st.caption("Restaurants in green (2–3 months) form the stable-growth priority universe.")
        else:
            st.info("Stability data not found — run priority_scoring_seasonality.ipynb.")

    st.markdown("---")

    # ── Row 3: Growth heatmap (top 20 by bookings) ───────────────────────────
    st.markdown("### Growth Rate Heatmap — Top 20 Restaurants by Volume")

    top20_names = (
        latest_all.nlargest(20, "monthly_bookings")["name"].tolist()
        if "monthly_bookings" in latest_all.columns else latest_all["name"].tolist()[:20]
    )

    heatmap_df = momentum_df[momentum_df["name"].isin(top20_names)].copy()
    heatmap_df["ym_label"] = heatmap_df["year_month"].dt.strftime("%b %Y")

    pivot = heatmap_df.pivot_table(
        index="name", columns="ym_label", values="booking_growth_rolling", aggfunc="mean"
    )

    # Sort columns chronologically
    all_months = sorted(heatmap_df["year_month"].unique())
    month_labels = [pd.Timestamp(m).strftime("%b %Y") for m in all_months]
    pivot = pivot.reindex(columns=[m for m in month_labels if m in pivot.columns])

    if not pivot.empty:
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[
                [0.0, "#e74c3c"],
                [0.4, "#1a1d27"],
                [0.6, "#1a1d27"],
                [1.0, "#2ecc71"],
            ],
            zmid=0,
            text=[[f"{v:.0%}" if pd.notna(v) else "—" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1%}<extra></extra>",
            colorbar=dict(
                title="Growth",
                tickformat=".0%",
                tickfont=dict(color="#7c82a0"),
                titlefont=dict(color="#7c82a0"),
            ),
        ))
        fig_heat.update_layout(
            **{k: v for k, v in CHART_THEME.items() if k not in ()},
            height=500,
            xaxis=dict(**CHART_THEME["xaxis"], tickangle=-45, tickfont=dict(size=10)),
            yaxis=dict(**CHART_THEME["yaxis"], tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Row 4: Top movers ─────────────────────────────────────────────────────
    st.markdown("### Top 10 Momentum Movers (Latest Month)")
    col3a, col3b = st.columns(2)

    with col3a:
        st.caption("📈 Highest Rolling Booking Growth")
        top_growth = latest_all.nlargest(10, "booking_growth_rolling")[["name", "booking_growth_rolling", "monthly_bookings"]].copy()
        top_growth["booking_growth_rolling"] = top_growth["booking_growth_rolling"].apply(lambda x: f"{x:.1%}")
        top_growth.columns = ["Restaurant", "Growth (3m)", "Monthly Bookings"]
        st.dataframe(top_growth.reset_index(drop=True), use_container_width=True, height=320)

    with col3b:
        st.caption("📉 Lowest Rolling Booking Growth")
        bot_growth = latest_all.nsmallest(10, "booking_growth_rolling")[["name", "booking_growth_rolling", "monthly_bookings"]].copy()
        bot_growth["booking_growth_rolling"] = bot_growth["booking_growth_rolling"].apply(lambda x: f"{x:.1%}")
        bot_growth.columns = ["Restaurant", "Growth (3m)", "Monthly Bookings"]
        st.dataframe(bot_growth.reset_index(drop=True), use_container_width=True, height=320)
