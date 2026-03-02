# -*- coding: utf-8 -*-
"""
pages/momentum.py
Page 2 - Momentum Dashboard
Segment distribution, growth heatmap, YoY vs MoM breakdown, stability analysis.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.loader import load_momentum, load_priority

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#111827", family="DM Sans"),
    xaxis=dict(gridcolor="#e0e0e0", showline=False, zeroline=False),
    yaxis=dict(gridcolor="#e0e0e0", showline=False, zeroline=False),
    margin=dict(l=0, r=0, t=30, b=0),
)
BASE_LAYOUT = {k: v for k, v in CHART_THEME.items() if k not in ("xaxis", "yaxis")}

SEG_COLOR_LIST = {
    "Rising Stars": "#2ecc71",
    "Emerging Opportunities": "#3b82f6",
    "Established Players": "#9b59b6",
    "Needs Attention": "#e74c3c",
}


def render():
    momentum_df = load_momentum()
    priority_df = load_priority()

    st.markdown("## Momentum Dashboard")
    st.markdown(
        "<p style='color:#6b7280; margin-top:-0.5rem;'>"
        "Segment distribution, growth trajectories, and seasonality-adjusted momentum signals."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    latest_all = momentum_df.sort_values("year_month").groupby("name", as_index=False).last()

    if "latest_segment" not in latest_all.columns and "latest_segment" in priority_df.columns:
        latest_all = latest_all.merge(
            priority_df[["name", "latest_segment"]].drop_duplicates("name"),
            on="name",
            how="left",
        )
    if "location" not in latest_all.columns and "location" in priority_df.columns:
        latest_all = latest_all.merge(
            priority_df[["name", "location"]].drop_duplicates("name"),
            on="name",
            how="left",
        )

    seg_col = "latest_segment" if "latest_segment" in latest_all.columns else "segment"
    has_segments = seg_col in latest_all.columns

    st.markdown("### Restaurant Search Highlight")
    s1, s2 = st.columns([1.4, 2.6])
    with s1:
        search_query = st.text_input(
            "Search restaurant name",
            placeholder="Type part of a restaurant name",
            key="momentum_search_query",
        )

    if search_query.strip():
        matched_restaurants = latest_all[
            latest_all["name"].str.contains(search_query.strip(), case=False, na=False)
        ]["name"].sort_values().tolist()
    else:
        matched_restaurants = latest_all["name"].sort_values().tolist()

    restaurant_options = ["None"] + matched_restaurants
    if "momentum_highlight_restaurant" not in st.session_state:
        st.session_state["momentum_highlight_restaurant"] = "None"
    if st.session_state["momentum_highlight_restaurant"] not in restaurant_options:
        st.session_state["momentum_highlight_restaurant"] = "None"

    with s2:
        selected_restaurant = st.selectbox(
            "Restaurant to highlight",
            options=restaurant_options,
            key="momentum_highlight_restaurant",
        )

    selected_row = (
        latest_all[latest_all["name"] == selected_restaurant].head(1)
        if selected_restaurant != "None"
        else pd.DataFrame()
    )
    selected_segment = (
        selected_row.iloc[0].get(seg_col)
        if has_segments and not selected_row.empty
        else None
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Active Restaurants", f"{latest_all['name'].nunique():,}")
    k2.metric("Total Monthly Bookings", f"{latest_all['monthly_bookings'].sum():,.0f}")
    k3.metric("Total Monthly Revenue", f"THB {latest_all['monthly_revenue'].sum()/1e6:.1f}M")

    if has_segments:
        rising = (latest_all[seg_col] == "Rising Stars").sum()
        emerging = (latest_all[seg_col] == "Emerging Opportunities").sum()
        k4.metric("Rising Stars", f"{rising}", f"+ Emerging: {emerging}")
    else:
        k4.metric("Avg Growth (3m)", f"{latest_all['booking_growth_rolling'].mean():.1%}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### Segment Distribution")
        if has_segments:
            seg_counts = latest_all[seg_col].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            colors = [SEG_COLOR_LIST.get(s, "#7c82a0") for s in seg_counts["segment"]]
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
                        font_color="#e8eaf0",
                    )
                ],
            )
            st.plotly_chart(fig_donut, width="stretch")
        else:
            st.info("Run notebooks to see segment distribution.")

    with col_right:
        st.markdown("### Performance vs Growth (Strategic Matrix)")
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

        if not selected_row.empty:
            hi = selected_row.iloc[0]
            if pd.notna(hi.get("score_perf")) and pd.notna(hi.get("score_growth")):
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[hi["score_perf"]],
                        y=[hi["score_growth"]],
                        mode="markers+text",
                        name="Selected Restaurant",
                        marker=dict(color="#cc0000", size=16, symbol="diamond", line=dict(color="#111827", width=2)),
                        text=[selected_restaurant],
                        textposition="top center",
                        hovertemplate=(
                            f"<b>{selected_restaurant}</b><br>"
                            f"Location: {hi.get('location_plot', 'Unknown')}<br>"
                            "Perf: %{x:.2f} | Growth: %{y:.2f}<extra></extra>"
                        ),
                        showlegend=True,
                    )
                )

        p75_perf = scatter_df["score_perf"].quantile(0.75)
        p75_growth = scatter_df["score_growth"].quantile(0.75)
        fig_scatter.add_vline(x=p75_perf, line_dash="dash", line_color="#2e3350", line_width=1)
        fig_scatter.add_hline(y=p75_growth, line_dash="dash", line_color="#2e3350", line_width=1)

        fig_scatter.update_layout(
            **BASE_LAYOUT,
            height=300,
            xaxis=dict(**CHART_THEME["xaxis"], title="Performance Score"),
            yaxis=dict(**CHART_THEME["yaxis"], title="Growth Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font_size=11),
        )
        st.plotly_chart(fig_scatter, width="stretch")
        if not selected_row.empty:
            perf_val = pd.to_numeric(pd.Series([hi.get("score_perf")]), errors="coerce").iloc[0]
            growth_val = pd.to_numeric(pd.Series([hi.get("score_growth")]), errors="coerce").iloc[0]
            st.caption(
                f"Selected: {selected_restaurant} | Segment: {hi.get(seg_col, 'Unknown')} | "
                f"Location: {hi.get('location_plot', 'Unknown')} | "
                f"Strategic matrix position: Perf {perf_val:.2f} / Growth {growth_val:.2f}"
                if pd.notna(perf_val) and pd.notna(growth_val)
                else f"Strategic matrix position: Perf {'-' if pd.isna(perf_val) else f'{perf_val:.2f}'} / "
                     f"Growth {'-' if pd.isna(growth_val) else f'{growth_val:.2f}'}"
            )

    st.markdown("---")

    col2a, col2b = st.columns(2)

    with col2a:
        st.markdown("### Growth Signal Used (YoY vs MoM)")
        if "growth_signal_used" in latest_all.columns:
            sig_counts = latest_all["growth_signal_used"].value_counts()
            fig_sig = go.Figure(
                go.Bar(
                    x=sig_counts.index,
                    y=sig_counts.values,
                    marker_color=["#f0a500", "#3b82f6"],
                    text=sig_counts.values,
                    textposition="outside",
                    textfont=dict(color="#e8eaf0"),
                    hovertemplate="%{x}: %{y} restaurants<extra></extra>",
                )
            )
            fig_sig.update_layout(
                **BASE_LAYOUT,
                height=260,
                showlegend=False,
                xaxis=dict(**CHART_THEME["xaxis"], title="Signal"),
                yaxis=dict(**CHART_THEME["yaxis"], title="Restaurants"),
            )
            st.plotly_chart(fig_sig, width="stretch")
            st.caption("YoY = seasonality-adjusted (preferred) | MoM = fallback for <12 months history")
        else:
            st.info("growth_signal_used column not found - re-run momentum_seasonality.ipynb.")

    with col2b:
        st.markdown("### Segment Stability (last 3 months)")
        if "growth_months" in priority_df.columns:
            stab = priority_df["growth_months"].value_counts().sort_index()
            fig_stab = go.Figure(
                go.Bar(
                    x=[f"{v} month{'s' if v != 1 else ''}" for v in stab.index],
                    y=stab.values,
                    marker_color=["#e74c3c", "#e67e22", "#2ecc71"],
                    text=stab.values,
                    textposition="outside",
                    textfont=dict(color="#e8eaf0"),
                    hovertemplate="In growth segment %{x}: %{y} restaurants<extra></extra>",
                )
            )
            fig_stab.update_layout(
                **BASE_LAYOUT,
                height=260,
                showlegend=False,
                xaxis=dict(**CHART_THEME["xaxis"], title="Consecutive growth months"),
                yaxis=dict(**CHART_THEME["yaxis"], title="Restaurants"),
            )
            st.plotly_chart(fig_stab, width="stretch")
            st.caption("Restaurants in green (2-3 months) form the stable-growth priority universe.")
        else:
            st.info("Stability data not found - run priority_scoring_seasonality.ipynb.")

    st.markdown("---")

    st.markdown("### Growth Rate Heatmap - Top 20 Restaurants by Volume")

    top20_names = (
        latest_all.nlargest(20, "monthly_bookings")["name"].tolist()
        if "monthly_bookings" in latest_all.columns
        else latest_all["name"].tolist()[:20]
    )

    heatmap_df = momentum_df[momentum_df["name"].isin(top20_names)].copy()
    heatmap_df["ym_label"] = heatmap_df["year_month"].dt.strftime("%b %Y")

    pivot = heatmap_df.pivot_table(
        index="name", columns="ym_label", values="booking_growth_rolling", aggfunc="mean"
    )

    all_months = sorted(heatmap_df["year_month"].unique())
    month_labels = [pd.Timestamp(m).strftime("%b %Y") for m in all_months]
    pivot = pivot.reindex(columns=[m for m in month_labels if m in pivot.columns])

    if not pivot.empty:
        fig_heat = go.Figure(
            go.Heatmap(
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
                text=[[f"{v:.0%}" if pd.notna(v) else "-" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont=dict(size=9),
                hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1%}<extra></extra>",
                colorbar=dict(
                    title=dict(text="Growth", font=dict(color="#7c82a0")),
                    tickformat=".0%",
                    tickfont=dict(color="#7c82a0"),
                ),
            )
        )
        fig_heat.update_layout(
            **BASE_LAYOUT,
            height=500,
            xaxis=dict(**CHART_THEME["xaxis"], tickangle=-45, tickfont=dict(size=10)),
            yaxis=dict(**CHART_THEME["yaxis"], tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_heat, width="stretch")

    st.markdown("---")

    st.markdown("### Top 10 Momentum Movers (Latest Month)")
    col3a, col3b = st.columns(2)

    with col3a:
        st.caption("Highest Rolling Booking Growth")
        top_growth = latest_all.nlargest(10, "booking_growth_rolling")[["name", "booking_growth_rolling", "monthly_bookings"]].copy()
        top_growth["booking_growth_rolling"] = top_growth["booking_growth_rolling"].apply(lambda x: f"{x:.1%}")
        top_growth.columns = ["Restaurant", "Growth (3m)", "Monthly Bookings"]
        st.dataframe(top_growth.reset_index(drop=True), width="stretch", height=320)

    with col3b:
        st.caption("Lowest Rolling Booking Growth")
        bot_growth = latest_all.nsmallest(10, "booking_growth_rolling")[["name", "booking_growth_rolling", "monthly_bookings"]].copy()
        bot_growth["booking_growth_rolling"] = bot_growth["booking_growth_rolling"].apply(lambda x: f"{x:.1%}")
        bot_growth.columns = ["Restaurant", "Growth (3m)", "Monthly Bookings"]
        st.dataframe(bot_growth.reset_index(drop=True), width="stretch", height=320)

    st.markdown("---")
    st.markdown("### Full Restaurant Momentum List (Latest Snapshot)")

    full_df = latest_all.copy()
    full_df["selected"] = full_df["name"].eq(selected_restaurant).map({True: "Yes", False: ""})
    if has_segments and seg_col in full_df.columns:
        full_df = full_df.rename(columns={seg_col: "momentum_segment"})

    display_cols = [
        "selected",
        "name",
        "momentum_segment",
        "location",
        "monthly_bookings",
        "monthly_revenue",
        "booking_growth_rolling",
        "score_perf",
        "score_growth",
        "growth_signal_used",
    ]
    display_cols = [c for c in display_cols if c in full_df.columns]
    full_display = full_df[display_cols].copy()
    if "booking_growth_rolling" in full_display.columns:
        full_display["booking_growth_rolling"] = full_display["booking_growth_rolling"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "-"
        )
    if "monthly_revenue" in full_display.columns:
        full_display["monthly_revenue"] = full_display["monthly_revenue"].apply(
            lambda x: f"THB {x:,.0f}" if pd.notna(x) else "-"
        )
    full_display = full_display.rename(
        columns={
            "selected": "Selected",
            "name": "Restaurant",
            "momentum_segment": "Momentum Segment",
            "location": "Location",
            "monthly_bookings": "Monthly Bookings",
            "monthly_revenue": "Monthly Revenue",
            "booking_growth_rolling": "Growth (3m)",
            "score_perf": "Performance Score",
            "score_growth": "Growth Score",
            "growth_signal_used": "Growth Signal",
        }
    )

    sort_cols = [c for c in ["Momentum Segment", "Monthly Bookings"] if c in full_display.columns]
    if sort_cols:
        ascending_vals = [True if c == "Momentum Segment" else False for c in sort_cols]
        full_display = full_display.sort_values(sort_cols, ascending=ascending_vals)

    st.dataframe(full_display.reset_index(drop=True), width="stretch", height=420)
