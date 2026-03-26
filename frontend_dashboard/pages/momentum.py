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
from theme import BASE_LAYOUT, CHART_THEME, GRID_COLOR, MUTED_TEXT, SOFT_DIVIDER, TEXT_COLOR

SEG_COLOR_LIST = {
    "Rising Stars": "#2ecc71",
    "Emerging Opportunities": "#3b82f6",
    "Established Players": "#9b59b6",
    "Needs Attention": "#e74c3c",
}


def fmt_thb_short(val):
    if val is None or pd.isna(val):
        return "-"
    if abs(val) >= 1_000_000:
        return f"THB {val/1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"THB {val/1_000:.0f}K"
    return f"THB {val:.0f}"


def fmt_pct(val):
    if val is None or pd.isna(val):
        return "-"
    return f"{val:.1%}"


def fmt_score_delta(current, previous):
    if previous is None or pd.isna(previous) or current is None or pd.isna(current):
        return None
    return f"{current - previous:+.2f}"


def fmt_level_delta(current, previous, currency: bool = False):
    if previous is None or pd.isna(previous) or current is None or pd.isna(current):
        return None
    delta = current - previous
    if currency:
        sign = "+" if delta >= 0 else "-"
        return f"{sign}{fmt_thb_short(abs(delta))} vs prev"
    return f"{delta:+,.0f} vs prev"


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def score_percentile(series: pd.Series, value) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    current = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(current) or values.empty:
        return float("nan")
    return float((values <= current).mean())


def resolve_growth_signal_columns(hist: pd.DataFrame, signal_used: str) -> tuple[str | None, str | None, str]:
    has_yoy = (
        "booking_growth_yoy_rolling" in hist.columns
        and numeric_series(hist, "booking_growth_yoy_rolling").notna().any()
    )
    if str(signal_used).strip().upper() == "YOY" and has_yoy:
        revenue_col = "gmv_growth_yoy_rolling" if "gmv_growth_yoy_rolling" in hist.columns else "gmv_growth_rolling"
        return "booking_growth_yoy_rolling", revenue_col, "YoY"

    booking_col = "booking_growth_mom_rolling" if "booking_growth_mom_rolling" in hist.columns else "booking_growth_rolling"
    revenue_col = "gmv_growth_mom_rolling" if "gmv_growth_mom_rolling" in hist.columns else "gmv_growth_rolling"
    label = "MoM" if booking_col == "booking_growth_mom_rolling" else "Rolling"
    return booking_col if booking_col in hist.columns else None, revenue_col if revenue_col in hist.columns else None, label


def render():
    momentum_df = load_momentum()
    priority_df = load_priority()

    st.markdown("## Momentum Dashboard")
    st.markdown(
        f"<p style='color:{MUTED_TEXT}; margin-top:-0.5rem;'>"
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
    selected_history = (
        momentum_df[momentum_df["name"] == selected_restaurant].sort_values("year_month").copy()
        if selected_restaurant != "None"
        else pd.DataFrame()
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Active Restaurants", f"{latest_all['name'].nunique():,}")
    k2.metric("Total Monthly Bookings", f"{latest_all['monthly_bookings'].sum():,.0f}")
    k3.metric("Total Monthly Revenue", f"THB {latest_all['monthly_gmv'].sum()/1e6:.1f}M")

    if has_segments:
        rising = (latest_all[seg_col] == "Rising Stars").sum()
        emerging = (latest_all[seg_col] == "Emerging Opportunities").sum()
        k4.metric("Rising Stars", f"{rising}", f"+ Emerging: {emerging}")
    else:
        k4.metric("Avg Growth (3m)", f"{latest_all['booking_growth_rolling'].mean():.1%}")

    if "is_seasonal" in latest_all.columns:
        n_seas = int(latest_all["is_seasonal"].fillna(False).sum())
        st.caption(f"🌊 {n_seas} restaurants flagged Seasonal — strong MoM but weak YoY. Timing-sensitive activation.")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### Segment Distribution")
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

    if not selected_history.empty:
        st.markdown("### Highlighted Restaurant Trajectory")
        selected_history = selected_history.copy()
        selected_history["month_label"] = selected_history["year_month"].dt.strftime("%b %Y")
        latest_point = selected_history.iloc[-1]
        previous_point = selected_history.iloc[-2] if len(selected_history) > 1 else None

        latest_perf = pd.to_numeric(pd.Series([latest_point.get("score_perf")]), errors="coerce").iloc[0]
        latest_growth = pd.to_numeric(pd.Series([latest_point.get("score_growth")]), errors="coerce").iloc[0]
        latest_bookings = pd.to_numeric(pd.Series([latest_point.get("monthly_bookings")]), errors="coerce").iloc[0]
        latest_gmv = pd.to_numeric(pd.Series([latest_point.get("monthly_gmv")]), errors="coerce").iloc[0]
        latest_growth_rate = pd.to_numeric(
            pd.Series(
                [
                    latest_point.get(
                        "booking_growth_rolling",
                        latest_point.get("booking_growth_mom_rolling"),
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]

        prev_perf = previous_point.get("score_perf") if previous_point is not None else None
        prev_growth = previous_point.get("score_growth") if previous_point is not None else None
        prev_bookings = previous_point.get("monthly_bookings") if previous_point is not None else None
        prev_gmv = previous_point.get("monthly_gmv") if previous_point is not None else None

        snap_a, snap_b, snap_c, snap_d, snap_e = st.columns(5)
        snap_a.metric("Performance Score", "-" if pd.isna(latest_perf) else f"{latest_perf:.2f}", delta=fmt_score_delta(latest_perf, prev_perf))
        snap_b.metric("Growth Score", "-" if pd.isna(latest_growth) else f"{latest_growth:.2f}", delta=fmt_score_delta(latest_growth, prev_growth))
        snap_c.metric("Latest Bookings", "-" if pd.isna(latest_bookings) else f"{latest_bookings:,.0f}", delta=fmt_level_delta(latest_bookings, prev_bookings))
        snap_d.metric("Latest Revenue", fmt_thb_short(latest_gmv), delta=fmt_level_delta(latest_gmv, prev_gmv, currency=True))
        snap_e.metric(
            "Latest Growth Signal",
            str(latest_point.get("growth_signal_used", "-")),
            delta=fmt_pct(latest_growth_rate) if pd.notna(latest_growth_rate) else None,
        )

        trend_col, path_col = st.columns(2)

        with trend_col:
            st.markdown("#### Score Trend")
            fig_scores = go.Figure()
            if "score_perf" in selected_history.columns:
                fig_scores.add_trace(
                    go.Scatter(
                        x=selected_history["year_month"],
                        y=pd.to_numeric(selected_history["score_perf"], errors="coerce"),
                        mode="lines+markers",
                        name="Performance Score",
                        line=dict(color="#3b82f6", width=3),
                        marker=dict(size=7),
                        hovertemplate="<b>%{x|%b %Y}</b><br>Performance: %{y:.2f}<extra></extra>",
                    )
                )
            if "score_growth" in selected_history.columns:
                fig_scores.add_trace(
                    go.Scatter(
                        x=selected_history["year_month"],
                        y=pd.to_numeric(selected_history["score_growth"], errors="coerce"),
                        mode="lines+markers",
                        name="Growth Score",
                        line=dict(color="#2ecc71", width=3),
                        marker=dict(size=7),
                        hovertemplate="<b>%{x|%b %Y}</b><br>Growth: %{y:.2f}<extra></extra>",
                    )
                )
            fig_scores.update_layout(
                **BASE_LAYOUT,
                height=280,
                xaxis=dict(**CHART_THEME["xaxis"], title="Month", tickangle=-30),
                yaxis=dict(**CHART_THEME["yaxis"], title="Score (0-1)"),
                legend=dict(orientation="h", y=1.02, x=0, font_size=10),
            )
            st.plotly_chart(fig_scores, width="stretch")

        with path_col:
            st.markdown("#### Matrix Path Over Time")
            path_df = selected_history.dropna(subset=["score_perf", "score_growth"]).copy()
            if len(path_df):
                hover_text = []
                for _, row in path_df.iterrows():
                    hover_text.append(
                        "<b>{month}</b><br>Perf: {perf:.2f}<br>Growth: {growth:.2f}<br>"
                        "Bookings: {bookings}<br>Revenue: {revenue}<br>Rolling growth: {roll}<br>Signal: {signal}".format(
                            month=row["month_label"],
                            perf=float(row["score_perf"]),
                            growth=float(row["score_growth"]),
                            bookings="-"
                            if pd.isna(pd.to_numeric(row.get("monthly_bookings"), errors="coerce"))
                            else f"{float(row.get('monthly_bookings')):,.0f}",
                            revenue=fmt_thb_short(pd.to_numeric(row.get("monthly_gmv"), errors="coerce")),
                            roll=fmt_pct(pd.to_numeric(row.get("booking_growth_rolling"), errors="coerce")),
                            signal=row.get("growth_signal_used", "-"),
                        )
                    )

                fig_path = go.Figure()
                fig_path.add_trace(
                    go.Scatter(
                        x=path_df["score_perf"],
                        y=path_df["score_growth"],
                        mode="lines+markers",
                        name=selected_restaurant,
                        line=dict(color="#94a3b8", width=2),
                        marker=dict(
                            size=9,
                            color=list(range(len(path_df))),
                            colorscale="Blues",
                            line=dict(color="#0f1117", width=1.5),
                            showscale=False,
                        ),
                        hovertext=hover_text,
                        hovertemplate="%{hovertext}<extra></extra>",
                    )
                )
                fig_path.add_trace(
                    go.Scatter(
                        x=[path_df["score_perf"].iloc[-1]],
                        y=[path_df["score_growth"].iloc[-1]],
                        mode="markers+text",
                        name="Latest",
                        marker=dict(color="#cc0000", size=16, symbol="diamond", line=dict(color="#111827", width=2)),
                        text=["Latest"],
                        textposition="top center",
                        hovertemplate="%{text}<extra></extra>",
                        showlegend=True,
                    )
                )
                fig_path.add_vline(
                    x=latest_all["score_perf"].quantile(0.75),
                    line_dash="dash",
                    line_color=GRID_COLOR,
                    line_width=1,
                )
                fig_path.add_hline(
                    y=latest_all["score_growth"].quantile(0.75),
                    line_dash="dash",
                    line_color=GRID_COLOR,
                    line_width=1,
                )
                fig_path.update_layout(
                    **BASE_LAYOUT,
                    height=280,
                    xaxis=dict(**CHART_THEME["xaxis"], title="Performance Score"),
                    yaxis=dict(**CHART_THEME["yaxis"], title="Growth Score"),
                    legend=dict(orientation="h", y=1.02, x=0, font_size=10),
                )
                st.plotly_chart(fig_path, width="stretch")
                st.caption(
                    "The line shows how the restaurant moved through the strategic matrix month by month. "
                    "Hover each point to see bookings, revenue, rolling growth, and the signal used."
                )
            else:
                st.info("Not enough scored history is available to chart the matrix path.")

        st.markdown("#### Why The Growth Score Is High")
        levels_df = selected_history.copy()
        levels_df["bookings_value"] = numeric_series(levels_df, "monthly_bookings")
        levels_df["revenue_value"] = numeric_series(levels_df, "monthly_gmv")
        levels_df["bookings_mom_change"] = levels_df["bookings_value"].diff()
        levels_df["revenue_mom_change"] = levels_df["revenue_value"].diff()
        levels_df["bookings_3m_avg"] = levels_df["bookings_value"].rolling(3, min_periods=1).mean()
        levels_df["revenue_3m_avg"] = levels_df["revenue_value"].rolling(3, min_periods=1).mean()

        booking_signal_col, revenue_signal_col, signal_label = resolve_growth_signal_columns(
            levels_df,
            latest_point.get("growth_signal_used", "-"),
        )
        levels_df["booking_signal_growth"] = (
            numeric_series(levels_df, booking_signal_col)
            if booking_signal_col
            else pd.Series(index=levels_df.index, dtype="float64")
        )
        levels_df["revenue_signal_growth"] = (
            numeric_series(levels_df, revenue_signal_col)
            if revenue_signal_col
            else pd.Series(index=levels_df.index, dtype="float64")
        )
        levels_df["mom_growth_composite"] = (
            numeric_series(levels_df, "booking_growth_mom_rolling")
            + numeric_series(levels_df, "gmv_growth_mom_rolling")
        ) / 2
        levels_df["yoy_growth_composite"] = (
            numeric_series(levels_df, "booking_growth_yoy_rolling")
            + numeric_series(levels_df, "gmv_growth_yoy_rolling")
        ) / 2
        levels_df["growth_input_blended"] = numeric_series(levels_df, "growth_rate_blended")
        if levels_df["growth_input_blended"].isna().all():
            levels_df["growth_input_blended"] = levels_df["mom_growth_composite"]
            yoy_mask = levels_df["yoy_growth_composite"].notna()
            levels_df.loc[yoy_mask, "growth_input_blended"] = (
                levels_df.loc[yoy_mask, "yoy_growth_composite"] * 0.70
                + levels_df.loc[yoy_mask, "mom_growth_composite"] * 0.30
            )

        latest_booking_signal = levels_df["booking_signal_growth"].iloc[-1] if len(levels_df) else None
        latest_revenue_signal = levels_df["revenue_signal_growth"].iloc[-1] if len(levels_df) else None
        latest_blended_growth = levels_df["growth_input_blended"].iloc[-1] if len(levels_df) else None
        growth_pctile = score_percentile(latest_all["score_growth"], latest_growth)
        growth_rank_label = f"P{int(round(growth_pctile * 100))}" if pd.notna(growth_pctile) else None
        blend_rule = (
            "70% YoY + 30% MoM blend"
            if signal_label == "YoY" and pd.notna(levels_df["yoy_growth_composite"].iloc[-1])
            else "100% MoM composite"
        )

        explainer_bits = [
            "`score_growth` is a relative 0-1 score, not a literal growth percentage.",
        ]
        if pd.notna(latest_growth) and growth_rank_label:
            explainer_bits.append(
                f"This restaurant's latest growth score is {latest_growth:.2f}, around {growth_rank_label} within the portfolio."
            )
        if pd.notna(latest_booking_signal) or pd.notna(latest_revenue_signal):
            explainer_bits.append(
                f"The latest {signal_label} rolling inputs are bookings {fmt_pct(latest_booking_signal)} and revenue {fmt_pct(latest_revenue_signal)}."
            )
        if pd.notna(latest_blended_growth):
            explainer_bits.append(
                f"Those inputs feed a raw blended growth rate of {fmt_pct(latest_blended_growth)} before normalization ({blend_rule})."
            )
        st.caption(" ".join(explainer_bits))

        level_col, driver_col = st.columns(2)

        with level_col:
            st.markdown("##### Raw Levels + 3M Trend")
            booking_custom = levels_df[["bookings_3m_avg", "bookings_mom_change"]].to_numpy()
            revenue_custom = levels_df[["revenue_3m_avg", "revenue_mom_change"]].to_numpy()

            fig_levels = go.Figure()
            fig_levels.add_trace(
                go.Bar(
                    x=levels_df["year_month"],
                    y=levels_df["bookings_value"],
                    name="Monthly Bookings",
                    marker_color="#3b82f6",
                    marker_opacity=0.70,
                    customdata=booking_custom,
                    hovertemplate=(
                        "<b>%{x|%b %Y}</b><br>"
                        "Bookings: %{y:,.0f}<br>"
                        "3m avg: %{customdata[0]:,.1f}<br>"
                        "MoM change: %{customdata[1]:+,.0f}<extra></extra>"
                    ),
                )
            )
            fig_levels.add_trace(
                go.Scatter(
                    x=levels_df["year_month"],
                    y=levels_df["bookings_3m_avg"],
                    mode="lines",
                    name="Bookings 3m Avg",
                    line=dict(color="#93c5fd", width=2, dash="dash"),
                    hovertemplate="<b>%{x|%b %Y}</b><br>Bookings 3m avg: %{y:,.1f}<extra></extra>",
                )
            )
            fig_levels.add_trace(
                go.Scatter(
                    x=levels_df["year_month"],
                    y=levels_df["revenue_value"],
                    mode="lines+markers",
                    name="Monthly Revenue",
                    line=dict(color="#f0a500", width=3),
                    marker=dict(size=7),
                    yaxis="y2",
                    customdata=revenue_custom,
                    hovertemplate=(
                        "<b>%{x|%b %Y}</b><br>"
                        "Revenue: THB %{y:,.0f}<br>"
                        "3m avg: THB %{customdata[0]:,.0f}<br>"
                        "MoM change: THB %{customdata[1]:+,.0f}<extra></extra>"
                    ),
                )
            )
            fig_levels.add_trace(
                go.Scatter(
                    x=levels_df["year_month"],
                    y=levels_df["revenue_3m_avg"],
                    mode="lines",
                    name="Revenue 3m Avg",
                    line=dict(color="#f7d488", width=2, dash="dash"),
                    yaxis="y2",
                    hovertemplate="<b>%{x|%b %Y}</b><br>Revenue 3m avg: THB %{y:,.0f}<extra></extra>",
                )
            )
            fig_levels.update_layout(
                **BASE_LAYOUT,
                height=320,
                xaxis=dict(**CHART_THEME["xaxis"], title="Month", tickangle=-30),
                yaxis=dict(**CHART_THEME["yaxis"], title="Bookings"),
                yaxis2=dict(
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    title="Revenue (THB)",
                    color="#f0a500",
                ),
                legend=dict(orientation="h", y=1.02, x=0, font_size=10),
            )
            st.plotly_chart(fig_levels, width="stretch")

        with driver_col:
            st.markdown(f"##### Growth Signal Behind Score ({signal_label})")
            fig_driver = go.Figure()
            fig_driver.add_trace(
                go.Scatter(
                    x=levels_df["year_month"],
                    y=levels_df["booking_signal_growth"],
                    mode="lines+markers",
                    name=f"Bookings {signal_label} Growth",
                    line=dict(color="#3b82f6", width=2.5),
                    marker=dict(size=6),
                    hovertemplate="<b>%{x|%b %Y}</b><br>Bookings growth: %{y:.1%}<extra></extra>",
                )
            )
            fig_driver.add_trace(
                go.Scatter(
                    x=levels_df["year_month"],
                    y=levels_df["revenue_signal_growth"],
                    mode="lines+markers",
                    name=f"Revenue {signal_label} Growth",
                    line=dict(color="#f0a500", width=2.5),
                    marker=dict(size=6),
                    hovertemplate="<b>%{x|%b %Y}</b><br>Revenue growth: %{y:.1%}<extra></extra>",
                )
            )
            fig_driver.add_trace(
                go.Scatter(
                    x=levels_df["year_month"],
                    y=levels_df["growth_input_blended"],
                    mode="lines",
                    name="Blended Growth Input",
                    line=dict(color="#e5e7eb", width=2, dash="dash"),
                    hovertemplate="<b>%{x|%b %Y}</b><br>Blended growth input: %{y:.1%}<extra></extra>",
                )
            )
            fig_driver.add_trace(
                go.Scatter(
                    x=levels_df["year_month"],
                    y=numeric_series(levels_df, "score_growth"),
                    mode="lines+markers",
                    name="Growth Score",
                    line=dict(color="#2ecc71", width=3),
                    marker=dict(size=7),
                    yaxis="y2",
                    hovertemplate="<b>%{x|%b %Y}</b><br>Growth score: %{y:.2f}<extra></extra>",
                )
            )
            fig_driver.add_hline(y=0, line_dash="dash", line_color=SOFT_DIVIDER, line_width=1)
            fig_driver.update_layout(
                **BASE_LAYOUT,
                height=320,
                xaxis=dict(**CHART_THEME["xaxis"], title="Month", tickangle=-30),
                yaxis=dict(**CHART_THEME["yaxis"], title=f"{signal_label} Growth", tickformat=".0%"),
                yaxis2=dict(
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    title="Growth Score (0-1)",
                    color="#2ecc71",
                    range=[0, 1],
                ),
                legend=dict(orientation="h", y=1.02, x=0, font_size=10),
            )
            st.plotly_chart(fig_driver, width="stretch")

        st.caption(
            "Left: raw bookings and revenue levels, with 3-month smoothing to show whether the lift is sustained. "
            "Right: the actual rolling growth inputs behind `score_growth`. "
            "When YoY is available the score uses a 50/50 bookings-revenue composite and blends 70% YoY with 30% MoM before normalizing to a 0-1 score."
        )

    st.markdown("---")

    col2a, col2b = st.columns(2)

    with col2a:
        st.markdown("### Growth Signal Used (YoY vs MoM)")
        if "growth_signal_used" in latest_all.columns:
            sig_counts = latest_all["growth_signal_used"].value_counts()
            fig_sig = go.Figure(go.Bar(
                x=sig_counts.index, y=sig_counts.values,
                marker_color=["#f0a500" if x == "YoY" else "#3b82f6" for x in sig_counts.index],
                text=sig_counts.values, textposition="outside",
                textfont=dict(color=TEXT_COLOR),
                hovertemplate="%{x}: %{y} restaurants<extra></extra>",
            ))
            fig_sig.update_layout(**BASE_LAYOUT, height=240, showlegend=False,
                xaxis=dict(**CHART_THEME["xaxis"], title="Signal"),
                yaxis=dict(**CHART_THEME["yaxis"], title="Restaurants"))
            st.plotly_chart(fig_sig, width="stretch")
            st.caption("YoY = seasonality-adjusted (≥12 months + ≥2 valid YoY months in last 3m) | MoM = fallback")
        else:
            st.info("growth_signal_used column not found — re-run momentum_seasonality.ipynb.")

    with col2b:
        st.markdown("### MoM vs YoY Score Distribution")
        has_mom_sc = "score_growth_mom" in latest_all.columns
        has_yoy_sc = "score_growth_yoy" in latest_all.columns
        if has_mom_sc or has_yoy_sc:
            fig_dist = go.Figure()
            if has_mom_sc:
                fig_dist.add_trace(go.Histogram(
                    x=latest_all["score_growth_mom"].dropna(), nbinsx=25,
                    name="MoM score", marker_color="#3b82f6", opacity=0.65,
                    hovertemplate="MoM: %{x:.2f} | %{y} restaurants<extra></extra>"))
            if has_yoy_sc:
                fig_dist.add_trace(go.Histogram(
                    x=latest_all["score_growth_yoy"].dropna(), nbinsx=25,
                    name="YoY score", marker_color="#f0a500", opacity=0.65,
                    hovertemplate="YoY: %{x:.2f} | %{y} restaurants<extra></extra>"))
            fig_dist.update_layout(**BASE_LAYOUT, height=240, barmode="overlay",
                xaxis=dict(**CHART_THEME["xaxis"], title="Score (0–1)"),
                yaxis=dict(**CHART_THEME["yaxis"], title="Restaurants"),
                legend=dict(orientation="h", y=1.05, x=0, font_size=10))
            st.plotly_chart(fig_dist, width="stretch")
            st.caption("Where both overlap, YoY (amber) strips out seasonal timing effects.")
        else:
            # Fallback: stability chart
            if "growth_months" in priority_df.columns:
                stab = priority_df["growth_months"].value_counts().sort_index()
                n = len(stab)
                bar_colors = (["#e74c3c"] * max(0, n-2) + ["#e67e22", "#2ecc71"])[:n]
                fig_stab = go.Figure(go.Bar(
                    x=[f"{v}m" for v in stab.index], y=stab.values,
                    marker_color=bar_colors, text=stab.values, textposition="outside",
                    textfont=dict(color=TEXT_COLOR),
                    hovertemplate="Positive MoM growth %{x}: %{y} restaurants<extra></extra>"))
                fig_stab.update_layout(**BASE_LAYOUT, height=240, showlegend=False,
                    xaxis=dict(**CHART_THEME["xaxis"], title="Positive-growth months"),
                    yaxis=dict(**CHART_THEME["yaxis"], title="Restaurants"))
                st.plotly_chart(fig_stab, width="stretch")
                st.caption("Re-run momentum_seasonality.ipynb to see MoM vs YoY score distribution.")
            else:
                st.info("Stability data not found — run priority_scoring_seasonality.ipynb.")

    st.markdown("---")

    st.markdown("### Growth Heatmap — Top 20 Restaurants by Volume")

    top20_names = (
        latest_all.nlargest(20, "monthly_bookings")["name"].tolist()
        if "monthly_bookings" in latest_all.columns
        else latest_all["name"].tolist()[:20]
    )
    heatmap_df = momentum_df[momentum_df["name"].isin(top20_names)].copy()
    heatmap_df["ym_label"] = heatmap_df["year_month"].dt.strftime("%b %Y")
    all_months   = sorted(heatmap_df["year_month"].unique())
    month_labels = [pd.Timestamp(m).strftime("%b %Y") for m in all_months]

    def _make_heatmap(values_col):
        pv = heatmap_df.pivot_table(index="name", columns="ym_label",
                                    values=values_col, aggfunc="mean")
        pv = pv.reindex(columns=[m for m in month_labels if m in pv.columns])
        if pv.empty:
            return None
        fig = go.Figure(go.Heatmap(
            z=pv.values, x=pv.columns.tolist(), y=pv.index.tolist(),
            colorscale=[[0.0,"#e74c3c"],[0.4,"#1a1d27"],[0.6,"#1a1d27"],[1.0,"#2ecc71"]],
            zmid=0,
            text=[[f"{v:.0%}" if pd.notna(v) else "–" for v in row] for row in pv.values],
            texttemplate="%{text}", textfont=dict(size=9),
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1%}<extra></extra>",
            colorbar=dict(title=dict(text="Growth", font=dict(color=MUTED_TEXT)),
                          tickformat=".0%", tickfont=dict(color=MUTED_TEXT))))
        fig.update_layout(**BASE_LAYOUT, height=500,
            xaxis=dict(**CHART_THEME["xaxis"], tickangle=-45, tickfont=dict(size=10)),
            yaxis=dict(**CHART_THEME["yaxis"], tickfont=dict(size=10)))
        return fig

    has_mom_rolling = "booking_growth_mom_rolling" in heatmap_df.columns
    has_yoy_rolling = "booking_growth_yoy_rolling" in heatmap_df.columns

    if has_mom_rolling and has_yoy_rolling:
        ht1, ht2 = st.tabs(["📅 MoM Rolling (3m)", "📆 YoY Rolling (3m)"])
        with ht1:
            st.caption("Month-over-month 3m rolling — short-term acceleration")
            fig_hm = _make_heatmap("booking_growth_mom_rolling")
            if fig_hm:
                st.plotly_chart(fig_hm, width="stretch")
        with ht2:
            st.caption("Year-over-year 3m rolling — seasonality-adjusted (blank = insufficient prior-year data)")
            fig_hy = _make_heatmap("booking_growth_yoy_rolling")
            if fig_hy:
                st.plotly_chart(fig_hy, width="stretch")
            else:
                st.info("YoY rolling data not available — re-run momentum_seasonality.ipynb.")
    else:
        fig_hm = _make_heatmap("booking_growth_rolling")
        if fig_hm:
            st.plotly_chart(fig_hm, width="stretch")

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
        "monthly_gmv",
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
    if "monthly_gmv" in full_display.columns:
        full_display["monthly_gmv"] = full_display["monthly_gmv"].apply(
            lambda x: f"THB {x:,.0f}" if pd.notna(x) else "-"
        )
    full_display = full_display.rename(
        columns={
            "selected": "Selected",
            "name": "Restaurant",
            "momentum_segment": "Momentum Segment",
            "location": "Location",
            "monthly_bookings": "Monthly Bookings",
            "monthly_gmv": "Monthly Revenue",
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
