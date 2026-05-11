import pandas as pd
import plotly.graph_objects as go
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
from theme import CHART_THEME, MUTED_TEXT, SOFT_DIVIDER


_CSS = """
<style>

.ov-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

.ov-step-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.ov-step-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 4px;
    background: linear-gradient(90deg, #cc0000, #ef4444);
    border-radius: 16px 16px 0 0;
}
.ov-step-num {
    font-weight: 700;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 0.5rem;
}
.ov-step-title {
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.6rem;
    line-height: 1.25;
}
.ov-step-desc {
    font-size: 0.88rem;
    color: var(--muted);
    line-height: 1.65;
    margin: 0 0 1rem;
}
.ov-tag-row { display: flex; flex-wrap: wrap; gap: 6px; }
.ov-tag {
    padding: 3px 10px;
    border-radius: 6px;
    white-space: nowrap;
}
.ov-tag-red  { background: #fff0ee; color: #993c1d; border: 1px solid #f0997b; }
.ov-tag-blue { background: #e6f1fb; color: #185fa5; border: 1px solid #85b7eb; }
.ov-tag-teal { background: #e1f5ee; color: #0f6e56; border: 1px solid #5dcaa5; }

.ov-howto-item {
    display: flex;
    gap: 1.1rem;
    align-items: flex-start;
    padding: 1.1rem 0;
    border-bottom: 1px solid var(--border);
}
.ov-howto-item:last-child { border-bottom: none; }
.ov-howto-num {
    flex: 0 0 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--accent);
    color: #fff;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 2px;
    flex-shrink: 0;
}
.ov-howto-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 4px;
}
.ov-howto-desc {
    font-size: 0.87rem;
    color: var(--muted);
    line-height: 1.6;
    margin: 0;
}

.ov-glossary {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
}
.ov-glossary-row {
    display: flex;
    gap: 1rem;
    padding: 0.85rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.87rem;
    align-items: flex-start;
}
.ov-glossary-row:last-child { border-bottom: none; }
.ov-glossary-term {
    flex: 0 0 190px;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.4;
}
.ov-glossary-def {
    color: var(--muted);
    line-height: 1.6;
}

</style>
"""


def render() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    # STEP CARDS
    st.markdown('### Our Approach ')
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown(
            """
            <div class="ov-step-card">
                <p class="ov-step-num">Step 1</p>
                <p class="ov-step-title">Rank high-potential restaurants</p>
                <p class="ov-step-desc">
                    Three engines work together to identify and rank which restaurants
                    deserve attention and investment based on momentum, peer benchmarking,
                    and marketing responsiveness.
                </p>
                <div class="ov-tag-row">
                    <span class="ov-tag ov-tag-red">Momentum Engine</span>
                    <span class="ov-tag ov-tag-blue">Clustering</span>
                    <span class="ov-tag ov-tag-teal">Priority Scoring</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="ov-step-card">
                <p class="ov-step-num">Step 2</p>
                <p class="ov-step-title">Define marketing strategy</p>
                <p class="ov-step-desc">
                    Measure how marketing activities translate to bookings, then get
                    tailored channel-level strategy recommendations for each restaurant
                    to maximise ROI.
                </p>
                <div class="ov-tag-row">
                    <span class="ov-tag ov-tag-red">Marketing Effectiveness</span>
                    <span class="ov-tag ov-tag-teal">Strategy Recommendations</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # HOW TO USE
    st.markdown('<hr class="ov-divider">', unsafe_allow_html=True)
    st.markdown('### A walkthrough on how to use this tool')

    steps = [
        (
            "Go to the Overview tab",
            "View how restaurants are prioritized based on key performance indicators such as Priority Score, GMV per GA View, and Number of Campaigns. This provides a quick snapshot of overall restaurant performance and opportunity size "
        ),
        (
            "Filter by segments or clusters",
            "Use the filter bar at the top of the page to narrow results by segment or cluster type. Applied filters persist across all pages, allowing you to maintain context while exploring different views.",
        ),
        (
            "Inspect a restaurant's detail view in the clustering tab",
            "Select a restaurant in the Clustering tab to explore its detailed profile, including cluster classification, behavioral patterns, and performance characteristics. Use these insights to better understand the restaurant’s positioning and potential growth drivers."
        ),
        (
            "Review the Strategy tab",
            "Generate a diagnosis on the main issues the restaurant is facing based on their metrics. Formulate a data-driven targeted strategy based on their metrics compared against clusters, segment, peers performance.",
        ),
        (
            "Data driven actions",
            "Use the insights gathered across the dashboard to design and tailor marketing strategies for specific restaurants, enabling more focused outreach and higher-impact campaigns."
        ),
    ]

    rows = "".join(
        f"""
        <div class="ov-howto-item">
            <div class="ov-howto-num">{i}</div>
            <div>
                <p class="ov-howto-title">{title}</p>
                <p class="ov-howto-desc">{desc}</p>
            </div>
        </div>
        """
        for i, (title, desc) in enumerate(steps, 1)
    )
    st.markdown(rows, unsafe_allow_html=True)


    st.markdown('<hr class="ov-divider">', unsafe_allow_html=True)

   
    terms = [
        ("Momentum Engine",
        "Algorithm that detects restaurants with accelerating booking and revenue trends over a given time window."),
        ("Clustering",
        "Machine learning grouping that places restaurants with similar characteristics "
        "(cuisine, location, size, price point) into comparable peer groups."),
        ("Priority Score",
        "A composite 0–100 score combining momentum, cluster rank, and marketing "
        "responsiveness used to shortlist and rank restaurants."),
        ("Marketing Effectiveness",
        "A measure of how much incremental booking uplift is attributable to specific "
        "marketing channel activities for each restaurant."),
        ("Strategy Recommendation",
        "Tailored, restaurant-level playbook suggesting which marketing channels and "
        "tactics are most likely to increase bookings and ROI."),
        ("Behavioural signals",
        "Engagement data including page views, search impressions, review activity, and "
        "time-on-page that indicate consumer interest in a restaurant."),
        ("Growth signal (YoY / MoM)",
        "The comparison window used to measure momentum — Year-on-Year (YoY) for "
        "long-term trends, Month-on-Month (MoM) for short-term acceleration."),
        ("Seasonal restaurant",
        "A restaurant whose booking volume shows a statistically significant periodic "
        "pattern tied to time of year, flagged by the momentum engine."),
    ]

    df = pd.DataFrame(terms, columns=["Term", "Definition"])

    st.markdown("### Key terms")
    st.dataframe(df, width="stretch", hide_index=True)
