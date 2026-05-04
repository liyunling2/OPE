import pandas as pd
# This function builds a detailed prompt for generating a strategic recommendation based on 
# restaurant performance data, peer comparisons, and campaign recommendations. 
# It synthesizes all relevant information into a structured format for an LLM to produce a concise, 
# evidence-backed recommendation.
def build_recommendation_prompt(
    restaurant_name: str,
    momentum: str,
    theme: str,
    active_gmv: float,
    active_conv: float,
    top_recs: pd.DataFrame,
    top_peers: pd.DataFrame,
) -> str:

    peer_lines = [] # Peer evidence block
    for _, p in top_peers.iterrows(): # Extract peer stats with safe defaults
        seg = p.get("latest_segment", "Unknown")
        gmv = p.get("monthly_gmv", 0) or 0
        best_ch = p.get("best_channel", "Unknown")
        inc_rev = p.get("avg_incremental_rev", None)
        reliability = p.get("lift_reliability", None)
        n_camp = p.get("n_campaigns", 0) or 0

        inc_str = f"THB {inc_rev:,.0f} avg incremental rev" if pd.notna(inc_rev) else "no revenue data"
        rel_str = f"{reliability*100:.0f}% lift reliability" if pd.notna(reliability) else "unknown reliability"

        peer_lines.append( # Format peer line with all available data
            f"  - {p.get('name', 'Unknown')} | Segment: {seg} | GMV: THB {gmv:,.0f} "
            f"| Best Channel: {best_ch} | {n_camp} campaigns | {inc_str} | {rel_str}"
        )
    peers_str = "\n".join(peer_lines) if peer_lines else "  No peer data available."

    strat_lines = [] # Strategy evidence block, ranked by score with safe defaults
    for i, (_, r) in enumerate(top_recs.iterrows(), 1): # Extract strategy stats with safe defaults
        uplift = f"{r['med_rev_uplift']:.1%}" if pd.notna(r.get("med_rev_uplift")) else "no uplift data"
        roi = f"{r['med_roi']:.1f}x ROI" if pd.notna(r.get("med_roi")) else "no ROI data"
        peers_n = r.get("peers_using", 0) # Number of peers using this strategy, default to 0 if missing
        inc = r.get("total_incremental", None) # Total incremental revenue, if available
        inc_str = f"THB {inc:,.0f} total incremental rev" if pd.notna(inc) else ""
        strat_lines.append( # Format strategy line with all available data
            f"  {i}. Campaign: '{r['strategy_name']}' via {r['channel']} "
            f"— {uplift} revenue uplift, {roi}, used by {peers_n} peer(s). {inc_str}"
        )
    strats_str = "\n".join(strat_lines) if strat_lines else "  No strategy data available." # Handle case with no strategy data

    return f"""You are a senior marketing strategist for HungryHub, a restaurant booking platform in Southeast Asia.
You are analysing peer campaign data to produce a specific, evidence-backed recommendation.

## Target Restaurant
- Name: {restaurant_name}
- Momentum Segment: {momentum}
- Thematic Cluster: {theme}
- Monthly GMV: THB {active_gmv:,.0f}
- View-to-Purchase Conversion Rate: {active_conv:.2%}

## Similar High-Performing Peers
{peers_str}

## Peer-Validated Campaigns (ranked by score)
{strats_str}

## Instructions
Write a strategic recommendation in exactly 3 paragraphs. Be specific — reference actual campaign names, channel types, and uplift figures from the data above. Do not be generic.

Paragraph 1 — Situation: What does this restaurant's GMV and momentum segment tell us about where it sits in its growth journey? What does the peer data reveal about what is working in this competitive set?

Paragraph 2 — Primary Recommendation: Which specific campaign should this restaurant prioritise first, and why? Reference the exact campaign name, the uplift figure, how many peers used it, and what that implies about channel-market fit for this restaurant type.

Paragraph 3 — Sequencing & Risk: Suggest how to sequence the top 2–3 campaigns over the next quarter. Flag one concrete risk (e.g. negative uplift in a specific campaign, low peer sample size, or single-channel dependency) visible in the data.

Rules: maximum 220 words, professional tone, no bullet points, synthesise numbers into insight rather than listing them raw. Write in a concise manner that would make it easy to understand for business stakeholders.
"""