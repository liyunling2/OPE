import pandas as pd
import streamlit as st
import os
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False


def display_value(value):
    if value is None or pd.isna(value):
        return "Not available in data"
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "nan", "none", "-", "<na>"}:
        return "Not available in data"
    return text


def factual_segment(row: dict, hist: pd.DataFrame | None = None):
    candidates = [row.get("latest_segment"), row.get("segment")]
    if hist is not None and len(hist):
        latest_hist = hist.sort_values("year_month").iloc[-1]
        candidates.extend([latest_hist.get("latest_segment"), latest_hist.get("segment")])

    for value in candidates:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"unknown", "nan", "none", "-", "<na>"}:
            return text
    return None


def fmt_thb(val):
    if val is None or pd.isna(val):
        return "-"
    val = float(val)
    if abs(val) >= 1_000_000:
        return "%.1fM THB" % (val / 1_000_000)
    if abs(val) >= 1_000:
        return "%.0fK THB" % (val / 1_000)
    return "%.0f THB" % val


def fmt_pct(val):
    if val is None or pd.isna(val):
        return "-"
    return "%.1f%%" % (float(val) * 100)


def fmt_num(val, digits=3):
    if val is None or pd.isna(val):
        return "-"
    return f"{float(val):,.{digits}f}"


def fmt_int(val):
    if val is None or pd.isna(val):
        return "-"
    return f"{int(round(float(val))):,}"

## previous gemini
# @st.cache_data(ttl=300, show_spinner=False)
# def generate_ai_playbook(prompt: str, _prompt_hash=None) -> str:
#     try:
#         # 1. Initialize the new client with your API key
#         load_dotenv()  # Load environment variables from .env file
#         client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
#         # 2. Generate content using the new client.models syntax
#         response = client.models.generate_content(
#             model="gemini-2.5-flash",
#             contents=prompt,
#         )
        
#         # 3. Return the generated text
#         if response and response.text:
#             return response.text
#         return "_(No response generated)_"
        
#     except Exception as e:
#         return f"_(Playbook generation unavailable: {e})_"

# def call_gemini(prompt: str) -> str:
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         raise RuntimeError("GEMINI_API_KEY is not set.")

#     client = genai.Client(api_key=api_key)
#     response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
#     return response.text.strip()
    

def _table_for_prompt(table: pd.DataFrame, table_type: str, top_n: int = 5) -> str:
    if table is None or table.empty:
        return "- insufficient data"

    lines = []
    for _, rec in table.head(top_n).iterrows():
        if table_type == "ga":
            lines.append(
                "- Rank {rank}: {name} | GMV/GA={gmv} | Add-to-cart={atc} | View-to-purchase={v2p} | Google Ads score={score}".format(
                    rank=int(rec.get("rank", 0)),
                    name=rec.get("strategy", "-"),
                    gmv=fmt_thb(rec.get("gmv_per_ga")),
                    atc=fmt_pct(rec.get("add_to_cart")),
                    v2p=fmt_pct(rec.get("view_to_purchase")),
                    score=fmt_num(rec.get("ga_strategy_score"), 3),
                )
            )
        else:
            lines.append(
                "- Rank {rank}: {name} | count={count} | revenue uplift={rev} | booking uplift={book} | score={score}".format(
                    rank=int(rec.get("rank", 0)),
                    name=rec.get("strategy_name", "-"),
                    count=fmt_int(rec.get("count")),
                    rev=fmt_pct(rec.get("avg_revenue_uplift_pct")),
                    book=fmt_pct(rec.get("avg_bookings_uplift_pct")),
                    score=fmt_num(rec.get("marketing_strategy_score"), 3),
                )
            )
    return "\n".join(lines) if lines else "- insufficient data"


def _restaurant_ga_snapshot(row: dict, hist: pd.DataFrame) -> str:
    source = row.copy()
    if hist is not None and len(hist):
        latest = hist.sort_values("year_month").iloc[-1].to_dict()
        source.update({k: v for k, v in latest.items() if k not in source or pd.isna(source.get(k))})
    return "\n".join(
        [
            f"- Items viewed: {fmt_int(source.get('ga_items_viewed'))}",
            f"- GMV/GA view: {fmt_thb(source.get('gmv_per_ga_view'))}",
            f"- Add-to-cart rate: {fmt_pct(source.get('ga_add_to_cart_rate'))}",
            f"- View-to-purchase rate: {fmt_pct(source.get('ga_view_to_purchase_rate'))}",
            f"- GA revenue/view: {fmt_thb(source.get('ga_revenue_per_view'))}",
        ]
    )


def call_cohere(selected: str,
    row: dict,
    hist: pd.DataFrame,
    segment: str | None,
    cluster_id,
    cluster_label,
    ga_rankings:pd.DataFrame,
    momentum_df:pd.DataFrame,
    peerdata:pd.DataFrame
    ) -> str:
    try:
        import cohere

        load_dotenv()
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            return "_(Playbook generation unavailable: COHERE_API_KEY is not set.)_"

        co = cohere.ClientV2(api_key) 

        prompt = build_ai_prompt(
                    selected=selected,
                    row=row,
                    hist=hist,
                    segment=segment,
                    cluster_id=cluster_id,
                    cluster_label=cluster_label,
                    ga_rankings=ga_rankings,
                    momentum_df=momentum_df,
                    peerdata=peerdata
        )
        # Analyze the text without RAG
        response = co.chat(
            model="command-a-03-2025",
            messages=[
                {
                    "role": "user", 
                    "content": f"{prompt}"
                }
            ]
        )
        if response and response.message.content[0].text:
            return response.message.content[0].text
        return "_(No response generated)_"
        
    except Exception as e:
        return f"_(Playbook generation unavailable: {e})_"
    

def build_ai_prompt(
    selected: str,
    row: dict,
    hist: pd.DataFrame,
    segment: str | None,
    cluster_id,
    cluster_label,
    ga_rankings:pd.DataFrame,
    momentum_df:pd.DataFrame,
    peerdata:pd.DataFrame
) -> str:
    priority_score = fmt_num(pd.to_numeric(pd.Series([row.get("priority_score")]), errors="coerce").iloc[0], 1)
    priority_tier = display_value(row.get("priority_tier"))

    return f"""
You are a senior marketing strategist at a restaurant reservation platform. Your job is to write a precise,
data-grounded strategy brief for ONE restaurant. Every claim must be traceable to a
specific number in the data below. Vague advice is a failure.
 
# ABSOLUTE RULES
1. USE ONLY the data provided. Do not invent benchmarks, percentages, or outcomes.
2. Every identified issue MUST include:
   (a) the restaurant's actual metric value
   (b) the cluster/segment/global benchmark it is being compared to
   (c) the quantified gap (pp or absolute)
   (d) WHY this gap is the priority — not just that it exists
   Comparison rule: compute every gap as ACTUAL - BENCHMARK. If the gap is positive
   or zero, the restaurant is at/above benchmark and that metric MUST NOT be described
   as "low", "below benchmark", "weak", "underperforming", or an issue. For percentage
   metrics, report the gap in percentage points. Example: 7.3% vs 6.5467% is +0.7533pp,
   so add-to-cart is above benchmark and cannot be an issue.
3. If multiple issues exist, RANK them by business impact (revenue loss > conversion
   gap > traffic gap) and state the ranking with justification.
4. The strategy for each issue must be directly traceable to the evidence:
   - Google Ads strategy → cite specific campaign type from Google Ads ranking table
   - CRM/KOL/FB action → cite specific strategy from marketing ranking table
   - Package choice → link to funnel stage the package addresses
5. Peer citations must name the specific strategy and its outcome metrics.
6. If data is missing for a metric, say "insufficient data" and exclude it from
   the issue ranking — do NOT estimate or assume.
7. Use bullet points only. One finding per bullet. No paragraphs.
8. Do not repeat the same metric twice across different sections.
9. PEER DATA must be used as citation evidence in the CRM / KOL / FB strategy:
   - Identify the top 1-2 performing peer restaurants from PEER RECOMMENDER'S DATA
     based on their revenue uplift or booking uplift metrics.
   - Name the peer restaurant and the specific campaign/strategy it ran.
   - Cite its exact outcome metrics (revenue uplift %, booking uplift %).
   - Use this as proof that the recommended strategy works for similar restaurants
     in the same cluster or segment.
   - Do NOT recommend the peer's strategy if it contradicts the issue identified —
     only cite peers whose campaigns address the same funnel gap.
# DATA
RESTAURANT IDENTITY
Restaurant:     {selected}
Cluster: {cluster_id} - {display_value(cluster_label)}
Segment: {display_value(segment)}
Priority Score: {priority_score}
Priority Tier: {priority_tier}
 
RESTAURANT METRICS (latest available)
GOOGLE ADS FUNNEL:
{_restaurant_ga_snapshot(row, hist)}

RESTAURANT CLUSTER COMPARISON (raw dataframe — do not modify, do not recalculate):
{ga_rankings}

RESTAURANT BOOKING HISTORY (raw dataframe — do not filter, do not aggregate, do not transform):
{momentum_df}

PEER RECOMMENDER'S DATA — restaurants in the same cluster with known campaign outcomes.
Use to identify the top 1-2 performing peers and cite their campaigns as evidence
in the CRM / KOL / FB strategy. Do not use as a benchmark table for gap analysis.
(raw dataframe — do not filter, do not aggregate, do not transform):
{peerdata}

AVAILABLE PACKAGES
1. BASIC   — Awareness Starter. Key tools: Blogger 20k×1/30k×1, Boost 2K THB, Pop-up Banner,
          Homepage guarantee, Line@, Push. Revenue guarantee: 30K THB / 90 days.
          Use when: traffic is the primary gap.
 
2. STANDARD — Growth & Conversion. Key tools: Blogger 50k×1/30k×1, TikTok/Reels VDO,
           Web Footer 2 days, Boost 3K THB, Line@, Push. Revenue guarantee: 40K THB / 120 days.
           Use when: traffic exists but add-to-cart or view-to-purchase is weak.
 
3. PREMIUM — High Impact. Key tools: Blogger 50k×2, TikTok/Reels VDO, Promotion Banner 1 week,
          Blog Advertorial, Boost 5K THB, Line@, Push. Revenue guarantee: 100K THB / 180 days.
          Use when: high priority tier AND both scale and conversion need addressing.
 

# REQUIRED OUTPUT FORMAT — follow EXACTLY, no deviations
> Return the response ONLY in valid JSON format. Do not give any comments to note..any reasoning must go into the json
> Arrays must always be returned, even if only one item exists 
> For EACH issue found, use the sub-structure below.
> If there is more than one issue, number them and rank by business impact.
> The most impactful issue (highest revenue consequence) must be Issue #1.


The JSON structure must strictly follow this schema:
{{
  "issues": [
    {{
      "issue_no": "integer",
      "description": "string"
    }}
  ],

  "strategy": [
    {{
      "title": "Overview",
      "description": "string"
    }},
    {{
      "title":       "Google Ads strategy",
      "description": "string"    // Google Ads actions aligned to funnel gaps
    }},
    {{
      "title":       "CRM / KOL / FB strategy",
      "description": "string"    // CRM, KOL, and Facebook actions with peer evidence
    }}
  ]
}}

## Instructions for each section:
1. issues
- Identify the restaurant's key business or funnel problems using the provided metrics, benchmarks, gaps, booking trends, GMV performance, risk signals, and peer comparisons. 
- Identify a maximum of 3 issues only.
- Rank issues by business impact:
  - Issue #1 must represent the highest revenue or conversion impact.
  - Lower-ranked issues should have smaller or secondary impact.
- Each issue description must:
  - Clearly explain the root problem.
  - Reference supporting evidence from the provided data.
  - Mention the affected funnel stage where relevant (traffic, add-to-cart, view-to-purchase).
  - Explain why the issue matters commercially.
  - Only describe a metric as a weakness when the actual value is lower than the stated benchmark.
    If actual is higher than benchmark, either call it a strength or omit it from issues.
- Keep descriptions concise but data-driven.
- Do not invent metrics or unsupported claims.

2. strategy
The strategy array must always contain exactly 3 objects in the fixed order below.
Do not rename titles, reorder entries, or use bullet points inside JSON string values.
All descriptions must be concise, data-driven paragraphs. Do not invent metrics or unsupported claims.

2a. "Overview"
- Write a single consolidated executive paragraph that synthesises the identified issues into one coherent growth narrative.
- The overview must:
    - Summarise the restaurant's primary growth challenge and the funnel stages most at risk.
    - Explain the overall strategic direction and why it is appropriate for this restaurant's segment or cluster.
    - Connect the Google Ads and CRM/KOL/FB strategies into a unified recommendation.
    - Directly describe the motivation of the strategy to address the issues identified
    - State the expected commercial outcome (bookings, GMV, conversion, or traffic quality improvement).
    - Do not repeat raw issue descriptions verbatim; synthesise and elevate into an executive recommendation.

2b. "Google Ads strategy"
- Write a concise paragraph covering all Google Ads campaign actions.
- The description must:
- Name the campaign type from the Google Ads ranking table (cluster scope preferred).
- State which funnel stage is targeted (Traffic / Add-to-cart / View-to-purchase).
- Cite the exact Score, Add-to-cart rate, or View-to-purchase rate as supporting evidence.
- Explain which funnel metric the campaign is expected to move and why.
- Note why this campaign type is suitable for the restaurant's segment or cluster.

2c. "CRM / KOL / FB strategy"
- Write a concise paragraph covering retention, influencer, and paid social actions.
- The description must:
    - Name the strategy from the marketing ranking table.
    - Cite exact Revenue uplift and Booking uplift figures from the table.
    - Reference a specific peer restaurant by name from PEER RECOMMENDER'S DATA.
        State the campaign/strategy it ran and cite its exact revenue uplift and booking
        uplift figures from that dataset. Only cite a peer whose campaign addresses the
        same funnel gap identified in the issues — do not cite a peer just because they
        performed well overall.
    - Explain which funnel gap the strategy closes.
    - Include one sentence on messaging direction — what the content or campaign must communicate to address the root issue.
"""


