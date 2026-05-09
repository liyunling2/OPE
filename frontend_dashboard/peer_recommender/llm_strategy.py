import pandas as pd
import streamlit as st
import os
from google import genai
from dotenv import load_dotenv
import cohere


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
                "- Rank {rank}: {name} | GMV/GA={gmv} | Add-to-cart={atc} | View-to-purchase={v2p} | GA score={score}".format(
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
    momentum_df:pd.DataFrame) -> str:
    try:
        api_key = os.getenv("COHERE_API_KEY")

        co = cohere.ClientV2(api_key) 

        prompt = build_ai_prompt(
                    selected=selected,
                    row=row,
                    hist=hist,
                    segment=segment,
                    cluster_id=cluster_id,
                    cluster_label=cluster_label,
                    ga_rankings=ga_rankings,
                    momentum_df=momentum_df
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
    momentum_df:pd.DataFrame
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
3. If multiple issues exist, RANK them by business impact (revenue loss > conversion
   gap > traffic gap) and state the ranking with justification.
4. The strategy for each issue must be directly traceable to the evidence:
   - GA campaign type strategy → cite specific campaign type from GA ranking table
   - CRM/KOL/FB action → cite specific strategy from marketing ranking table
   - Package choice → link to funnel stage the package addresses
5. Peer citations must name the specific strategy and its outcome metrics.
6. If data is missing for a metric, say "insufficient data" and exclude it from
   the issue ranking — do NOT estimate or assume.
7. Use bullet points only. One finding per bullet. No paragraphs.
8. Do not repeat the same metric twice across different sections.
 
# DATA
RESTAURANT IDENTITY
Restaurant:     {selected}
Cluster: {cluster_id} - {display_value(cluster_label)}
Segment: {display_value(segment)}
Priority Score: {priority_score}
Priority Tier: {priority_tier}
 
 
RESTAURANT METRICS (latest available)
GA FUNNEL:
{_restaurant_ga_snapshot(row, hist)}


RESTAURANT CLUSTER COMPARISON (raw dataframe — do not modify, do not recalculate):
{ga_rankings}

RESTAURANT BOOKING HISTORY (raw dataframe — do not filter, do not aggregate, do not transform):
{momentum_df}


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

  "strategy": "string",

  "package_recommendation_summary": {{
    "recommended_package": "string",
    "rationale": "string"
  }}
}}

## Instructions for each section:
1. issues
- Identify the restaurant's key business or funnel problems using the provided metrics, benchmarks, gaps, booking trends, GMV performance, risk signals, and peer comparisons.
- Rank issues by business impact:
  - Issue #1 must represent the highest revenue or conversion impact.
  - Lower-ranked issues should have smaller or secondary impact.
- Each issue description must:
  - Clearly explain the root problem.
  - Reference supporting evidence from the provided data.
  - Mention the affected funnel stage where relevant (traffic, add-to-cart, view-to-purchase).
  - Explain why the issue matters commercially.
- Keep descriptions concise but data-driven.
- Do not invent metrics or unsupported claims.

2. strategy
- Provide a single consolidated strategic recommendation covering the identified issues.
- The strategy should:
  - Explain what actions should be taken.
  - Reference relevant GA campaign types, CRM strategies, KOL activities, or Facebook actions when applicable.
  - Align recommendations to the identified funnel gaps.
  - Mention why the strategy is suitable for this restaurant segment or cluster.
  - Focus on practical growth actions that improve bookings, GMV, conversion, or traffic quality.
- Write as a concise executive recommendation paragraph.

3. package_recommendation_summary
- Recommend ONLY one package:
  - Basic
  - Standard
  - Premium
- The recommendation must align with the restaurant's highest-priority issue and required growth support.
- Prefer:
  - Basic for smaller or lower-risk growth gaps.
  - Standard for moderate funnel or acquisition gaps.
  - Premium for major conversion, retention, or revenue recovery problems requiring stronger CRM/retargeting support.
"""
