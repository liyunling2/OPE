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


def call_cohere(selected: str, row: dict, hist: pd.DataFrame, segment: str, cluster_id: int, cluster_label: str, ga_cluster_table: list , ga_segment_table: list, ga_global_table: list, m_cluster_table: list, m_segment_table: list, m_global_table: list ) -> str:
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
                    ga_cluster_table=ga_cluster_table,
                    ga_segment_table=ga_segment_table,
                    ga_global_table=ga_global_table,
                    m_cluster_table=m_cluster_table,
                    m_segment_table=m_segment_table,
                    m_global_table=m_global_table,
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
    ga_cluster_table: pd.DataFrame,
    ga_segment_table: pd.DataFrame,
    ga_global_table: pd.DataFrame,
    m_cluster_table: pd.DataFrame,
    m_segment_table: pd.DataFrame,
    m_global_table: pd.DataFrame,
) -> str:
    priority_score = fmt_num(pd.to_numeric(pd.Series([row.get("priority_score")]), errors="coerce").iloc[0], 1)
    preferred_channel = display_value(row.get("recommended_channel"))
    priority_tier = display_value(row.get("priority_tier"))

    return f"""
You are a senior marketing strategist advising a restaurant owner.

IMPORTANT RULES:
- Use ONLY the data provided below.
- DO NOT invent any numbers or assumptions.
- You MUST produce an answer; never return empty.
- If data is missing, say "insufficient data".
- Always mention the restaurant name at least once.
- Always reference REAL metrics from the GA and CRM/KOL/FB tables.
- Be specific to THIS restaurant; no generic advice.

STYLE RULES:
- Use bullet points only; no long paragraphs.
- Max 1-2 lines per point.
- Use simple language.
- Only metric names can be technical.

---------------------
RESTAURANT CONTEXT
---------------------
Restaurant: {selected}
Cluster: {cluster_id} - {display_value(cluster_label)}
Segment: {display_value(segment)}
Priority Score: {priority_score}
Priority Tier: {priority_tier}
Preferred Channel: {preferred_channel}

---------------------
RESTAURANT GA DIAGNOSTIC
---------------------
{_restaurant_ga_snapshot(row, hist)}

---------------------
GA STRATEGY RANKINGS
Formula: GA Strategy Score = (GMV/GA x 0.40) + (Add to Cart x 0.30) + (View to Purchase x 0.30)
Note: GA Count = estimated sessions allocated to the selected scope using scope GA-view share by month. It is not reused platform-wide sessions.
---------------------
Cluster Level:
{_table_for_prompt(ga_cluster_table, 'ga')}

Segment Level:
{_table_for_prompt(ga_segment_table, 'ga')}

Global Level:
{_table_for_prompt(ga_global_table, 'ga')}

---------------------
CRM / KOL / FB STRATEGY RANKINGS
Formula: Marketing Strategy Score = (Average Revenue Uplift x 0.60) + (Average Booking Uplift x 0.40)
---------------------
Cluster Level:
{_table_for_prompt(m_cluster_table, 'marketing')}

Segment Level:
{_table_for_prompt(m_segment_table, 'marketing')}

Global Level:
{_table_for_prompt(m_global_table, 'marketing')}

---------------------
AVAILABLE MARKETING PACKAGES (WITH CAPABILITIES)
---------------------

BASIC PACKAGE (Awareness Starter – Low Cost, Entry Level)
Purpose:
- Drive initial awareness and light engagement
- Suitable for low traffic restaurants

Key Capabilities:
- Revenue Guarantee (30K+ THB)
- Revenue Guarantee (90 Day)
- Send Blogger to Review x1 (20k followers)
- Send Blogger to Review x2 (30k followers)
- Boost post THB 2,000 Baht
- Pop-up Banner: Individual
- Photoshooting
- Guaranteed in Restaurants list home page
- HH Facebook Post : Individual post
- Line@ Broadcasts :  Individual post
- Push Notification

Limitations:
- No strong video content
- Limited reach and weak conversion tools

Use When:
- Traffic is low
- Need visibility, not deep conversion

STANDARD PACKAGE (Growth & Conversion – Balanced)
Purpose:
- Improve customer intent and conversion
- Best for mid-funnel problems (low add-to-cart, weak engagement)

Key Capabilities:
- Revenue Guarantee (40K+ THB)
- Revenue Guarantee (120 Day)
- Send Blogger to Review x1 (50k followers)
- Send Blogger to Review x2 (30k followers)
- Boost post THB 3,000 Baht
- Pop-up Banner: Individual
- Guaranteed in Restaurants list home page
- Photoshooting
- HH Facebook Post : Individual post
- Line@ Broadcasts :  Individual post
- Push Notification
- Web Footer (2 days)
- Tiktok VDO or Instagram Reels VDO

Strengths:
- Combines awareness + conversion tools
- Strong for improving menu appeal and intent

Use When:
- Traffic exists but conversion is weak
- Add-to-cart or engagement is low

PREMIUM PACKAGE (High Impact – Scale + Conversion)
Purpose:
- Maximise reach AND conversion at scale
- Suitable for high-priority restaurants

Key Capabilities:
- Revenue Guarantee (100K+ THB)
- Revenue Guarantee (180 Day)
- Send Blogger to Review x1 (50k followers)
- Send Blogger to Review x2 (50k followers)
- Boost post THB 5,000 Baht
- Pop-up Banner: Individual
- Guaranteed in Restaurants list home page
- Photoshooting
- HH Facebook Post : Individual post
- Line@ Broadcasts :  Individual post
- Push Notification
- Web Footer (2 days)
- Tiktok VDO or Instagram Reels VDO
- Guaranteed in Restaurants Promotion Banner (1 week)
- Blog: Advertorial: Individual

Strengths:
- Highest visibility + strongest conversion ecosystem
- Combines brand + performance marketing

Limitations:
- Highest cost → must justify ROI

Use When:
- High priority tier
- Need scale OR strong brand push
- Conversion AND reach both need improvement

---------------------
TASK
---------------------
Give a decisive, business-focused recommendation for THIS restaurant.

FORMAT STRICTLY:

## 1. Key Issue
- State the MAIN problem using GA and/or marketing ranking evidence.
- Include the metric, benchmark/context, and business pain.

## 2. Recommended Package
- Package: Basic / Standard / Premium
- Explain why this package fixes the exact funnel problem.
- Mention specific package features.
- Explain why cheaper option fails.
- Explain why more expensive option is unnecessary, unless Premium is chosen.

Decision rules:
- Mid-funnel issue, e.g. add-to-cart low → Standard.
- Lower-funnel issue, e.g. view-to-purchase low → Standard or Premium.
- Traffic issue → Basic or Standard.
- Only choose Premium if high priority tier AND scale or strong brand push is needed.

## 3. GA Support Strategy
- Choose up to 2 GA activities from the GA rankings.
- Explain what each improves: traffic, add-to-cart, view-to-purchase, or revenue per view.
- Explain what the GA metrics suggest about traffic quality, add-to-cart intent, view-to-purchase conversion, or revenue per view.

## 4. CRM / KOL / FB Priority Actions
- Choose EXACTLY 2 strategies from the CRM/KOL/FB rankings.
- For each: strategy name, concrete campaign idea, and what it improves.

## 5. Execution Plan
- Blogger Strategy:
- Content:
- Messaging:
- Paid + CRM:
- Funnel Impact:

## 6. Why This Will Work
- Reference GA data.
- Reference strategy evidence.
- Connect to outcome.

## 7. 30-Day Plan
- Launch:
- Optimise:
- Scale:

## 8. Expected Impact
- Translate into more orders, better conversion, or higher revenue per visitor.
- Do not promise exact results.
"""

