# -*- coding: utf-8 -*-
import os
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data.loader import (
    load_priority,
    load_momentum,
    get_restaurant_history,
    get_restaurant_priority_row,
    recommend_strategies_for_restaurant,
)
from theme import AXIS, BORDER_COLOR, MUTED_TEXT, SURFACE_COLOR, TEXT_COLOR
from google import genai
from dotenv import load_dotenv
load_dotenv()

def layout(height=300, **kwargs):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, family="DM Sans"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=height,
    )
    base.update(kwargs)
    return base


def fmt_thb(val):
    if pd.isna(val): return "-"
    if val >= 1_000_000: return "%.1fM THB" % (val / 1_000_000)
    if val >= 1_000:     return "%.0fK THB" % (val / 1_000)
    return "%.0f THB" % val

def fmt_pct(val):
    if val is None or pd.isna(val): return "No data"
    return "%.1f%%" % (val * 100)

def fmt_ratio(val, digits=3):
    if val is None or pd.isna(val): return "-"
    return f"{float(val):,.{digits}f}"

def fmt_int(val):
    if val is None or pd.isna(val): return "-"
    return f"{int(round(float(val))):,}"

def fmt_month(val):
    if val is None or pd.isna(val): return "-"
    return pd.to_datetime(val).strftime("%b %Y")


def clean_tier_label(tier: str) -> str:
    """Strip emoji characters and mojibake prefixes from tier label strings.
    priority_list.csv may contain emoji-prefixed tiers (e.g. emoji-prefixed labels)
    which render as mojibake (ðŸ"´) when the CSV is written/read with mismatched
    encoding on Windows. We strip everything up to and including the first
    space-preceded word boundary so only the readable text remains.
    """
    import re
    s = str(tier).strip()
    # Remove any leading non-ASCII characters and surrounding whitespace
    s = re.sub(r'^[^\x00-\x7F\s]+\s*', '', s)
    # Also catch common mojibake prefixes like ðŸ"´, ðŸŸ , ðŸŸ¡
    s = re.sub(r'^[Ã°Å¸â€œÂ´\s\x80-\xff]+\s*', '', s)
    return s.strip()




def factual_segment(row: dict, hist: pd.DataFrame | None = None):
    """Return the segment only if it exists in the data. Do not invent 'Unknown'."""
    candidates = [
        row.get("latest_segment"),
        row.get("segment"),
    ]
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

def display_value(value):
    """Display only real values; otherwise show a clear data-unavailable marker."""
    if value is None or pd.isna(value):
        return "Not available in data"
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "nan", "none", "-", "<na>"}:
        return "Not available in data"
    return text


def _to_float(val):
    num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
    if pd.isna(num):
        return None
    return float(num)


def _safe_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _latest_metric(hist: pd.DataFrame, column: str, fallback=None):
    if column in hist.columns:
        series = pd.to_numeric(hist.sort_values("year_month")[column], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[-1])
    return _to_float(fallback)


def _recent_metric_avg(hist: pd.DataFrame, column: str, periods: int = 3):
    if column not in hist.columns:
        return None
    series = pd.to_numeric(hist.sort_values("year_month")[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.tail(periods).mean())


def _metric_benchmark(value, series: pd.Series) -> dict:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"median": None, "p75": None, "percentile": None, "label": "No benchmark", "delta_vs_median": None}

    median = float(clean.median())
    p75 = float(clean.quantile(0.75))
    percentile = None
    if value is not None:
        percentile = float((clean <= value).mean() * 100)

    if percentile is None:
        label = "No benchmark"
    elif percentile >= 75:
        label = "Top quartile"
    elif percentile >= 50:
        label = "Above median"
    elif percentile >= 25:
        label = "Below median"
    else:
        label = "Bottom quartile"

    delta_vs_median = None
    if value is not None and median not in (None, 0):
        delta_vs_median = (float(value) / median) - 1

    return {
        "median": median,
        "p75": p75,
        "percentile": percentile,
        "label": label,
        "delta_vs_median": delta_vs_median,
    }


def _target_value(current, floor=None, growth: float = 0.10, protect_floor: bool = False):
    candidates = []
    if current is not None:
        candidates.append(current * (0.95 if protect_floor else (1 + growth)))
    if floor is not None:
        candidates.append(floor)
    if not candidates:
        return None
    return max(candidates)


def _format_target_value(value, formatter: str) -> str:
    if formatter == "thb":
        return fmt_thb(value)
    if formatter == "pct":
        return fmt_pct(value)
    if formatter == "int":
        return fmt_int(value)
    return fmt_ratio(value)


def build_ga_context(row: dict, hist: pd.DataFrame, priority_df: pd.DataFrame) -> dict:
    hist = hist.sort_values("year_month").copy()
    ga_hist = hist.copy()
    if "ga_items_viewed" in ga_hist.columns:
        ga_hist = ga_hist[pd.to_numeric(ga_hist["ga_items_viewed"], errors="coerce").fillna(0) > 0]

    latest_ga_month = row.get("ga_data_month")
    if len(ga_hist) and "year_month" in ga_hist.columns:
        latest_ga_month = ga_hist["year_month"].iloc[-1]

    metrics = [
        "gmv_per_ga_view",
        "bookings_per_ga_view",
        "ga_add_to_cart_rate",
        "ga_view_to_purchase_rate",
        "ga_purchase_to_cart_rate",
        "ga_revenue_per_view",
        "ga_items_viewed",
    ]

    out = {
        "has_ga_data": bool(row.get("has_ga_data", False)),
        "latest_month": latest_ga_month,
        "benchmarks": {},
    }
    for metric in metrics:
        latest_value = _latest_metric(ga_hist, metric, row.get(metric))
        avg_3m = _recent_metric_avg(ga_hist, metric)
        benchmark = _metric_benchmark(latest_value, _safe_numeric_series(priority_df, metric))
        out[metric] = latest_value
        out[f"{metric}_3m_avg"] = avg_3m
        out["benchmarks"][metric] = benchmark

    out["has_ga_data"] = out["has_ga_data"] or any(out.get(metric) is not None for metric in metrics)

    if not out["has_ga_data"]:
        out["primary_signal_code"] = "no_ga_data"
        out["primary_signal_short"] = "No GA diagnostic"
        out["primary_signal"] = "GA metrics are not available for this restaurant yet."
        out["strategy_stance"] = "Lean on historical lift evidence until GA tracking is available."
        out["kpi_targets"] = []
        return out

    gmv = out.get("gmv_per_ga_view")
    views = out.get("ga_items_viewed")
    atc = out.get("ga_add_to_cart_rate")
    v2p = out.get("ga_view_to_purchase_rate")
    p2c = out.get("ga_purchase_to_cart_rate")
    rev_per_view = out.get("ga_revenue_per_view")
    bookings_per_view = out.get("bookings_per_ga_view")

    gmv_b = out["benchmarks"]["gmv_per_ga_view"]
    views_b = out["benchmarks"]["ga_items_viewed"]
    atc_b = out["benchmarks"]["ga_add_to_cart_rate"]
    v2p_b = out["benchmarks"]["ga_view_to_purchase_rate"]
    p2c_b = out["benchmarks"]["ga_purchase_to_cart_rate"]

    high_gmv = gmv is not None and gmv_b["p75"] is not None and gmv >= gmv_b["p75"]
    low_gmv = gmv is not None and gmv_b["median"] is not None and gmv < gmv_b["median"]
    high_views = views is not None and views_b["p75"] is not None and views >= views_b["p75"]
    low_views = views is not None and views_b["median"] is not None and views < views_b["median"]
    high_atc = atc is not None and atc_b["p75"] is not None and atc >= atc_b["p75"]
    low_atc = atc is not None and atc_b["median"] is not None and atc < atc_b["median"]
    high_v2p = v2p is not None and v2p_b["p75"] is not None and v2p >= v2p_b["p75"]
    low_v2p = v2p is not None and v2p_b["median"] is not None and v2p < v2p_b["median"]
    low_p2c = p2c is not None and p2c_b["median"] is not None and p2c < p2c_b["median"]

    if high_gmv and low_views:
        out["primary_signal_code"] = "scale_efficient_demand"
        out["primary_signal_short"] = "Scale qualified traffic"
        out["primary_signal"] = (
            "GMV per GA view is already strong, but traffic volume is still below the portfolio median. "
            "Scale qualified views without diluting conversion quality."
        )
        out["strategy_stance"] = (
            "Favor reach and prospecting tactics that bring in more qualified traffic, while using GMV per GA view as the guardrail."
        )
    elif high_atc and (low_v2p or low_p2c):
        out["primary_signal_code"] = "convert_existing_intent"
        out["primary_signal_short"] = "Close lower-funnel leak"
        out["primary_signal"] = (
            "Users are showing intent in GA, but too few of those views are converting into purchases. "
            "Retargeting, lifecycle, and offer sequencing should focus on finishing the purchase."
        )
        out["strategy_stance"] = (
            "Prioritize lower-funnel conversion tactics that recover existing demand before paying for more top-funnel traffic."
        )
    elif high_views and low_atc:
        out["primary_signal_code"] = "improve_pre_cart_message"
        out["primary_signal_short"] = "Lift view-to-cart intent"
        out["primary_signal"] = (
            "Traffic volume is present, but the menu or offer is not converting views into adds to cart strongly enough. "
            "Creative, proposition, and landing-page quality need tightening."
        )
        out["strategy_stance"] = (
            "Use creative and merchandising tactics to improve intent from existing traffic before broadening reach."
        )
    elif low_gmv and low_v2p:
        out["primary_signal_code"] = "repair_conversion_efficiency"
        out["primary_signal_short"] = "Repair conversion efficiency"
        out["primary_signal"] = (
            "Both monetization per GA view and purchase conversion are below the portfolio median. "
            "This is a lower-funnel efficiency issue, not just a traffic scale issue."
        )
        out["strategy_stance"] = (
            "Bias toward tighter conversion, retargeting, and offer control tactics until GMV per GA view normalizes."
        )
    elif high_gmv and high_v2p:
        out["primary_signal_code"] = "protect_efficiency_then_scale"
        out["primary_signal_short"] = "Protect efficiency, then scale"
        out["primary_signal"] = (
            "This restaurant is monetizing GA traffic well and converting strongly. "
            "It can scale, but only if the channel mix preserves the current efficiency profile."
        )
        out["strategy_stance"] = (
            "Scale carefully with quality controls and keep GMV per GA view above the current top-half benchmark."
        )
    else:
        out["primary_signal_code"] = "balanced_ga_profile"
        out["primary_signal_short"] = None
        out["primary_signal"] = (
            "GA metrics are not flashing a single acute leak. "
            "Focus on incremental efficiency gains while leaning on the strongest historical strategy evidence."
        )
        out["strategy_stance"] = (
            "Use the best historical playbooks, but keep GMV per GA view and view-to-purchase moving up together."
        )

    targets = []

    def add_target(label: str, value, formatter: str, why: str):
        if value is None:
            return
        targets.append({"label": label, "value": value, "formatter": formatter, "why": why})

    code = out["primary_signal_code"]
    if code == "scale_efficient_demand":
        add_target("GA item views", _target_value(views, floor=views_b["median"], growth=0.20), "int", "Scale qualified traffic volume beyond the portfolio median.")
        add_target("GMV / GA view", _target_value(gmv, floor=gmv_b["median"], protect_floor=True), "thb", "Protect monetization efficiency while traffic scales.")
        add_target("View to purchase", _target_value(v2p, floor=v2p_b["median"], growth=0.05), "pct", "Keep conversion quality intact as reach expands.")
    elif code == "convert_existing_intent":
        add_target("View to purchase", _target_value(v2p, floor=v2p_b["median"], growth=0.15), "pct", "Convert existing interest into completed purchases.")
        add_target("Purchase / cart", _target_value(p2c, floor=p2c_b["median"], growth=0.10), "pct", "Reduce checkout drop-off once users add to cart.")
        add_target("GMV / GA view", _target_value(gmv, floor=gmv_b["median"], growth=0.10), "thb", "Raise monetization from current traffic before scaling reach.")
    elif code == "improve_pre_cart_message":
        add_target("Add to cart rate", _target_value(atc, floor=atc_b["median"], growth=0.15), "pct", "Improve menu and offer relevance from existing traffic.")
        add_target("GMV / GA view", _target_value(gmv, floor=gmv_b["median"], growth=0.10), "thb", "Translate stronger intent into better value per visit.")
        add_target("GA revenue / view", _target_value(rev_per_view, floor=None, growth=0.10), "thb", "Lift revenue productivity before buying more sessions.")
    elif code == "repair_conversion_efficiency":
        add_target("GMV / GA view", _target_value(gmv, floor=gmv_b["median"], growth=0.20), "thb", "Bring monetization efficiency back toward the portfolio middle.")
        add_target("View to purchase", _target_value(v2p, floor=v2p_b["median"], growth=0.20), "pct", "Repair lower-funnel conversion.")
        add_target("Bookings / GA view", _target_value(bookings_per_view, floor=out["benchmarks"]["bookings_per_ga_view"]["median"], growth=0.15), "ratio", "Get more bookings from each view.")
    elif code == "protect_efficiency_then_scale":
        add_target("GMV / GA view", _target_value(gmv, floor=gmv_b["p75"], protect_floor=True), "thb", "Stay in the top quartile while scaling.")
        add_target("GA item views", _target_value(views, floor=views_b["median"], growth=0.15), "int", "Add qualified traffic without weakening economics.")
        add_target("Bookings / GA view", _target_value(bookings_per_view, floor=out["benchmarks"]["bookings_per_ga_view"]["median"], protect_floor=True), "ratio", "Hold booking yield per view as budgets rise.")
    else:
        add_target("GMV / GA view", _target_value(gmv, floor=gmv_b["median"], growth=0.10), "thb", "Keep monetization per visit trending up.")
        add_target("Add to cart rate", _target_value(atc, floor=atc_b["median"], growth=0.10), "pct", "Improve commercial intent from traffic already on page.")
        add_target("GA item views", _target_value(views, floor=views_b["median"], growth=0.10), "int", "Add qualified demand without breaking conversion.")

    out["kpi_targets"] = targets
    return out


def _evidence_level(activities, restaurants) -> str:
    """Evidence label based only on observed sample coverage."""
    acts = pd.to_numeric(pd.Series([activities]), errors="coerce").iloc[0]
    rests = pd.to_numeric(pd.Series([restaurants]), errors="coerce").iloc[0]
    acts = 0 if pd.isna(acts) else float(acts)
    rests = 0 if pd.isna(rests) else float(rests)

    if acts >= 12 and rests >= 6:
        return "High"
    if acts >= 6 and rests >= 3:
        return "Medium"
    return "Low"


def rank_strategies_factually(recs: pd.DataFrame, min_sample_size: int = 3) -> pd.DataFrame:
    """
    Rank strategies for the selected restaurant context using only observed historical evidence.

    The recommendation source is contextual:
    - cluster = restaurants in the selected restaurant's cluster
    - segment = restaurants in the selected restaurant's momentum segment
    - global = all restaurants, used as fallback evidence

    Formula:
    Contextual Strategy Score = Observed Impact × Evidence Strength × Scope Match

    Observed Impact =
        50% average revenue uplift +
        30% average bookings uplift +
        20% success rate

    Evidence Strength =
        60% activity coverage +
        40% restaurant coverage

    Scope Match =
        1.00 for cluster evidence,
        0.75 for segment evidence,
        0.50 for global evidence.

    No GA keyword matching, no ROI, no priority-tier boost, no hardcoded actions.
    """
    if recs.empty:
        return recs

    out = recs.copy()

    rev = pd.to_numeric(out.get("avg_revenue_uplift_pct"), errors="coerce").fillna(0)
    bookings = pd.to_numeric(out.get("avg_bookings_uplift_pct"), errors="coerce").fillna(0)
    success = pd.to_numeric(out.get("success_rate"), errors="coerce").fillna(0)
    activities = pd.to_numeric(out.get("activities"), errors="coerce").fillna(0)
    restaurants = pd.to_numeric(out.get("restaurants"), errors="coerce").fillna(0)

    scope = out.get("recommendation_scope", pd.Series("global", index=out.index)).fillna("global").astype(str).str.lower()
    scope_match = scope.map({"cluster": 1.00, "segment": 0.75, "global": 0.50}).fillna(0.50)

    # Coverage is capped at 1 so one very large sample does not dominate only because of volume.
    activity_coverage = (activities / max(float(min_sample_size) * 4.0, 1.0)).clip(0, 1)
    restaurant_coverage = (restaurants / 6.0).clip(0, 1)

    out["observed_impact_score"] = ((0.50 * rev) + (0.30 * bookings) + (0.20 * success)) * 100
    out["evidence_strength"] = ((0.60 * activity_coverage) + (0.40 * restaurant_coverage)) * 100
    out["scope_match"] = scope_match * 100
    out["strategy_score"] = (
        out["observed_impact_score"]
        * (out["evidence_strength"] / 100)
        * (out["scope_match"] / 100)
    )
    out["evidence_level"] = [
        _evidence_level(a, r) for a, r in zip(activities, restaurants)
    ]

    out = out.sort_values(
        ["strategy_score", "scope_match", "observed_impact_score", "evidence_strength", "activities", "restaurants"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    return out

def build_ga_snapshot_markdown(ga_context: dict) -> str:
    if not ga_context.get("has_ga_data"):
        return "No GA diagnostics available for this restaurant."

    gmv_b = ga_context["benchmarks"]["gmv_per_ga_view"]
    views_b = ga_context["benchmarks"]["ga_items_viewed"]
    gmv_delta = gmv_b.get("delta_vs_median")
    delta_text = "-"
    if gmv_delta is not None:
        delta_text = "%+.0f%% vs median" % (gmv_delta * 100)

    return "\n".join(
        [
            f"- **Latest GA month:** {fmt_month(ga_context.get('latest_month'))}",
            f"- **GMV / GA view:** {fmt_thb(ga_context.get('gmv_per_ga_view'))} ({gmv_b.get('label', 'No benchmark')}; median {fmt_thb(gmv_b.get('median'))}; {delta_text})",
            f"- **Traffic volume:** {fmt_int(ga_context.get('ga_items_viewed'))} GA item views ({views_b.get('label', 'No benchmark')}; median {fmt_int(views_b.get('median'))})",
            f"- **Funnel:** Add to cart {fmt_pct(ga_context.get('ga_add_to_cart_rate'))} | View to purchase {fmt_pct(ga_context.get('ga_view_to_purchase_rate'))} | Purchase / cart {fmt_pct(ga_context.get('ga_purchase_to_cart_rate'))}",
            f"- **Revenue productivity:** {fmt_thb(ga_context.get('ga_revenue_per_view'))} GA revenue per view | {fmt_ratio(ga_context.get('bookings_per_ga_view'))} bookings per view",
        ]
    )

def _playbook_actions(strategy_name: str, ga_context=None) -> str:
    text = str(strategy_name).lower()
    signal = (ga_context or {}).get("primary_signal_code", "")

    if signal == "scale_efficient_demand" and any(k in text for k in ["creator", "influencer", "kol", "awareness", "prospecting"]):
        return "Scale qualified traffic with strict audience filters and protect GMV per GA view as the pacing guardrail."
    if signal == "convert_existing_intent" and any(k in text for k in ["retarget", "crm", "conversion", "lifecycle", "nurture"]):
        return "Use short-window retargeting, cart recovery, and CRM reminder flows to close the current GA conversion leak."
    if signal == "improve_pre_cart_message" and any(k in text for k in ["creator", "influencer", "kol", "promo", "awareness"]):
        return "Refresh creative, offers, and menu proof points to lift add-to-cart intent from traffic already landing on page."
    if signal == "repair_conversion_efficiency" and any(k in text for k in ["retarget", "crm", "performance", "promo"]):
        return "Tighten lower-funnel targeting and promotional controls until GMV per GA view returns above the portfolio median."
    if signal == "protect_efficiency_then_scale" and any(k in text for k in ["crm", "loyalty", "retention", "retarget"]):
        return "Protect strong conversion economics with loyalty and remarketing layers before expanding reach."
    if "reactivation" in text:
        return "Target lapsed diners with return incentives, 7/14-day reminder flow, and table-time scarcity messaging."
    if "retarget" in text:
        return "Retarget menu viewers and past engagers with a 5-7 day conversion window and audience-frequency caps."
    if "loyalty" in text or "retention" in text:
        return "Launch member-only value bundles and repeat-visit nudges tied to 30-day revisit behavior."
    if "prospecting" in text or "awareness" in text:
        return "Run new-customer acquisition campaigns by lookalike audiences with strict CAC and first-booking targets."
    if "creator" in text or "influencer" in text or "kol" in text:
        return "Deploy creator-led social proof bursts with limited-time codes and track first-time bookings by creator."
    if "promo" in text or "conversion" in text:
        return "Use time-bound promotional offers with daily pacing controls and stop-loss rules on weak cohorts."
    return "Execute controlled A/B testing with one clear CTA, strict audience split, and weekly budget reallocation."


def build_grounded_brief(row: dict, hist: pd.DataFrame, recs: pd.DataFrame, ga_context: dict) -> str:
    lines = []
    lines.append("## Data-Driven Recommended Playbook")
    lines.append(
        "Strategies are ranked using observed historical outcomes only: revenue uplift, bookings uplift, "
        "success rate, number of activities, and number of restaurants."
    )

    if recs.empty:
        lines.append("- No robust historical strategy signal is available yet for this restaurant's context.")
    else:
        for idx, (_, rec) in enumerate(recs.head(3).iterrows(), start=1):
            lines.append(
                "{idx}. **{name}** ({scope})\n"
                "- Strategy score: {score}\n"
                "- Observed impact: {impact}\n"
                "- Evidence strength: {strength} ({level})\n"
                "- Revenue uplift: {rev_uplift}\n"
                "- Bookings uplift: {book_uplift}\n"
                "- Success rate: {success}\n"
                "- Evidence base: {acts} activities across {rests} restaurants".format(
                    idx=idx,
                    name=rec.get("strategy_name", "-"),
                    scope=str(rec.get("recommendation_scope", "historical")).title(),
                    score=fmt_ratio(pd.to_numeric(rec.get("strategy_score"), errors="coerce"), 2),
                    impact=fmt_ratio(pd.to_numeric(rec.get("observed_impact_score"), errors="coerce"), 2),
                    strength=fmt_pct(pd.to_numeric(rec.get("evidence_strength"), errors="coerce") / 100),
                    level=rec.get("evidence_level", "Low"),
                    rev_uplift=fmt_pct(pd.to_numeric(rec.get("avg_revenue_uplift_pct"), errors="coerce")),
                    book_uplift=fmt_pct(pd.to_numeric(rec.get("avg_bookings_uplift_pct"), errors="coerce")),
                    success=fmt_pct(pd.to_numeric(rec.get("success_rate"), errors="coerce")),
                    acts=int(pd.to_numeric(rec.get("activities"), errors="coerce") or 0),
                    rests=int(pd.to_numeric(rec.get("restaurants"), errors="coerce") or 0),
                )
            )

    return "\n".join(lines)

def build_prompt(row: dict, hist: pd.DataFrame, recs: pd.DataFrame, grounded_brief: str, ga_context: dict) -> str:
    top_rows = []
    for _, rec in recs.head(5).iterrows():
        top_rows.append(
            "- {name} | scope={scope} | strategy_score={score} | observed_impact={impact} | evidence_strength={strength} | revenue_uplift={rev} | booking_uplift={book} | success_rate={success} | activities={acts} | restaurants={rests}".format(
                name=rec.get("strategy_name", "-"),
                scope=rec.get("recommendation_scope", "-"),
                score=fmt_ratio(pd.to_numeric(rec.get("strategy_score"), errors="coerce"), 2),
                impact=fmt_ratio(pd.to_numeric(rec.get("observed_impact_score"), errors="coerce"), 2),
                strength=fmt_pct(pd.to_numeric(rec.get("evidence_strength"), errors="coerce") / 100),
                rev=fmt_pct(pd.to_numeric(rec.get("avg_revenue_uplift_pct"), errors="coerce")),
                book=fmt_pct(pd.to_numeric(rec.get("avg_bookings_uplift_pct"), errors="coerce")),
                success=fmt_pct(pd.to_numeric(rec.get("success_rate"), errors="coerce")),
                acts=fmt_int(rec.get("activities")),
                rests=fmt_int(rec.get("restaurants")),
            )
        )
    top_block = "\n".join(top_rows) if top_rows else "- No recommendation rows available"

    ga_block = build_ga_snapshot_markdown(ga_context)

    return f"""
    You are a senior marketing strategist advising a restaurant owner.

    CRITICAL RULES:
    - Use ONLY provided data
    - DO NOT invent numbers or assumptions
    - If unsure → say "insufficient data"
    - Be specific to THIS restaurant (no generic advice)
    - Do NOT mention or recreate priority-tier labels such as "Activate", "Proven", "Untapped", or "Review"
    - Do NOT mention GA fit, ROI, action templates, KPI targets, or execution guardrails

    STYLE RULES:
    - Use bullet points (NO long paragraphs)
    - Max 1–2 lines per point
    - Use SIMPLE language (no jargon)
    - Only metric names can be technical

    ---------------------
    RESTAURANT CONTEXT
    ---------------------
    Name: {display_value(row.get("name"))}
    Segment: {display_value(factual_segment(row, hist))}

    ---------------------
    GA METRICS
    ---------------------
    {ga_block}

    ---------------------
    FACTUAL STRATEGY RANKING
    ---------------------
    {top_block}

    ---------------------
    RANKING FORMULA
    ---------------------
    Strategy Score = Observed Impact × Evidence Strength
    Observed Impact = 50% revenue uplift + 30% bookings uplift + 20% success rate
    Evidence Strength = 60% activity coverage + 40% restaurant coverage

    ---------------------
    TASK
    ---------------------

    ## 1. Key Issue
    - State the main problem using the available GA or performance metrics.
    - If there is not enough data, say "insufficient data".

    ## 2. Recommended Package
    - Choose Basic / Standard / Premium.
    - Justify using only the factual strategy ranking and available metrics.
    - Explain why cheaper or more expensive packages are not suitable.

    ## 3. Why This Strategy Ranks Highest
    - Explain using strategy score, observed impact, evidence strength, revenue uplift, bookings uplift, success rate, activities, and restaurants.
    - Do not use GA fit or ROI.

    ## 4. 30-Day Plan
    - Launch
    - Optimise
    - Review

    ## 5. Expected Impact
    - Translate factual historical uplift into likely business outcome.
    - Do not promise exact results.
    """

def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()


def render():
    priority_df = load_priority()
    momentum_df = load_momentum()

    st.markdown("## Strategy Engine")
    st.markdown(
        f"<p style='color:{MUTED_TEXT};margin-top:-0.5rem;'>"
        "Historical lift evidence is overlaid with restaurant-level GA diagnostics, "
        "using GMV / GA view as the main efficiency guardrail."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if len(priority_df) == 0:
        st.warning("No priority data. Run priority_scoring_seasonality.ipynb first.")
        return

    preselected = st.session_state.get("strategy_restaurant", None)
    all_names   = priority_df.sort_values("priority_score", ascending=False)["name"].tolist()
    default_idx = all_names.index(preselected) if preselected and preselected in all_names else 0

    col_sel, col_cfg, col_info = st.columns([2, 1.2, 2.8])
    with col_sel:
        selected = st.selectbox(
            "Restaurant", all_names, index=default_idx,
            format_func=lambda n: "#%d  %s" % (all_names.index(n) + 1, n),
        )
        st.session_state["strategy_restaurant"] = selected

    with col_cfg:
        min_sample_size = st.number_input("Min sample", min_value=1, max_value=20, value=3, step=1)
        top_n = st.slider("Top strategies", min_value=1, max_value=6, value=3, step=1)

    row           = get_restaurant_priority_row(priority_df, selected)
    hist          = get_restaurant_history(momentum_df, selected)
    ga_context    = build_ga_context(row, hist, priority_df)
    candidate_n = max(int(top_n) * 3, 10)
    strategy_recs = recommend_strategies_for_restaurant(
        selected, top_n=candidate_n, min_sample_size=int(min_sample_size)
    )
    strategy_recs = rank_strategies_factually(strategy_recs, min_sample_size=int(min_sample_size)).head(int(top_n))

    score      = row.get("priority_score", 0)
    segment_value = factual_segment(row, hist)
    segment_line = (
        "<div style='font-size:0.75rem;color:{muted};margin-top:2px;'>Segment: {seg}</div>".format(
            muted=MUTED_TEXT,
            seg=segment_value,
        )
        if segment_value
        else "<div style='font-size:0.75rem;color:{muted};margin-top:2px;'>Segment not available in data</div>".format(
            muted=MUTED_TEXT
        )
    )

    with col_info:
        st.markdown(
            "<div style='background:{surface};border:1px solid {border};"
            "border-left:4px solid {c};border-radius:8px;padding:1rem 1.4rem;"
            "box-shadow:0 1px 2px rgba(0,0,0,0.05);'>"
            "<div style='display:flex;justify-content:space-between;align-items:center;'>"
            "<div>"
            "<div style='font-size:1.3rem;color:{text};font-weight:700;'>{n}</div>"
            "{segment_line}"
            "</div>"
            "<div style='text-align:right;'>"
            "<div style='font-size:1.8rem;color:#cc0000;font-weight:700;'>{s:.0f}</div>"
            "<div style='font-size:0.7rem;color:{muted};'>SCORE</div>"
            "</div>"
            "</div></div>".format(
                c=BORDER_COLOR,
                n=selected,
                segment_line=segment_line,
                s=score,
                surface=SURFACE_COLOR,
                border=BORDER_COLOR,
                text=TEXT_COLOR,
                muted=MUTED_TEXT,
            ),
            unsafe_allow_html=True,
        )

    # ── Seasonal warning — plain st.warning, no HTML, no emoji escape issues ──
    is_seasonal_flag = bool(row.get("is_seasonal", False))
    if is_seasonal_flag:
        st.warning(
            "Seasonal pattern detected — strong recent MoM but YoY is below portfolio median. "
            "Consider timing activation to align with the seasonal peak."
        )

    if len(hist):
        lat = hist.sort_values("year_month").iloc[-1]
        perf_a, perf_b, perf_c, perf_d = st.columns(4)
        perf_a.metric("Bookings", "%d" % int(pd.to_numeric(lat.get("monthly_bookings"), errors="coerce") or 0))
        perf_b.metric("Revenue", fmt_thb(pd.to_numeric(lat.get("monthly_gmv"), errors="coerce")))
        perf_c.metric("MoM Growth", fmt_pct(pd.to_numeric(lat.get("booking_growth_mom_rolling", lat.get("booking_growth_rolling")), errors="coerce")))
        perf_d.metric("YoY Growth", fmt_pct(pd.to_numeric(lat.get("booking_growth_yoy_rolling", lat.get("booking_growth_yoy")), errors="coerce")))

        if ga_context.get("has_ga_data"):
            gmv_delta = ga_context["benchmarks"]["gmv_per_ga_view"].get("delta_vs_median")
            delta_text = None if gmv_delta is None else "%+.0f%% vs median" % (gmv_delta * 100)
            ga_a, ga_b, ga_c, ga_d, ga_e, ga_f = st.columns(6)
            ga_a.metric("Traffic Volume", fmt_int(ga_context.get("ga_items_viewed")))
            ga_b.metric("GMV / GA View", fmt_thb(ga_context.get("gmv_per_ga_view")), delta=delta_text)
            ga_c.metric("Bookings / GA View", fmt_ratio(ga_context.get("bookings_per_ga_view")))
            ga_d.metric("GA Add to Cart", fmt_pct(ga_context.get("ga_add_to_cart_rate")))
            ga_e.metric("View to Purchase", fmt_pct(ga_context.get("ga_view_to_purchase_rate")))
            ga_f.metric("GA Revenue / View", fmt_thb(ga_context.get("ga_revenue_per_view")))
        else:
            st.info("No GA metrics are available for this restaurant yet.")

    st.markdown("---")
    chart_col, strat_col = st.columns([1.05, 0.95])

    with chart_col:
        st.markdown("### Booking Trend + GMV / GA View")
        if len(hist):
            hs = hist.sort_values("year_month").copy()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hs["year_month"], y=hs["monthly_bookings"],
                marker_color="#3b82f6", marker_opacity=0.70, name="Bookings",
                hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,} bookings<extra></extra>",
            ))
            if "gmv_per_ga_view" in hs.columns and pd.to_numeric(hs["gmv_per_ga_view"], errors="coerce").notna().any():
                fig.add_trace(go.Scatter(
                    x=hs["year_month"], y=pd.to_numeric(hs["gmv_per_ga_view"], errors="coerce"),
                    mode="lines+markers", name="GMV / GA View",
                    line=dict(color="#2ecc71", width=3), marker=dict(size=7), yaxis="y2",
                    hovertemplate="GMV / GA View: %{y:.2f} THB<extra></extra>",
                ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_COLOR, family="DM Sans"),
                margin=dict(l=0, r=0, t=30, b=0), height=280,
                xaxis=dict(**AXIS, tickangle=-30),
                yaxis=dict(**AXIS, title="Bookings"),
                yaxis2=dict(
                    overlaying="y", side="right", showgrid=False, color="#2ecc71", title="GMV / GA View",
                    tickfont=dict(color=MUTED_TEXT, size=9),
                ),
                legend=dict(orientation="h", y=1.02, x=0, font_size=10, bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, width="stretch")

        st.markdown("### GA Traffic + Funnel")
        if len(hist) and ga_context.get("has_ga_data"):
            hs = hist.sort_values("year_month").copy()
            hs["ga_items_viewed"] = pd.to_numeric(hs.get("ga_items_viewed"), errors="coerce")
            fig_ga = go.Figure()
            fig_ga.add_trace(go.Bar(
                x=hs["year_month"], y=hs["ga_items_viewed"],
                name="GA Item Views", marker_color="#3b82f6", marker_opacity=0.60,
                hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f} item views<extra></extra>",
            ))
            for col, label, color in [
                ("ga_add_to_cart_rate", "Add to Cart", "#f0a500"),
                ("ga_view_to_purchase_rate", "View to Purchase", "#2ecc71"),
                ("ga_purchase_to_cart_rate", "Purchase / Cart", "#e74c3c"),
            ]:
                if col in hs.columns and pd.to_numeric(hs[col], errors="coerce").notna().any():
                    fig_ga.add_trace(go.Scatter(
                        x=hs["year_month"], y=pd.to_numeric(hs[col], errors="coerce"),
                        mode="lines+markers", name=label,
                        line=dict(color=color, width=2), marker=dict(size=6), yaxis="y2",
                        hovertemplate=f"{label}: %{{y:.1%}}<extra></extra>",
                    ))
            fig_ga.update_layout(
                **layout(
                    280,
                    xaxis=dict(**AXIS, tickangle=-30),
                    yaxis=dict(**AXIS, title="GA item views"),
                    yaxis2=dict(
                        overlaying="y", side="right", showgrid=False, tickformat=".0%",
                        color=MUTED_TEXT, title="Funnel rate",
                    ),
                    legend=dict(orientation="h", y=1.02, x=0, font_size=10, bgcolor="rgba(0,0,0,0)"),
                )
            )
            st.plotly_chart(fig_ga, width="stretch")
        else:
            st.info("GA traffic and funnel history is not available.")

        st.markdown("### Ranked Strategy Evidence")
        st.caption(
            "Strategies are ranked for the selected restaurant context. "
            "Cluster evidence is preferred, then segment evidence, then global fallback. "
            "No GA-fit keyword rules, ROI, priority-tier boost, or hardcoded action templates are used."
        )
        with st.expander("__How Strategy Score Is Calculated__", expanded=False):
            st.markdown("""
            ### Strategy Score Calculation

            ##### 1. Final Score
            `strategy_score = observed_impact_score × evidence_strength × scope_match`

            ##### 2. Observed Impact
            `observed_impact_score = (revenue_uplift × 0.50) + (bookings_uplift × 0.30) + (success_rate × 0.20)`

            ##### 3. Evidence Strength
            `evidence_strength = (activity_coverage × 0.60) + (restaurant_coverage × 0.40)`

            ##### 4. Scope Match
            `scope_match =`
            - `1.00 (cluster)`
            - `0.75 (segment)`
            - `0.50 (global)`
            """)
        if len(strategy_recs):
            display_cols = [
                "strategy_name",
                "recommendation_scope",
                "activities",
                "restaurants",
                "success_rate",
                "avg_revenue_uplift_pct",
                "avg_bookings_uplift_pct",
                "observed_impact_score",
                "evidence_strength",
                "scope_match",
                "evidence_level",
                "strategy_score",
            ]
            display_cols = [c for c in display_cols if c in strategy_recs.columns]
            recs_display = strategy_recs[display_cols].copy()
            recs_display = recs_display.rename(columns={
                "strategy_name": "Strategy",
                "recommendation_scope": "Scope",
                "activities": "Activities",
                "restaurants": "Restaurants",
                "success_rate": "Success Rate",
                "avg_revenue_uplift_pct": "Revenue Uplift",
                "avg_bookings_uplift_pct": "Bookings Uplift",
                "observed_impact_score": "Observed Impact",
                "evidence_strength": "Evidence Strength",
                "scope_match": "Scope Match",
                "evidence_level": "Evidence Level",
                "strategy_score": "Strategy Score",
            })
            if "Scope" in recs_display.columns:
                recs_display["Scope"] = recs_display["Scope"].astype(str).str.title()
            for col in ["Success Rate", "Revenue Uplift", "Bookings Uplift"]:
                if col in recs_display.columns:
                    recs_display[col] = recs_display[col].apply(fmt_pct)
            if "Evidence Strength" in recs_display.columns:
                recs_display["Evidence Strength"] = (pd.to_numeric(recs_display["Evidence Strength"], errors="coerce") / 100).apply(fmt_pct)
            for col in ["Observed Impact", "Strategy Score"]:
                if col in recs_display.columns:
                    recs_display[col] = pd.to_numeric(recs_display[col], errors="coerce").round(2)
            st.dataframe(recs_display, width="stretch", hide_index=True, height=280)
        else:
            st.info("No strategy recommendations available for this restaurant yet.")


    with strat_col:
        st.markdown("### Grounded Strategy Brief")
        grounded_key  = "grounded_strategy_%s" % selected
        ai_key        = "ai_strategy_%s" % selected
        grounded_brief = build_grounded_brief(row, hist, strategy_recs, ga_context)
        st.session_state[grounded_key] = grounded_brief

        st.markdown(st.session_state[grounded_key])
        st.download_button(
            label="Download Grounded Brief",
            data="MARKETING STRATEGY\n%s\n\n%s" % (selected, st.session_state[grounded_key]),
            file_name="grounded_strategy_%s.txt" % selected.replace(" ", "_"),
            mime="text/plain",
            width="stretch",
        )

    # ===== FULL WIDTH AI SECTION =====
    st.markdown("---")
    st.markdown("### Optional AI Narrative")

    st.caption("Uses `GEMINI_API_KEY` if configured. The grounded brief remains the source of truth.")

    a_col, b_col = st.columns([2, 1])

    with a_col:
        generate_ai = st.button("Generate AI Narrative", key="gen_ai_%s" % selected, width="stretch")

    with b_col:
        if st.button("Clear AI", key="clr_ai_%s" % selected, width="stretch"):
            st.session_state[ai_key] = None

    if generate_ai:
        with st.spinner("Generating AI narrative..."):
            try:
                prompt = build_prompt(row, hist, strategy_recs, grounded_brief, ga_context)
                st.session_state[ai_key] = call_gemini(prompt)
            except Exception as e:
                st.error("AI generation failed: %s" % e)
                st.session_state[ai_key] = None

    if st.session_state.get(ai_key):
        st.markdown(st.session_state[ai_key])
        st.download_button(
            label="Download AI Narrative",
            data="AI STRATEGY NARRATIVE\n%s\n\n%s" % (selected, st.session_state[ai_key]),
            file_name="ai_strategy_%s" % selected.replace(" ", "_"),
            mime="text/plain",
            width="stretch",
        )
