import pandas as pd
import requests
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ==================== CONFIGURATION ====================
GOOGLE_API_KEY = "AIzaSyAFVk_x1rnmuPvPue9spUPiVpvoc6iHtQw"
API_DELAY = 0.2
RESULTS_PATH = Path("places_api_new_results.csv")
PRINT_PREVIEW_N = 10  # how many rows to print in final preview
# =======================================================


# ----------------------------
# Cuisine inference from types
# ----------------------------
def _build_cuisine_rules() -> Dict[str, Tuple[str, float]]:
    """
    Maps Google Places types -> (cuisine, weight)
    """
    return {
        # Japanese
        "japanese_restaurant": ("Japanese", 2.0),
        "sushi_restaurant": ("Japanese", 2.0),
        "ramen_restaurant": ("Japanese", 2.0),

        # Chinese
        "chinese_restaurant": ("Chinese", 2.0),
        "hot_pot_restaurant": ("Chinese", 2.0),
        "dim_sum_restaurant": ("Chinese", 2.0),

        # Korean / Thai / Vietnamese / Indian
        "korean_restaurant": ("Korean", 2.0),
        "thai_restaurant": ("Thai", 2.0),
        "vietnamese_restaurant": ("Vietnamese", 2.0),
        "indian_restaurant": ("Indian", 2.0),

        # SEA
        "malaysian_restaurant": ("Malaysian", 2.0),
        "indonesian_restaurant": ("Indonesian", 2.0),

        # Western / Europe
        "italian_restaurant": ("Italian", 2.0),
        "pizza_restaurant": ("Italian", 1.6),
        "french_restaurant": ("French", 2.0),
        "spanish_restaurant": ("Spanish", 2.0),
        "tapas_restaurant": ("Spanish", 2.0),
        "greek_restaurant": ("Greek", 2.0),
        "mediterranean_restaurant": ("Mediterranean", 2.0),

        # Americas
        "american_restaurant": ("American", 1.6),
        "mexican_restaurant": ("Mexican", 2.0),
        "latin_american_restaurant": ("Latin American", 2.0),
        "brazilian_restaurant": ("Brazilian", 2.0),
        "argentinian_restaurant": ("Argentinian", 2.0),

        # Middle Eastern
        "middle_eastern_restaurant": ("Middle Eastern", 2.0),
        "turkish_restaurant": ("Turkish", 2.0),
        "lebanese_restaurant": ("Lebanese", 2.0),

        # Protein / theme
        "steak_house": ("Steakhouse", 2.0),
        "barbecue_restaurant": ("BBQ", 2.0),
        "seafood_restaurant": ("Seafood", 2.0),
        "chicken_restaurant": ("Chicken", 1.6),
        "hamburger_restaurant": ("Burgers", 1.6),
        "sandwich_shop": ("Sandwiches", 1.2),

        # Lifestyle / casual
        "cafe": ("Cafe", 1.0),
        "coffee_shop": ("Cafe", 1.0),
        "bakery": ("Bakery", 1.0),
        "dessert_shop": ("Dessert", 1.0),
        "ice_cream_shop": ("Dessert", 1.0),
        "bar": ("Bar", 0.8),
        "fast_food_restaurant": ("Fast Food", 1.2),

        # Dietary
        "vegetarian_restaurant": ("Vegetarian", 2.0),
        "vegan_restaurant": ("Vegan", 2.0),
    }


def _parse_types(raw_types_value) -> List[str]:
    """
    raw_types stored as comma-separated string in CSV.
    """
    if raw_types_value is None or (isinstance(raw_types_value, float) and pd.isna(raw_types_value)):
        return []
    s = str(raw_types_value).strip()
    if not s:
        return []
    return [t.strip().lower() for t in s.split(",") if t.strip()]


def infer_cuisine_from_types(types: List[str]) -> Tuple[Optional[str], float]:
    """
    Returns (cuisine, confidence) inferred from Google 'types'
    """
    if not types:
        return (None, 0.0)

    rules = _build_cuisine_rules()
    score_by: Dict[str, float] = {}
    hits_by: Dict[str, int] = {}

    for t in types:
        if t in rules:
            cuisine, weight = rules[t]
            score_by[cuisine] = score_by.get(cuisine, 0.0) + weight
            hits_by[cuisine] = hits_by.get(cuisine, 0) + 1

    if not score_by:
        # if generic restaurant but no cuisine-specific tag
        generic = {"restaurant", "food", "meal_takeaway", "meal_delivery"}
        if any(t in generic for t in types):
            return ("General", 0.15)
        return (None, 0.2)

    cuisine = max(score_by.keys(), key=lambda c: (score_by[c], hits_by.get(c, 0)))
    score = score_by[cuisine]
    hits = hits_by.get(cuisine, 1)

    confidence = min(1.0, 0.35 + 0.15 * hits + 0.10 * score)
    return (cuisine, round(confidence, 2))


# -----------------------------------
# Google Places (New) types-only pull
# -----------------------------------
def fetch_raw_types_only(query_string: str, api_key: str, timeout: int = 10) -> Dict:
    """
    Calls Places API (New) searchText but requests ONLY places.types to save credits.
    Returns dict: {found: bool, raw_types: str|None, error: str|None}
    """
    url = "https://places.googleapis.com/v1/places:searchText"
    field_mask = ["places.types"]  # ONLY types -> cheaper

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": ",".join(field_mask),
    }
    payload = {"textQuery": query_string}

    out = {"found": False, "raw_types": None, "error": None}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        data = resp.json()

        if "error" in data:
            out["error"] = data["error"].get("message", "Unknown API Error")
            return out

        places = data.get("places", [])
        if not places:
            out["error"] = "No results found"
            return out

        types = places[0].get("types", []) or []
        out["found"] = True
        out["raw_types"] = ",".join(types) if types else None
        return out

    except Exception as e:
        out["error"] = str(e)
        return out


# ==================== MAIN ====================
if __name__ == "__main__":
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found in current folder.")

    df = pd.read_csv(RESULTS_PATH)

    # Required column
    if "input_string" not in df.columns:
        raise KeyError("Expected column 'input_string' in places_api_new_results.csv")

    # Ensure output columns exist (these are our additions)
    if "raw_types" not in df.columns:
        df["raw_types"] = pd.NA
    if "Cuisine" not in df.columns:
        df["Cuisine"] = pd.NA
    if "Cuisine_confidence" not in df.columns:
        df["Cuisine_confidence"] = pd.NA

    # Only pull types for rows missing raw_types (saves credits)
    mask_missing_types = df["raw_types"].isna() | (df["raw_types"].astype(str).str.strip() == "")
    rows_to_update = df.index[mask_missing_types].tolist()

    print(f"Found {len(df)} total rows.")
    print(f"Need to pull raw_types for {len(rows_to_update)} rows (types-only).")

    # Pull raw_types only for those rows
    for n, idx in enumerate(rows_to_update, start=1):
        query = str(df.at[idx, "input_string"]).strip()
        if not query:
            continue

        print(f"\n[{n}/{len(rows_to_update)}] types-only lookup: {query}")
        res = fetch_raw_types_only(query, GOOGLE_API_KEY)

        print("   API found:", res["found"])
        print("   raw_types:", res["raw_types"])
        print("   error:", res["error"])

        # Only update minimal new fields
        if res["found"] and res["raw_types"]:
            df.at[idx, "raw_types"] = res["raw_types"]
            print("   ✅ raw_types saved")
        else:
            print("   ⚠️ raw_types NOT saved")

        # Preserve existing error column: fill only if empty
        if "error" in df.columns:
            if (pd.isna(df.at[idx, "error"]) or str(df.at[idx, "error"]).strip() == "") and res["error"]:
                df.at[idx, "error"] = res["error"]

        time.sleep(API_DELAY)

    # Recompute cuisine from raw_types (no extra API cost)
    cuisines: List[Optional[str]] = []
    confs: List[float] = []

    for raw in df["raw_types"].tolist():
        types = _parse_types(raw)
        cuisine, conf = infer_cuisine_from_types(types)
        cuisines.append(cuisine)
        confs.append(conf)

    df["Cuisine"] = cuisines
    df["Cuisine_confidence"] = confs

    # Preview
    print(f"\n🍽️ Cuisine inference preview (first {PRINT_PREVIEW_N} rows):")
    preview_cols = ["input_string", "raw_types", "Cuisine", "Cuisine_confidence"]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head(PRINT_PREVIEW_N).to_string(index=False))

    # Summary
    print("\n📊 Summary:")
    print("Total rows:", len(df))
    print("Rows with raw_types:", int(df["raw_types"].notna().sum()))
    print("Rows with Cuisine:", int(df["Cuisine"].notna().sum()))
    try:
        print("Avg Cuisine_confidence:", round(float(pd.to_numeric(df["Cuisine_confidence"], errors="coerce").mean()), 3))
    except Exception:
        print("Avg Cuisine_confidence: (could not compute)")

    # Show missing types (if any)
    missing = df[df["raw_types"].isna() | (df["raw_types"].astype(str).str.strip() == "")]
    if len(missing) > 0:
        print("\n❌ Still missing raw_types (showing up to 10):")
        cols_show = ["input_string"]
        if "error" in df.columns:
            cols_show.append("error")
        print(missing[cols_show].head(10).to_string(index=False))

    # Save back: includes ALL existing columns + our additions
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n✅ Saved updates into {RESULTS_PATH} (kept existing columns, added/updated raw_types + Cuisine + Cuisine_confidence).")
