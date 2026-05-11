"""
Infer restaurant cities for the OPE restaurant list.

The script first uses cheap local evidence (restaurant names and existing
Places API addresses), then falls back to DuckDuckGo HTML search for unresolved
restaurants. Results are written incrementally so interrupted runs can resume.

Examples:
    python pull_cities.py
    python pull_cities.py --limit 25 --dry-run
    python pull_cities.py --workers 8
    python pull_cities.py --no-web
    python pull_cities.py --repair-errors
"""

from __future__ import annotations

import argparse
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


SCRIPT_DIR = Path(__file__).resolve().parent
OPE_DIR = SCRIPT_DIR.parent

DEFAULT_INPUT_PATH = SCRIPT_DIR / "unique_restaurant_list.csv"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "restaurant_city_scraped.csv"
DEFAULT_PLACES_PATH = OPE_DIR / "_1_eda" / "data_output" / "places_api_new_results.csv"

DEFAULT_SEARCH_PROVIDERS = (
    {
        "name": "bing",
        "url": "https://www.bing.com/search",
        "method": "GET",
    },
)

DUCKDUCKGO_PROVIDER = (
    {
        "name": "duckduckgo",
        "url": "https://html.duckduckgo.com/html/",
        "method": "POST",
    },
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

REQUEST_TIMEOUT = 8
SLEEP_MIN = 1.5
SLEEP_MAX = 3.5
FLUSH_EVERY = 25
DEFAULT_WORKERS = 6

WORKER_STATE = threading.local()


# Common city/province/location strings seen in restaurant names, Google Places
# formatted addresses, and search snippets.
KNOWN_LOCATIONS = [
    "Bangkok",
    "กรุงเทพมหานคร",
    "Krung Thep Maha Nakhon",
    "Chiang Mai",
    "Chiangmai",
    "Phuket",
    "Pattaya",
    "Muang Pattaya",
    "Chonburi",
    "Nonthaburi",
    "Pathum Thani",
    "Samut Prakan",
    "Samut Sakhon",
    "Nakhon Pathom",
    "Ayutthaya",
    "Phra Nakhon Si Ayutthaya",
    "Hua Hin",
    "Cha-am",
    "Krabi",
    "Koh Samui",
    "Ko Samui",
    "Surat Thani",
    "Hat Yai",
    "Songkhla",
    "Nakhon Ratchasima",
    "Korat",
    "Khon Kaen",
    "Udon Thani",
    "Rayong",
    "Trat",
    "Kanchanaburi",
    "Ratchaburi",
    "Chiang Rai",
    "Lampang",
    "Phitsanulok",
    "Nakhon Si Thammarat",
    "Nakhon Nayok",
    "Chanthaburi",
    "Chai Nat",
    "Chon Buri",
    "Si Racha",
    "Sriracha",
    "Nan",
    "Buriram",
    "Koh Chang",
    "Khao Yai",
    "Nakhon Sawan",
    "Phang-nga",
    "Ranong",
    "Trang",
    "Ubon Ratchathani",
    "Sattahip",
    "Singapore",
    "Putrajaya",
    "Kuala Lumpur",
    "Selangor",
    "Johor Bahru",
    "Penang",
]

LOCATION_ALIASES = {
    "krung thep maha nakhon": "Bangkok",
    "กรุงเทพมหานคร": "Bangkok",
    "chiangmai": "Chiang Mai",
    "muang pattaya": "Pattaya",
    "ko samui": "Koh Samui",
    "samui": "Koh Samui",
    "korat": "Nakhon Ratchasima",
    "phra nakhon si ayutthaya": "Ayutthaya",
    "chon buri": "Chonburi",
    "si racha": "Chonburi",
    "sriracha": "Chonburi",
    "koh chang": "Trat",
    "khao yai": "Nakhon Ratchasima",
    "phang-nga": "Phang Nga",
    "sattahip": "Chonburi",
    "khonkaen": "Khon Kaen",
}

BANGKOK_AREAS = [
    "Ari",
    "Asok",
    "Asoke",
    "Bang Na",
    "Bangna",
    "Bang Rak",
    "Chatuchak",
    "Chidlom",
    "Chit Lom",
    "Ekkamai",
    "Huai Khwang",
    "Khlong Toei",
    "Lat Phrao",
    "Min Buri",
    "Minburi",
    "Muang Thong Thani",
    "On Nut",
    "Pathum Wan",
    "Phra Khanong",
    "Phrom Phong",
    "Ploen Chit",
    "Ploenchit",
    "Rama 3",
    "Rama 9",
    "Ratchada",
    "Riverside",
    "Sathorn",
    "Sathon",
    "Siam",
    "Silom",
    "Sukhumvit",
    "Thong Lo",
    "Thonglor",
    "Watthana",
]

PLACE_HINTS = {
    "asiatique": "Bangkok",
    "baiyoke": "Bangkok",
    "bang bon": "Bangkok",
    "bang sue": "Bangkok",
    "bangsue": "Bangkok",
    "bang yai": "Nonthaburi",
    "banthat thong": "Bangkok",
    "banthatthong": "Bangkok",
    "bangkapi": "Bangkok",
    "bangkae": "Bangkok",
    "bantadthong": "Bangkok",
    "bang khae": "Bangkok",
    "bang kapi": "Bangkok",
    "bangkhunnon": "Bangkok",
    "bangpoon": "Pathum Thani",
    "bangrak": "Bangkok",
    "bangi": "Selangor",
    "bearing": "Samut Prakan",
    "big c ladprao": "Bangkok",
    "bkk": "Bangkok",
    "block 28": "Bangkok",
    "centralworld": "Bangkok",
    "central world": "Bangkok",
    "central embassy": "Bangkok",
    "central eastville": "Bangkok",
    "central festival samui": "Koh Samui",
    "central ladprao": "Bangkok",
    "central rama 2": "Bangkok",
    "central salaya": "Nakhon Pathom",
    "central ubon ratchathani": "Ubon Ratchathani",
    "central village": "Samut Prakan",
    "central westgate": "Nonthaburi",
    "central westville": "Nonthaburi",
    "chaengwattana": "Nonthaburi",
    "chaengwatta": "Nonthaburi",
    "charn issara": "Bangkok",
    "charn at the avenue": "Bangkok",
    "chinatown": "Bangkok",
    "chokchai 4": "Bangkok",
    "chula": "Bangkok",
    "cosmo bazaar": "Nonthaburi",
    "emquartier": "Bangkok",
    "eastin grand hotel phayathai": "Bangkok",
    "ekamai": "Bangkok",
    "exchange tower": "Bangkok",
    "fashion island": "Bangkok",
    "fortune town": "Bangkok",
    "future park rangsit": "Pathum Thani",
    "gaysorn": "Bangkok",
    "gateway bangsue": "Bangkok",
    "grande centre point surawong": "Bangkok",
    "iconsiam": "Bangkok",
    "ics": "Bangkok",
    "im park chula": "Bangkok",
    "index ladkrabang": "Bangkok",
    "jaransanitwong": "Bangkok",
    "jw marriott": "Bangkok",
    "kanchanaphisek": "Bangkok",
    "kasetnawamin": "Bangkok",
    "kaset-nawamin": "Bangkok",
    "khlong luang": "Pathum Thani",
    "klong 2": "Pathum Thani",
    "krungthep kreetha": "Bangkok",
    "koh kret": "Nonthaburi",
    "kubphuket": "Phuket",
    "lat phrao": "Bangkok",
    "ladprao": "Bangkok",
    "ladkrabang": "Bangkok",
    "lat krabang": "Bangkok",
    "lebua": "Bangkok",
    "le meridien": "Bangkok",
    "larn luang": "Bangkok",
    "lam luk ka": "Pathum Thani",
    "luang prabang": "Luang Prabang",
    "macpherson": "Singapore",
    "mackenzie road": "Singapore",
    "market village suvarnabhumi": "Samut Prakan",
    "mbk": "Bangkok",
    "mbk center": "Bangkok",
    "muang thong thani": "Nonthaburi",
    "naradhiwas": "Bangkok",
    "ngamwongwan": "Nonthaburi",
    "nong chok": "Bangkok",
    "north ratchaphruek": "Nonthaburi",
    "naret": "Bangkok",
    "nawamin": "Bangkok",
    "nawamin-si burapha": "Bangkok",
    "onnut": "Bangkok",
    "people park onnut": "Bangkok",
    "phong phet": "Nonthaburi",
    "phan khwai": "Bangkok",
    "phutthamonthon sai 2": "Nakhon Pathom",
    "phutthamonthon sai 4": "Bangkok",
    "pinklao": "Bangkok",
    "phra ram 9": "Bangkok",
    "phran nok": "Bangkok",
    "prachachuen": "Nonthaburi",
    "prannok": "Bangkok",
    "praditmanutham": "Bangkok",
    "prince theatre": "Bangkok",
    "promenade": "Bangkok",
    "ptt active park": "Nonthaburi",
    "rajamangala": "Bangkok",
    "rat uthit": "Bangkok",
    "rama 2": "Bangkok",
    "rama 4": "Bangkok",
    "ramkhamhaeng": "Bangkok",
    "rangsit": "Pathum Thani",
    "ratchadapisek": "Bangkok",
    "ratchaphruek": "Nonthaburi",
    "ratchapruek": "Nonthaburi",
    "ratchathewi": "Bangkok",
    "ratchayothin": "Bangkok",
    "sanampao": "Bangkok",
    "saphan khwai": "Bangkok",
    "sai burapha": "Bangkok",
    "serithai": "Bangkok",
    "samyan": "Bangkok",
    "sc park": "Bangkok",
    "seacon square": "Bangkok",
    "seacon bangkae": "Bangkok",
    "seasons mall": "Bangkok",
    "si phraya": "Bangkok",
    "silq hotel": "Bangkok",
    "skyview hotel": "Bangkok",
    "shaw centre": "Singapore",
    "srinagarindra": "Bangkok",
    "surawong": "Bangkok",
    "star vista": "Singapore",
    "suvarnabhumi": "Samut Prakan",
    "the paseo mall lat krabang": "Bangkok",
    "the mall bangkapi": "Bangkok",
    "the mall bangkae": "Bangkok",
    "the mall bang kapi": "Bangkok",
    "the mall ngamwongwan": "Nonthaburi",
    "the nine center": "Bangkok",
    "the bright rama 2": "Bangkok",
    "talat phlu": "Bangkok",
    "tiwanon": "Nonthaburi",
    "town in town": "Bangkok",
    "qsncc": "Bangkok",
    "vue lifestyle mall": "Bangkok",
    "vibhavadi": "Bangkok",
    "wongwian yai": "Bangkok",
    "workpoint alley": "Pathum Thani",
    "yingcharoen square": "Bangkok",
    "zpell": "Pathum Thani",
}


def clean_text(text: Any) -> str:
    """Return normalized text, preserving empty strings for missing values."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def canonical_location(location: str) -> str:
    """Map alternate spellings and Thai administrative names to reporting city."""
    return LOCATION_ALIASES.get(location.lower(), location)


def normalized_key(text: Any) -> str:
    """Normalize restaurant/place names for exact joins across slightly different punctuation."""
    return re.sub(r"[^a-z0-9]+", "", clean_text(text).lower())


def contains_phrase(text_lower: str, phrase: str) -> bool:
    """Case-insensitive phrase match that tolerates punctuation around words."""
    escaped = re.escape(phrase.lower()).replace(r"\ ", r"[\s\-]+")
    return bool(re.search(rf"(?<![a-z]){escaped}(?![a-z])", text_lower))


def meaningful_name_tokens(name: str) -> set[str]:
    """Return useful tokens for judging whether a search result is about a restaurant."""
    stopwords = {
        "and",
        "at",
        "bar",
        "by",
        "cafe",
        "eatery",
        "hotel",
        "in",
        "of",
        "restaurant",
        "rooftop",
        "the",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", clean_text(name).lower())
        if len(token) >= 3 and token not in stopwords
    }


def is_relevant_search_result(result_text: str, restaurant_name: str) -> bool:
    """Keep only search results that appear to refer to the requested restaurant."""
    tokens = meaningful_name_tokens(restaurant_name)
    if not tokens:
        return False

    text_lower = clean_text(result_text).lower()
    matched_tokens = {token for token in tokens if token in text_lower}
    needed = 1 if len(tokens) == 1 else 2
    return len(matched_tokens) >= needed or clean_text(restaurant_name).lower() in text_lower


def infer_city_from_text(text: Any, *, source_kind: str = "text") -> tuple[str | None, str, str | None]:
    """
    Infer city from arbitrary text.

    Returns:
        city, confidence, matched_keyword
    """
    text_clean = clean_text(text)
    if not text_clean:
        return None, "low", None

    text_lower = text_clean.lower()
    confidence = "high" if source_kind in {"name", "places"} else "medium"

    for location in sorted(KNOWN_LOCATIONS, key=len, reverse=True):
        if contains_phrase(text_lower, location):
            return canonical_location(location), confidence, location

    for area in sorted(BANGKOK_AREAS, key=len, reverse=True):
        if contains_phrase(text_lower, area):
            return "Bangkok", "medium", area

    for hint, city in sorted(PLACE_HINTS.items(), key=lambda item: len(item[0]), reverse=True):
        if contains_phrase(text_lower, hint):
            return city, "medium", hint

    return None, "low", None


def load_places_city_hints(path: Path) -> dict[str, dict[str, str | None]]:
    """Build local city hints keyed by normalized restaurant input string."""
    if not path.exists():
        return {}

    places = pd.read_csv(path)
    if "input_string" not in places.columns:
        return {}

    hints: dict[str, dict[str, str | None]] = {}
    for _, row in places.iterrows():
        keys = {
            clean_text(row.get("input_string")).lower(),
            clean_text(row.get("official_name")).lower() if "official_name" in places.columns else "",
            normalized_key(row.get("input_string")),
            normalized_key(row.get("official_name")) if "official_name" in places.columns else "",
        }
        keys = {key for key in keys if key}
        if not keys:
            continue

        evidence = " ".join(
            clean_text(row.get(col))
            for col in ("city", "formatted_address", "official_name")
            if col in places.columns
        )
        city, confidence, matched_keyword = infer_city_from_text(evidence, source_kind="places")
        if city:
            for key in keys:
                hints[key] = {
                    "city": city,
                    "confidence": confidence,
                    "matched_keyword": matched_keyword,
                    "source_text": evidence[:1000],
                }

    return hints


def build_session(search_providers: tuple[dict[str, str], ...] = DEFAULT_SEARCH_PROVIDERS) -> requests.Session:
    """Create a requests session with conservative retry behavior."""
    retry = Retry(
        total=1,
        connect=1,
        read=1,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.headers.update(HEADERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.search_providers = search_providers
    return session


def get_worker_session(search_providers: tuple[dict[str, str], ...]) -> requests.Session:
    """Return one reusable HTTP session per worker thread."""
    session = getattr(WORKER_STATE, "session", None)
    if session is None or session.search_providers != search_providers:
        session = build_session(search_providers)
        WORKER_STATE.session = session
    return session


def parse_bing_results(html: str, restaurant_name: str) -> tuple[list[str], list[str]]:
    """Extract relevant and all result text from a Bing search page."""
    soup = BeautifulSoup(html, "html.parser")
    relevant_texts = []
    all_texts = []

    for result in soup.select("li.b_algo")[:8]:
        result_text = " ".join(
            clean_text(element.get_text(" ", strip=True))
            for element in (
                result.select_one("h2"),
                result.select_one(".b_caption p"),
                result.select_one("cite"),
            )
            if element is not None
        )
        if result_text:
            all_texts.append(result_text)
            if is_relevant_search_result(result_text, restaurant_name):
                relevant_texts.append(result_text)

    return relevant_texts, all_texts


def parse_duckduckgo_results(html: str, restaurant_name: str) -> tuple[list[str], list[str]]:
    """Extract relevant and all result text from a DuckDuckGo HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    relevant_texts = []
    all_texts = []

    for result in soup.select(".result")[:5]:
        result_text = " ".join(
            clean_text(element.get_text(" ", strip=True))
            for element in (
                result.select_one(".result__title"),
                result.select_one(".result__snippet"),
                result.select_one(".result__url"),
            )
            if element is not None
        )
        if result_text:
            all_texts.append(result_text)
            if is_relevant_search_result(result_text, restaurant_name):
                relevant_texts.append(result_text)

    return relevant_texts, all_texts


def fetch_search_results(
    provider: dict[str, str],
    query: str,
    restaurant_name: str,
    session: requests.Session,
) -> dict[str, Any]:
    """Fetch and parse one search provider."""
    if provider["method"] == "POST":
        response = session.post(provider["url"], data={"q": query}, timeout=REQUEST_TIMEOUT)
    else:
        response = session.get(provider["url"], params={"q": query}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    if provider["name"] == "bing":
        relevant_texts, all_texts = parse_bing_results(response.text, restaurant_name)
    else:
        relevant_texts, all_texts = parse_duckduckgo_results(response.text, restaurant_name)

    return {
        "provider": provider["name"],
        "relevant_texts": relevant_texts,
        "all_texts": all_texts,
    }


def search_restaurant_city(name: str, session: requests.Session) -> dict[str, str | None]:
    """Search restaurant name and infer city from web results."""
    query = f"{clean_text(name)} restaurant city address"
    providers_without_results = []

    for provider in session.search_providers:
        try:
            parsed = fetch_search_results(provider, query, name, session)
        except requests.RequestException as exc:
            providers_without_results.append(provider["name"])
            continue

        source_text = " | ".join(parsed["relevant_texts"])
        city, confidence, matched_keyword = infer_city_from_text(source_text, source_kind="search")
        if city:
            return {
                "city": city,
                "confidence": confidence,
                "matched_keyword": matched_keyword,
                "source": parsed["provider"],
                "search_query": query,
                "source_text": source_text[:1000],
                "error": None,
            }

        if parsed["relevant_texts"] or parsed["all_texts"]:
            return {
                "city": None,
                "confidence": "low",
                "matched_keyword": None,
                "source": parsed["provider"],
                "search_query": query,
                "source_text": (source_text or " | ".join(parsed["all_texts"]))[:1000],
                "error": None,
            }

        providers_without_results.append(parsed["provider"])

    return {
        "city": None,
        "confidence": "low",
        "matched_keyword": None,
        "source": "not_found_web",
        "search_query": query,
        "source_text": None,
        "error": None,
    }


def infer_restaurant_city(
    restaurant_name: str,
    places_hints: dict[str, dict[str, str | None]],
    session: requests.Session | None,
) -> dict[str, str | None]:
    """Resolve one restaurant city from name, local Places hints, then web search."""
    city, confidence, matched_keyword = infer_city_from_text(restaurant_name, source_kind="name")
    if city:
        return {
            "city": city,
            "confidence": confidence,
            "matched_keyword": matched_keyword,
            "source": "restaurant_name",
            "search_query": None,
            "source_text": clean_text(restaurant_name),
            "error": None,
        }

    place_hint = places_hints.get(clean_text(restaurant_name).lower()) or places_hints.get(normalized_key(restaurant_name))
    if place_hint:
        return {
            "city": place_hint["city"],
            "confidence": place_hint["confidence"],
            "matched_keyword": place_hint["matched_keyword"],
            "source": "places_api_existing_output",
            "search_query": None,
            "source_text": place_hint["source_text"],
            "error": None,
        }

    if session is None:
        return {
            "city": None,
            "confidence": "low",
            "matched_keyword": None,
            "source": "not_found_local",
            "search_query": None,
            "source_text": None,
            "error": None,
        }

    return search_restaurant_city(restaurant_name, session)


def read_restaurant_input(path: Path) -> pd.DataFrame:
    """Load and validate the restaurant input CSV."""
    restaurants = pd.read_csv(path)
    required_cols = {"restaurant_id", "name"}
    missing_cols = required_cols - set(restaurants.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in input file: {sorted(missing_cols)}")

    restaurants = restaurants[["restaurant_id", "name"]].copy()
    restaurants["name"] = restaurants["name"].map(clean_text)
    restaurants = restaurants.dropna(subset=["restaurant_id", "name"])
    restaurants = restaurants[restaurants["name"].ne("")]
    return restaurants.drop_duplicates(subset=["restaurant_id", "name"])


def read_existing_output(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def is_retryable_output(row: pd.Series) -> bool:
    """Rows with web/search failures should be reprocessed, not treated as complete."""
    confidence = clean_text(row.get("confidence")).lower()
    error = clean_text(row.get("error")).lower()
    source = clean_text(row.get("source")).lower()
    city = clean_text(row.get("scraped_city"))
    source_text = clean_text(row.get("source_text"))
    weak_blank = not city and not source_text and source in {"web_search", "not_found_web", "not_found_local"}
    return confidence == "error" or "403 client error" in error or weak_blank


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
    rows.clear()


def build_output_row(
    row: dict[str, Any],
    places_hints: dict[str, dict[str, str | None]],
    search_providers: tuple[dict[str, str], ...],
    no_web: bool,
) -> dict[str, Any]:
    """Resolve one input row into the output schema."""
    session = None if no_web else get_worker_session(search_providers)
    result = infer_restaurant_city(row["name"], places_hints, session)

    return {
        "restaurant_id": row["restaurant_id"],
        "name": row["name"],
        "scraped_city": result["city"],
        "confidence": result["confidence"],
        "matched_keyword": result["matched_keyword"],
        "source": result["source"],
        "search_query": result["search_query"],
        "source_text": result["source_text"],
        "error": result["error"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input restaurant CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV.")
    parser.add_argument(
        "--places",
        type=Path,
        default=DEFAULT_PLACES_PATH,
        help="Existing Places API output used for local address hints.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N remaining rows.")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel lookup workers for web searches. Use 1 for sequential mode.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing the output CSV.")
    parser.add_argument("--force", action="store_true", help="Reprocess rows already present in the output CSV.")
    parser.add_argument(
        "--repair-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reprocess rows whose previous output has confidence=error or a 403 error.",
    )
    parser.add_argument("--no-web", action="store_true", help="Skip web search fallback and use only local hints.")
    parser.add_argument(
        "--include-duckduckgo",
        action="store_true",
        help="Also try DuckDuckGo HTML search. It is disabled by default because it often returns 403.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    restaurants = read_restaurant_input(args.input)
    existing_output = read_existing_output(args.output)

    if args.force or existing_output.empty or "restaurant_id" not in existing_output.columns:
        done_ids: set[str] = set()
        retry_ids: set[str] = set()
    else:
        retry_mask = existing_output.apply(is_retryable_output, axis=1) if args.repair_errors else pd.Series(False, index=existing_output.index)
        retry_ids = set(existing_output.loc[retry_mask, "restaurant_id"].astype(str))
        done_ids = set(existing_output.loc[~retry_mask, "restaurant_id"].astype(str))

        if retry_ids and not args.dry_run:
            existing_output.loc[~retry_mask].to_csv(args.output, index=False)

    rows_to_process = restaurants[
        args.force | ~restaurants["restaurant_id"].astype(str).isin(done_ids)
    ].copy()
    if args.limit is not None:
        rows_to_process = rows_to_process.head(args.limit)

    places_hints = load_places_city_hints(args.places)
    search_providers = DEFAULT_SEARCH_PROVIDERS + (DUCKDUCKGO_PROVIDER if args.include_duckduckgo else ())
    workers = max(1, args.workers)

    print(f"Input restaurants: {len(restaurants)}")
    print(f"Existing rows skipped: {0 if args.force else len(done_ids)}")
    print(f"Existing error rows to repair: {0 if args.force else len(retry_ids)}")
    print(f"Local Places hints loaded: {len(places_hints)}")
    print(f"Rows to process: {len(rows_to_process)}")
    print(f"Web search providers: {'none' if args.no_web else ', '.join(provider['name'] for provider in search_providers)}")
    print(f"Workers: {workers}")
    if args.dry_run:
        print("Dry run enabled: no output will be written.")

    pending_rows: list[dict[str, Any]] = []
    input_rows = rows_to_process.to_dict("records")

    def handle_output(output_row: dict[str, Any]) -> None:
        if args.dry_run:
            print(output_row)
        else:
            pending_rows.append(output_row)
            if len(pending_rows) >= FLUSH_EVERY:
                append_rows(args.output, pending_rows)

    if workers == 1:
        for index, row in enumerate(tqdm(input_rows, total=len(input_rows)), start=1):
            output_row = build_output_row(row, places_hints, search_providers, args.no_web)
            handle_output(output_row)

            if (
                not args.no_web
                and output_row["source"] in {"bing", "duckduckgo", "web_search"}
                and index < len(input_rows)
            ):
                time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(build_output_row, row, places_hints, search_providers, args.no_web)
                for row in input_rows
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                handle_output(future.result())

    if not args.dry_run:
        append_rows(args.output, pending_rows)
        print(f"Saved output to: {args.output}")

    print("Done.")


if __name__ == "__main__":
    main()
