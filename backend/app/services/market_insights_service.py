"""
Market insights: given a category (property_type, optional subtype) and
a location, computes real aggregate stats (avg/min/max price, price per
marla) from the existing listings dataset, plus a short AI-generated
one-line insight.

Reuses search_engine.py's extract_core_location/normalize_location_text
for location matching -- the SAME word-level, phase-stripped matching
logic that Step 2's hard filter uses, rather than a second, possibly
inconsistent implementation. If "DHA Phase 6" matches "DHA Defence,
Lahore" for search, it should match the same way here.

The AI insight is a genuinely optional layer on top of real computed
statistics: the stats themselves are always returned even if Gemini
fails or isn't configured -- graceful degradation, same principle as
everywhere else in this pipeline. A one-line insight is a nice-to-have,
never a dependency for the numbers being useful.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from starlette.concurrency import run_in_threadpool

from app.config import get_settings

_PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from search_engine import load_listings, extract_core_location, normalize_location_text  # noqa: E402

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

settings = get_settings()

INSIGHT_MODEL_NAME = "gemini-3.5-flash-lite"

INSIGHT_SYSTEM_PROMPT = """\
You are a real estate market analyst for a Pakistani property platform. \
You will be given real, computed statistics for a specific property \
category and location -- not raw listings. Write EXACTLY ONE short, \
natural sentence (under 30 words) giving a useful, honest take on \
these numbers for a prospective buyer.

Base your sentence ONLY on the numbers given. Do not invent trends, \
comparisons to other areas, or historical context not present in the \
data. If the sample size (listing_count) is small (under 10), you may \
note that the estimate is based on limited data. Return only the \
sentence, no preamble, no quotation marks.
"""


def compute_market_stats(property_type: str, location: str, subtype: Optional[str] = None) -> dict:
    """
    Pure computation (pandas aggregation) -- no network, directly
    testable. Returns the raw stats dict; ai_insight is added
    separately by generate_market_insights() below.
    """
    df = load_listings(settings.listings_path)

    mask = df["property_type"].str.lower() == property_type.lower()
    if subtype:
        mask &= df["subtype"].str.lower() == subtype.lower()

    core_location = extract_core_location(location)
    query_words = [w for w in core_location.split() if w]
    if query_words:
        normalized_listing_loc = df["location"].apply(normalize_location_text)
        for word in query_words:
            mask &= normalized_listing_loc.str.contains(word, na=False, regex=False)

    matching = df[mask]

    if matching.empty:
        return {
            "property_type": property_type,
            "location": location,
            "listing_count": 0,
            "avg_price_pkr": None,
            "min_price_pkr": None,
            "max_price_pkr": None,
            "avg_price_per_marla_pkr": None,
        }

    price_per_marla = matching["price_pkr"] / matching["area_marla"].replace(0, float("nan"))

    return {
        "property_type": property_type,
        "location": location,
        "listing_count": int(len(matching)),
        "avg_price_pkr": round(float(matching["price_pkr"].mean()), 2),
        "min_price_pkr": int(matching["price_pkr"].min()),
        "max_price_pkr": int(matching["price_pkr"].max()),
        "avg_price_per_marla_pkr": round(float(price_per_marla.mean()), 2) if not price_per_marla.dropna().empty else None,
    }


def _call_gemini_for_insight(stats: dict, api_key: str) -> Optional[str]:
    if genai is None or not api_key or stats["listing_count"] == 0:
        return None
    try:
        client = genai.Client(api_key=api_key)
        config = genai_types.GenerateContentConfig(system_instruction=INSIGHT_SYSTEM_PROMPT)
        response = client.models.generate_content(
            model=INSIGHT_MODEL_NAME,
            contents=str(stats),
            config=config,
        )
        text = getattr(response, "text", None)
        return text.strip() if text else None
    except Exception:  # noqa: BLE001
        return None


async def generate_market_insights(property_type: str, location: str, subtype: Optional[str] = None) -> dict:
    """Async entry point for the router: computes stats (threadpooled,
    since it touches the pandas DataFrame) then optionally adds an AI
    insight (also threadpooled, since it's a blocking network call)."""
    stats = await run_in_threadpool(compute_market_stats, property_type, location, subtype)
    insight = await run_in_threadpool(_call_gemini_for_insight, stats, settings.gemini_api_key)
    return {**stats, "ai_insight": insight}
