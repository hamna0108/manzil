#!/usr/bin/env python3
"""
summary_generation.py
========================

Step 5 of the Real Estate Property Finder pipeline: re-ranks results so
listings with VERIFIED nearby POIs (from Step 4) surface first, then
generates a short natural-language summary of the top picks using
Gemini -- grounded strictly in the actual retrieved/verified data
(Retrieval-Augmented Generation).

The grounding boundary is deliberate and load-bearing: hard facts
(price, whether a POI is confirmed nearby, distances) must come ONLY
from the structured listing data passed into the prompt. The model is
explicitly forbidden from inventing anything not present -- agent
names, exact distances beyond what's given, additional amenities, or
claiming a POI is confirmed when verification returned False or None.
Soft interpretive framing (e.g. "these lean toward the cosy feel you
wanted") is allowed, since that describes the search's own reasoning,
not a new factual claim about the property.

Architecture mirrors poi_verification.py: network I/O (the Gemini call)
is dependency-injected so the re-ranking and prompt-building logic can
be fully unit-tested offline -- see test_summary_generation.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("summary_generation")

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"

# Only the top N results ever go into the summary prompt, independent of
# whatever top_k the search itself used -- keeps the prompt compact and
# cheap even if a caller requests a larger result set.
SUMMARY_MAX_LISTINGS = 5

SUMMARY_SYSTEM_PROMPT = """\
You are writing a short, warm summary of real estate search results for \
a Pakistani property search platform. You will be given the user's \
original search intent and a list of matching listings as structured \
data.

### Strict grounding rules (these are not optional):
- Every FACTUAL claim (price, size, location, whether a nearby amenity \
is confirmed) must come directly from the structured data provided. \
NEVER invent details that aren't in the data: no fabricated agent \
names, no invented exact distances beyond what's given, no additional \
amenities not listed.
- Each listing includes a "poi_verification" section. Only say an \
amenity is confirmed nearby if its "verified" field is exactly true. \
If "verified" is false, you may say it wasn't found nearby, or simply \
omit mentioning it -- do not claim it IS nearby. If "verified" is null \
(meaning it couldn't be checked), do NOT claim it is either present or \
absent -- omit it entirely or note it wasn't verified.
- If a listing has no poi_verification data at all, do not mention \
nearby amenities for it.
- You MAY use interpretive, subjective language to describe how well \
listings match the user's stated preferences (e.g. "these lean toward \
the cosy feel you wanted") -- this describes the search's own \
reasoning, not a new fact about the property, and is expected.
- If a "relaxation_note" field is provided explaining that search \
constraints had to be relaxed to find these results, acknowledge that \
honestly and naturally in your summary -- do not contradict it or \
pretend an exact match was found when it wasn't.

### Output rules:
- Write 2-4 sentences, warm and natural, not a bulleted list.
- Do not repeat raw JSON field names in your prose.
- If NO listings are provided, do not attempt to write a summary --
  this case is handled separately and you should never receive it.
"""


# --------------------------------------------------------------------------- #
# Pure logic: re-ranking (no network, fully unit-testable)
# --------------------------------------------------------------------------- #
def _verified_poi_count(listing: dict) -> int:
    """Counts how many of a listing's requested POIs were CONFIRMED
    (verified=True). False and None both count as zero -- neither is a
    confirmed match, so neither should be rewarded in ranking."""
    verification = listing.get("poi_verification") or {}
    if not isinstance(verification, dict):
        return 0
    return sum(1 for v in verification.values() if isinstance(v, dict) and v.get("verified") is True)


def rerank_by_poi_verification(results: list[dict]) -> list[dict]:
    """
    Re-orders results so listings with more CONFIRMED nearby POIs rank
    higher. Uses a stable sort keyed only on verified-POI count, so
    listings with an equal count keep their existing relative order
    (the Stage 3 semantic/tiebreak ordering) rather than being
    re-shuffled arbitrarily.

    Safe to call unconditionally, even if Step 4 was never run: a
    listing with no "poi_verification" key (or an empty one) simply
    scores 0 and the function becomes a no-op, preserving the original
    order exactly.
    """
    if not results:
        return results
    # Python's sort is stable, so this correctly preserves relative
    # order among equal-scoring listings.
    return sorted(results, key=_verified_poi_count, reverse=True)


# --------------------------------------------------------------------------- #
# Pure logic: prompt construction (no network, fully unit-testable)
# --------------------------------------------------------------------------- #
def _format_listing_for_prompt(listing: dict) -> dict:
    """
    Extracts ONLY the fields the summary is allowed to ground claims in
    -- deliberately not passing the whole listing dict through, so
    there's no accidental path for internal fields (raw embeddings,
    similarity scores, database ids) to leak into the prompt or the
    generated text.
    """
    poi_verification = listing.get("poi_verification") or {}
    clean_poi = {
        term: {
            "verified": v.get("verified"),
            "distance_m": v.get("distance_m"),
            "name": v.get("name"),
        }
        for term, v in poi_verification.items()
        if isinstance(v, dict)
    }
    return {
        "title": listing.get("title"),
        "property_type": listing.get("property_type"),
        "subtype": listing.get("subtype"),
        "location": listing.get("location"),
        "price_pkr": listing.get("price_pkr"),
        "area_marla": listing.get("area_marla"),
        "bedrooms": listing.get("bedrooms"),
        "bathrooms": listing.get("bathrooms"),
        "poi_verification": clean_poi,
    }


def build_summary_prompt(
    query: str,
    results: list[dict],
    intent: dict,
    message: Optional[str] = None,
) -> str:
    """
    Builds the user-turn content for the Gemini summary call. Pure
    string construction -- no network, directly testable.
    """
    capped_results = results[:SUMMARY_MAX_LISTINGS]
    listings_data = [_format_listing_for_prompt(r) for r in capped_results]

    payload = {
        "original_query": query,
        "requested_soft_signals": intent.get("soft_signals") or [],
        "requested_poi": intent.get("poi") or [],
        "relaxation_note": message,  # None if no relaxation happened
        "listings": listings_data,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Network I/O (dependency-injected for testability -- see
# test_summary_generation.py for the mocking strategy)
# --------------------------------------------------------------------------- #
def _call_gemini(prompt_content: str, api_key: str) -> Optional[str]:
    """Real Gemini call. Not unit-tested directly (requires network);
    kept deliberately thin so bugs are unlikely to hide here."""
    if genai is None:
        logger.error("google-genai SDK is not installed.")
        return None
    if not api_key:
        logger.error("No Gemini API key provided.")
        return None

    try:
        client = genai.Client(api_key=api_key)
        config = genai_types.GenerateContentConfig(
            system_instruction=SUMMARY_SYSTEM_PROMPT,
            # Deliberately NOT forcing temperature=0 here, unlike Step 1's
            # extraction -- this is user-facing narrative copy where
            # natural phrasing is desirable, not a structured value we
            # need byte-for-byte reproducibility on.
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt_content,
            config=config,
        )
        text = getattr(response, "text", None)
        return text.strip() if text else None
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini summary call failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_summary(
    query: str,
    results: list[dict],
    intent: dict,
    message: Optional[str],
    gemini_api_key: str,
    generate_fn=_call_gemini,
) -> Optional[str]:
    """
    Generates the grounded natural-language summary for a set of
    (already re-ranked) results.

    generate_fn is dependency-injected so this is unit-testable without
    a real API key -- pass a mock function returning canned text.

    Returns None (not an exception, not a placeholder string) if:
      - results is empty (nothing to summarize)
      - the Gemini call fails for any reason
    Callers must treat None as "no summary available this time" and
    still show the raw results -- a missing summary should never block
    the user from seeing their search results.
    """
    if not results:
        logger.info("No results to summarize -- skipping summary generation.")
        return None

    prompt_content = build_summary_prompt(query, results, intent, message)
    summary = generate_fn(prompt_content, gemini_api_key)

    if summary is None:
        logger.warning("Summary generation failed or returned nothing -- results will be shown without a summary.")
    return summary


def enrich_with_summary(
    response: dict,
    query: str,
    intent: dict,
    gemini_api_key: str,
    generate_fn=_call_gemini,
) -> dict:
    """
    Top-level Step 5 integration point: takes a search response (as
    returned by search_engine.py's search_listings(), optionally already
    enriched by poi_verification.py's verify_pois_for_results()),
    re-ranks by verified POIs, and adds a "summary" key.

    Safe to call even if Step 4 was skipped -- re-ranking becomes a
    no-op (see rerank_by_poi_verification's docstring) and the summary
    just won't reference any POI confirmations.
    """
    reranked_results = rerank_by_poi_verification(response.get("results", []))
    summary = generate_summary(
        query=query,
        results=reranked_results,
        intent=intent,
        message=response.get("message"),
        gemini_api_key=gemini_api_key,
        generate_fn=generate_fn,
    )

    enriched = dict(response)
    enriched["results"] = reranked_results
    enriched["summary"] = summary
    return enriched


# --------------------------------------------------------------------------- #
# Live smoke test -- requires a real Gemini API key, cannot run in an
# offline sandbox.
# --------------------------------------------------------------------------- #
def _live_smoke_test():
    """Run with: GEMINI_API_KEY=... python summary_generation.py"""
    import os

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("Set GEMINI_API_KEY to run this smoke test.")
        return

    fake_results = [
        {
            "id": 0, "title": "5 Marla Cosy House DHA Phase 6", "property_type": "House", "subtype": "House",
            "location": "DHA Phase 6, Lahore", "price_pkr": 24500000, "area_marla": 5.0,
            "bedrooms": 3, "bathrooms": 2,
            "poi_verification": {
                "school": {"verified": True, "distance_m": 450, "name": "Beaconhouse School"},
                "hospital": {"verified": False, "distance_m": None, "name": None},
            },
        },
        {
            "id": 1, "title": "5 Marla House DHA Phase 6 General", "property_type": "House", "subtype": "House",
            "location": "DHA Phase 6, Lahore", "price_pkr": 23000000, "area_marla": 5.0,
            "bedrooms": 3, "bathrooms": 2,
            "poi_verification": {
                "school": {"verified": None, "distance_m": None, "name": None, "reason": "Overpass API unavailable"},
                "hospital": {"verified": False, "distance_m": None, "name": None},
            },
        },
    ]
    intent = {"soft_signals": ["cosy"], "poi": ["school", "hospital"]}

    reranked = rerank_by_poi_verification(fake_results)
    print("Re-ranked order (by verified POI count):", [r["id"] for r in reranked])
    assert reranked[0]["id"] == 0, "Listing with the confirmed school should rank first"

    summary = generate_summary("cosy 5 marla house DHA Phase 6 near a school", reranked, intent, None, api_key)
    print("\nGenerated summary:\n", summary)


if __name__ == "__main__":
    _live_smoke_test()
