#!/usr/bin/env python3
"""
test_summary_generation.py
=============================

Tests summary_generation.py's re-ranking and prompt-construction logic
entirely offline using mocked Gemini calls -- no real network calls, no
API key needed. Mirrors the testing approach used in
test_poi_verification.py and test_intent_extractor.py.

What this DOES verify: re-ranking correctness and stability, prompt
content (specifically that grounding-critical data like verified=False/
None makes it into the prompt so the model can respect it), and every
graceful-degradation branch.

What this does NOT verify: whether Gemini actually follows the
grounding instructions in practice. Run summary_generation.py directly
(with a real GEMINI_API_KEY) for that -- and read the output critically,
since prompt compliance is a model behavior, not something a unit test
can prove in the abstract.

Usage:
    python test_summary_generation.py
"""

import json

from summary_generation import (
    rerank_by_poi_verification,
    _verified_poi_count,
    build_summary_prompt,
    _format_listing_for_prompt,
    generate_summary,
    enrich_with_summary,
)

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}  {detail}")


def section(title: str):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# --------------------------------------------------------------------------- #
# 1. _verified_poi_count -- the scoring function ranking depends on
# --------------------------------------------------------------------------- #
section("_verified_poi_count()")

check("no poi_verification key -> 0", _verified_poi_count({}) == 0)
check("empty poi_verification -> 0", _verified_poi_count({"poi_verification": {}}) == 0)
check(
    "one verified=True, one False -> counts only the True",
    _verified_poi_count({"poi_verification": {
        "school": {"verified": True}, "hospital": {"verified": False},
    }}) == 1,
)
check(
    "verified=None does NOT count (not confirmed, same as False for ranking)",
    _verified_poi_count({"poi_verification": {"school": {"verified": None}}}) == 0,
)
check(
    "two verified=True -> counts both",
    _verified_poi_count({"poi_verification": {
        "school": {"verified": True}, "park": {"verified": True},
    }}) == 2,
)
check(
    "malformed poi_verification (not a dict) doesn't crash",
    _verified_poi_count({"poi_verification": "garbage"}) == 0,
)
check(
    "malformed individual entry (not a dict) doesn't crash",
    _verified_poi_count({"poi_verification": {"school": "garbage"}}) == 0,
)


# --------------------------------------------------------------------------- #
# 2. rerank_by_poi_verification -- ordering AND stability
# --------------------------------------------------------------------------- #
section("rerank_by_poi_verification()")

check("empty list -> empty list", rerank_by_poi_verification([]) == [])

no_poi_data = [
    {"id": 0, "title": "A"}, {"id": 1, "title": "B"}, {"id": 2, "title": "C"},
]
reranked_noop = rerank_by_poi_verification(no_poi_data)
check(
    "no poi_verification anywhere -> order UNCHANGED (safe no-op when Step 4 wasn't run)",
    [r["id"] for r in reranked_noop] == [0, 1, 2],
)

mixed = [
    {"id": 0, "title": "zero verified", "poi_verification": {"school": {"verified": False}}},
    {"id": 1, "title": "two verified", "poi_verification": {"school": {"verified": True}, "park": {"verified": True}}},
    {"id": 2, "title": "one verified", "poi_verification": {"school": {"verified": True}}},
    {"id": 3, "title": "zero verified, unknown", "poi_verification": {"school": {"verified": None}}},
]
reranked = rerank_by_poi_verification(mixed)
check(
    "results ordered by verified count descending: [2 verified, 1 verified, 0, 0]",
    [r["id"] for r in reranked] == [1, 2, 0, 3],
    f"got order: {[r['id'] for r in reranked]}",
)

# Stability check: two listings with the SAME verified count must keep
# their ORIGINAL relative order (the Stage 3 ranking that came before
# Step 5), not be reordered arbitrarily.
same_count = [
    {"id": "first", "poi_verification": {"school": {"verified": True}}},
    {"id": "second", "poi_verification": {"school": {"verified": True}}},
    {"id": "third", "poi_verification": {"school": {"verified": True}}},
]
reranked_stable = rerank_by_poi_verification(same_count)
check(
    "STABILITY: equal verified-counts preserve original relative order (doesn't reshuffle prior ranking)",
    [r["id"] for r in reranked_stable] == ["first", "second", "third"],
    f"got: {[r['id'] for r in reranked_stable]}",
)


# --------------------------------------------------------------------------- #
# 3. build_summary_prompt / _format_listing_for_prompt -- the
#    grounding-critical content that goes to the model
# --------------------------------------------------------------------------- #
section("build_summary_prompt() and _format_listing_for_prompt()")

listing_with_internal_fields = {
    "id": 0, "title": "5 Marla House", "property_type": "House", "subtype": "House",
    "location": "DHA, Lahore", "price_pkr": 20000000, "area_marla": 5.0,
    "bedrooms": 3, "bathrooms": 2,
    "similarity_score": 0.83,  # internal field -- should NOT leak into prompt
    "title_embedding": [0.1] * 384,  # internal field -- should NOT leak into prompt
    "url": "https://example.com/listing",  # not needed for summary text
    "poi_verification": {
        "school": {"verified": True, "distance_m": 450, "name": "Beaconhouse School"},
        "hospital": {"verified": False, "distance_m": None, "name": None},
        "park": {"verified": None, "distance_m": None, "name": None, "reason": "Overpass API unavailable"},
    },
}
formatted = _format_listing_for_prompt(listing_with_internal_fields)
check("factual fields ARE included", formatted["price_pkr"] == 20000000 and formatted["area_marla"] == 5.0)
check("internal fields (similarity_score, embedding) are NOT included -- no leakage risk", "similarity_score" not in formatted and "title_embedding" not in formatted)

check("verified=True case preserved exactly", formatted["poi_verification"]["school"]["verified"] is True)
check(
    "CRITICAL: verified=False is preserved as False, not silently dropped or upgraded to True",
    formatted["poi_verification"]["hospital"]["verified"] is False,
)
check(
    "CRITICAL: verified=None is preserved as None (distinct from False) -- model must be able to see 'unknown'",
    formatted["poi_verification"]["park"]["verified"] is None,
)

prompt = build_summary_prompt(
    query="cosy 5 marla house near a school",
    results=[listing_with_internal_fields],
    intent={"soft_signals": ["cosy"], "poi": ["school", "hospital", "park"]},
    message="Your budget was slightly below market rate; showing the closest matches.",
)
prompt_data = json.loads(prompt)  # must be valid JSON -- if this raises, the prompt itself is malformed
check("prompt is valid JSON (parseable, well-formed)", True)
check("prompt includes the original query", prompt_data["original_query"] == "cosy 5 marla house near a school")
check("prompt includes the relaxation note (for honest acknowledgment)", "budget" in prompt_data["relaxation_note"])
check("prompt does NOT include internal fields in the listings", "similarity_score" not in json.dumps(prompt_data["listings"]))

many_listings = [dict(listing_with_internal_fields, id=i) for i in range(10)]
capped_prompt = build_summary_prompt("query", many_listings, {}, None)
capped_data = json.loads(capped_prompt)
check(
    "prompt caps listings at SUMMARY_MAX_LISTINGS (5) even if more are passed in",
    len(capped_data["listings"]) == 5,
    f"got {len(capped_data['listings'])} listings in prompt",
)


# --------------------------------------------------------------------------- #
# 4. generate_summary -- graceful degradation, every failure mode
# --------------------------------------------------------------------------- #
section("generate_summary() -- graceful degradation")

def mock_gemini_success(prompt, api_key):
    return "These cosy 5 marla homes in DHA Phase 6 fit your budget nicely, with one confirmed just 450m from Beaconhouse School."

def mock_gemini_failure(prompt, api_key):
    return None  # simulates API failure

def mock_should_not_be_called(prompt, api_key):
    raise AssertionError("Gemini should NOT have been called for empty results")


summary = generate_summary("query", [listing_with_internal_fields], {}, None, "fake-key", generate_fn=mock_gemini_success)
check("successful generation returns the summary text", summary is not None and "cosy" in summary)

empty_summary = generate_summary("query", [], {}, None, "fake-key", generate_fn=mock_should_not_be_called)
check("EMPTY results list -> returns None WITHOUT calling Gemini at all (no wasted call, no hallucination risk)", empty_summary is None)

failed_summary = generate_summary("query", [listing_with_internal_fields], {}, None, "fake-key", generate_fn=mock_gemini_failure)
check("Gemini call failure -> returns None (not an exception, not a placeholder string)", failed_summary is None)


# --------------------------------------------------------------------------- #
# 5. enrich_with_summary -- full integration, including the
#    re-rank-then-summarize ordering
# --------------------------------------------------------------------------- #
section("enrich_with_summary() -- full integration")

fake_response = {
    "message": None,
    "relaxed_constraints": [],
    "results": [
        {"id": 0, "title": "No POI match", "poi_verification": {"school": {"verified": False}}},
        {"id": 1, "title": "Confirmed school nearby", "poi_verification": {"school": {"verified": True}}},
    ],
}
enriched = enrich_with_summary(fake_response, "query", {"poi": ["school"]}, "fake-key", generate_fn=mock_gemini_success)
check("enrich_with_summary re-ranks results (confirmed match moves to #1)", enriched["results"][0]["id"] == 1)
check("enrich_with_summary adds a 'summary' key", "summary" in enriched and enriched["summary"] is not None)
check("original response keys (message, relaxed_constraints) are preserved", "message" in enriched and "relaxed_constraints" in enriched)

empty_response = {"message": "No listings found.", "relaxed_constraints": [], "results": []}
enriched_empty = enrich_with_summary(empty_response, "query", {}, "fake-key", generate_fn=mock_should_not_be_called)
check("enrich_with_summary on empty results -> summary is None, Gemini never called", enriched_empty["summary"] is None)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
section("SUMMARY")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

if failed:
    print(f"\nRESULT: FAILED -- {failed} test(s) did not pass.")
    import sys
    sys.exit(1)
else:
    print(f"\nRESULT: ALL {passed} TESTS PASSED -- Step 5 logic is verified (offline).")
    print("REMINDER: run 'GEMINI_API_KEY=... python summary_generation.py' with a real key")
    print("to check the ACTUAL generated prose reads well and respects grounding in practice.")
