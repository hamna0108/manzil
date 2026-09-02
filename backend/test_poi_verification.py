#!/usr/bin/env python3
"""
test_poi_verification.py
============================

Tests poi_verification.py's PURE LOGIC entirely offline, using mocked
Overpass responses -- no real network calls, no internet dependency.
This is possible because fetch_overpass() is dependency-injected
everywhere it's used (verify_pois_for_listing takes a fetch_fn
parameter), the same pattern used for Qwen/Gemini routing in
intent_extractor.py.

What this DOES verify: query building, distance calculation, POI
attribution, the honest True/False/None distinction, caching, and
graceful degradation on every failure mode.

What this does NOT verify: whether the real Overpass API actually
behaves the way we assume. Run poi_verification.py directly (its
verify_live_smoke_test()) on a machine with internet access for that.

Usage:
    python test_poi_verification.py
"""

import json
import os
import tempfile
import os
import tempfile

from poi_verification import (
    build_overpass_query,
    resolve_poi_type,
    haversine_km,
    attribute_and_score,
    parse_overpass_response,
    verify_pois_for_listing,
    verify_pois_for_results,
    fetch_overpass,
    _cache_key,
    _has_valid_coordinates,
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
# 1. POI term resolution -- known terms, synonyms, unknown terms
# --------------------------------------------------------------------------- #
section("resolve_poi_type()")

check("'school' resolves", resolve_poi_type("school") is not None)
check("'mosque' resolves", resolve_poi_type("mosque") is not None)
check("'masjid' (synonym) resolves to the same tag as 'mosque'",
      resolve_poi_type("masjid")["tags"] == resolve_poi_type("mosque")["tags"])
check("case-insensitive: 'SCHOOL' resolves", resolve_poi_type("SCHOOL") is not None)
check("whitespace-tolerant: '  school  ' resolves", resolve_poi_type("  school  ") is not None)
check("unknown term returns None, NOT a default/guess", resolve_poi_type("underwater_basket_weaving_studio") is None)
check("empty string returns None", resolve_poi_type("") is None)
check("None input returns None (doesn't crash)", resolve_poi_type(None) is None)


# --------------------------------------------------------------------------- #
# 2. Query building -- pure string construction, no network
# --------------------------------------------------------------------------- #
section("build_overpass_query()")

query = build_overpass_query(31.5, 74.3, ["school"])
check("query is built for a resolvable term", query is not None)
check("query contains the coordinates", "31.5" in query and "74.3" in query)
check("query contains the correct radius in meters (1.5km -> 1500m)", "1500" in query)
check("query includes node/way/relation (not just node)", all(x in query for x in ["node(", "way(", "relation("]))

multi_query = build_overpass_query(31.5, 74.3, ["school", "hospital"])
check("multi-POI query includes both radii (1500m for school, 3000m for hospital)", "1500" in multi_query and "3000" in multi_query)

none_query = build_overpass_query(31.5, 74.3, ["totally_unknown_type"])
check("query is None when nothing resolves (no point hitting the API for nothing)", none_query is None)

mixed_query = build_overpass_query(31.5, 74.3, ["school", "totally_unknown_type"])
check("mixed known+unknown still builds a query for the known part", mixed_query is not None)


# --------------------------------------------------------------------------- #
# 3. Distance calculation
# --------------------------------------------------------------------------- #
section("haversine_km()")

check("distance to self is ~0", haversine_km(31.5, 74.3, 31.5, 74.3) < 0.001)
dist = haversine_km(31.5204, 74.3587, 33.6844, 73.0479)  # Lahore -> Islamabad, real ~275km
check("Lahore->Islamabad distance is realistic (~260-290km)", 260 <= dist <= 290, f"got {dist:.1f}km")


# --------------------------------------------------------------------------- #
# 4. Attribution and scoring -- the core logic, with a realistic mocked
#    Overpass response shape (mix of node/way/relation, with/without names)
# --------------------------------------------------------------------------- #
section("attribute_and_score()")

MOCK_ELEMENTS = [
    {  # a real node, close, WITH a name
        "type": "node", "id": 1, "lat": 31.501, "lon": 74.301,
        "tags": {"amenity": "school", "name": "Beaconhouse School"},
    },
    {  # a WAY (building polygon) using 'center' from out:center, further away
        "type": "way", "id": 2, "center": {"lat": 31.52, "lon": 74.32},
        "tags": {"amenity": "school", "name": "City School"},
    },
    {  # matches a DIFFERENT poi type (hospital), should not count as a school match
        "type": "node", "id": 3, "lat": 31.5005, "lon": 74.3005,
        "tags": {"amenity": "hospital", "name": "Services Hospital"},
    },
    {  # no name tag at all -- must not crash, should fall back sensibly
        "type": "node", "id": 4, "lat": 31.5008, "lon": 74.3008,
        "tags": {"amenity": "school"},
    },
]

scored = attribute_and_score(MOCK_ELEMENTS, listing_lat=31.5, listing_lon=74.3, poi_terms=["school", "hospital"])

check("school is verified True", scored["school"]["verified"] is True)
check("hospital is verified True (independently attributed, not mixed up with school)", scored["hospital"]["verified"] is True)
check("hospital's matched name is the hospital, not a school", scored["hospital"]["name"] == "Services Hospital")

# Closest school should be chosen: node id=4 (no name, ~0.13km) is closer
# than id=1 (named, ~0.14km) which is closer than id=2 (way, much further).
# Let's verify the CLOSEST one wins regardless of whether it has a name.
check("closest matching element wins (by distance), not just the first one found",
      scored["school"]["distance_m"] < 200, f"got {scored['school']['distance_m']}m")

unresolvable_scored = attribute_and_score(MOCK_ELEMENTS, 31.5, 74.3, ["totally_unknown_type"])
check("requesting an unresolvable term returns nothing for it (caller adds verified=None separately)", "totally_unknown_type" not in unresolvable_scored)

no_match_scored = attribute_and_score([], 31.5, 74.3, ["school"])
check("empty element list -> verified False (genuinely checked, nothing found)", no_match_scored["school"]["verified"] is False)
check("empty element list -> distance_m is None", no_match_scored["school"]["distance_m"] is None)

# Regression test: real live Overpass data returned a name with trailing
# whitespace ("Bridge accademy ") -- raw OSM tag data isn't always clean,
# and it shouldn't leak untrimmed into what gets shown to a user.
WHITESPACE_NAME_ELEMENT = [{
    "type": "node", "id": 99, "lat": 31.5005, "lon": 74.3005,
    "tags": {"amenity": "school", "name": "Bridge accademy "},
}]
whitespace_scored = attribute_and_score(WHITESPACE_NAME_ELEMENT, 31.5, 74.3, ["school"])
check(
    "REGRESSION: trailing whitespace in raw OSM name data gets stripped",
    whitespace_scored["school"]["name"] == "Bridge accademy",
    repr(whitespace_scored["school"]["name"]),
)


# --------------------------------------------------------------------------- #
# 5. parse_overpass_response -- malformed/unexpected response shapes
# --------------------------------------------------------------------------- #
section("parse_overpass_response() -- malformed input handling")

check("normal response parses correctly", len(parse_overpass_response({"elements": [{"id": 1}]})) == 1)
check("missing 'elements' key doesn't crash", parse_overpass_response({}) == [])
check("elements not a list doesn't crash", parse_overpass_response({"elements": "not a list"}) == [])
check("completely wrong type (not a dict) doesn't crash", parse_overpass_response("garbage") == [])
check("None doesn't crash", parse_overpass_response(None) == [])


# --------------------------------------------------------------------------- #
# 6. verify_pois_for_listing -- full orchestration, mocked fetch_fn,
#    exercising every honest-state branch explicitly
# --------------------------------------------------------------------------- #
section("verify_pois_for_listing() -- the True/False/None honesty contract")

def mock_fetch_success(query):
    return {"elements": MOCK_ELEMENTS}

def mock_fetch_failure(query):
    return None  # simulates every Overpass endpoint being down

def mock_fetch_should_not_be_called(query):
    raise AssertionError("fetch_fn was called when it should NOT have been (e.g. no valid coords, or cache hit)")


good_listing = {"latitude": 31.5, "longitude": 74.3}
result = verify_pois_for_listing(good_listing, ["school", "unknown_poi_xyz"], cache={}, fetch_fn=mock_fetch_success)
check("resolvable term with successful fetch -> verified True", result["school"]["verified"] is True)
check("unresolvable term -> verified None with an explanatory reason", result["unknown_poi_xyz"]["verified"] is None and "reason" in result["unknown_poi_xyz"])

missing_coords_listing = {"latitude": None, "longitude": None}
result = verify_pois_for_listing(missing_coords_listing, ["school"], cache={}, fetch_fn=mock_fetch_should_not_be_called)
check("listing with no coordinates -> verified None, network NEVER called (no wasted API call)", result["school"]["verified"] is None)

bad_coords_listing = {"latitude": 10.0, "longitude": 10.0}  # nowhere near Pakistan
result = verify_pois_for_listing(bad_coords_listing, ["school"], cache={}, fetch_fn=mock_fetch_should_not_be_called)
check("listing with garbage coordinates (outside Pakistan) -> verified None, network NEVER called", result["school"]["verified"] is None)

result = verify_pois_for_listing(good_listing, ["school"], cache={}, fetch_fn=mock_fetch_failure)
check("Overpass API down (all endpoints failed) -> verified None, NOT False", result["school"]["verified"] is None)
check("API-down reason is distinct from 'unknown POI type' reason", "Overpass API unavailable" in result["school"]["reason"])

empty_poi_result = verify_pois_for_listing(good_listing, [], cache={}, fetch_fn=mock_fetch_should_not_be_called)
check("empty poi_terms list -> returns empty dict, network never touched", empty_poi_result == {})


# --------------------------------------------------------------------------- #
# 7. Caching -- second identical request must NOT hit the network
# --------------------------------------------------------------------------- #
section("caching behavior")

call_count = {"n": 0}
def counting_fetch(query):
    call_count["n"] += 1
    return {"elements": MOCK_ELEMENTS}

cache = {}
verify_pois_for_listing(good_listing, ["school"], cache=cache, fetch_fn=counting_fetch)
check("first call hits the network once", call_count["n"] == 1)

verify_pois_for_listing(good_listing, ["school"], cache=cache, fetch_fn=counting_fetch)
check("second identical call uses the cache, network NOT hit again", call_count["n"] == 1, f"network was called {call_count['n']} times")

key1 = _cache_key(31.50001, 74.30001, ["school"])
key2 = _cache_key(31.50002, 74.30002, ["school"])
check("cache key rounds coordinates (near-identical points share a cache entry)", key1 == key2)

key3 = _cache_key(31.5, 74.3, ["school", "hospital"])
key4 = _cache_key(31.5, 74.3, ["hospital", "school"])
check("cache key is order-independent for the poi_terms list", key3 == key4)


# --------------------------------------------------------------------------- #
# 8. verify_pois_for_results -- the actual integration point with
#    search_engine.py's result list
# --------------------------------------------------------------------------- #
section("verify_pois_for_results() -- integration with search results")

fake_results = [
    {"id": 0, "title": "House A", "latitude": 31.5, "longitude": 74.3},
    {"id": 1, "title": "House B", "latitude": 31.5, "longitude": 74.3},
]

enriched = verify_pois_for_results(fake_results, poi_terms=["school"], fetch_fn=mock_fetch_success)
check("every result gets a poi_verification key", all("poi_verification" in r for r in enriched))
check("original result fields are preserved (title still present)", enriched[0]["title"] == "House A")

no_poi_enriched = verify_pois_for_results(fake_results, poi_terms=[], fetch_fn=mock_fetch_should_not_be_called)
check("empty poi_terms -> no-op, network never touched even across multiple results", all(r["poi_verification"] == {} for r in no_poi_enriched))


# --------------------------------------------------------------------------- #
# 9. THE ACTUAL BUG FIX: endpoint cooldown prevents repeated timeout
#    waits across multiple listings when the network is genuinely down
# --------------------------------------------------------------------------- #
section("endpoint cooldown -- fixes the 'slow repeated failure' bug")

import time as time_module
from poi_verification import reset_endpoint_health, OVERPASS_ENDPOINTS, _is_endpoint_in_cooldown, _mark_endpoint_failed

reset_endpoint_health()  # start clean, don't leak state from other tests

for endpoint in OVERPASS_ENDPOINTS:
    _mark_endpoint_failed(endpoint)

check(
    "after a failure, endpoints are correctly marked as in-cooldown",
    all(_is_endpoint_in_cooldown(e) for e in OVERPASS_ENDPOINTS),
)

start = time_module.time()
result = fetch_overpass("fake query that will never be sent")
elapsed = time_module.time() - start
check(
    "BUGFIX VERIFIED: fetch_overpass returns near-instantly when all endpoints are in cooldown (no wasted timeout)",
    elapsed < 0.5,
    f"took {elapsed:.2f}s -- should skip cooldown endpoints without any network attempt",
)
check("fetch_overpass correctly returns None when all endpoints are in cooldown", result is None)

reset_endpoint_health()  # clean up for subsequent tests


# --------------------------------------------------------------------------- #
# 10. Time budget: verify_pois_for_results bounds total worst-case time
# --------------------------------------------------------------------------- #
section("time budget -- bounds worst-case total latency")

def mock_slow_fetch(query):
    time_module.sleep(0.3)  # simulate a slow-but-eventually-failing network call
    return None

many_listings = [{"latitude": 31.5, "longitude": 74.3, "id": i} for i in range(10)]

start = time_module.time()
# Use a dedicated, isolated cache file for this test -- otherwise an
# earlier test section's successful mock result (same coordinates) gets
# cached to the shared DEFAULT_CACHE_PATH and this test would silently
# hit that stale cache instead of ever calling mock_slow_fetch, making
# the timing assertion meaningless. (This bit us during development --
# caught by the "0 skipped" assertion failing, not assumed away.)
isolated_cache_path = tempfile.mktemp(suffix="_poi_cache_test.json")
budget_results = verify_pois_for_results(
    many_listings, ["school"], cache_path=isolated_cache_path, fetch_fn=mock_slow_fetch, max_total_seconds=1.0
)
elapsed = time_module.time() - start
os.remove(isolated_cache_path) if os.path.exists(isolated_cache_path) else None

check(
    "BUGFIX VERIFIED: total time is bounded by max_total_seconds, not len(listings) * per-call time",
    elapsed < 2.0,  # would be 3.0+ seconds (10 * 0.3s) without the budget cutoff
    f"took {elapsed:.2f}s for 10 listings at 0.3s each (would be ~3s uncapped)",
)
check("all 10 listings still get a result (none silently dropped)", len(budget_results) == 10)
budget_exceeded_count = sum(
    1 for r in budget_results
    if r["poi_verification"].get("school", {}).get("reason") == "skipped -- verification time budget exceeded for this search"
)
check("at least some listings past the budget are honestly marked as skipped (not silently guessed)", budget_exceeded_count > 0, f"got {budget_exceeded_count} skipped")


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
    print(f"\nRESULT: ALL {passed} TESTS PASSED -- Step 4 logic is verified (offline).")
    print("REMINDER: run 'python poi_verification.py' on a machine with real internet")
    print("access to smoke-test the actual live Overpass API call before trusting this in production.")
