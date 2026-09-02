#!/usr/bin/env python3
"""
poi_verification.py
======================

Step 4 of the Real Estate Property Finder pipeline: verifies whether
the POIs a user asked for ("near a school", "close to a hospital")
genuinely exist near a listing, using real OpenStreetMap data via the
Overpass API.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from typing import Any, Optional

import requests
import urllib3.util.connection as _urllib3_connection

# --------------------------------------------------------------------------- #
# Force IPv4-only DNS resolution for outbound HTTPS requests.
#
# ROOT CAUSE: `nslookup` for both overpass-api.de and overpass.kumi.systems
# returns IPv6 addresses before IPv4 ones. Python's requests/urllib3 (unlike
# browsers, which use "Happy Eyeballs" to race IPv4 vs IPv6 and use whichever
# answers first) simply tries addresses in DNS order. On networks where IPv6
# is advertised but not fully/reliably routed (common on many ISPs), this
# means every connection attempt first tries a dead IPv6 path and hangs
# until timeout before ever trying IPv4 -- which is exactly the ConnectTimeout
# / ReadTimeout pattern observed in testing. Forcing AF_INET (IPv4) here
# skips the broken path entirely. This only affects this process's socket
# resolution, not the system's network configuration.
# --------------------------------------------------------------------------- #
def _allowed_gai_family():
    return socket.AF_INET

_urllib3_connection.allowed_gai_family = _allowed_gai_family

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("poi_verification")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# IMPORTANT: lz4.overpass-api.de / z.overpass-api.de / overpass-api.de are all
# load-balancer aliases in front of the SAME backend cluster and WAF. They are
# NOT independent fallbacks -- if that WAF blocks/rate-limits you, all three
# fail together (this is exactly what was happening: 406 / timeout / 406).
#
# overpass.kumi.systems is a genuinely separate, independently-hosted public
# Overpass mirror, so it's included as a real fallback. Public mirrors do
# occasionally change or go down -- check https://overpass-api.de/ and the
# OSM wiki "Overpass API" page periodically for the current list.
# CONFIRMED via independent OS-level testing (Test-NetConnection) that
# overpass-api.de / lz4.overpass-api.de are network-blocked on this machine's
# current connection: ICMP ping succeeds but TCP:443 is refused/dropped --
# that combination means a firewall/ISP is deliberately blocking HTTPS to
# that host. No code fix can route around that; it needs a VPN or a network
# config change on the client side. Until then, we deprioritize it (try it
# LAST) so we don't burn 10-20s per search on a host we know is unreachable.
#
# Endpoint list cross-checked against the OFFICIAL current instance table at
# https://wiki.openstreetmap.org/wiki/Overpass_API (checked 2026-08-25):
#   - overpass.kumi.systems was RENAMED to overpass.private.coffee. Using
#     the canonical domain. Their listed policy: "no rate limit in place,
#     feel free to use our service in any project."
#   - overpass.openstreetmap.fr is NOT in the official global-coverage
#     table -- that's why it returned 403 "white-listed usages only".
#     Removed; it was never a general-purpose public mirror.
#   - maps.mail.ru (VK Maps, Russia) IS officially listed with "no requests
#     limitations" -- added as a third, independently-operated fallback.
# Public mirrors do change over time -- re-check that wiki page periodically.
# Endpoints confirmed reachable and working on this network today
# (2026-08-25): maps.mail.ru and overpass.private.coffee both succeeded.
# overpass-api.de / lz4.overpass-api.de have failed 100% of attempts across
# every test (TCP connect refused/dropped) -- they are excluded from the
# default rotation so we stop paying a guaranteed ~24s tax on every search.
# Set OVERPASS_INCLUDE_BLOCKED_DE=1 in the environment to add them back in
# (e.g. once your network/ISP situation changes) without editing this file.
OVERPASS_ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
if os.environ.get("OVERPASS_INCLUDE_BLOCKED_DE") == "1":
    OVERPASS_ENDPOINTS += [
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ]

OVERPASS_QUERY_TIMEOUT_S = 25
# (connect_timeout, read_timeout) instead of one flat number. This lets us
# tell "couldn't even establish a TCP connection" (network/firewall problem)
# apart from "connected fine, but the server was slow to answer" (server
# load problem) -- the two have very different fixes.
CONNECT_TIMEOUT_S = 8
READ_TIMEOUT_S = 30
INTER_REQUEST_DELAY_S = 1.0

# Upstream-gateway errors (the mirror's proxy is up but its backend isn't)
# are often transient on free public instances -- worth one quick retry
# before writing the endpoint off for this search.
RETRYABLE_STATUS_CODES = {502, 503, 504}
RETRY_DELAY_S = 3.0

# Use ONLY an honest, identifying User-Agent per the Overpass API usage
# policy (https://dev.overpass-api.de/overpass-doc/en/preface/commons.html).
# Do NOT spoof a Referer claiming to be overpass-turbo -- that pattern is
# actively fingerprinted and blocked by the WAF (this was the actual cause
# of the 406s), and it also isn't true, which risks getting the UA banned
# outright if a maintainer notices.
USER_AGENT = "ManzilPropertySearch/1.0 (hamnaharram12@gmail.com)"

# --------------------------------------------------------------------------- #
# Endpoint health tracking
# --------------------------------------------------------------------------- #
_endpoint_failure_times: dict[str, float] = {}

# Transient failures (timeouts, 5xx, connection errors) -- the server is
# just busy/unavailable, safe to retry relatively soon.
TRANSIENT_COOLDOWN_S = 60.0

# Hard blocks (401/403/406/429) -- the WAF or rate limiter has explicitly
# rejected us. Retrying quickly only makes the block worse/longer, so back
# off much more aggressively.
BLOCKED_COOLDOWN_S = 300.0

# Status codes that mean "you have been explicitly blocked/throttled",
# as opposed to "the server had a transient problem". This is the fix for
# the bug where 406s never triggered a cooldown and were retried on every
# single listing.
HARD_BLOCK_STATUS_CODES = {401, 403, 406, 429}

def _is_endpoint_in_cooldown(endpoint: str) -> bool:
    cooldown_until = _endpoint_failure_times.get(endpoint)
    if cooldown_until is None:
        return False
    return time.time() < cooldown_until

def _mark_endpoint_failed(endpoint: str, cooldown_s: float = TRANSIENT_COOLDOWN_S) -> None:
    _endpoint_failure_times[endpoint] = time.time() + cooldown_s

def _mark_endpoint_healthy(endpoint: str) -> None:
    _endpoint_failure_times.pop(endpoint, None)

def reset_endpoint_health() -> None:
    _endpoint_failure_times.clear()

PK_LAT_RANGE = (23.5, 37.5)
PK_LON_RANGE = (60.5, 77.5)

DEFAULT_CACHE_PATH = "poi_cache.json"

# --------------------------------------------------------------------------- #
# POI term -> OSM tag mapping
# --------------------------------------------------------------------------- #
POI_TAG_MAP: dict[str, dict] = {
    "school":       {"tags": [("amenity", "school"), ("building", "school"), ("education", "school")], "radius_km": 5.0, "label": "school"},
    "university":   {"tags": [("amenity", "university"), ("amenity", "college"), ("building", "university"), ("building", "college")], "radius_km": 5.0, "label": "university"},
    "college":      {"tags": [("amenity", "college"), ("amenity", "university"), ("building", "college"), ("office", "educational_institution")], "radius_km": 5.0, "label": "college"},
    "hospital":     {"tags": [("amenity", "hospital"), ("healthcare", "hospital"), ("building", "hospital"), ("amenity", "clinic")], "radius_km": 5.0, "label": "hospital"},
    "clinic":       {"tags": [("amenity", "clinic"), ("amenity", "doctors"), ("healthcare", "clinic")], "radius_km": 3.0, "label": "clinic"},
    "pharmacy":     {"tags": [("amenity", "pharmacy"), ("healthcare", "pharmacy"), ("shop", "chemist")], "radius_km": 3.0, "label": "pharmacy"},
    "mosque":       {"tags": [("amenity", "place_of_worship"), ("building", "mosque")], "radius_km": 3.0, "label": "mosque"},
    "masjid":       {"tags": [("amenity", "place_of_worship"), ("building", "mosque")], "radius_km": 3.0, "label": "mosque"},
    "church":       {"tags": [("amenity", "place_of_worship"), ("building", "church")], "radius_km": 3.0, "label": "church"},
    "park":         {"tags": [("leisure", "park"), ("leisure", "garden"), ("landuse", "recreation_ground")], "radius_km": 4.0, "label": "park"},
    "market":       {"tags": [("shop", "supermarket"), ("amenity", "marketplace"), ("shop", "convenience"), ("shop", "mall")], "radius_km": 4.0, "label": "market"},
    "supermarket":  {"tags": [("shop", "supermarket"), ("shop", "convenience"), ("building", "retail")], "radius_km": 4.0, "label": "supermarket"},
    "mall":         {"tags": [("shop", "mall"), ("building", "retail"), ("building", "commercial")], "radius_km": 5.0, "label": "shopping mall"},
    "bank":         {"tags": [("amenity", "bank")], "radius_km": 3.0, "label": "bank"},
    "atm":          {"tags": [("amenity", "atm")], "radius_km": 3.0, "label": "ATM"},
    "restaurant":   {"tags": [("amenity", "restaurant"), ("amenity", "fast_food"), ("amenity", "cafe")], "radius_km": 3.0, "label": "restaurant"},
    "gym":          {"tags": [("leisure", "fitness_centre"), ("sport", "fitness"), ("building", "sports_centre")], "radius_km": 4.0, "label": "gym"},
    "airport":      {"tags": [("aeroway", "aerodrome"), ("aeroway", "terminal")], "radius_km": 25.0, "label": "airport"},
    "highway":      {"tags": [("highway", "motorway"), ("highway", "trunk"), ("highway", "primary"), ("highway", "secondary")], "radius_km": 5.0, "label": "main road"},
    "main road":    {"tags": [("highway", "motorway"), ("highway", "trunk"), ("highway", "primary"), ("highway", "secondary")], "radius_km": 5.0, "label": "main road"},
    "metro station": {"tags": [("railway", "station"), ("public_transport", "station")], "radius_km": 5.0, "label": "metro station"},
    "train station": {"tags": [("railway", "station"), ("public_transport", "station")], "radius_km": 5.0, "label": "train station"},
    "bus stop":     {"tags": [("highway", "bus_stop"), ("public_transport", "platform"), ("amenity", "bus_station")], "radius_km": 3.0, "label": "bus stop"},
    "police station": {"tags": [("amenity", "police")], "radius_km": 5.0, "label": "police station"},
    "fire station": {"tags": [("amenity", "fire_station")], "radius_km": 5.0, "label": "fire station"},
}

def resolve_poi_type(poi_term: str) -> Optional[dict]:
    if not poi_term:
        return None
    return POI_TAG_MAP.get(str(poi_term).strip().lower())

# --------------------------------------------------------------------------- #
# Pure logic: query building
# --------------------------------------------------------------------------- #
def build_overpass_query(lat: float, lon: float, poi_terms: list[str]) -> Optional[str]:
    blocks = []
    for term in poi_terms:
        resolved = resolve_poi_type(term)
        if resolved is None:
            continue
        radius_m = int(resolved["radius_km"] * 1000)
        for key, value in resolved["tags"]:
            blocks.append(f'node(around:{radius_m},{lat},{lon})["{key}"="{value}"];')
            blocks.append(f'way(around:{radius_m},{lat},{lon})["{key}"="{value}"];')
            blocks.append(f'relation(around:{radius_m},{lat},{lon})["{key}"="{value}"];')

    if not blocks:
        return None

    body = "\n  ".join(blocks)
    return f"[out:json][timeout:{OVERPASS_QUERY_TIMEOUT_S}];\n(\n  {body}\n);\nout center tags;"

def build_batch_overpass_query(
    locations: list[tuple[float, float]],
    poi_terms: list[str],
) -> Optional[str]:
    """
    Build ONE Overpass query covering multiple listing locations at once,
    instead of one query per listing. Overpass's `around` filter accepts a
    single center point, so we emit one block per (location, tag) pair, but
    they're all combined into a single HTTP request -- this is the key
    difference. The server-side union automatically deduplicates any POI
    that happens to be within range of more than one listing.

    Sending 1 request instead of N is important on free public mirrors:
    each request counts against the server's shared rate limit, so batching
    directly reduces how often we get throttled/overloaded (429/502/read
    timeouts) under real usage.
    """
    blocks = []
    for lat, lon in locations:
        for term in poi_terms:
            resolved = resolve_poi_type(term)
            if resolved is None:
                continue
            radius_m = int(resolved["radius_km"] * 1000)
            for key, value in resolved["tags"]:
                blocks.append(f'node(around:{radius_m},{lat},{lon})["{key}"="{value}"];')
                blocks.append(f'way(around:{radius_m},{lat},{lon})["{key}"="{value}"];')
                blocks.append(f'relation(around:{radius_m},{lat},{lon})["{key}"="{value}"];')

    if not blocks:
        return None

    body = "\n  ".join(blocks)
    return f"[out:json][timeout:{OVERPASS_QUERY_TIMEOUT_S}];\n(\n  {body}\n);\nout center tags;"

# --------------------------------------------------------------------------- #
# Pure logic: distance + attribution
# --------------------------------------------------------------------------- #
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))

def _element_coords(element: dict) -> Optional[tuple[float, float]]:
    if "lat" in element and "lon" in element:
        return element["lat"], element["lon"]
    center = element.get("center")
    if center and "lat" in center and "lon" in center:
        return center["lat"], center["lon"]
    return None

def attribute_and_score(
    elements: list[dict],
    listing_lat: float,
    listing_lon: float,
    poi_terms: list[str],
) -> dict[str, dict]:
    results = {}

    for term in poi_terms:
        resolved = resolve_poi_type(term)
        if resolved is None:
            continue

        matching_tag_pairs = set(resolved["tags"])
        radius_km = resolved["radius_km"]
        best: Optional[dict] = None
        best_distance = float("inf")

        for element in elements:
            tags = element.get("tags", {})
            if not any(tags.get(k) == v for k, v in matching_tag_pairs):
                continue

            coords = _element_coords(element)
            if coords is None:
                continue

            dist_km = haversine_km(listing_lat, listing_lon, coords[0], coords[1])
            if dist_km > radius_km + 1e-6:
                continue

            if dist_km < best_distance:
                best_distance = dist_km
                raw_name = tags.get("name")
                clean_name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
                best = {
                    "verified": True,
                    "distance_m": round(dist_km * 1000),
                    "name": clean_name or resolved["label"].title(),
                    "osm_id": element.get("id"),
                }

        results[term] = best or {"verified": False, "distance_m": None, "name": None, "osm_id": None}

    return results

def parse_overpass_response(raw_json: dict) -> list[dict]:
    if not isinstance(raw_json, dict):
        return []
    elements = raw_json.get("elements")
    return elements if isinstance(elements, list) else []

# --------------------------------------------------------------------------- #
# Network I/O
# --------------------------------------------------------------------------- #
def fetch_overpass(query: str, session: Optional[requests.Session] = None) -> Optional[dict]:
    sess = session or requests.Session()

    # Honest headers only. No spoofed Referer -- see the comment on
    # USER_AGENT above for why that was actively causing the 406s.
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    any_endpoint_attempted = False

    for endpoint in OVERPASS_ENDPOINTS:
        if _is_endpoint_in_cooldown(endpoint):
            logger.info("Skipping %s (still in cooldown).", endpoint)
            continue

        any_endpoint_attempted = True
        try:
            logger.info("=========================================")
            logger.info(f"📍 SENDING MAP REQUEST TO: {endpoint}")

            response = sess.post(
                endpoint,
                data={"data": query},
                headers=headers,
                timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
            )

            if response.status_code in RETRYABLE_STATUS_CODES:
                logger.warning(
                    "Overpass %s gave a transient upstream error (%d) -- retrying once in %.0fs "
                    "before moving on.",
                    endpoint, response.status_code, RETRY_DELAY_S,
                )
                time.sleep(RETRY_DELAY_S)
                try:
                    response = sess.post(
                        endpoint,
                        data={"data": query},
                        headers=headers,
                        timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
                    )
                except requests.exceptions.RequestException as exc:
                    logger.warning("Retry against %s also failed (%s).", endpoint, exc)
                    _mark_endpoint_failed(endpoint, cooldown_s=TRANSIENT_COOLDOWN_S)
                    continue

            if response.status_code in HARD_BLOCK_STATUS_CODES:
                logger.warning(
                    "Overpass %s explicitly blocked/throttled us! Code: %d. "
                    "Backing off for %.0fs. Reason: %s",
                    endpoint, response.status_code, BLOCKED_COOLDOWN_S,
                    response.text[:150].replace("\n", " "),
                )
                _mark_endpoint_failed(endpoint, cooldown_s=BLOCKED_COOLDOWN_S)
                continue

            if response.status_code >= 400:
                logger.warning(
                    "Overpass %s returned an error. Code: %d. Reason: %s",
                    endpoint, response.status_code, response.text[:150].replace("\n", " "),
                )
                if response.status_code >= 500:
                    _mark_endpoint_failed(endpoint, cooldown_s=TRANSIENT_COOLDOWN_S)
                continue

            response.raise_for_status()
            _mark_endpoint_healthy(endpoint)

            response_data = response.json()
            if "remark" in response_data:
                logger.error(f"❌ SERVER HIDDEN ERROR: {response_data['remark']}")

            elements_found = len(response_data.get("elements", []))
            logger.info(f"✅ SUCCESS! Map server returned {elements_found} POI elements.")
            logger.info("=========================================")
            return response_data

        except requests.exceptions.ConnectTimeout:
            logger.warning(
                "Overpass endpoint %s: could not even open a TCP connection within %ss "
                "-- this points to a network/firewall/DNS problem, not server load.",
                endpoint, CONNECT_TIMEOUT_S,
            )
            _mark_endpoint_failed(endpoint, cooldown_s=TRANSIENT_COOLDOWN_S)
        except requests.exceptions.ReadTimeout:
            logger.warning(
                "Overpass endpoint %s: connected fine but no response within %ss "
                "-- server is slow/overloaded, or a proxy is silently stalling the response.",
                endpoint, READ_TIMEOUT_S,
            )
            _mark_endpoint_failed(endpoint, cooldown_s=TRANSIENT_COOLDOWN_S)
        except requests.exceptions.Timeout:
            logger.warning("Overpass endpoint %s timed out.", endpoint)
            _mark_endpoint_failed(endpoint, cooldown_s=TRANSIENT_COOLDOWN_S)
        except requests.exceptions.RequestException as exc:
            logger.warning("Overpass endpoint %s failed (%s).", endpoint, exc)
            _mark_endpoint_failed(endpoint, cooldown_s=TRANSIENT_COOLDOWN_S)
        except (ValueError, json.JSONDecodeError):
            logger.warning("Overpass endpoint %s returned unparseable JSON.", endpoint)
            _mark_endpoint_failed(endpoint, cooldown_s=TRANSIENT_COOLDOWN_S)

    if not any_endpoint_attempted:
        logger.error("All Overpass endpoints are in cooldown from a previous block/failure.")
    else:
        logger.error("All Overpass endpoints failed on this attempt.")
    return None

# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def _cache_key(lat: float, lon: float, poi_terms: list[str]) -> str:
    rounded_lat = round(lat, 4)
    rounded_lon = round(lon, 4)
    terms_key = ",".join(sorted(set(t.strip().lower() for t in poi_terms if t)))
    return f"{rounded_lat}:{rounded_lon}:{terms_key}"

def load_poi_cache(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load POI cache at %s (%s). Starting fresh.", path, exc)
    return {}

def save_poi_cache(cache: dict, path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.error("Failed to save POI cache to %s: %s", path, exc)

# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _has_valid_coordinates(lat: Any, lon: Any) -> bool:
    if lat is None or lon is None:
        return False
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return PK_LAT_RANGE[0] <= lat <= PK_LAT_RANGE[1] and PK_LON_RANGE[0] <= lon <= PK_LON_RANGE[1]

def verify_pois_for_listing(
    listing: dict,
    poi_terms: list[str],
    cache: dict,
    fetch_fn=fetch_overpass,
    cache_path: str = DEFAULT_CACHE_PATH,
) -> dict[str, dict]:
    if not poi_terms:
        return {}

    results: dict[str, dict] = {}
    unresolvable = [t for t in poi_terms if resolve_poi_type(t) is None]
    resolvable = [t for t in poi_terms if t not in unresolvable]

    for term in unresolvable:
        results[term] = {"verified": None, "distance_m": None, "name": None, "reason": "unknown POI type -- not in our verification vocabulary"}

    lat, lon = listing.get("latitude"), listing.get("longitude")
    if not _has_valid_coordinates(lat, lon):
        for term in resolvable:
            results[term] = {"verified": None, "distance_m": None, "name": None, "reason": "listing has no valid coordinates"}
        return results

    lat, lon = float(lat), float(lon)

    cache_key = _cache_key(lat, lon, resolvable)
    if resolvable and cache_key in cache:
        cached = cache[cache_key]
        results.update(cached)
        return results

    if not resolvable:
        return results

    query = build_overpass_query(lat, lon, resolvable)
    if query is None:
        for term in resolvable:
            results[term] = {"verified": None, "distance_m": None, "name": None, "reason": "unknown POI type"}
        return results

    raw_response = fetch_fn(query)
    if raw_response is None:
        for term in resolvable:
            results[term] = {"verified": None, "distance_m": None, "name": None, "reason": "Overpass API unavailable"}
        return results

    elements = parse_overpass_response(raw_response)
    scored = attribute_and_score(elements, lat, lon, resolvable)
    results.update(scored)

    if resolvable:
        cache[cache_key] = scored
        save_poi_cache(cache, cache_path)

    return results

def verify_pois_for_results(
    results: list[dict],
    poi_terms: list[str],
    cache_path: str = DEFAULT_CACHE_PATH,
    fetch_fn=fetch_overpass,
    max_total_seconds: float = 90.0,
) -> list[dict]:
    """
    Verify POIs for a batch of listings with as FEW network requests as
    possible. Every listing that needs fresh data (i.e. isn't already
    cached) is folded into a single Overpass query, rather than issuing one
    request per listing -- this is what keeps us under free public mirrors'
    rate limits in practice. Falls back to marking listings unverified
    (never crashes/hangs the caller) if the batch fetch fails or the time
    budget runs out.
    """
    if not poi_terms:
        for r in results:
            r["poi_verification"] = {}
        return results

    cache = load_poi_cache(cache_path)
    start_time = time.time()

    enriched = [dict(listing) for listing in results]

    # Bucket listings into: unresolvable terms (no network needed), invalid
    # coordinates (no network needed), already cached (no network needed),
    # and everything else (needs to go in the batch request).
    unresolvable = [t for t in poi_terms if resolve_poi_type(t) is None]
    resolvable = [t for t in poi_terms if t not in unresolvable]

    to_fetch_idx: list[int] = []
    to_fetch_coords: list[tuple[float, float]] = []

    for i, listing in enumerate(enriched):
        verification: dict[str, dict] = {}
        for term in unresolvable:
            verification[term] = {"verified": None, "distance_m": None, "name": None, "reason": "unknown POI type -- not in our verification vocabulary"}

        lat, lon = listing.get("latitude"), listing.get("longitude")
        if not resolvable:
            listing["poi_verification"] = verification
            continue

        if not _has_valid_coordinates(lat, lon):
            for term in resolvable:
                verification[term] = {"verified": None, "distance_m": None, "name": None, "reason": "listing has no valid coordinates"}
            listing["poi_verification"] = verification
            continue

        lat, lon = float(lat), float(lon)
        cache_key = _cache_key(lat, lon, resolvable)
        if cache_key in cache:
            verification.update(cache[cache_key])
            listing["poi_verification"] = verification
        else:
            # Placeholder -- filled in after the batch fetch below.
            listing["poi_verification"] = verification
            to_fetch_idx.append(i)
            to_fetch_coords.append((lat, lon))

    if not to_fetch_idx:
        return enriched

    if (time.time() - start_time) > max_total_seconds:
        for i in to_fetch_idx:
            for term in resolvable:
                enriched[i]["poi_verification"][term] = {
                    "verified": None, "distance_m": None, "name": None,
                    "reason": "skipped -- verification time budget exceeded for this search",
                }
        return enriched

    query = build_batch_overpass_query(to_fetch_coords, resolvable)
    if query is None:
        for i in to_fetch_idx:
            for term in resolvable:
                enriched[i]["poi_verification"][term] = {"verified": None, "distance_m": None, "name": None, "reason": "unknown POI type"}
        return enriched

    raw_response = fetch_fn(query)
    if raw_response is None:
        for i in to_fetch_idx:
            for term in resolvable:
                enriched[i]["poi_verification"][term] = {"verified": None, "distance_m": None, "name": None, "reason": "Overpass API unavailable"}
        return enriched

    elements = parse_overpass_response(raw_response)

    # One shared element set, scored independently per listing -- a POI
    # only counts for a listing if it's within THAT listing's radius, even
    # though the fetch itself covered every listing in the batch at once.
    for i, (lat, lon) in zip(to_fetch_idx, to_fetch_coords):
        scored = attribute_and_score(elements, lat, lon, resolvable)
        enriched[i]["poi_verification"].update(scored)
        cache[_cache_key(lat, lon, resolvable)] = scored

    save_poi_cache(cache, cache_path)
    return enriched

def verify_live_smoke_test():
    print("Running LIVE smoke test against real Overpass API (requires internet)...")
    test_listing = {"latitude": 31.5085, "longitude": 74.3453}
    poi_terms = ["market", "school", "totally_made_up_poi_type"]

    cache = {}
    result = verify_pois_for_listing(test_listing, poi_terms, cache)
    print(json.dumps(result, indent=2))

    assert "totally_made_up_poi_type" in result
    assert result["totally_made_up_poi_type"]["verified"] is None
    print("\nUnresolvable POI type correctly returned verified=None (not False). Good.")

if __name__ == "__main__":
    verify_live_smoke_test()