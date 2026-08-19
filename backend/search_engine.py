#!/usr/bin/env python3
"""
search_engine.py
==================

Steps 2 & 3 of the Real Estate Property Finder pipeline.

Takes the structured intent JSON produced by `intent_extractor.py` (Step 1)
and searches it against the preprocessed, embedded dataset produced by
`preprocess_dataset.py` (`listings.json`):

    Stage 1 - Hard filtering (Pandas):
        Apply strict logical filters (property_type, price_max, marla
        range, location substring match) with a graceful degradation
        safety net so the pipeline never returns a blank page.

    Stage 2 - Semantic vector ranking (SentenceTransformers):
        Embed the query's soft_signals and rank Stage 1 candidates by
        cosine similarity against their precomputed title_embedding.

    Stage 3 - Top-N selection:
        Return the top `top_k` candidates as clean JSON-serializable
        dicts with property details plus a similarity_score.

Usage:
    from search_engine import search_listings
    results = search_listings(intent, listings_path="listings.json", top_k=5)

    # or run the built-in demo harness:
    python search_engine.py
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("search_engine")


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

# Module-level cache so the (fairly expensive to load) embedding model and
# the listings dataset are only loaded once per process, even across
# multiple search_listings() calls.
_MODEL_CACHE: dict[str, Any] = {}
_LISTINGS_CACHE: dict[str, pd.DataFrame] = {}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_listings(listings_path: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Load listings.json into a Pandas DataFrame, pre-converting the stored
    `title_embedding` lists into NumPy arrays for fast vector math.

    Results are cached in-process by path so repeated calls (e.g. across
    multiple user queries in the same run) don't re-read/re-parse the file.
    """
    if use_cache and listings_path in _LISTINGS_CACHE:
        return _LISTINGS_CACHE[listings_path]

    logger.info("Loading listings from '%s'...", listings_path)
    try:
        with open(listings_path, "r", encoding="utf-8") as fh:
            raw_records = json.load(fh)
    except FileNotFoundError:
        logger.error("Listings file not found: %s", listings_path)
        raise
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse listings JSON at '%s': %s", listings_path, exc)
        raise

    if not raw_records:
        logger.warning("Listings file '%s' is empty.", listings_path)

    df = pd.DataFrame(raw_records)

    required_cols = [
        "id",
        "title",
        "url",
        "property_type",
        "subtype",
        "location",
        "price_pkr",
        "area_marla",
        "bedrooms",
        "bathrooms",
        "latitude",
        "longitude",
        "title_embedding",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(
            "Listings dataset is missing expected columns: %s. "
            "Filling with safe defaults.",
            missing,
        )
        for col in missing:
            df[col] = None

    # Pre-convert embedding lists -> NumPy arrays once, up front, so Stage 2
    # never has to repeat this conversion per-query.
    def _to_vector(value: Any) -> Optional[np.ndarray]:
        if isinstance(value, list) and len(value) > 0:
            return np.asarray(value, dtype=np.float32)
        return None

    df["title_embedding_vec"] = df["title_embedding"].apply(_to_vector)

    # Normalize location to string for reliable substring matching later.
    df["location"] = df["location"].astype(str)

    logger.info("Loaded %d listings from '%s'.", len(df), listings_path)

    if use_cache:
        _LISTINGS_CACHE[listings_path] = df

    return df


def _get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """Load (and cache) the SentenceTransformer embedding model."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    if SentenceTransformer is None:
        logger.error(
            "sentence_transformers is not installed. "
            "Semantic ranking will fall back to a neutral score."
        )
        _MODEL_CACHE[model_name] = None
        return None

    try:
        logger.info("Loading embedding model '%s'...", model_name)
        model = SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load embedding model '%s': %s", model_name, exc)
        model = None

    _MODEL_CACHE[model_name] = model
    return model


# --------------------------------------------------------------------------- #
# Stage 1: Hard filtering
# --------------------------------------------------------------------------- #
def normalize_location_text(text: str) -> str:
    """
    Normalize a location string for matching purposes only (never for
    display) -- strips commas and collapses whitespace so that
    "DHA, Multan" and "DHA Multan" are treated as the same location.

    This matters because the intent-extraction model (Qwen or Gemini)
    isn't perfectly consistent about comma placement, and a plain
    substring match would otherwise silently fail on a comma that
    carries no real meaning (e.g. "DHA Multan" not found inside
    "DHA, Multan, Punjab, Pakistan" due to the comma breaking the
    substring). Fixing it here, once, at the matching layer is far more
    robust than trying to force perfect punctuation consistency out of
    an LLM across every training example.
    """
    if not text:
        return ""
    return " ".join(str(text).replace(",", " ").split()).lower()


# Matches granular qualifiers that real scraped location data usually
# does NOT capture at this level of detail (confirmed empirically: 93%
# of DHA+Lahore listings are stored as the single generic string "DHA
# Defence, Lahore", with zero phase-number granularity in the location
# field). Stripping these before the Stage 1 hard filter means a query
# for "DHA Phase 6" correctly matches "DHA Defence, Lahore" listings
# instead of returning zero results -- the phase-6-specific detail, if
# it exists at all, only lives in free-text titles, so it's handled as
# a Stage 3 semantic ranking signal instead (see build_soft_signal_query).
#
# The trailing token is deliberately bounded to 1-4 characters: phase/
# sector/block identifiers are short (numbers, roman numerals, single
# letters, hyphenated codes like "F-7") -- an unbounded match would
# incorrectly swallow the next real word too, e.g. stripping "Islamabad"
# out of "Sector, Islamabad" because it also matches [a-z0-9]+.
_GRANULAR_QUALIFIER_PATTERN = re.compile(
    r"\b(phase|sector|block|street|st)\s*[-#]?\s*[a-z0-9]{1,4}\b", re.IGNORECASE
)


def extract_core_location(text: str) -> str:
    """
    Strips phase/sector/block/street-level qualifiers from a location
    string, leaving the scheme + city level that the real dataset
    reliably supports for hard filtering. E.g. "DHA Phase 6, Lahore" ->
    "DHA, Lahore". Falls back to the normalized full text if stripping
    would leave nothing (e.g. a location that's ONLY a phase number with
    no scheme name at all).
    """
    normalized = normalize_location_text(text)
    if not normalized:
        return ""
    stripped = _GRANULAR_QUALIFIER_PATTERN.sub("", normalized)
    stripped = " ".join(stripped.split())  # collapse any double-spaces left behind
    return stripped if stripped else normalized


SIZE_EXPAND_PCT = 0.18   # widen marla_min/marla_max window by ~18% each direction
PRICE_EXPAND_PCT = 0.12  # raise price_max ceiling by ~12%
NEARBY_RADIUS_KM = 10.0  # "nearby areas" tier: physical distance, not just shared city name


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Great-circle distance in km between one reference point (lat1, lon1)
    and an array of points (lat2, lon2). Vectorized for use directly
    against a DataFrame column, so this stays fast even at 146k+ rows.
    """
    R = 6371.0  # Earth's radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_location_centroid(df: pd.DataFrame, location: str) -> Optional[tuple[float, float]]:
    """
    Finds the geographic centroid (average lat/lon) of listings whose
    location text matches the query's CORE location -- regardless of
    property_type or subtype, since a physical place has one set of
    coordinates no matter what's being sold there. This gives us a real
    reference point for "nearby areas" without needing a live geocoding
    API call: we reuse coordinates the dataset already has.

    Returns None if no matching listings have valid coordinates (e.g. a
    location string with no matches at all, or all matches lack lat/lon).
    """
    core = extract_core_location(location)
    query_words = [w for w in core.split() if w]
    if not query_words:
        return None

    normalized_listing_loc = df["location"].apply(normalize_location_text)
    word_mask = pd.Series(True, index=df.index)
    for word in query_words:
        word_mask &= normalized_listing_loc.str.contains(word, na=False, regex=False)

    matching = df[word_mask]
    coords = matching[["latitude", "longitude"]].dropna()
    if coords.empty:
        return None

    return (float(coords["latitude"].mean()), float(coords["longitude"].mean()))


def apply_hard_filters(
    df: pd.DataFrame,
    intent: dict,
    size_expand_pct: float = 0.0,
    price_expand_pct: float = 0.0,
    location_mode: str = "full",
    centroid: Optional[tuple[float, float]] = None,
    radius_km: float = NEARBY_RADIUS_KM,
) -> pd.DataFrame:
    """
    Apply strict logical filters derived from the intent JSON.

    property_type and subtype are ALWAYS applied at full strictness --
    there is no relaxation path for either. A Commercial Plot search
    returning a Residential Plot isn't a close match, it's a wrong
    answer with a price tag attached; the same logic applies even more
    strongly to property_type itself (a House buyer has no use for a
    Plot, regardless of how good the price is). Everything else --
    size, budget, location -- is genuinely negotiable and is relaxed
    parametrically by filter_with_fallback() instead of being switched
    off entirely.

    Args:
        size_expand_pct: widens the marla_min/marla_max window outward
            by this fraction (e.g. 0.18 = 18% wider on each side).
        price_expand_pct: raises price_max by this fraction (e.g. 0.12
            = accept prices up to 12% over the stated budget).
        location_mode: "full" (scheme+city, word-level match -- the
            normal default), "radius" (physical distance from `centroid`
            within `radius_km` -- real "nearby areas", not just shared
            city name), "city_only" (drop the scheme name, match on city
            alone -- last resort), or "none" (no location constraint).
        centroid: (lat, lon) reference point, required when
            location_mode="radius". See compute_location_centroid().
        radius_km: search radius in km, used only when
            location_mode="radius".
    """
    mask = pd.Series(True, index=df.index)

    property_type = intent.get("property_type")
    if property_type:
        mask &= df["property_type"].str.lower() == str(property_type).lower()

    subtype = intent.get("subtype")
    if subtype and str(subtype).lower() != "any" and "subtype" in df.columns:
        mask &= df["subtype"].str.lower() == str(subtype).lower()

    price_max = intent.get("price_max")
    if price_max is not None:
        effective_price_max = price_max * (1 + price_expand_pct)
        mask &= df["price_pkr"] <= effective_price_max

    marla_min = intent.get("marla_min")
    marla_max = intent.get("marla_max")
    if marla_min is not None:
        effective_min = marla_min * (1 - size_expand_pct)
        mask &= df["area_marla"] >= effective_min
    if marla_max is not None:
        effective_max = marla_max * (1 + size_expand_pct)
        mask &= df["area_marla"] <= effective_max

    location = intent.get("location")
    if location and location_mode == "radius":
        # Physical distance from a real reference point -- this is what
        # "nearby areas" should actually mean (a buyer open to Bahria
        # Orchard because it's 3km from DHA, not because it happens to
        # share the word "Lahore"). Listings without coordinates can't
        # be evaluated for distance and are excluded, not silently kept.
        if centroid is None:
            logger.warning("location_mode='radius' requested but no centroid available -- skipping location filter for this attempt.")
        else:
            has_coords = df["latitude"].notna() & df["longitude"].notna()
            distances = pd.Series(np.inf, index=df.index)
            distances[has_coords] = haversine_km(
                centroid[0], centroid[1],
                df.loc[has_coords, "latitude"].to_numpy(),
                df.loc[has_coords, "longitude"].to_numpy(),
            )
            mask &= distances <= radius_km
    elif location and location_mode != "none":
        # Match on the CORE location (scheme + city, phase/sector/block
        # stripped) rather than the full phrase -- confirmed against
        # real data that phase-level detail mostly isn't present in
        # the location field at all, so hard-filtering on it would
        # incorrectly return zero results for very common searches
        # like "DHA Phase 6". The full phrase (with phase intact) is
        # still used for Stage 3 semantic ranking -- see
        # build_soft_signal_query -- so phase-specific listings still
        # surface first when the detail exists in the title.
        #
        # Matching is WORD-LEVEL (every query word must appear
        # somewhere in the listing's location), not one contiguous
        # substring: real addresses interleave extra words between
        # the scheme name and city (e.g. "DHA Defence, Lahore" has
        # "Defence" between "DHA" and "Lahore").
        core_query_loc = extract_core_location(location)
        query_words = [w for w in core_query_loc.split() if w]

        if location_mode == "city_only":
            # Drop everything except the LAST word, which is the city in
            # every location string this pipeline produces (Step 0 builds
            # location as "<address>, <city>", and extract_core_location
            # preserves word order). This is the true last-resort tier:
            # same city, any scheme/society, no distance guarantee.
            query_words = query_words[-1:]

        if query_words:
            normalized_listing_loc = df["location"].apply(normalize_location_text)
            word_mask = pd.Series(True, index=df.index)
            for word in query_words:
                word_mask &= normalized_listing_loc.str.contains(word, na=False, regex=False)
            mask &= word_mask

    return df[mask]


def filter_with_fallback(df: pd.DataFrame, intent: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply hard filters with a graceful-degradation safety net: if strict
    filtering yields zero candidates, progressively relax constraints
    in order of how negotiable they actually are to a real buyer:

        1. Size          - widen the area window ~18% each direction
        2. Budget        - raise the price ceiling ~12%
        3. Nearby areas  - physical radius (~10km) around the requested
                            location's real coordinates, e.g. a DHA Lahore
                            search may pick up a listing 3km away under a
                            different scheme name. This is a genuine
                            "adjacent areas" tier, not just a shared city
                            name.
        4. City-wide      - drop to city-level text matching (any scheme,
                            any distance within the city). Absolute last
                            resort before giving up.

    property_type and subtype are NEVER relaxed, at any tier. If a
    buyer wants a Commercial Plot, a Residential Plot is a 100% failure
    no matter how good the price/size/location match is -- so if every
    tier above is exhausted and still nothing matches, this returns an
    EMPTY result set rather than ever crossing into the wrong property
    type or subtype. Honest "we found nothing" beats a confidently
    wrong answer.

    Returns:
        (candidates_df, relaxations_applied) -- relaxations_applied is
        a list of constraint names that had to be loosened. An empty
        DataFrame with a non-empty relaxations list means every tier
        was tried and none worked; build_result_message() uses this to
        tell the user honestly rather than silently substituting an
        unrelated listing.
    """
    logger.info("Applying hard filters (property_type, subtype strict; price, area, location negotiable)...")
    candidates = apply_hard_filters(df, intent)

    if len(candidates) > 0:
        logger.info("Hard filtering found %d candidates on first pass.", len(candidates))
        return candidates, []

    logger.warning("Strict filtering returned 0 candidates. Engaging graceful degradation...")

    size_set = intent.get("marla_min") is not None or intent.get("marla_max") is not None
    price_set = intent.get("price_max") is not None
    location = intent.get("location")
    location_set = bool(location)

    # Compute the geographic reference point ONCE, up front, from the
    # full unfiltered dataset (any property_type/subtype) -- a physical
    # place has one set of coordinates regardless of what's for sale
    # there. Reused across the "nearby" tier without re-computing.
    centroid = compute_location_centroid(df, location) if location_set else None
    if location_set and centroid is None:
        logger.info("No coordinates available to compute a 'nearby areas' centroid for %r -- that tier will be skipped.", location)

    relaxation_steps = [
        ("size", {"size_expand_pct": SIZE_EXPAND_PCT}, size_set),
        ("budget", {"size_expand_pct": SIZE_EXPAND_PCT, "price_expand_pct": PRICE_EXPAND_PCT}, price_set),
        (
            "nearby areas",
            {
                "size_expand_pct": SIZE_EXPAND_PCT, "price_expand_pct": PRICE_EXPAND_PCT,
                "location_mode": "radius", "centroid": centroid, "radius_km": NEARBY_RADIUS_KM,
            },
            location_set and centroid is not None,
        ),
        (
            "city-wide",
            {"size_expand_pct": SIZE_EXPAND_PCT, "price_expand_pct": PRICE_EXPAND_PCT, "location_mode": "city_only"},
            location_set,
        ),
    ]

    applied: list[str] = []
    for step_name, relax_kwargs, was_constrained in relaxation_steps:
        # Only report/attempt a step if it's actually applicable -- e.g.
        # skip "nearby areas" entirely if we couldn't compute a centroid,
        # rather than silently no-op'ing it and confusingly reporting it
        # as "tried".
        if not was_constrained:
            continue
        applied.append(step_name)
        candidates = apply_hard_filters(df, intent, **relax_kwargs)
        if len(candidates) > 0:
            logger.warning("Found %d candidates after relaxing: %s.", len(candidates), applied)
            return candidates, applied

    # Every negotiable tier exhausted. property_type/subtype are never
    # relaxed -- return empty rather than show the wrong kind of property.
    logger.warning(
        "No candidates found even after expanding size, budget, and location. "
        "Returning zero results (property_type/subtype are never relaxed)."
    )
    return candidates, applied


def build_result_message(intent: dict, relaxations: list[str], candidates: pd.DataFrame) -> Optional[str]:
    """
    Compose a plain-language message telling the user exactly what
    happened when their exact query didn't match anything -- e.g. an
    unrealistic price for the requested area, or no listings in the
    requested location -- and what was shown instead.

    Returns None when the strict query matched cleanly (no message needed).
    """
    if not relaxations:
        return None

    property_type = intent.get("property_type") or "property"
    location = intent.get("location")
    price_max = intent.get("price_max")

    # Describe what was requested, in plain language.
    requested_bits = [property_type]
    marla_min, marla_max = intent.get("marla_min"), intent.get("marla_max")
    if marla_min and marla_max and marla_min == marla_max:
        requested_bits.insert(0, f"{marla_min:g} Marla")
    if location:
        requested_bits.append(f"in {location}")
    if price_max:
        requested_bits.append(f"under PKR {price_max:,}")
    requested_desc = " ".join(requested_bits)

    if candidates.empty:
        # Every negotiable tier (size, budget, location) was tried and
        # NOTHING matched -- property_type/subtype are never relaxed, so
        # this is an honest "we genuinely have nothing" rather than a
        # silent substitution of the wrong kind of property.
        return (
            f"We couldn't find any '{requested_desc}' listings, even after expanding "
            f"the size, budget, and nearby areas we searched. There may genuinely be "
            f"nothing matching this combination right now."
        )

    loosened_desc = {
        "size": f"the size (by up to {int(SIZE_EXPAND_PCT * 100)}%)",
        "budget": f"the budget (by up to {int(PRICE_EXPAND_PCT * 100)}%)",
        "nearby areas": f"the search area to within {int(NEARBY_RADIUS_KM)}km of {intent.get('location') or 'your requested area'}",
        "city-wide": "the search area to anywhere in the city",
    }
    loosened_readable = [loosened_desc[step] for step in relaxations if step in loosened_desc]

    message = f"No exact matches for '{requested_desc}'."

    if "budget" in relaxations and price_max and not candidates.empty:
        cheapest = candidates["price_pkr"].min()
        if cheapest > price_max:
            message += (
                f" Your budget of PKR {price_max:,} is below the typical market rate for this — "
                f"the closest matches start from PKR {cheapest:,}."
            )

    if loosened_readable:
        message += f" We relaxed {', '.join(loosened_readable)} to show you the closest available options."

    return message


# --------------------------------------------------------------------------- #
# Stage 2: Semantic vector ranking
# --------------------------------------------------------------------------- #
def extract_granular_qualifier(location: str) -> str:
    """
    Returns just the phase/sector/block-level qualifier text stripped out
    by extract_core_location() (e.g. "DHA Phase 6, Lahore" -> "Phase 6"),
    or "" if there was none. Used to boost Stage 3 ranking toward
    listings whose TITLE mentions the specific phase/sector the hard
    filter had to ignore -- see apply_hard_filters() for why the hard
    filter can't use this directly.
    """
    if not location:
        return ""
    matches = _GRANULAR_QUALIFIER_PATTERN.findall(location)
    # findall with a capturing group returns only the group; re-run a
    # full-match search to get the whole qualifier phrase, not just the
    # captured keyword.
    full_matches = _GRANULAR_QUALIFIER_PATTERN.finditer(location)
    return " ".join(m.group(0) for m in full_matches)


def build_soft_signal_query(intent: dict) -> str:
    """
    Combine soft_signals (and optionally poi) into a single query string
    for Stage 3 semantic ranking.

    Also appends any phase/sector/block qualifier stripped from the
    location during hard filtering (e.g. "Phase 6"), so listings whose
    title happens to mention that specific phase rank higher -- this is
    how phase-level specificity gets recovered as a soft signal after
    Stage 1 had to treat it as unmatchable structured data. See
    apply_hard_filters() and extract_core_location() for the full
    reasoning.
    """
    soft_signals = intent.get("soft_signals") or []
    query_terms = [str(term).strip() for term in soft_signals if str(term).strip()]

    granular = extract_granular_qualifier(intent.get("location") or "")
    if granular:
        query_terms.append(granular)

    return " ".join(query_terms)


def cosine_similarity_matrix(query_vec: np.ndarray, candidate_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and a matrix
    of candidate vectors, returning scores normalized to [0.0, 1.0].
    """
    query_norm = np.linalg.norm(query_vec)
    candidate_norms = np.linalg.norm(candidate_vecs, axis=1)

    # Avoid division by zero for any zero vectors.
    safe_query_norm = query_norm if query_norm > 1e-12 else 1e-12
    safe_candidate_norms = np.where(candidate_norms > 1e-12, candidate_norms, 1e-12)

    dot_products = candidate_vecs @ query_vec
    cosine_sim = dot_products / (safe_candidate_norms * safe_query_norm)

    # Cosine similarity is naturally in [-1, 1]; rescale to [0, 1] so the
    # score reads intuitively as a "match strength" percentage.
    normalized = (cosine_sim + 1.0) / 2.0
    return np.clip(normalized, 0.0, 1.0)


def rank_by_semantic_similarity(
    candidates: pd.DataFrame,
    intent: dict,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> pd.DataFrame:
    """
    Compute a similarity_score for every candidate based on cosine
    similarity between the embedded soft_signals query and each
    listing's precomputed title_embedding.

    If there are no soft_signals in the intent, or embeddings/model are
    unavailable, every candidate gets a neutral score (0.5) so ranking
    degrades to "no particular semantic preference" rather than failing.
    """
    candidates = candidates.copy()
    query_text = build_soft_signal_query(intent)

    if not query_text:
        logger.info("No soft_signals present in intent; skipping semantic ranking.")
        candidates["similarity_score"] = 0.5
        return candidates

    model = _get_embedding_model(model_name)
    if model is None:
        logger.warning(
            "Embedding model unavailable; assigning neutral similarity scores."
        )
        candidates["similarity_score"] = 0.5
        return candidates

    logger.info("Embedding soft-signal query: %r", query_text)
    try:
        query_embedding = model.encode(query_text, convert_to_numpy=True).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to embed soft-signal query: %s", exc)
        candidates["similarity_score"] = 0.5
        return candidates

    # Rows with a missing/invalid title_embedding get a neutral score and
    # are excluded from the vectorized similarity computation.
    valid_mask = candidates["title_embedding_vec"].apply(lambda v: v is not None)
    candidates["similarity_score"] = 0.5

    if valid_mask.any():
        valid_vecs = np.vstack(candidates.loc[valid_mask, "title_embedding_vec"].to_list())
        scores = cosine_similarity_matrix(query_embedding, valid_vecs)
        candidates.loc[valid_mask, "similarity_score"] = scores

    invalid_count = (~valid_mask).sum()
    if invalid_count:
        logger.warning(
            "%d candidates had no valid title_embedding; assigned neutral score.",
            invalid_count,
        )

    return candidates


# --------------------------------------------------------------------------- #
# Stage 3: Top-N selection & formatting
# --------------------------------------------------------------------------- #
def _compute_tiebreak(candidates: pd.DataFrame, intent: dict) -> tuple[pd.Series, bool]:
    """
    Determines what to sort by (and in which direction) when
    similarity_score ties -- which happens on every query with no
    soft_signals, since every candidate gets the same neutral 0.5 score.

    The naive fix ("always sort price/area descending to surface
    premium listings") has a real bug: marla_min/marla_max can represent
    either an open-ended ceiling ("under 10 marla") OR an exact stated
    value ("5 marla" -> min=max=5.0). Descending sort on an EXACT value
    systematically favors listings at the top of the relaxation
    tolerance band over the listing that's the literal exact match the
    user asked for -- e.g. a 5.7 marla listing would outrank an exact
    5.0 marla match for a "5 marla" query. The correct behavior depends
    on which of these the user actually stated:

        - price_max is always a ceiling (our schema has no price_min) ->
          descending is correct: closer to budget = more premium.
        - marla exact value (min == max) -> sort by DISTANCE to that
          value, closest first -- not by raw magnitude.
        - marla ceiling only (max set, min unset, "under X") ->
          descending, same reasoning as price.
        - marla floor only (min set, max unset, "at least X") ->
          ascending toward the floor -- someone who said "at least 5
          marla" isn't necessarily asking for the single biggest
          property in the dataset.
        - marla explicit range (min != max, both set) -> descending,
          leaning toward "more for your money" within the stated range.
        - nothing numeric stated at all -> no real signal to lean on;
          defaults to price descending to avoid systematically
          surfacing the cheapest/smallest matches, same reasoning as
          the price-ceiling case, just without a real ceiling to anchor to.

    Returns (tiebreak_series, ascending).
    """
    marla_min = intent.get("marla_min")
    marla_max = intent.get("marla_max")
    price_max = intent.get("price_max")

    if price_max is not None:
        return candidates["price_pkr"], False

    if marla_min is not None and marla_max is not None:
        if abs(marla_min - marla_max) < 1e-9:
            target = marla_min
            return (candidates["area_marla"] - target).abs(), True
        return candidates["area_marla"], False
    if marla_max is not None:
        return candidates["area_marla"], False
    if marla_min is not None:
        return candidates["area_marla"], True

    return candidates["price_pkr"], False


def format_results(ranked_candidates: pd.DataFrame, intent: dict, top_k: int) -> list[dict]:
    """
    Select the top_k candidates and format them as clean JSON dicts.

    Sort order: similarity_score (descending) first, then a
    context-dependent tiebreak from _compute_tiebreak() -- e.g. price
    descending for a budget-ceiling query, or distance-to-target for an
    exact area value -- and finally price_pkr ascending as a
    deterministic last resort, so ordering is never left to arbitrary
    DataFrame row order even when both prior keys tie (e.g. multiple
    listings that are all exactly the requested area).
    """
    if ranked_candidates.empty:
        return []

    tiebreak_series, tiebreak_ascending = _compute_tiebreak(ranked_candidates, intent)
    sortable = ranked_candidates.copy()
    sortable["_tiebreak"] = tiebreak_series

    top_candidates = sortable.sort_values(
        ["similarity_score", "_tiebreak", "price_pkr"],
        ascending=[False, tiebreak_ascending, True],
    ).head(top_k)

    results = []
    for _, row in top_candidates.iterrows():
        try:
            results.append(
                {
                    "id": int(row["id"]),
                    "title": str(row["title"]),
                    "url": str(row["url"]) if "url" in row and pd.notna(row["url"]) else "",
                    "property_type": str(row["property_type"]),
                    "subtype": str(row["subtype"]) if "subtype" in row and pd.notna(row["subtype"]) else None,
                    "location": str(row["location"]),
                    "price_pkr": int(row["price_pkr"]),
                    "area_marla": float(row["area_marla"]),
                    "bedrooms": int(row["bedrooms"]),
                    "bathrooms": int(row["bathrooms"]),
                    "latitude": float(row["latitude"]) if pd.notna(row["latitude"]) else None,
                    "longitude": float(row["longitude"]) if pd.notna(row["longitude"]) else None,
                    "similarity_score": round(float(row["similarity_score"]), 4),
                }
            )
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Skipping malformed candidate row during formatting: %s", exc)

    return results


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def search_listings(
    intent: dict,
    listings_path: str = "listings.json",
    top_k: int = DEFAULT_TOP_K,
    embedding_model: str = EMBEDDING_MODEL_NAME,
) -> dict:
    """
    Search the preprocessed listings dataset using a structured intent
    dict (as produced by intent_extractor.extract_query_intent).

    Args:
        intent: Structured query intent (property_type, subtype, location,
            marla_min/max, price_max, soft_signals, poi).
        listings_path: Path to the preprocessed listings.json file.
        top_k: Number of top-ranked results to return.
        embedding_model: sentence-transformers model name for Stage 2.

    Returns:
        A dict of the form:
            {
                "message": str or None,       # user-facing note, e.g. why
                                               # constraints were relaxed;
                                               # None when the exact query
                                               # matched cleanly
                "relaxed_constraints": list[str],  # which filters were
                                               # loosened to find results,
                                               # empty if none were
                "results": list[dict]         # up to top_k listings,
                                               # sorted by similarity_score
            }
        "results" is an empty list only if the listings dataset itself is
        empty or fails to load.
    """
    if not isinstance(intent, dict):
        logger.error("Invalid intent provided (not a dict): %r", intent)
        return {"message": "Invalid search request.", "relaxed_constraints": [], "results": []}

    try:
        df = load_listings(listings_path)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.error("Could not load listings dataset; returning no results.")
        return {
            "message": "The listings database is currently unavailable. Please try again shortly.",
            "relaxed_constraints": [],
            "results": [],
        }

    if df.empty:
        logger.warning("Listings dataset is empty; no results to return.")
        return {"message": "No listings are available in the database yet.", "relaxed_constraints": [], "results": []}

    candidates, relaxations = filter_with_fallback(df, intent)
    message = build_result_message(intent, relaxations, candidates)

    if relaxations:
        logger.info(
            "Results reflect relaxed constraints (%s) because no exact match was found.",
            ", ".join(relaxations),
        )

    ranked = rank_by_semantic_similarity(candidates, intent, model_name=embedding_model)
    results = format_results(ranked, intent, top_k=top_k)

    logger.info("Returning top %d result(s) for this query.", len(results))
    return {"message": message, "relaxed_constraints": relaxations, "results": results}


# --------------------------------------------------------------------------- #
# Demo / manual test harness
# --------------------------------------------------------------------------- #
def _pretty_print_results(results: list[dict]) -> None:
    if not results:
        print("(no results)")
        return
    for rank, item in enumerate(results, start=1):
        print(
            f"  #{rank} [{item['similarity_score']:.3f}] "
            f"{item['title']} — {item['property_type']} — {item['location']} — "
            f"PKR {item['price_pkr']:,} — {item['area_marla']} Marla"
        )


def main() -> None:
    listings_path = os.environ.get("LISTINGS_PATH", "listings.json")

    sample_intents = [
        {
            "property_type": "House",
            "subtype": None,
            "location": "DHA Phase 6",
            "marla_min": 5.0,
            "marla_max": 5.0,
            "price_max": 25000000,
            "soft_signals": ["cosy", "sunlit"],
            "poi": ["school", "park"],
        },
        {
            "property_type": "Flat",
            "subtype": None,
            "location": "Gulberg Lahore",
            "marla_min": None,
            "marla_max": None,
            "price_max": 8500000,
            "soft_signals": ["modern"],
            "poi": [],
        },
        {
            # Deliberately unrealistic: 4 Kanal (80 Marla) house for PKR
            # 10 Lakh is far below market rate. Exercises the graceful
            # degradation + honest messaging path end-to-end.
            "property_type": "House",
            "subtype": None,
            "location": "Islamabad",
            "marla_min": 80.0,
            "marla_max": 80.0,
            "price_max": 1000000,
            "soft_signals": [],
            "poi": [],
        },
    ]

    logger.info(
        "=== Running search_engine.py demo with %d sample intent(s) against '%s' ===",
        len(sample_intents),
        listings_path,
    )

    for i, intent in enumerate(sample_intents, start=1):
        print(f"\n--- Sample Query {i} ---")
        print(f"Intent: {json.dumps(intent, ensure_ascii=False)}")
        try:
            response = search_listings(intent, listings_path=listings_path, top_k=5)
            if response["message"]:
                print(f"Message: {response['message']}")
            print(f"Top {len(response['results'])} result(s):")
            _pretty_print_results(response["results"])
        except Exception as exc:  # noqa: BLE001
            logger.error("Search failed for sample query %d: %s", i, exc)

    logger.info("=== Demo complete ===")


if __name__ == "__main__":
    main()
