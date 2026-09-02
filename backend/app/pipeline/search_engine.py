#!/usr/bin/env python3
"""
search_engine.py
==================

Steps 2 & 3 of the Real Estate Property Finder pipeline (Approach 1).

Takes structured intent JSON produced by Step 1 and searches it against
the preprocessed listings dataset (listings.json).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
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
# Constants & Caches
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

_MODEL_CACHE: dict[str, Any] = {}
_LISTINGS_CACHE: dict[str, dict[str, Any]] = {}
CACHE_TTL_SECONDS = 60


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_listings(listings_path: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Load listings.json into a Pandas DataFrame.
    Checks the disk every 60 seconds to catch daily scraper updates,
    and strictly filters out any property older than 30 days.
    """
    now = time.time()
    if use_cache and listings_path in _LISTINGS_CACHE:
        cached_data = _LISTINGS_CACHE[listings_path]
        if (now - cached_data["loaded_at"]) <= CACHE_TTL_SECONDS:
            return cached_data["df"]

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

    df = pd.DataFrame(raw_records) if raw_records else pd.DataFrame()

# 30-Day Freshness Filter (Preserves listings without explicit dates)
    if not df.empty and "date_listed" in df.columns:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
        parsed_dates = pd.to_datetime(df["date_listed"], utc=True, errors="coerce")
        # Keep if within last 30 days OR if no date was recorded
        df = df[(parsed_dates >= cutoff) | (parsed_dates.isna())].copy()

    required_cols = [
        "id", "title", "url", "property_type", "subtype", "location",
        "price_pkr", "area_marla", "bedrooms", "bathrooms", "latitude",
        "longitude", "title_embedding",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        for col in missing:
            df[col] = None

    def _to_vector(value: Any) -> Optional[np.ndarray]:
        if isinstance(value, list) and len(value) > 0:
            return np.asarray(value, dtype=np.float32)
        return None

    if "title_embedding" in df.columns:
        df["title_embedding_vec"] = df["title_embedding"].apply(_to_vector)

    if "location" in df.columns:
        df["location"] = df["location"].astype(str)

    logger.info("Loaded %d fresh listings from '%s'.", len(df), listings_path)

    if use_cache:
        _LISTINGS_CACHE[listings_path] = {"df": df, "loaded_at": now}

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
    """Normalize a location string for matching purposes only."""
    if not text:
        return ""
    return " ".join(str(text).replace(",", " ").split()).lower()


_GRANULAR_QUALIFIER_PATTERN = re.compile(
    r"\b(phase|sector|block|street|st)\s*[-#]?\s*[a-z0-9]{1,4}\b", re.IGNORECASE
)


def extract_core_location(text: str) -> str:
    """Strips phase/sector/block qualifiers for high-level matching."""
    normalized = normalize_location_text(text)
    if not normalized:
        return ""
    stripped = _GRANULAR_QUALIFIER_PATTERN.sub("", normalized)
    stripped = " ".join(stripped.split())
    return stripped if stripped else normalized


SIZE_EXPAND_PCT = 0.18
PRICE_EXPAND_PCT = 0.12
NEARBY_RADIUS_KM = 10.0


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in km between reference point and an array of points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_location_centroid(df: pd.DataFrame, location: str) -> Optional[tuple[float, float]]:
    """Finds geographic centroid of listings matching the core location."""
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
    """Apply strict logical filters derived from the intent JSON."""
    mask = pd.Series(True, index=df.index)

    subtype = intent.get("subtype")
    property_type = intent.get("property_type")

    # 1. Property Type & Subtype Filtering (Cross-Category Normalized)
    # 1. Property Type & Subtype Filtering (Cross-Category Normalized)
    if subtype and str(subtype).lower() != "any" and "subtype" in df.columns:
        target_subtype = str(subtype).strip().lower()

        if target_subtype == "commercial plot":
            is_direct_subtype = df["subtype"].astype(str).str.lower() == "commercial plot"
            is_commercial_title_plot = (
                df["property_type"].astype(str).str.lower().isin(["commercial", "plots", "plot"])
            ) & (
                df["title"].astype(str).str.contains(r"\b(?:plot|plots|land)\b", case=False, na=False)
            )
            mask &= (is_direct_subtype | is_commercial_title_plot)

        # --- SHOP MATCHING BRANCH ---
        elif target_subtype == "shop":
            is_direct_shop = df["subtype"].astype(str).str.lower() == "shop"
            is_commercial_title_shop = (
                df["property_type"].astype(str).str.lower() == "commercial"
            ) & (
                df["title"].astype(str).str.contains(r"\b(?:shop|dukan|store|showroom)\b", case=False, na=False)
            )
            mask &= (is_direct_shop | is_commercial_title_shop)

        # --- RESIDENTIAL PLOT MATCHING BRANCH ---
        elif target_subtype == "residential plot":
            is_direct_res = df["subtype"].astype(str).str.lower() == "residential plot"
            # Zameen often categorizes standard residential plots simply as "Plots"
            is_generic_plot = df["subtype"].astype(str).str.lower().isin(["plot", "plots"])
            # Ensure we don't accidentally grab commercial plots marked generically
            is_not_commercial = ~df["title"].astype(str).str.contains(r"\b(?:commercial|shop|industrial)\b", case=False, na=False)
            mask &= (is_direct_res | (is_generic_plot & is_not_commercial))
        # -----------------------------

        else:
            mask &= df["subtype"].astype(str).str.lower() == target_subtype

    elif property_type:
        target_type = str(property_type).strip().lower()
        if target_type in ["plot", "plots"]:
            mask &= df["property_type"].astype(str).str.lower().isin(["plot", "plots"])
        elif target_type in ["house", "flat", "home", "homes"]:
            mask &= df["property_type"].astype(str).str.lower().isin(["home", "homes", "house", "flat"])
        else:
            mask &= df["property_type"].astype(str).str.lower() == target_type

    # 2. Price Ceiling
    price_max = intent.get("price_max")
    if price_max is not None:
        effective_price_max = price_max * (1 + price_expand_pct)
        mask &= df["price_pkr"] <= effective_price_max

    # 3. Area Range
    marla_min = intent.get("marla_min")
    marla_max = intent.get("marla_max")
    if marla_min is not None:
        effective_min = marla_min * (1 - size_expand_pct)
        mask &= df["area_marla"] >= effective_min
    if marla_max is not None:
        effective_max = marla_max * (1 + size_expand_pct)
        mask &= df["area_marla"] <= effective_max

    # 4. Location Matching
    location = intent.get("location")
    if location and location_mode == "radius":
        if centroid is None:
            logger.warning("location_mode='radius' requested but centroid unavailable.")
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
        core_query_loc = extract_core_location(location)
        query_words = [w for w in core_query_loc.split() if w]

        if location_mode == "city_only":
            query_words = query_words[-1:]

        if query_words:
            normalized_listing_loc = df["location"].apply(normalize_location_text)
            word_mask = pd.Series(True, index=df.index)
            for word in query_words:
                word_mask &= normalized_listing_loc.str.contains(word, na=False, regex=False)
            mask &= word_mask

    return df[mask]


def filter_with_fallback(df: pd.DataFrame, intent: dict) -> tuple[pd.DataFrame, list[str]]:
    """Apply hard filters with graceful relaxation safety nets."""
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

    centroid = compute_location_centroid(df, location) if location_set else None
    if location_set and centroid is None:
        logger.info("No coordinates available to compute a 'nearby areas' centroid for %r.", location)

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
        if not was_constrained:
            continue
        applied.append(step_name)
        candidates = apply_hard_filters(df, intent, **relax_kwargs)
        if len(candidates) > 0:
            logger.warning("Found %d candidates after relaxing: %s.", len(candidates), applied)
            return candidates, applied

    logger.warning(
        "No candidates found even after expanding size, budget, and location. "
        "Returning zero results (property_type/subtype are never relaxed)."
    )
    return candidates, applied


def build_result_message(intent: dict, relaxations: list[str], candidates: pd.DataFrame) -> Optional[str]:
    """Compose a plain-language explanation if constraints were relaxed."""
    if not relaxations:
        return None

    property_type = intent.get("property_type") or "property"
    location = intent.get("location")
    price_max = intent.get("price_max")

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
    if not location:
        return ""
    full_matches = _GRANULAR_QUALIFIER_PATTERN.finditer(location)
    return " ".join(m.group(0) for m in full_matches)


def build_soft_signal_query(intent: dict) -> str:
    soft_signals = intent.get("soft_signals") or []
    pois = intent.get("poi") or []
    all_signals = soft_signals + pois
    query_terms = [str(term).strip() for term in all_signals if str(term).strip()]

    granular = extract_granular_qualifier(intent.get("location") or "")
    if granular:
        query_terms.append(granular)

    return " ".join(query_terms)


def cosine_similarity_matrix(query_vec: np.ndarray, candidate_vecs: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec)
    candidate_norms = np.linalg.norm(candidate_vecs, axis=1)

    safe_query_norm = query_norm if query_norm > 1e-12 else 1e-12
    safe_candidate_norms = np.where(candidate_norms > 1e-12, candidate_norms, 1e-12)

    dot_products = candidate_vecs @ query_vec
    cosine_sim = dot_products / (safe_candidate_norms * safe_query_norm)
    normalized = (cosine_sim + 1.0) / 2.0
    return np.clip(normalized, 0.0, 1.0)


def rank_by_semantic_similarity(
    candidates: pd.DataFrame,
    intent: dict,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> pd.DataFrame:
    candidates = candidates.copy()
    query_text = build_soft_signal_query(intent)

    if not query_text:
        logger.info("No soft_signals present in intent; skipping semantic ranking.")
        candidates["similarity_score"] = 0.5
        return candidates

    model = _get_embedding_model(model_name)
    if model is None:
        logger.warning("Embedding model unavailable; assigning neutral similarity scores.")
        candidates["similarity_score"] = 0.5
        return candidates

    logger.info("Embedding soft-signal query: %r", query_text)
    try:
        query_embedding = model.encode(query_text, convert_to_numpy=True).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to embed soft-signal query: %s", exc)
        candidates["similarity_score"] = 0.5
        return candidates

    valid_mask = candidates["title_embedding_vec"].apply(lambda v: v is not None)
    candidates["similarity_score"] = 0.5

    if valid_mask.any():
        valid_vecs = np.vstack(candidates.loc[valid_mask, "title_embedding_vec"].to_list())
        scores = cosine_similarity_matrix(query_embedding, valid_vecs)
        candidates.loc[valid_mask, "similarity_score"] = scores

    invalid_count = (~valid_mask).sum()
    if invalid_count:
        logger.warning("%d candidates had no valid title_embedding; assigned neutral score.", invalid_count)

    return candidates


# --------------------------------------------------------------------------- #
# Stage 3: Top-N selection & formatting
# --------------------------------------------------------------------------- #
def _safe_int(val: Any, default: int = 0) -> int:
    """Converts NaN, None, and unparseable values to a safe integer."""
    if pd.isna(val) or val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: Any) -> Optional[float]:
    """Converts NaN or invalid floats to None."""
    if pd.isna(val) or val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _compute_tiebreak(candidates: pd.DataFrame, intent: dict) -> tuple[pd.Series, bool]:
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
    """Select the top_k candidates and format them as clean, type-safe JSON dicts."""
    if ranked_candidates.empty:
        return []

    sortable = ranked_candidates.copy()

    # 1. Exact Subtype Boost: Prioritizes explicit intent (e.g. Shop over Hotel)
    target_subtype = str(intent.get("subtype") or "").strip().lower()
    if target_subtype and target_subtype != "any":
        is_exact_subtype = sortable["subtype"].astype(str).str.lower() == target_subtype
        is_title_match = sortable["title"].astype(str).str.contains(
            rf"\b{re.escape(target_subtype)}\b", case=False, na=False
        )
        sortable["_exact_match"] = is_exact_subtype | is_title_match
    else:
        sortable["_exact_match"] = True

    # 2. Tiebreaker computation
    tiebreak_series, tiebreak_ascending = _compute_tiebreak(ranked_candidates, intent)
    sortable["_tiebreak"] = tiebreak_series

    # 3. Sort: similarity -> exact target match -> tiebreaker -> price
    top_candidates = sortable.sort_values(
        ["similarity_score", "_exact_match", "_tiebreak", "price_pkr"],
        ascending=[False, False, tiebreak_ascending, True],
    ).head(top_k)

    requested_pois = intent.get("poi") or []

    results = []
    for _, row in top_candidates.iterrows():
        try:
            # Only homes/flats have meaningful bedrooms/bathrooms
            prop_type = str(row.get("property_type", "")).strip().lower()
            is_residential = prop_type in ["house", "flat", "homes", "home"]

            results.append(
                {
                    "id": int(row["id"]),
                    "title": str(row["title"]),
                    "url": str(row["url"]) if "url" in row and pd.notna(row["url"]) else "",
                    "property_type": str(row["property_type"]),
                    "subtype": str(row["subtype"]) if "subtype" in row and pd.notna(row["subtype"]) else None,
                    "location": str(row["location"]),
                    "price_pkr": _safe_int(row.get("price_pkr"), default=0),
                    "area_marla": _safe_float(row.get("area_marla")),
                    # Beds/baths stay strictly 0 for Commercial & Plots to avoid "1 Beds" badges
                    "bedrooms": _safe_int(row.get("bedrooms"), default=0) if is_residential else 0,
                    "bathrooms": _safe_int(row.get("bathrooms"), default=0) if is_residential else 0,
                    "latitude": _safe_float(row.get("latitude")),
                    "longitude": _safe_float(row.get("longitude")),
                    "similarity_score": round(float(row["similarity_score"]), 4),
                    "poi": requested_pois,
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