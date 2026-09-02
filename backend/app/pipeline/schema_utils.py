#!/usr/bin/env python3
"""
schema_utils.py
==================

Single source of truth for the intent extraction schema, used by every
script that needs to know its shape: intent_extractor.py (production
routing), finetune_qwen.py (training), evaluate_qwen.py (grading).

Before this module existed, SUBTYPES_BY_CATEGORY and SYSTEM_PROMPT were
each defined independently in three different files -- a real risk, since
adding a new subtype or tweaking the prompt in only one place would
silently desync training from production. Everything schema-related now
lives here exactly once.
"""

from __future__ import annotations

from typing import Optional

INTENT_SCHEMA = {
    "property_type": None,   # str: "House" | "Flat" | "Farmhouse" | "Plot" | null
    "subtype": None,         # str: specific Zameen.com subtype, or null if unspecified
    "location": None,        # str or null
    "marla_min": None,       # float or null
    "marla_max": None,       # float or null
    "price_max": None,       # int (raw PKR) or null
    "soft_signals": [],      # list[str]
    "poi": [],               # list[str]
}

REQUIRED_KEYS = set(INTENT_SCHEMA.keys())

VALID_PROPERTY_TYPES = {"House", "Flat", "Farmhouse", "Plot", "Commercial"}

SUBTYPES_BY_CATEGORY = {
    "House": {"House", "Upper Portion", "Lower Portion", "Room"},
    "Flat": {"Flat", "Penthouse"},
    "Farmhouse": {"Farmhouse"},
    "Plot": {"Residential Plot", "Commercial Plot", "Agricultural Land", "Industrial Land", "Plot File"},
    "Commercial": {"Shop", "Office", "Building", "Warehouse", "Factory"},
}

# Categories where an unspecified subtype is ambiguous enough that the
# calling application should ask the user to clarify before searching,
# rather than guessing. Kept deliberately small: only Plot has subtypes
# that are drastically different in use-case/price (Commercial vs.
# Residential Plot serve completely different buyers). House subtypes
# default to "any" instead of asking.
CLARIFICATION_REQUIRED_CATEGORIES = {"Plot"}

SYSTEM_PROMPT = """\
You are a strict information-extraction engine for a Pakistani real estate \
search platform (similar to Zameen.com). Your ONLY job is to read a user's \
natural language property search query and convert it into a single JSON \
object that exactly matches the required schema. You do not chat, explain, \
apologize, or add any text outside the JSON object.

### Domain knowledge you MUST apply correctly:

1. AREA UNITS (convert everything to Marla, a floating point number):
   - "Marla" is the base unit. E.g. "5 Marla" -> marla_min/marla_max = 5.0
   - "Kanal" must be converted: 1 Kanal = 20 Marla.
   - "Sq. Yd." must be converted: 1 Sq. Yd. = 0.04 Marla.
   - If the user gives a single area value (not a range), set marla_min AND
     marla_max to that same converted value UNLESS the phrasing implies a
     minimum only (e.g. "at least 10 marla" -> marla_min=10.0, marla_max=null)
     or a maximum only (e.g. "under 10 marla" -> marla_min=null,
     marla_max=10.0).
   - If the user gives a range ("between 5 and 10 marla"), set marla_min and
     marla_max accordingly.
   - If no area is mentioned at all, both fields are null.

2. PRICE UNITS (convert everything to a raw integer in PKR):
   - "Crore" or "Cr" -> multiply the number by 10,000,000
   - "Lakh" or "Lac" -> multiply the number by 100,000
   - "Thousand" -> multiply the number by 1,000
   - price_max should represent the user's maximum budget. Phrases like
     "under", "below", "max", "up to", "within" all indicate price_max.
   - If no price is mentioned, price_max is null.

3. PROPERTY TYPE & SUBTYPE (two separate fields):
   - property_type is the broad category, exactly one of:
     "House", "Flat", "Farmhouse", "Plot", or null.
     * "House", "home", "villa", "bungalow" -> "House"
     * "Flat", "apartment" -> "Flat"
     * "Penthouse" -> property_type "Flat" (penthouse is a Flat subtype)
     * "Farm house", "farmhouse" -> "Farmhouse". IMPORTANT: Farmhouse is
       its OWN top-level category, never nest it under "House".
     * "Plot", "land" (any kind) -> "Plot"
     * If the property type is not mentioned or unclear, use null.
   - subtype is the specific listing type WITHIN that category, and must
     only be set if the user's wording actually specifies it:
     * House subtypes: "House", "Upper Portion", "Lower Portion", "Room".
     * Flat subtypes: "Flat", "Penthouse".
     * Farmhouse subtype: always "Farmhouse".
     * Plot subtypes: "Residential Plot", "Commercial Plot",
       "Agricultural Land", "Industrial Land", "Plot File". 
     * CRITICAL RULE FOR PLOTS: NEVER GUESS OR ASSUME THE SUBTYPE. If the 
       user just says "plot" or "land" without explicitly stating 
       "Commercial", "Residential", etc., you MUST leave subtype as null.
     * NEVER invent a subtype string outside these lists.

4. LOCATION:
   - Extract the most specific location phrase mentioned, exactly as
     stated. Do NOT guess or add a city that wasn't mentioned. Do NOT
     expand abbreviations into something more specific than stated.
   - If no location is mentioned, use null.

5. SOFT_SIGNALS: subjective/descriptive words only (not hard filters),
   translated to clean English regardless of input language (e.g. Roman
   Urdu "sasta" -> "cheap", "purani"/"purana" -> "old"). Empty list [] if
   none mentioned.

6. POI: explicitly requested nearby amenities, normalized to lowercase
   singular nouns (e.g. "schools" -> "school"). Empty list [] if none.

### Output rules:
- Return ONLY a single valid JSON object. No markdown fences, no
  commentary, no leading or trailing text.
- The JSON object MUST contain exactly these keys: property_type,
  subtype, location, marla_min, marla_max, price_max, soft_signals, poi.
- Use JSON `null` for any field with no information in the query, except
  soft_signals and poi, which must be `[]` when nothing is mentioned.
- Numbers must be actual JSON numbers, not strings.
"""


def _case_insensitive_match(value: str, valid_set: set[str]) -> Optional[str]:
    """
    Returns the canonically-cased match for `value` within `valid_set`
    (case-insensitive lookup), or None if no match exists at all.

    Model output casing isn't perfectly consistent across generations
    (e.g. "commercial plot" vs "Commercial Plot") even when the semantic
    content is exactly right. Rejecting on casing alone wastes a Gemini
    fallback call for something we can safely self-correct -- so
    validation checks case-insensitively, and normalize_intent() below
    snaps the value back to the canonical casing.
    """
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    for candidate in valid_set:
        if candidate.lower() == lowered:
            return candidate
    return None


def is_schema_valid(resp) -> tuple[bool, str]:
    """
    Checks whether a raw parsed JSON response is well-formed against
    INTENT_SCHEMA. This is the EXACT condition that determines whether
    Qwen's output is trusted directly or the pipeline falls back to
    Gemini -- keeping this in one place means the fallback trigger used
    in production (intent_extractor.py) and the metric reported by
    evaluate_qwen.py can never silently diverge.
    """
    if not isinstance(resp, dict):
        return False, "not a JSON object"

    missing = REQUIRED_KEYS - set(resp.keys())
    if missing:
        return False, f"missing keys: {missing}"

    cat = resp.get("property_type")
    if cat is not None and _case_insensitive_match(cat, VALID_PROPERTY_TYPES) is None:
        return False, f"invalid property_type: {cat!r}"

    # Normalize category casing before checking subtype membership, so a
    # correctly-typed-but-mis-cased category doesn't cascade into a false
    # "subtype invalid for category" rejection.
    normalized_cat = _case_insensitive_match(cat, VALID_PROPERTY_TYPES) if cat else None

    sub = resp.get("subtype")
    allowed_subtypes = SUBTYPES_BY_CATEGORY.get(normalized_cat, set()) if normalized_cat else set()
    if sub is not None and _case_insensitive_match(sub, allowed_subtypes) is None:
        return False, f"subtype {sub!r} invalid for category {cat!r}"

    if not isinstance(resp.get("soft_signals"), list):
        return False, "soft_signals is not a list"
    if not isinstance(resp.get("poi"), list):
        return False, "poi is not a list"

    for field in ("marla_min", "marla_max", "price_max"):
        v = resp.get(field)
        if v is not None and not isinstance(v, (int, float)):
            return False, f"{field} is not numeric"

    if resp.get("location") is not None and not isinstance(resp.get("location"), str):
        return False, "location is not a string"

    return True, "ok"


def _plot_subtype_is_explicit(query: str, subtype: str) -> bool:
  query_lower = query.lower()
  subtype_lower = subtype.lower()
  if subtype_lower == "residential plot":
    return "residential" in query_lower
  if subtype_lower == "commercial plot":
    return "commercial" in query_lower
  if subtype_lower == "agricultural land":
    return "agricultural" in query_lower or "farm land" in query_lower or "farmland" in query_lower
  if subtype_lower == "industrial land":
    return "industrial" in query_lower
  if subtype_lower == "plot file":
    return "plot file" in query_lower or "file plot" in query_lower
  return False


def normalize_intent(parsed: dict, raw_query: str | None = None) -> dict:
    """
    Coerces a schema-valid (or close-to-valid) raw dict into a clean
    INTENT_SCHEMA-shaped dict with safe fallbacks for anything malformed.
    """
    intent = dict(INTENT_SCHEMA)
    if not isinstance(parsed, dict):
        return intent

    # --- Rule-Based Commercial Patch (No Retraining Required) ---
    if raw_query:
        q_low = raw_query.lower()
        if any(w in q_low for w in ["shop", "dukan", "dukaan", "store"]):
            parsed["property_type"] = "Commercial"
            parsed["subtype"] = "Shop"
        elif any(w in q_low for w in ["office", "dafter", "daftar"]):
            parsed["property_type"] = "Commercial"
            parsed["subtype"] = "Office"
        elif any(w in q_low for w in ["warehouse", "godown", "godam", "factory"]):
            parsed["property_type"] = "Commercial"
            parsed["subtype"] = "Warehouse"
        elif any(w in q_low for w in ["plaza", "commercial building"]):
            parsed["property_type"] = "Commercial"
            parsed["subtype"] = "Building"
    # -------------------------------------------------------------

    cat = parsed.get("property_type")
    intent["property_type"] = _case_insensitive_match(cat, VALID_PROPERTY_TYPES) if cat else None

    # A model can emit a valid Plot subtype even when the query only says
    # "plot". Require subtype evidence from the user's actual wording.
    if (
      intent["property_type"] == "Plot"
      and intent["subtype"] is not None
      and raw_query is not None
      and not _plot_subtype_is_explicit(raw_query, intent["subtype"])
    ):
      intent["subtype"] = None

    location = parsed.get("location")
    intent["location"] = location.strip() if isinstance(location, str) and location.strip() else None

    for key in ("marla_min", "marla_max"):
        v = parsed.get(key)
        intent[key] = float(v) if isinstance(v, (int, float)) else None

    price = parsed.get("price_max")
    intent["price_max"] = int(round(price)) if isinstance(price, (int, float)) else None

    for key in ("soft_signals", "poi"):
        v = parsed.get(key)
        intent[key] = [str(x).strip().lower() for x in v if str(x).strip()] if isinstance(v, list) else []

    return intent


def needs_clarification(intent: dict) -> dict | None:
    """
    Checks whether the extracted intent is ambiguous enough that the
    calling application should ask the user a clarifying question before
    searching. See CLARIFICATION_REQUIRED_CATEGORIES for the rationale.
    """
    property_type = intent.get("property_type")
    subtype = intent.get("subtype")

    if property_type in CLARIFICATION_REQUIRED_CATEGORIES and not subtype:
        options = sorted(SUBTYPES_BY_CATEGORY.get(property_type, set())) + ["Any"]
        return {
            "field": "subtype",
            "question": f"What kind of {property_type.lower()} are you looking for?",
            "options": options,
        }
    return None
