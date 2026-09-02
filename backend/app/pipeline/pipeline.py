#!/usr/bin/env python3
"""
pipeline.py
=============

Connects Step 1 (intent extraction, with clarification handling) to
Steps 2 & 3 (hard filtering + semantic ranking), so a raw user query
turns into ranked results end-to-end.

Two-turn conversation model:

    Turn 1: run_query(query, ...)
        -> extracts intent
        -> if ambiguous (e.g. "plot" with no subtype), returns
           status="needs_clarification" with a question + options,
           WITHOUT searching yet
        -> otherwise searches immediately and returns status="ok"
           with results

    Turn 2 (only if turn 1 asked for clarification):
        answer_clarification(intent, field, value, ...)
        -> fills in the answered field, then searches

This mirrors how a real chat UI would use it: call run_query() once;
if the response says needs_clarification, show the question and options
to the user, then call answer_clarification() with whatever they picked.
"""

from __future__ import annotations

import logging
from typing import Optional

from intent_extractor import extract_query_intent
from schema_utils import needs_clarification
from search_engine import search_listings, DEFAULT_TOP_K

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


def handle_search(intent: dict, listings_path: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Given an already-extracted intent, either request clarification or
    run the actual search. Separated from run_query() so it can be
    tested directly with a hand-built intent dict, without needing a
    real Qwen/Gemini call -- extraction routing is already covered by
    test_intent_extractor.py; this tests the NEW glue logic only.
    """
    clarification = needs_clarification(intent)
    if clarification:
        logger.info("Intent is ambiguous (%s) -- requesting clarification instead of searching.", clarification["field"])
        return {
            "status": "needs_clarification",
            "intent": intent,
            "clarification": clarification,
            "message": None,
            "relaxed_constraints": [],
            "results": [],
        }

    search_response = search_listings(intent, listings_path=listings_path, top_k=top_k)
    return {
        "status": "ok",
        "intent": intent,
        "clarification": None,
        **search_response,
    }


def run_query(
    query: str,
    gemini_api_key: str,
    qwen_path: Optional[str] = None,
    listings_path: str = "listings.json",
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Full pipeline, turn 1: raw query -> intent -> (clarification OR results).

    Returns a dict:
        {
            "status": "needs_clarification" | "ok",
            "intent": dict,                    # the extracted intent so far
            "clarification": dict | None,       # question+options if status is needs_clarification
            "message": str | None,              # e.g. relaxed-constraints explanation
            "relaxed_constraints": list[str],
            "results": list[dict],
        }
    """
    intent = extract_query_intent(query, gemini_api_key=gemini_api_key, qwen_path=qwen_path)
    return handle_search(intent, listings_path=listings_path, top_k=top_k)


def answer_clarification(
    intent: dict,
    field: str,
    value: str,
    listings_path: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Full pipeline, turn 2: the user answered a clarification question
    (e.g. picked "Commercial Plot" or "Any"). Fills in the answered
    field on the existing intent and completes the search.

    IMPORTANT: "Any" is stored as the literal string "Any", NOT
    converted to None. This matters: needs_clarification() treats a
    None/empty subtype as "not yet answered" and would otherwise
    re-trigger the same clarification question forever after the user
    explicitly chose "Any" -- None can't distinguish "never asked" from
    "explicitly answered as unconstrained". search_engine.py's hard
    filter already treats the string "any" (case-insensitive) as
    "skip this constraint", so no change is needed there.
    """
    updated_intent = dict(intent)
    updated_intent[field] = value.strip()
    logger.info("Clarification answered: %s = %r -- completing search.", field, updated_intent[field])
    return handle_search(updated_intent, listings_path=listings_path, top_k=top_k)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    import os
    import json

    api_key = os.environ.get("GEMINI_API_KEY", "")
    qwen_path = os.environ.get("QWEN_MODEL_PATH")
    listings_path = os.environ.get("LISTINGS_PATH", "listings.json")

    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        return

    print("\n--- Full pipeline: unambiguous query (should search immediately) ---")
    resp = run_query("5 marla house in DHA Phase 6 under 2.5 crore", api_key, qwen_path, listings_path)
    print(f"Status: {resp['status']}")
    print(f"Results: {len(resp['results'])}")

    print("\n--- Full pipeline: ambiguous plot query (should ask for clarification) ---")
    resp = run_query("I want a plot in DHA Lahore under 1 crore", api_key, qwen_path, listings_path)
    print(f"Status: {resp['status']}")
    if resp["status"] == "needs_clarification":
        print(f"Question: {resp['clarification']['question']}")
        print(f"Options: {resp['clarification']['options']}")

        print("\n--- Simulating user picking 'Commercial Plot' ---")
        resp2 = answer_clarification(resp["intent"], "subtype", "Commercial Plot", listings_path)
        print(f"Status: {resp2['status']}")
        print(f"Results: {len(resp2['results'])}")
        for r in resp2["results"]:
            print(f"  - {r['title']} [{r['subtype']}]")


if __name__ == "__main__":
    main()
