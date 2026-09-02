from __future__ import annotations

import os
import json
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from intent_extractor import extract_query_intent
from schema_utils import needs_clarification
from search_engine import search_listings, DEFAULT_TOP_K

# --- NEW IMPORTS ---
from poi_verification import verify_pois_for_results
from summary_generation import enrich_with_summary
# -------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


def handle_search(
    intent: dict, 
    query: str,              # Added so we can pass it to the summarizer
    gemini_api_key: str,     # Added so we can pass it to the summarizer
    listings_path: str, 
    top_k: int = DEFAULT_TOP_K
) -> dict:
    """
    Given an already-extracted intent, either request clarification or
    run the actual search, verify POIs, and generate a summary.
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
            "summary": None
        }

    # Step 2 & 3: Search Listings
    search_response = search_listings(intent, listings_path=listings_path, top_k=top_k)

    # Step 4: POI Verification
    poi_terms = intent.get("poi", [])
    if poi_terms and search_response.get("results"):
        search_response["results"] = verify_pois_for_results(search_response["results"], poi_terms)

    # Step 5: Summary Generation (and Re-ranking)
    search_response = enrich_with_summary(search_response, query, intent, gemini_api_key)

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
    """Full pipeline, turn 1: raw query -> intent -> (clarification OR results)."""
    intent = extract_query_intent(query, gemini_api_key=gemini_api_key, qwen_path=qwen_path)
    
    # Pass query and gemini_api_key into handle_search
    return handle_search(intent, query, gemini_api_key, listings_path=listings_path, top_k=top_k)


def answer_clarification(
    query: str,              # Needed for turn 2 summary
    gemini_api_key: str,     # Needed for turn 2 summary
    intent: dict,
    field: str,
    value: str,
    listings_path: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Full pipeline, turn 2: the user answered a clarification question."""
    updated_intent = dict(intent)
    updated_intent[field] = value.strip()
    logger.info("Clarification answered: %s = %r -- completing search.", field, updated_intent[field])
    
    # Pass query and gemini_api_key into handle_search
    return handle_search(updated_intent, query, gemini_api_key, listings_path=listings_path, top_k=top_k)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    qwen_path = os.environ.get("QWEN_MODEL_PATH")
    
    # Check if we should use ../data/listings.json or listings.json
    default_listings = "../data/listings.json" if os.path.exists("../data/listings.json") else "listings.json"
    listings_path = os.environ.get("LISTINGS_PATH", default_listings)

    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        return

    print("\n--- Full pipeline: unambiguous query (should search immediately) ---")
    q1 = "5 marla house in bahria town near school in 1.8 crore"
    resp = run_query(q1, api_key, qwen_path, listings_path)
    print(f"Status: {resp['status']}")
    print(f"Results: {len(resp['results'])}")
    
    # --- PRINT CLAUDE'S FINAL OUTPUTS ---
    print("\n--- AI Summary ---")
    print(resp.get("summary"))
    
    print("\n--- Top Verified POIs ---")
    for r in resp.get("results", []):
        print(f" - {r['title']} | POIs: {r.get('poi_verification')}")

    print("\n\n--- Full pipeline: ambiguous plot query (should ask for clarification) ---")
    q2 = "I want a plot in DHA Lahore under 1 crore near a park"
    resp = run_query(q2, api_key, qwen_path, listings_path)
    print(f"Status: {resp['status']}")
    
    if resp["status"] == "needs_clarification":
        print(f"Question: {resp['clarification']['question']}")
        print(f"Options: {resp['clarification']['options']}")

        print("\n--- Simulating user picking 'Commercial Plot' ---")
        resp2 = answer_clarification(q2, api_key, resp["intent"], "subtype", "Commercial Plot", listings_path)
        print(f"Status: {resp2['status']}")
        
        # --- PRINT CLAUDE'S FINAL OUTPUTS ---
        print("\n--- AI Summary ---")
        print(resp2.get("summary"))


if __name__ == "__main__":
    main()