"""
Bridges FastAPI's async world to our existing, synchronous, blocking
Steps 0-5 pipeline (intent_extractor.py, search_engine.py, pipeline.py,
poi_verification.py, summary_generation.py).

Two things make this file necessary rather than just `import`ing the
pipeline directly in a router:

1. THREADING: every pipeline call is blocking (requests to Gemini/
   Overpass, blocking model inference). Calling it directly inside an
   `async def` route would freeze the event loop for every other
   concurrent request for the whole duration of the search. Every
   public function here wraps the actual pipeline call in
   starlette's run_in_threadpool, so the blocking work happens on a
   worker thread instead of the event loop.

2. IMPORT PATH: the pipeline files use flat imports (`from schema_utils
   import ...`) by original design -- they were built to run standalone.
   Rather than rewriting their internals (and re-verifying the 150+
   tests already passing against the current versions), this module
   adds the pipeline/ directory to sys.path before importing anything
   from it, so the flat-import style keeps working unchanged inside the
   package.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.config import get_settings

_PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from pipeline import handle_search, answer_clarification as _answer_clarification  # noqa: E402
from intent_extractor import extract_query_intent  # noqa: E402
from search_engine import load_listings  # noqa: E402
from poi_verification import verify_pois_for_results  # noqa: E402
from summary_generation import enrich_with_summary  # noqa: E402

settings = get_settings()

# Loaded once at import time (module-level singleton), not per-request --
# load_listings() already has its own internal caching keyed by path, so
# this just avoids the (cheap but non-zero) dict lookup + path resolution
# on every single search, and makes it obvious the dataset is meant to be
# a shared, long-lived, read-mostly resource for the life of the process.
def _get_listings_df():
    return load_listings(settings.listings_path)


async def run_search(
    query: str,
    top_k: int = 5,
    intent_override: dict[str, Any] | None = None,
    clarification_field: str | None = None,
    clarification_value: str | None = None,
    run_poi_verification: bool = True,
    generate_summary: bool = True,
) -> dict:
    """
    The single entry point the /search route calls. Handles both turns
    of the clarification flow:

      - Turn 1 (intent_override is None): extract intent from `query`
        fresh, then filter/rank. May return status="needs_clarification".
      - Turn 2 (intent_override + clarification_field/value provided):
        skip extraction entirely, apply the clarification answer to the
        echoed-back intent, then filter/rank. Never re-runs Qwen/Gemini.

    Runs Steps 4 (POI verification) and 5 (re-rank + summary) after a
    successful search, UNLESS there are zero results (nothing to verify
    or summarize) or the caller opts out via the two flags -- useful
    for a "fast" search mode that skips the slower verification/summary
    steps if a future frontend wants that tradeoff.
    """

    def _sync_search() -> dict:
        if intent_override is not None and clarification_field and clarification_value is not None:
            return _answer_clarification(
                intent_override, clarification_field, clarification_value, settings.listings_path, top_k=top_k
            )

        intent = extract_query_intent(query, gemini_api_key=settings.gemini_api_key, qwen_path=settings.qwen_model_path or None)
        return handle_search(intent, listings_path=settings.listings_path, top_k=top_k)

    response = await run_in_threadpool(_sync_search)

    if response["status"] != "ok" or not response["results"]:
        return response

    if run_poi_verification and response["intent"].get("poi"):
        response["results"] = await run_in_threadpool(
            verify_pois_for_results, response["results"], response["intent"]["poi"]
        )

    if generate_summary:
        response = await run_in_threadpool(
            enrich_with_summary, response, query, response["intent"], settings.gemini_api_key
        )
    else:
        response.setdefault("summary", None)

    return response
