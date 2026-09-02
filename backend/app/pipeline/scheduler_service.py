"""
scheduler_service.py
====================
Manages automated daily scraping, deduplication, geocoding, 
and embedding generation inside the FastAPI application runtime.
"""

import asyncio
import logging
import os
from pathlib import Path
from scraper import main_async, build_arg_parser

logger = logging.getLogger("etl_scheduler")

# Track sync job execution metrics for observability
ETL_STATUS = {
    "last_run_started": None,
    "last_run_completed": None,
    "status": "idle",  # "idle", "running", "failed"
    "last_error": None,
}


async def run_daily_etl_job():
    """Executes the daily scraper run in an isolated async task."""
    global ETL_STATUS

    if ETL_STATUS["status"] == "running":
        logger.warning("ETL run skipped: A sync job is already in progress.")
        return

    ETL_STATUS["status"] = "running"
    ETL_STATUS["last_run_started"] = asyncio.get_event_loop().time()
    ETL_STATUS["last_error"] = None

    logger.info("Starting scheduled daily ETL pipeline...")

    parser = build_arg_parser()
    locationiq_key = os.environ.get("LOCATIONIQ_API_KEY", "")
    output_path = Path("data/listings.json")

    # Command arguments matching your daily production pipeline
    cli_args = [
        "--mode", "daily",
        "--output", str(output_path),
        "--headless",
        "--locationiq-api-key", locationiq_key,
    ]

    args = parser.parse_args(cli_args)

    try:
        # Run the full scraping, geocoding, and MiniLM embedding pipeline
        await main_async(args)
        ETL_STATUS["status"] = "idle"
        ETL_STATUS["last_run_completed"] = asyncio.get_event_loop().time()
        logger.info("Daily ETL pipeline completed successfully.")
    except Exception as exc:  # noqa: BLE001
        ETL_STATUS["status"] = "failed"
        ETL_STATUS["last_error"] = str(exc)
        logger.exception("Daily ETL pipeline failed: %s", exc)