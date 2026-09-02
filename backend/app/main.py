import asyncio
import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import auth, history, market_insights, saved_listings, search, voice

logger = logging.getLogger("etl_scheduler")
settings = get_settings()

# In-memory ETL status for monitoring and reporting
ETL_STATUS = {
    "status": "idle",  # "idle" | "running" | "failed"
    "last_run_started": None,
    "last_run_completed": None,
    "last_error": None,
}

scheduler = AsyncIOScheduler()


async def run_daily_etl_job():
    """Executes the daily scraper run in an isolated worker subprocess with geocoding."""
    global ETL_STATUS

    if ETL_STATUS["status"] == "running":
        logger.warning("ETL run skipped: A sync job is already in progress.")
        return

    ETL_STATUS["status"] = "running"
    ETL_STATUS["last_run_started"] = asyncio.get_event_loop().time()
    ETL_STATUS["last_error"] = None

    logger.info("Starting automated daily ETL pipeline in dedicated worker process...")

    python_exe = sys.executable
    scraper_path = os.path.join(os.getcwd(), "scraper.py")
    output_path = os.path.join("data", "listings.json")

    # Read the key from Settings or the OS environment
    locationiq_key = getattr(settings, "locationiq_api_key", None) or os.environ.get("LOCATIONIQ_API_KEY", "")

    cmd = [
        python_exe,
        scraper_path,
        "--mode", "daily",
        "--output", output_path,
        "--headless",
        "--geocode-provider", "locationiq",
        "--locationiq-api-key", locationiq_key,
        "--geocode-requests-per-minute", "60.0",  # <--- ADD THIS LINE
    ]

    # Force UTF-8 environment on Windows to prevent console charmap errors
    worker_env = os.environ.copy()
    worker_env["PYTHONIOENCODING"] = "utf-8"
    worker_env["LOCATIONIQ_API_KEY"] = locationiq_key

    try:
        process = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=worker_env,
            encoding="utf-8",
        )
        if process.stdout:
            logger.info("Daily scraper worker output:\n%s", process.stdout)
        ETL_STATUS["status"] = "idle"
        ETL_STATUS["last_run_completed"] = asyncio.get_event_loop().time()
        logger.info("Automated daily ETL pipeline finished successfully.")
    except subprocess.CalledProcessError as exc:
        ETL_STATUS["status"] = "failed"
        ETL_STATUS["last_error"] = exc.stderr
        logger.error("Automated daily ETL pipeline failed: %s", exc.stderr)
    except Exception as exc:  # noqa: BLE001
        ETL_STATUS["status"] = "failed"
        ETL_STATUS["last_error"] = str(exc)
        logger.exception("Unexpected error in daily ETL job: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to handle startup and shutdown of the scheduler."""
    # Runs automatically every day at 3:00 AM (Asia/Karachi PKT)
    scheduler.add_job(
        run_daily_etl_job,
        trigger=CronTrigger(hour=3, minute=50, timezone="Asia/Karachi"),
        id="daily_zameen_etl",
        name="Scrape 24h listings, geocode, and embed",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Background ETL scheduler started (Daily at 3:50 AM PKT).")

    yield

    scheduler.shutdown(wait=False)
    logger.info("Background ETL scheduler stopped.")


app = FastAPI(
    title="Real Estate Property Finder API",
    description=(
        "Natural-language property search backed by a fine-tuned Qwen "
        "model, semantic ranking, live POI verification, and grounded AI summaries."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(search.router)
app.include_router(saved_listings.router)
app.include_router(history.router)
app.include_router(market_insights.router)
app.include_router(voice.router)


@app.get("/health", tags=["meta"])
async def health():
    """Cheap liveness check."""
    return {"status": "ok"}


# --- Admin & Observability Endpoints ---


@app.get("/admin/etl-status", tags=["Admin / ETL"])
async def get_etl_status():
    """Returns the health and status of the daily scraping sync."""
    return ETL_STATUS


@app.post("/admin/sync-now", tags=["Admin / ETL"])
async def trigger_manual_sync(background_tasks: BackgroundTasks):
    """Triggers the daily sync on demand via API/Swagger UI."""
    if ETL_STATUS["status"] == "running":
        return {"status": "in_progress", "message": "A scrape job is already active."}

    background_tasks.add_task(run_daily_etl_job)
    return {"status": "started", "message": "Daily sync triggered in background."}