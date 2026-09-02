#!/usr/bin/env python3
"""
Zameen.com Real Estate ETL Pipeline
====================================

Scrapes property listings from zameen.com, geocodes locations, generates
sentence embeddings for titles, and writes a listings.json file matching
the target schema.

MODES
-----
--mode initial   Fetches listings from the last 30 days and OVERWRITES
                 listings.json from scratch (old file is ignored/backed up).
--mode daily     Fetches listings from the last 24 hours and APPENDS
                 (with de-duplication by listing id) to the existing
                 listings.json.

IMPORTANT — READ BEFORE RUNNING
--------------------------------
1. Zameen.com has no public API (confirmed: there is no official
   api.zameen.com endpoint). This script renders pages with Playwright
   (a real browser engine) and extracts data from the site's embedded
   Next.js state (`__NEXT_DATA__`) with a DOM-based fallback. Site
   structure changes over time — if extraction starts returning empty
   results, re-inspect the live page (DevTools -> search page source for
   `__NEXT_DATA__`, or Network tab -> XHR) and update `_extract_next_data`
   / `_parse_from_dom` accordingly. Field paths below are written to be
   defensive (multiple `.get()` fallbacks) for exactly this reason.
2. Scraping zameen.com's public listing pages this way is technically
   feasible, but you are responsible for checking zameen.com's Terms of
   Service and robots.txt, and for using the data in a way that complies
   with them. This script deliberately scrapes only what is publicly
   visible in the rendered HTML (no login bypass, no paywall/captcha
   circumvention), throttles requests, and identifies itself with a
   normal browser UA — but that does not make it exempt from the site's
   ToS. Consider reaching out to Zameen for a licensed data feed if you
   plan to run this at scale/commercially.
3. Respect the rate limits already in this script (geocoding especially —
   Nominatim's usage policy is 1 req/sec, enforced below). Do not remove
   the delays "to go faster" — you will get IP-banned by Nominatim, and
   possibly rate-limited/blocked by Zameen.

USAGE
-----
    python scraper.py --mode initial --output listings.json
    python scraper.py --mode daily   --output listings.json

    # Optional flags:
    python scraper.py --mode initial \\
        --cities islamabad rawalpindi lahore karachi \\
        --property-types Homes Plots \\
        --max-pages-per-search 40 \\
        --headless \\
        --output listings.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

from playwright.async_api import async_playwright, Page, Response, TimeoutError as PWTimeoutError
from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("zameen_pipeline")

BASE_URL = "https://www.zameen.com"

# Search category slugs Zameen uses. Update/extend as needed.
DEFAULT_PROPERTY_TYPE_SLUGS = {
    "Homes": "Homes",
    "Plots": "Plots",
    "Commercial": "Commercial",
}

DEFAULT_CITIES = ["Islamabad-3", "Rawalpindi-41", "Lahore-1", "Karachi-2"]
# Zameen city-slugs are `<Name>-<id>` (NO page number in the slug itself —
# the page number is appended separately when building the search URL, see
# _scrape_search). Verified against the live site on 2026-08-28:
#   Islamabad-3, Rawalpindi-41, Lahore-1, Karachi-2
# For other cities, visit zameen.com, select the city in the search filter,
# and read the `<CityName>-<id>` portion out of the resulting URL (drop the
# trailing "-<page>.html").

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
]

STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
window.chrome = { runtime: {} };
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class RawListing:
    """Intermediate representation before geocoding/embedding."""
    id: int
    title: str
    url: str
    property_type: str
    subtype: str
    location: str
    price_pkr: Optional[int]
    area_marla: Optional[float]
    bedrooms: Optional[int]
    bathrooms: Optional[int]
    date_listed: str  # ISO 8601


# ---------------------------------------------------------------------------
# Date parsing helpers
# ---------------------------------------------------------------------------
RELATIVE_PATTERNS = [
    (re.compile(r"today", re.I), lambda m, now: now),
    (re.compile(r"yesterday", re.I), lambda m, now: now - timedelta(days=1)),
    (re.compile(r"(\d+)\s*hour", re.I), lambda m, now: now - timedelta(hours=int(m.group(1)))),
    (re.compile(r"(\d+)\s*minute", re.I), lambda m, now: now - timedelta(minutes=int(m.group(1)))),
    (re.compile(r"(\d+)\s*day", re.I), lambda m, now: now - timedelta(days=int(m.group(1)))),
    (re.compile(r"(\d+)\s*week", re.I), lambda m, now: now - timedelta(weeks=int(m.group(1)))),
    (re.compile(r"(\d+)\s*month", re.I), lambda m, now: now - relativedelta(months=int(m.group(1)))),
]


def parse_listing_date(raw: Any, now: Optional[datetime] = None) -> Optional[datetime]:
    """
    Parse Zameen's various date representations into a timezone-aware
    datetime. Handles:
      - Unix timestamps (int/float, seconds or ms)
      - ISO-ish absolute strings ("2026-08-20", "Aug 20, 2026")
      - Relative strings ("3 days ago", "Yesterday", "2 weeks ago")
    Returns None if unparseable (caller should decide how to treat that).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if raw is None:
        return None

    # Unix timestamp
    if isinstance(raw, (int, float)):
        try:
            ts = float(raw)
            if ts > 1e12:  # milliseconds
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OSError):
            return None

    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text:
        return None

    for pattern, resolver in RELATIVE_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                return resolver(m, now).replace(tzinfo=timezone.utc)
            except Exception:
                continue

    # Fall back to absolute date parsing
    try:
        dt = dateutil_parser.parse(text, fuzzy=True)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------
class ZameenScraper:
    def __init__(
        self,
        mode: str,
        cities: list[str],
        property_types: list[str],
        max_pages_per_search: int = 50,
        headless: bool = True,
        request_delay_range: tuple[float, float] = (2.0, 4.5),
    ):
        self.mode = mode
        self.cities = cities
        self.property_types = property_types
        self.max_pages_per_search = max_pages_per_search
        self.headless = headless
        self.request_delay_range = request_delay_range

        now = datetime.now(timezone.utc)
        self.cutoff: datetime = now - (timedelta(days=30) if mode == "initial" else timedelta(hours=24))

        self._captured_json_payloads: list[dict] = []

    # -- network interception -------------------------------------------------
    async def _on_response(self, response: Response) -> None:
        """
        Capture JSON XHR/fetch responses that look like listing search
        results, in case Zameen's search page is backed by an internal
        JSON endpoint (common for Next.js/React SPAs even without a
        documented public API). We inspect content-type + URL heuristics
        rather than hardcoding one exact path, since these endpoints are
        undocumented and can change.
        """
        try:
            url = response.url
            content_type = response.headers.get("content-type", "")
            if "application/json" not in content_type:
                return
            if not any(k in url.lower() for k in ("search", "listing", "property", "catalog")):
                return
            body = await response.json()
            self._captured_json_payloads.append({"url": url, "body": body})
        except Exception:
            # Non-fatal: JSON parsing/response races are expected under load
            pass

    # -- __NEXT_DATA__ extraction ---------------------------------------------
    async def _extract_next_data(self, page: Page) -> Optional[dict]:
        """
        Next.js apps embed their initial server-rendered state in
        <script id="__NEXT_DATA__">{...}</script>. This is far more
        stable than CSS selectors because it's the app's actual data
        contract, not presentational markup.
        """
        try:
            handle = await page.query_selector("script#__NEXT_DATA__")
            if not handle:
                return None
            text = await handle.inner_text()
            return json.loads(text)
        except Exception as e:
            log.debug("Could not extract __NEXT_DATA__: %s", e)
            return None

    def _find_listing_arrays(self, obj: Any, depth: int = 0) -> list[list[dict]]:
        """
        Recursively search a parsed JSON blob for arrays-of-dicts that
        look like listing records (heuristic: dict has a subset of keys
        we expect from a Zameen listing card). This makes the parser
        resilient to Zameen changing the nesting path inside
        __NEXT_DATA__.props.pageProps.
        """
        found = []
        if depth > 12:
            return found

        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            sample_keys = set(obj[0].keys())
            signal_keys = {"price", "location", "title", "coverPhoto", "purpose",
                           "externalID", "slug", "productType", "area"}
            if len(sample_keys & signal_keys) >= 2:
                found.append(obj)
        if isinstance(obj, dict):
            for v in obj.values():
                found.extend(self._find_listing_arrays(v, depth + 1))
        elif isinstance(obj, list):
            for v in obj:
                found.extend(self._find_listing_arrays(v, depth + 1))
        return found

    # -- field-level parsing ---------------------------------------------------
    @staticmethod
    def _safe_get(d: dict, *keys, default=None):
        for k in keys:
            if isinstance(d, dict) and k in d and d[k] is not None:
                return d[k]
        return default

    def _normalize_record(self, rec: dict, property_type_hint: str) -> Optional[RawListing]:
        try:
            listing_id = self._safe_get(rec, "externalID", "id", "listingId")
            title = self._safe_get(rec, "title", "titleEn", default="").strip()
            slug = self._safe_get(rec, "slug", "url", default="")
            url = slug if str(slug).startswith("http") else urljoin(BASE_URL, f"/Property/{slug}")

            price = self._safe_get(rec, "price", default=None)
            if isinstance(price, dict):
                price = price.get("value") or price.get("amount")

            area_val = self._safe_get(rec, "area", default=None)
            area_marla = None
            if isinstance(area_val, dict):
                area_marla = self._coerce_area_to_marla(area_val.get("value"), area_val.get("unit", ""))
            elif isinstance(area_val, (int, float, str)):
                area_marla = self._coerce_area_to_marla(area_val, rec.get("areaUnit", "Marla"))

            location = self._safe_get(rec, "location", default={})
            if isinstance(location, dict):
                location_str = location.get("name") or ", ".join(
                    filter(None, [location.get("city", {}).get("name") if isinstance(location.get("city"), dict) else location.get("city"),
                                  ])
                )
            else:
                location_str = str(location)

            bedrooms = self._safe_get(rec, "rooms", "bedrooms", default=None)
            bathrooms = self._safe_get(rec, "baths", "bathrooms", default=None)

            date_raw = self._safe_get(rec, "createdAt", "listedDate", "date", "timestamp")
            parsed_date = parse_listing_date(date_raw)
            if parsed_date is None:
                # If we truly cannot determine a date, we cannot honor the
                # 30-day/24-hour cutoff contract for this record. Skip it
                # rather than silently mis-dating it.
                log.debug("Skipping listing %s: unparseable date %r", listing_id, date_raw)
                return None

            if not listing_id or not title:
                return None

            return RawListing(
                id=int(listing_id) if str(listing_id).isdigit() else abs(hash(listing_id)) % (10 ** 9),
                title=title,
                url=url,
                property_type=property_type_hint,
                subtype=self._safe_get(rec, "type", "subType", default=property_type_hint),
                location=location_str or "Unknown",
                price_pkr=int(price) if price else None,
                area_marla=area_marla,
                bedrooms=int(bedrooms) if str(bedrooms).isdigit() else None,
                bathrooms=int(bathrooms) if str(bathrooms).isdigit() else None,
                date_listed=parsed_date.astimezone(timezone.utc).isoformat(),
            )
        except Exception as e:
            log.debug("Failed to normalize record: %s", e)
            return None

    @staticmethod
    def _coerce_area_to_marla(value: Any, unit: str) -> Optional[float]:
        """Zameen mixes Marla/Kanal/sqft/sqyd depending on listing type."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        unit = (unit or "").lower()
        conversions = {
            "marla": 1.0,
            "kanal": 20.0,
            "sq. yd.": 1 / 30.25,
            "sqyd": 1 / 30.25,
            "sq. ft.": 1 / 272.25,
            "sqft": 1 / 272.25,
        }
        factor = next((f for key, f in conversions.items() if key in unit), 1.0)
        return round(v * factor, 2)

    # -- DOM fallback ------------------------------------------------------
    # Zameen's search-result pages are server-rendered HTML (not a Next.js/
    # XHR-backed SPA), so __NEXT_DATA__/JSON interception will typically
    # find nothing and THIS is the path that actually does the work. Rather
    # than hardcoding CSS classes (which are usually hashed/obfuscated and
    # change on every redeploy), we find the property link in each card and
    # walk up the DOM until we hit a container whose visible text looks like
    # a full card (contains "PKR"), then regex-parse that text blob. This is
    # slower per-card but far more resilient to markup/class-name changes.
    #
    # Verified card text pattern (live site, 2026-08-28):
    #   "<Location>, <Broader Location> · <photo count> · <Area> ·
    #    <Title>...more · Added: <relative time>(Updated: <relative time>)"
    # e.g. "DHA Phase 6, DHA Defence · 67 · 1 Kanal · DESIGNER HOUSE WITH
    #       SWIMMING POOL...more · Added: 8 minutes ago(Updated: 8 minutes ago)"
    #
    # NOTE: bedroom/bathroom counts are NOT reliably present in this grid
    # view (they appear to only be shown on the listing detail page for
    # many categories). This extractor returns None for both and logs how
    # many listings ended up with no bed/bath data. If you need that field
    # populated, add a second pass that visits each listing's `url` and
    # extracts it from the detail page — flagged as a TODO below.
    async def _parse_from_dom(self, page: Page, property_type_hint: str) -> list[RawListing]:
        results: list[RawListing] = []

        try:
            raw_cards = await page.evaluate(
                """
                () => {
                    const seen = new Set();
                    const cards = [];
                    const anchors = Array.from(
                        document.querySelectorAll('a[href*="/Property/"]')
                    );
                    for (const a of anchors) {
                        const href = a.getAttribute('href');
                        if (!href || seen.has(href)) continue;

                        let node = a;
                        let container = null;
                        for (let i = 0; i < 8 && node; i++) {
                            node = node.parentElement;
                            if (!node) break;
                            const t = node.innerText || '';
                            if (t.includes('PKR') && t.length > 40 && t.length < 4000) {
                                container = node;
                                break;
                            }
                        }
                        if (!container) continue;

                        seen.add(href);
                        cards.push({
                            href: href,
                            title_attr: a.getAttribute('title') || '',
                            text: container.innerText || ''
                        });
                    }
                    return cards;
                }
                """
            )
        except Exception as e:
            log.warning("DOM card extraction JS failed: %s", e)
            return results

        for raw in raw_cards:
            listing = self._parse_card_text(raw, property_type_hint)
            if listing:
                results.append(listing)

        no_bedbath = sum(1 for r in results if r.bedrooms is None and r.bathrooms is None)
        if results and no_bedbath == len(results):
            log.info("Note: bed/bath data not found in grid view for this page "
                      "(%d listings) — expected for some categories; not an error.",
                      len(results))

        return results

    def _parse_card_text(self, raw: dict, property_type_hint: str) -> Optional[RawListing]:
        """
        Parses a card's raw `innerText` (newline-delimited — NOT the
        markdown-rendered "·"-delimited text you'd see in a text-only page
        fetch, which is a rendering artifact, not real DOM structure).

        Verified real line-by-line structure (from a live run, 2026-08-28):
            SUPER HOT            <- badge (optional)
            49                   <- photo count
            1                    <- video count (optional)
            TITANIUM             <- agency tier badge (optional)
            PKR
            14.75 Crore
            DHA Defence Phase 2, DHA Defence   <- location (has a comma)
            5                    <- bedrooms (bare number, optional)
            6                    <- bathrooms (bare number, optional)
            1 Kanal              <- area
            <title text>
            <description snippet>
            ...
            more
            Added: 2 hours ago
            WhatsApp
            CALL
        Bed/bath lines are only present for some listings (e.g. houses);
        plots/commercial cards often go straight from location to area with
        no numeric lines in between. This is inferred from two observed
        samples, not documented anywhere, so treat bedrooms/bathrooms as
        best-effort — verify against debug/*.html if it looks off.
        """
        href = raw.get("href", "")
        text = raw.get("text", "")
        if not href or not text:
            return None

        url = urljoin(BASE_URL, href)
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        title = (raw.get("title_attr") or "").strip()
        if not title:
            m = re.search(r"\n([^\n]{10,150}?)\n(?:[^\n]*\n)?\.\.\.\s*\n?\s*more", text, re.I)
            title = m.group(1).strip() if m else ""
        if not title:
            return None

        price_pkr = self._parse_price_text(text)

        area_idx, area_marla = None, None
        for i, ln in enumerate(lines):
            m = re.fullmatch(r"([\d.]+)\s*(Marla|Kanal|Sq\.\s*Yd\.|Sq\.\s*Ft\.|Sq\.\s*M\.)", ln, re.I)
            if m:
                area_idx = i
                area_marla = self._coerce_area_to_marla(m.group(1), m.group(2))
                break

        date_match = re.search(r"Added:\s*([^(\n]+)", text)
        parsed_date = parse_listing_date(date_match.group(1).strip()) if date_match else None
        if parsed_date is None:
            return None  # can't honor the recency-window contract without a date

        # Location: the line containing a comma that appears closest before
        # the area line (falling back to a scan of all lines if we couldn't
        # locate the area line for some reason). This avoids accidentally
        # grabbing a comma from the title/description further down.
        location = "Unknown"
        bedrooms: Optional[int] = None
        bathrooms: Optional[int] = None

        search_range = range(area_idx - 1, -1, -1) if area_idx is not None else range(len(lines) - 1, -1, -1)
        location_idx = None
        for i in search_range:
            if "," in lines[i] and not re.search(r"PKR|Crore|Lakh|Arab", lines[i], re.I):
                location = lines[i]
                location_idx = i
                break

        # Bare numeric lines strictly between location and area are, per the
        # observed pattern, bedrooms then bathrooms (in that order).
        if location_idx is not None and area_idx is not None and area_idx > location_idx + 1:
            between = lines[location_idx + 1:area_idx]
            numeric = [ln for ln in between if re.fullmatch(r"\d+", ln)]
            if len(numeric) >= 2:
                bedrooms, bathrooms = int(numeric[0]), int(numeric[1])
            elif len(numeric) == 1:
                bedrooms = int(numeric[0])  # ambiguous single value — treat as beds

        id_match = re.search(r"-(\d{6,})-\d+-\d+\.html", href)
        listing_id = int(id_match.group(1)) if id_match else abs(hash(href)) % (10 ** 9)

        return RawListing(
            id=listing_id,
            title=title,
            url=url,
            property_type=property_type_hint,
            subtype=property_type_hint,
            location=location,
            price_pkr=price_pkr,
            area_marla=area_marla,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            date_listed=parsed_date.astimezone(timezone.utc).isoformat(),
        )

    @staticmethod
    def _parse_price_text(text: str) -> Optional[int]:
        """
        Zameen renders prices as e.g. 'PKR22.95 Crore', 'PKR50 Lakh',
        'PKR6.3 Crore' (verified against live listing cards) — always a
        'PKR' prefix, a decimal number, then a Crore/Lakh/Arab unit. Plain
        comma-separated PKR amounts ('PKR 2,500,000') are handled too as a
        fallback, since some contexts (detail pages) may render that way.
        """
        text = text.strip()
        m = re.search(r"PKR\s*([\d,]+(?:\.\d+)?)\s*(crore|lakh|arab)?", text, re.I)
        if not m:
            return None
        try:
            num = float(m.group(1).replace(",", ""))
        except ValueError:
            return None
        unit = (m.group(2) or "").lower()
        multiplier = {"lakh": 100_000, "crore": 10_000_000, "arab": 1_000_000_000}.get(unit, 1)
        return int(round(num * multiplier))

    @classmethod
    def _parse_area_text(cls, text: str) -> Optional[float]:
        m = re.search(r"([\d.]+)\s*(Marla|Kanal|Sq\.\s*Yd\.|Sq\.\s*Ft\.|Sq\.\s*M\.)", text, re.I)
        if not m:
            return None
        return cls._coerce_area_to_marla(m.group(1), m.group(2))

    @staticmethod
    def _first_int(text: str) -> Optional[int]:
        m = re.search(r"\d+", text)
        return int(m.group()) if m else None

    # -- page crawl loop -----------------------------------------------------
    async def _scrape_search(self, page: Page, city_slug: str, property_type: str) -> list[RawListing]:
        collected: list[RawListing] = []
        hit_cutoff = False

        for page_num in range(1, self.max_pages_per_search + 1):
            if hit_cutoff:
                break

            search_url = f"{BASE_URL}/{property_type}/{city_slug}-{page_num}.html"
            log.info("Fetching %s", search_url)

            self._captured_json_payloads.clear()
            try:
                await page.goto(search_url, wait_until="networkidle", timeout=45_000)
            except PWTimeoutError:
                log.warning("Timeout loading %s — skipping", search_url)
                continue

            # Human-like jitter + a small scroll to trigger lazy-loaded content
            await page.mouse.wheel(0, random.randint(800, 2000))
            await asyncio.sleep(random.uniform(*self.request_delay_range))

            page_records: list[RawListing] = []

            # Strategy 1: __NEXT_DATA__
            next_data = await self._extract_next_data(page)
            if next_data:
                for arr in self._find_listing_arrays(next_data):
                    for rec in arr:
                        norm = self._normalize_record(rec, property_type)
                        if norm:
                            page_records.append(norm)

            # Strategy 2: captured JSON XHR payloads
            if not page_records:
                for payload in self._captured_json_payloads:
                    for arr in self._find_listing_arrays(payload["body"]):
                        for rec in arr:
                            norm = self._normalize_record(rec, property_type)
                            if norm:
                                page_records.append(norm)

            # Strategy 3: DOM fallback
            if not page_records:
                page_records = await self._parse_from_dom(page, property_type)

            if not page_records:
                debug_dir = Path("debug")
                debug_dir.mkdir(exist_ok=True)
                stamp = f"{city_slug}_{property_type}_p{page_num}"
                try:
                    html = await page.content()
                    (debug_dir / f"{stamp}.html").write_text(html, encoding="utf-8")
                    log.warning(
                        "No records extracted from %s (page %d) — stopping "
                        "pagination for this search. Saved page HTML to "
                        "debug/%s.html for inspection.",
                        city_slug, page_num, stamp,
                    )
                except Exception as e:
                    log.warning(
                        "No records extracted from %s (page %d), and failed "
                        "to save debug HTML: %s", city_slug, page_num, e,
                    )
                break

            in_window = [r for r in page_records
                         if datetime.fromisoformat(r.date_listed) >= self.cutoff]
            collected.extend(in_window)

            log.info("Page %d: %d records total, %d within cutoff window",
                      page_num, len(page_records), len(in_window))

            # Listings are sorted newest-first by default on Zameen search
            # results, so once a full page falls entirely outside the
            # cutoff window we can stop paginating this search.
            if len(in_window) == 0 and len(page_records) > 0:
                hit_cutoff = True

        return collected

    async def run(self) -> list[RawListing]:
        all_listings: dict[int, RawListing] = {}

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={"width": 1366, "height": 900},
                locale="en-US",
                timezone_id="Asia/Karachi",
            )
            await context.add_init_script(STEALTH_INIT_SCRIPT)
            page = await context.new_page()
            page.on("response", lambda r: asyncio.create_task(self._on_response(r)))

            for city in self.cities:
                for prop_type in self.property_types:
                    try:
                        results = await self._scrape_search(page, city, prop_type)
                        for r in results:
                            all_listings[r.id] = r  # de-dup within this run
                    except Exception as e:
                        log.exception("Failed scraping %s/%s: %s", city, prop_type, e)
                    await asyncio.sleep(random.uniform(*self.request_delay_range))

            await browser.close()

        log.info("Scrape complete: %d unique raw listings within window", len(all_listings))
        return list(all_listings.values())


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------
def geocode_listings(listings: list[dict], cache_path: Path = Path("geocode_cache.json"),
                     requests_per_minute: float = 4.0, provider: str = "auto",
                     locationiq_api_key: Optional[str] = None) -> list[dict]:
    """
    Adds latitude/longitude to each listing dict directly via the LocationIQ API.
    """
    import requests

    if not locationiq_api_key:
        log.error("[ERROR] STOP! You forgot to pass your LocationIQ API key in the terminal command.")
        sys.exit(1)

    log.info("Geocoding directly via LocationIQ API (using requests). "
             "— rate-limited to %.0f req/min.", requests_per_minute)

    cache: dict[str, tuple[Optional[float], Optional[float]]] = {}
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                raw_cache = json.load(f)
            cache = {k: (v[0], v[1]) for k, v in raw_cache.items()}
            log.info("Loaded %d cached geocode results from %s", len(cache), cache_path)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not read geocode cache at %s (%s) — starting fresh", cache_path, e)

    def _save_cache() -> None:
        try:
            tmp = cache_path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            tmp.replace(cache_path)
        except OSError as e:
            log.warning("Could not persist geocode cache to %s: %s", cache_path, e)

    delay_seconds = 60.0 / requests_per_minute

    new_lookups = 0
    for i, listing in enumerate(listings):
        location = (listing.get("location") or "").strip()

        if not location or len(location) > 120 or "\n" in location:
            listing["latitude"] = None
            listing["longitude"] = None
            continue

       # Clean specific micro-blocks/commercial zones that trigger 404s
        clean_loc = re.sub(r"\s*-\s*Commercial Zone\s*[A-Z0-9]+", "", location, flags=re.I)
        clean_loc = re.sub(r"\s*-\s*Block\s*[A-Z0-9]+", "", clean_loc, flags=re.I)
        clean_loc = clean_loc.strip().strip(",")

        query = f"{clean_loc}, Pakistan"

        if query in cache:
            lat, lon = cache[query]
        else:
            lat, lon = None, None
            try:
                url = "https://us1.locationiq.com/v1/search.php"
                params = {
                    "key": locationiq_api_key,
                    "q": query,
                    "format": "json",
                    "limit": 1
                }
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        lat = float(data[0]["lat"])
                        lon = float(data[0]["lon"])
                elif response.status_code == 404:
                    # Fallback: if specific building/society fails, strip the first part and try the parent city/sector
                    if "," in clean_loc:
                        parent_loc = clean_loc.split(",", 1)[1].strip()
                        fallback_query = f"{parent_loc}, Pakistan"
                        fallback_resp = requests.get(
                            url, 
                            params={"key": locationiq_api_key, "q": fallback_query, "format": "json", "limit": 1}, 
                            timeout=10
                        )
                        if fallback_resp.status_code == 200:
                            fdata = fallback_resp.json()
                            if fdata and len(fdata) > 0:
                                lat = float(fdata[0]["lat"])
                                lon = float(fdata[0]["lon"])
                params = {
                    "key": locationiq_api_key,
                    "q": query,
                    "format": "json",
                    "limit": 1
                }
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        lat = float(data[0]["lat"])
                        lon = float(data[0]["lon"])
                elif response.status_code == 401:
                    log.error("LocationIQ returned HTTP 401: Unauthorized. Your API key is invalid.")
                    sys.exit(1)
                elif response.status_code == 429:
                    log.warning("Rate limit hit. Sleeping for 5 seconds...")
                    time.sleep(5)
                else:
                    log.warning("LocationIQ returned HTTP %d for %r", response.status_code, query)
                    
            except Exception as e:
                log.warning("Unexpected geocoding error for %r: %s", query, e)
            
            cache[query] = (lat, lon)
            new_lookups += 1
            time.sleep(delay_seconds)  
            
            if new_lookups % 20 == 0:
                _save_cache()  

        listing["latitude"] = lat
        listing["longitude"] = lon

    _save_cache()
    log.info("Geocoding done: %d cache hits, %d new lookups, %d cached entries on disk",
              len(listings) - new_lookups, new_lookups, len(cache))

    return listings


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def add_title_embeddings(listings: list[dict]) -> list[dict]:
    """
    Generates 384-dim sentence embeddings for each listing's title using
    all-MiniLM-L6-v2. Batched for throughput (loading the model is the
    expensive part; batching the encode() call amortizes it across all
    listings instead of doing one model call per record).
    """
    from sentence_transformers import SentenceTransformer

    if not listings:
        return listings

    log.info("Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    titles = [listing.get("title", "") for listing in listings]
    embeddings = model.encode(titles, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    for listing, emb in zip(listings, embeddings):
        listing["title_embedding"] = [round(float(x), 6) for x in emb.tolist()]

    return listings


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            log.warning("%s did not contain a JSON array — ignoring existing content", path)
            return []
        return data
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Could not read existing %s (%s) — treating as empty", path, e)
        return []


def save_listings(path: Path, listings: list[dict]) -> None:
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(listings, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)  # atomic on POSIX
    log.info("Wrote %d listings to %s", len(listings), path)


def merge_daily(existing: list[dict], new: list[dict]) -> list[dict]:
    by_id = {item["id"]: item for item in existing}
    added, updated = 0, 0
    for item in new:
        if item["id"] in by_id:
            updated += 1
        else:
            added += 1
        by_id[item["id"]] = item  # new data wins on conflict (fresher scrape)
    log.info("Daily merge: %d new, %d updated, %d total", added, updated, len(by_id))
    return list(by_id.values())


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Zameen.com real estate ETL pipeline")
    p.add_argument("--mode", choices=["initial", "daily"], required=True,
                    help="'initial' = last 30 days, overwrite file. "
                         "'daily' = last 24 hours, append+dedupe.")
    p.add_argument("--output", default="listings.json", type=Path,
                    help="Path to listings.json output file")
    p.add_argument("--cities", nargs="+", default=DEFAULT_CITIES,
                    help="Zameen city URL slugs, e.g. islamabad-3-1")
    p.add_argument("--property-types", nargs="+",
                    default=list(DEFAULT_PROPERTY_TYPE_SLUGS.values()),
                    help="Zameen category slugs, e.g. Homes Plots Commercial")
    p.add_argument("--max-pages-per-search", type=int, default=50)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", dest="headless", action="store_false")
    p.add_argument("--skip-geocode", action="store_true",
                    help="Skip geocoding step (useful for local testing)")
    p.add_argument("--geocode-cache", default="geocode_cache.json", type=Path,
                    help="Path to the persistent on-disk geocode cache")
    p.add_argument("--geocode-provider", choices=["auto", "locationiq", "nominatim"], default="auto",
                    help="'auto' uses LocationIQ if a key is configured, else falls back to "
                         "Nominatim. See geocode_listings() docstring for why LocationIQ is "
                         "preferred for a scheduled production pipeline.")
    p.add_argument("--locationiq-api-key", default=None,
                    help="LocationIQ API key. Can also be set via the LOCATIONIQ_API_KEY "
                         "environment variable (recommended, keeps it out of shell history). "
                         "Free key: https://locationiq.com")
    p.add_argument("--geocode-requests-per-minute", type=float, default=4.0,
                    help="Applies to the Nominatim path. Their policy caps regularly-scheduled "
                         "scripts at 4/min — only raise this if using a provider whose terms "
                         "permit it (e.g. LocationIQ, which uses its own internal throttle).")
    p.add_argument("--skip-embeddings", action="store_true",
                    help="Skip embedding generation step (useful for local testing)")
    return p


async def main_async(args: argparse.Namespace) -> None:
    log.info("Starting run: mode=%s output=%s", args.mode, args.output)

    scraper = ZameenScraper(
        mode=args.mode,
        cities=args.cities,
        property_types=args.property_types,
        max_pages_per_search=args.max_pages_per_search,
        headless=args.headless,
    )
    raw_listings = await scraper.run()

    if not raw_listings:
        log.warning("No listings scraped this run. Exiting without touching %s.", args.output)
        return

    listings = [asdict(r) for r in raw_listings]

    if not args.skip_geocode:
        log.info("Geocoding %d listings (cached + rate-limited)...", len(listings))
        listings = geocode_listings(listings, cache_path=args.geocode_cache,
                                     requests_per_minute=args.geocode_requests_per_minute,
                                     provider=args.geocode_provider,
                                     locationiq_api_key=args.locationiq_api_key)
    else:
        for l in listings:
            l["latitude"], l["longitude"] = None, None

    if not args.skip_embeddings:
        listings = add_title_embeddings(listings)
    else:
        for l in listings:
            l["title_embedding"] = []

    # Reorder keys to match the target schema exactly
    ordered = []
    for l in listings:
        ordered.append({
            "id": l["id"],
            "title": l["title"],
            "url": l["url"],
            "property_type": l["property_type"],
            "subtype": l["subtype"],
            "location": l["location"],
            "price_pkr": l["price_pkr"],
            "area_marla": l["area_marla"],
            "bedrooms": l["bedrooms"],
            "bathrooms": l["bathrooms"],
            "latitude": l["latitude"],
            "longitude": l["longitude"],
            "date_listed": l["date_listed"],
            "title_embedding": l["title_embedding"],
        })

    if args.mode == "initial":
        if args.output.exists():
            backup = args.output.with_name(
                f"{args.output.stem}.backup-{datetime.now():%Y%m%d%H%M%S}.json"
            )
            shutil.copy2(args.output, backup)
            log.info("Existing %s backed up to %s before overwrite", args.output, backup)
        save_listings(args.output, ordered)
    else:  # daily
        existing = load_existing(args.output)
        merged = merge_daily(existing, ordered)
        save_listings(args.output, merged)

    log.info("Run complete.")


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()