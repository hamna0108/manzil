import os
from pathlib import Path
from dotenv import load_dotenv
from scraper import geocode_listings

# .env se key load karein
load_dotenv()
api_key = os.getenv("LOCATIONIQ_API_KEY")

print(f"Loaded API Key: {api_key[:6]}... (length: {len(api_key) if api_key else 0})")

# Dummy listings sirf geocoding test karne ke liye
sample_listings = [
    {"id": 1, "title": "Test House 1", "location": "Gulberg 3, Lahore"},
    {"id": 2, "title": "Test House 2", "location": "F-7 Markaz, Islamabad"},
]

# Test run (cache alag test file mein save hogi taake main cache disturb na ho)
results = geocode_listings(
    sample_listings,
    cache_path=Path("test_geocode_cache.json"),
    requests_per_minute=60.0,
    locationiq_api_key=api_key
)

print("\n--- RESULTS ---")
for item in results:
    print(f"Location: {item['location']} -> Lat: {item['latitude']}, Lon: {item['longitude']}")