"""
Application settings, loaded from environment variables / .env.
Uses pydantic-settings so config is validated at startup rather than
failing halfway through a request when a variable turns out missing.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/property_finder"

    # --- Auth ---
    jwt_secret_key: str = "change-me-in-.env-this-default-is-not-secure"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # --- Pipeline (Steps 0-5) ---
    gemini_api_key: str = ""
    locationiq_api_key: str = ""
    qwen_model_path: str = "C:/Users/Gull/Desktop/Real Estate/backend/llama.cpp/qwen-intent-q4.gguf"
    listings_path: str = "./data/listings.json"

    # --- CORS ---
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    """Cached so .env is only parsed once per process, not per request."""
    return Settings()
