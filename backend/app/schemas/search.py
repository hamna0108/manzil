from typing import Any

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str

    top_k: int = 5

    # --- Clarification round-trip (turn 2 only) ---
    # If the first /search call returned status="needs_clarification",
    # the client re-calls /search with the SAME `intent` object echoed
    # back here, plus the field/value the user picked. This keeps the
    # whole flow stateless (no server-side session storage) and avoids
    # re-running Qwen/Gemini extraction a second time for the same query.
    intent_override: dict[str, Any] | None = None
    clarification_field: str | None = None
    clarification_value: str | None = None


class POIVerification(BaseModel):
    verified: bool | None
    distance_m: float | None = None
    name: str | None = None
    reason: str | None = None


class ListingResult(BaseModel):
    id: int
    title: str
    url: str
    property_type: str
    subtype: str | None
    location: str
    price_pkr: int
    area_marla: float
    bedrooms: int
    bathrooms: int
    latitude: float | None
    longitude: float | None
    similarity_score: float
    poi_verification: dict[str, POIVerification] = {}


class ClarificationInfo(BaseModel):
    field: str
    question: str
    options: list[str]


class SearchResponse(BaseModel):
    status: str  # "ok" | "needs_clarification"
    intent: dict[str, Any]
    clarification: ClarificationInfo | None = None
    message: str | None = None
    relaxed_constraints: list[str] = []
    results: list[ListingResult] = []
    summary: str | None = None
