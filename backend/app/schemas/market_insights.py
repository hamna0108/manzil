from pydantic import BaseModel


class MarketInsightsResponse(BaseModel):
    property_type: str
    location: str
    listing_count: int
    avg_price_pkr: float | None
    min_price_pkr: int | None
    max_price_pkr: int | None
    avg_price_per_marla_pkr: float | None
    ai_insight: str | None