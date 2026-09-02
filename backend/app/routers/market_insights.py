from fastapi import APIRouter, Query

from app.schemas.market_insights import MarketInsightsResponse
from app.services.market_insights_service import generate_market_insights

router = APIRouter(prefix="/market-insights", tags=["market-insights"])


@router.get("", response_model=MarketInsightsResponse)
async def market_insights(
    property_type: str = Query(..., description="e.g. House, Flat, Farmhouse, Plot"),
    location: str = Query(..., description="e.g. 'DHA Lahore', 'Bahria Town Rawalpindi'"),
    subtype: str | None = Query(None, description="e.g. Commercial Plot, Upper Portion"),
):
    """
    Open endpoint (no auth required) -- real, computed stats (avg/min/
    max price, price per marla) for a category+location combination,
    plus an optional one-line AI insight. Stats are always returned
    even if the AI insight generation fails or isn't configured.
    """
    return await generate_market_insights(property_type=property_type, location=location, subtype=subtype)
