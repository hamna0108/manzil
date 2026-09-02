import uuid
from datetime import datetime

from pydantic import BaseModel


class SaveListingRequest(BaseModel):
    listing_id: int
    title: str
    url: str | None = None
    property_type: str | None = None
    subtype: str | None = None
    location: str | None = None
    price_pkr: int | None = None
    area_marla: float | None = None


class SavedListingResponse(BaseModel):
    id: uuid.UUID
    listing_id: int
    title: str
    url: str | None
    property_type: str | None
    subtype: str | None
    location: str | None
    price_pkr: int | None
    area_marla: float | None
    saved_at: datetime

    model_config = {"from_attributes": True}
