import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class SearchHistoryResponse(BaseModel):
    id: uuid.UUID
    query_text: str
    intent_json: dict[str, Any]
    result_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class SaveSearchRequest(BaseModel):
    name: str | None = None
    query_text: str
    intent_json: dict[str, Any]
    alerts_enabled: bool = False


class SavedSearchResponse(BaseModel):
    id: uuid.UUID
    name: str | None
    query_text: str
    intent_json: dict[str, Any]
    alerts_enabled: bool
    created_at: datetime

    model_config = {"from_attributes": True}
