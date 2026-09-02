from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user_optional
from app.database import get_db
from app.models.search_history import SearchHistory
from app.models.user import User
from app.schemas.search import SearchRequest, SearchResponse
from app.services.pipeline_bridge import run_search

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    current_user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """
    Runs the full Steps 0-5 pipeline end-to-end. Open to anonymous
    users (current_user may be None) -- search history is only written
    when a valid, currently-logged-in user made the request.

    Clarification flow (see schemas/search.py's SearchRequest for the
    full reasoning): if this returns status="needs_clarification", the
    client re-calls this same endpoint with `intent` from the response
    echoed back as `intent_override`, plus `clarification_field` and
    `clarification_value` set to whatever the user picked.
    """
    result = await run_search(
        query=payload.query,
        top_k=payload.top_k,
        intent_override=payload.intent_override,
        clarification_field=payload.clarification_field,
        clarification_value=payload.clarification_value,
    )

    # Only log history for a real, currently-authenticated user, and
    # only for a completed search (not a clarification round-trip,
    # which isn't a "real" search result yet) -- an anonymous or
    # not-yet-resolved search leaves no trace, per the requirement that
    # history is an opt-in-by-login feature.
    if current_user is not None and result["status"] == "ok":
        db.add(SearchHistory(
            user_id=current_user.id,
            query_text=payload.query,
            intent_json=result["intent"],
            result_count=len(result["results"]),
        ))
        await db.commit()

    return result
