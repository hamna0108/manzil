import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database import get_db
from app.models.saved_search import SavedSearch
from app.models.search_history import SearchHistory
from app.models.user import User
from app.schemas.history import (
    SaveSearchRequest,
    SavedSearchResponse,
    SearchHistoryResponse,
)

router = APIRouter(tags=["history"])


@router.get("/search-history", response_model=list[SearchHistoryResponse])
async def get_search_history(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SearchHistory)
        .where(SearchHistory.user_id == current_user.id)
        .order_by(SearchHistory.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


@router.post("/saved-searches", response_model=SavedSearchResponse, status_code=status.HTTP_201_CREATED)
async def save_search(
    payload: SaveSearchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    saved = SavedSearch(user_id=current_user.id, **payload.model_dump())
    db.add(saved)
    await db.commit()
    await db.refresh(saved)
    return saved


@router.get("/saved-searches", response_model=list[SavedSearchResponse])
async def get_saved_searches(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SavedSearch).where(SavedSearch.user_id == current_user.id).order_by(SavedSearch.created_at.desc())
    )
    return result.scalars().all()


@router.delete("/saved-searches/{saved_search_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_saved_search(
    saved_search_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SavedSearch).where(SavedSearch.id == saved_search_id, SavedSearch.user_id == current_user.id)
    )
    saved = result.scalar_one_or_none()
    if saved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved search not found.")

    await db.delete(saved)
    await db.commit()
