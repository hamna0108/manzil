import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database import get_db
from app.models.saved_listing import SavedListing
from app.models.user import User
from app.schemas.saved_listing import SaveListingRequest, SavedListingResponse

router = APIRouter(prefix="/saved-listings", tags=["saved-listings"])


@router.post("", response_model=SavedListingResponse, status_code=status.HTTP_201_CREATED)
async def save_listing(
    payload: SaveListingRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    saved = SavedListing(user_id=current_user.id, **payload.model_dump())
    db.add(saved)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This listing is already saved.")
    await db.refresh(saved)
    return saved


@router.get("", response_model=list[SavedListingResponse])
async def get_saved_listings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SavedListing).where(SavedListing.user_id == current_user.id).order_by(SavedListing.saved_at.desc())
    )
    return result.scalars().all()


@router.delete("/{listing_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unsave_listing(
    listing_id: int,  # <-- UUID ki jagah 'int' kar diya
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        # <-- SavedListing.id ki jagah SavedListing.listing_id check karein
        select(SavedListing).where(SavedListing.listing_id == listing_id, SavedListing.user_id == current_user.id)
    )
    saved = result.scalar_one_or_none()
    if saved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved listing not found.")

    await db.delete(saved)
    await db.commit()
