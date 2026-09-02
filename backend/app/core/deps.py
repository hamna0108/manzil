"""
Shared FastAPI dependencies: DB session, required auth (for save/history/
saved-search endpoints), and OPTIONAL auth (for /search, which must stay
usable by anonymous users but should still log history when a valid
token happens to be present).
"""

import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.database import get_db
from app.models.user import User

# auto_error=True (default) -- used where auth is REQUIRED. FastAPI/Swagger
# also uses this to render the "Authorize" button in the docs UI.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=True)

# Separate scheme with auto_error=False for endpoints where a token is
# OPTIONAL -- lets /search stay open to anonymous users while still
# resolving a User if a valid token IS provided.
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """REQUIRED auth. Raises 401 if the token is missing/invalid/expired,
    or if it decodes fine but the user no longer exists."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    user_id = decode_access_token(token)
    if user_id is None:
        raise credentials_exception

    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exception
    return user


async def get_current_user_optional(
    token: str | None = Depends(oauth2_scheme_optional),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """OPTIONAL auth for /search. Returns None (never raises) if there's
    no token, an invalid token, or an expired token -- an anonymous or
    stale-token request should still get search results, just without
    history logging."""
    if token is None:
        return None
    user_id = decode_access_token(token)
    if user_id is None:
        return None
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return None
    result = await db.execute(select(User).where(User.id == user_uuid))
    return result.scalar_one_or_none()
