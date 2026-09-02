"""
Async SQLAlchemy 2.0 engine + session factory. Async (not sync SQLAlchemy)
because the FastAPI app is async throughout -- keeping the DB layer
async too avoids a second, different kind of blocking-call problem
alongside the sync-pipeline-in-threadpool issue handled in
services/pipeline_bridge.py.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    """All ORM models inherit from this."""
    pass


async def get_db():
    """FastAPI dependency -- yields a session, always closes it after
    the request, even if the route raises."""
    async with AsyncSessionLocal() as session:
        yield session
