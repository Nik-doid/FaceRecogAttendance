"""Database engines and session factories.

The API is async (asyncpg) so FastAPI request handlers never block the event loop.
The background recognition worker is synchronous (OpenCV/ONNX are sync), so it uses
its own sync engine (psycopg) against the same database. Both share the ORM models.

Tests can point DATABASE_URL at SQLite (aiosqlite / sqlite) — no code changes needed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import settings

_async_engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)
_async_session_factory = async_sessionmaker(
    _async_engine, class_=AsyncSession, expire_on_commit=False
)

_sync_engine = create_engine(
    settings.database_sync_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)
_sync_session_factory = sessionmaker(
    _sync_engine, class_=Session, expire_on_commit=False
)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one async session per request, always closed."""
    async with _async_session_factory() as session:
        yield session


def get_sync_session() -> Iterator[Session]:
    """Context-manager generator for the sync worker sessions."""
    with _sync_session_factory() as session:
        yield session


def async_session() -> AsyncSession:
    """Standalone async session factory for code outside request handlers."""
    return _async_session_factory()


def sync_session() -> Session:
    """Standalone sync session factory for the worker."""
    return _sync_session_factory()


async def close_engines() -> None:
    """Dispose connection pools on shutdown."""
    await _async_engine.dispose()
    _sync_engine.dispose()
