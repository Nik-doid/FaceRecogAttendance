"""FastAPI dependencies: container access, DB helpers, auth, rate limiting.

The control endpoints are async ``def`` handlers. The repositories are written for
the synchronous SQLAlchemy Session (they are also used by the sync worker), so every
DB call is dispatched to a thread via ``asyncio.to_thread`` to keep the event loop
unblocked.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Annotated, TypeVar

from fastapi import Depends
from sqlalchemy.orm import Session

from app.container import Container
from app.core.security import get_current_user
from app.database.session import sync_session

R = TypeVar("R")

_container: Container | None = None


def set_container(container: Container) -> None:
    global _container
    _container = container


def get_container() -> Container:
    if _container is None:
        raise RuntimeError("Container has not been initialized; call set_container at startup")
    return _container


async def run_db[R](fn: Callable[..., R], *args: object, **kwargs: object) -> R:
    """Run a blocking repository call off the event loop."""
    return await asyncio.to_thread(fn, *args, **kwargs)


@contextmanager
def db_session() -> Iterator[Session]:
    """Synchronous session for use inside async endpoints (closed by caller)."""
    session = sync_session()
    try:
        yield session
    finally:
        session.close()


require_admin = get_current_user

# Typed dependency aliases for FastAPI route signatures.
ContainerDep = Annotated[Container, Depends(get_container)]
AdminDep = Annotated[str, Depends(require_admin)]
