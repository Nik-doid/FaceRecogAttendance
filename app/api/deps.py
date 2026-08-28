"""FastAPI dependencies: container access, auth, rate limiting."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.container import Container
from app.core.security import get_current_user

_container: Container | None = None


def set_container(container: Container) -> None:
    global _container
    _container = container


def get_container() -> Container:
    if _container is None:
        raise RuntimeError("Container has not been initialized; call set_container at startup")
    return _container


require_admin = get_current_user

# Typed dependency aliases for FastAPI route signatures.
ContainerDep = Annotated[Container, Depends(get_container)]
AdminDep = Annotated[str, Depends(require_admin)]
