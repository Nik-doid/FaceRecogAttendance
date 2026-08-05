"""Shared response schemas."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    database: str
    camera: str
    index_size: int


class MessageResponse(BaseModel):
    message: str
    detail: str | None = None


class Page[T](BaseModel):
    items: list[T]
    total: int
