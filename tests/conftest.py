"""Shared fixtures: sqlite-backed DB, test settings, DI container, TestClient.

The service's own database is swapped to SQLite by monkeypatching the engine
globals in ``app.database.session``; both the async API side and the sync worker
side then talk to the same SQLite file.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

import app.database.session as session_module
from app.api.deps import set_container
from app.api.router import api_router
from app.config.settings import Settings
from app.container import Container
from app.database.base import Base
from app.runtime import Models
from tests.fakes import build_ai

E1 = np.full(8, 0.5, dtype="float32")
E2 = np.full(8, -0.5, dtype="float32")


@pytest.fixture(scope="session")
def models() -> Models:
    """One SCRFD + one ArcFace for the whole test run; loading them costs seconds."""
    from app.config.settings import settings as env_settings  # noqa: PLC0415
    from app.runtime import load_models  # noqa: PLC0415

    return load_models(env_settings)


@pytest.fixture(autouse=True)
def reset_rate_limiter(monkeypatch):
    """Give each test a fresh, permissive rate limiter (avoids cross-test bleed)."""
    import app.core.security as security_module
    from app.core.security import RateLimiter

    monkeypatch.setattr(security_module, "_rate_limiter", RateLimiter(100, 60))


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(
        app_env="development",
        log_level="DEBUG",
        camera_id="cam-test",
        rtsp_url="",
        camera_autostart=False,
        frame_skip=1,
        minimum_face_size=10,
        recognition_threshold=0.8,
        duplicate_timeout_seconds=60,
        liveness_enabled=False,
        tracking_enabled=False,
        require_engagement=False,
        attendance_broker="null",
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
        database_sync_url=f"sqlite:///{tmp_path}/test.db",
        storage_path=tmp_path / "storage",
        unknown_faces_dir=tmp_path / "storage" / "unknown",
        snapshots_dir=tmp_path / "storage" / "snapshots",
        snapshot_enabled=True,
        employee_photos_source=[tmp_path / "photos"],
        jwt_secret_key="test-secret-that-is-long-enough-1234567890",
    )


@pytest.fixture
def db(test_settings: Settings, monkeypatch) -> Iterator[object]:
    sync_engine = create_engine(test_settings.database_sync_url)
    async_engine = create_async_engine(test_settings.database_url)
    sync_factory = sessionmaker(bind=sync_engine, expire_on_commit=False)
    async_factory = async_sessionmaker(bind=async_engine, expire_on_commit=False)

    monkeypatch.setattr(session_module, "_sync_engine", sync_engine)
    monkeypatch.setattr(session_module, "_sync_session_factory", sync_factory)
    monkeypatch.setattr(session_module, "_async_engine", async_engine)
    monkeypatch.setattr(session_module, "_async_session_factory", async_factory)

    Base.metadata.create_all(sync_engine)
    yield sync_engine
    sync_engine.dispose()
    async_engine.sync_engine.dispose()


@pytest.fixture
def container(db, test_settings: Settings) -> Container:
    return Container(settings=test_settings, ai=build_ai(E1, E2))


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    set_container(container)
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    with TestClient(app) as c:
        yield c
    set_container(None)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    from app.core.security import create_access_token

    token = create_access_token("ops-tester")
    return {"Authorization": f"Bearer {token}"}


class FakeLoop:
    def __init__(self) -> None:
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running


def use_fake_camera(container: Container) -> None:
    """Swap the container's camera service for one backed by a scriptable fake loop."""
    from app.services.camera_service import CameraService

    container.camera_service = CameraService(lambda: FakeLoop())


def wait_until(predicate, timeout: float = 10.0, interval: float = 0.05) -> bool:
    """Poll ``predicate`` until True or timeout. Returns whether it became True."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False
