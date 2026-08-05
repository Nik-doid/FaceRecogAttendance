"""Runtime tuning settings backed by the ``settings`` DB table.

Env config is the default; the DB rows can override it without redeploy. The worker
reads the merged result each time it needs a value, so changing a row takes effect
on the next frame.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config.settings import Settings as AppSettings
from app.repositories.setting_repo import SettingRepository

DEFAULT_KEYS = (
    "recognition_threshold",
    "duplicate_timeout_seconds",
    "minimum_face_size",
    "frame_skip",
    "silentface_threshold",
)


@dataclass(frozen=True)
class RuntimeSettings:
    recognition_threshold: float
    duplicate_timeout_seconds: int
    minimum_face_size: int
    frame_skip: int
    silentface_threshold: float

    @staticmethod
    def _as_float(value: str, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _as_int(value: str, fallback: int) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return fallback


class SettingsService:
    def __init__(self, repo: SettingRepository) -> None:
        self._repo = repo

    def seed_defaults(self, session: Session, settings: AppSettings) -> None:
        """Insert DB rows from env defaults if not already present (startup)."""
        self._repo.seed(
            session,
            {
                "recognition_threshold": str(settings.recognition_threshold),
                "duplicate_timeout_seconds": str(settings.duplicate_timeout_seconds),
                "minimum_face_size": str(settings.minimum_face_size),
                "frame_skip": str(settings.frame_skip),
                "silentface_threshold": str(settings.silentface_threshold),
            },
        )

    def get_runtime(self, session: Session, env: AppSettings) -> RuntimeSettings:
        """Merge DB overrides over env defaults."""
        values = self._repo.get_many(session, list(DEFAULT_KEYS))
        rt = RuntimeSettings(
            recognition_threshold=RuntimeSettings._as_float(
                values.get("recognition_threshold", ""), env.recognition_threshold
            ),
            duplicate_timeout_seconds=RuntimeSettings._as_int(
                values.get("duplicate_timeout_seconds", ""), env.duplicate_timeout_seconds
            ),
            minimum_face_size=RuntimeSettings._as_int(
                values.get("minimum_face_size", ""), env.minimum_face_size
            ),
            frame_skip=RuntimeSettings._as_int(
                values.get("frame_skip", ""), env.frame_skip
            ),
            silentface_threshold=RuntimeSettings._as_float(
                values.get("silentface_threshold", ""), env.silentface_threshold
            ),
        )
        return rt
