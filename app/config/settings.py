"""Application configuration.

All configuration is environment-driven (see `.env.example`). Pydantic v2 validates
and coerces values at import time so a misconfigured deployment fails fast on boot
rather than at some random moment mid-pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

APP_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed access to every environment variable the service needs.

    Env vars are read with no prefix (``RECOGNITION_THRESHOLD=...`` maps directly to
    the ``recognition_threshold`` field). Values can also come from a local ``.env``
    file in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=APP_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Service -------------------------------------------------------------
    app_name: str = "face-recognition-service"
    app_env: Literal["development", "staging", "production"] = "production"
    log_level: str = "INFO"

    # --- Camera --------------------------------------------------------------
    camera_id: str = "cam-01"
    rtsp_url: str = ""
    camera_source: Literal["rtsp", "device"] = "rtsp"
    camera_device_index: int = 0
    camera_autostart: bool = False
    frame_skip: int = 2
    max_frame_width: int = 1280

    # --- Recognition ---------------------------------------------------------
    recognition_threshold: float = 0.60
    duplicate_timeout_seconds: int = 300
    minimum_face_size: int = 80
    liveness_enabled: bool = True
    silentface_threshold: float = 0.50
    tracking_enabled: bool = False

    # --- AI models -----------------------------------------------------------
    models_dir: Path = APP_ROOT / "models"
    detect_model: str = "det_10g.onnx"
    recognize_model: str = "w600k_r50.onnx"
    silentface_model_path: Path | None = None
    onnx_providers: Annotated[list[str], NoDecode] = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    # --- Employee photo source (the existing system) --------------------------
    employee_photos_source: Annotated[list[Path], NoDecode] = [
        APP_ROOT / "uploads" / "employees"
    ]

    # --- Attendance reporting (message queue to the existing system) ----------
    attendance_broker: Literal["rabbitmq", "null"] = "rabbitmq"
    attendance_mq_url: str = "amqp://guest:guest@localhost:5672/"
    attendance_exchange: str = "attendance.events"
    attendance_routing_key: str = "checkin"
    attendance_queue: str = "attendance.checkin"
    attendance_publish_retries: int = 3

    # --- This service's own database -----------------------------------------
    database_url: str = "postgresql+asyncpg://face:face@localhost:5432/face_recognition"
    database_sync_url: str = "postgresql+psycopg://face:face@localhost:5432/face_recognition"

    # --- Storage ---------------------------------------------------------------
    storage_path: Path = APP_ROOT / "storage"
    unknown_faces_dir: Path = APP_ROOT / "storage" / "unknown_faces"
    snapshots_dir: Path = APP_ROOT / "storage" / "snapshots"
    snapshot_enabled: bool = True

    # --- Security ---------------------------------------------------------------
    jwt_secret_key: str = "insecure-dev-secret-change-me-please-use-a-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_token: str = ""
    rate_limit_max_requests: int = 20
    rate_limit_window_seconds: int = 60

    # --- API --------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Derived paths ------------------------------------------------------------
    @property
    def face_index_dump_path(self) -> Path:
        return self.storage_path / "face_index"

    # --- Validation -------------------------------------------------------------
    @field_validator("onnx_providers", mode="before")
    @classmethod
    def _parse_list(cls, v: object) -> object:
        """Accept both JSON arrays and comma-separated strings for list fields."""
        if isinstance(v, str):
            return [part.strip() for part in v.split(",") if part.strip()]
        return v

    @field_validator("employee_photos_source", mode="before")
    @classmethod
    def _parse_path_list(cls, v: object) -> object:
        if isinstance(v, str):
            return [Path(part.strip()) for part in v.split(",") if part.strip()]
        return v


settings = Settings()
