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
    detect_thresh: float = 0.40
    duplicate_timeout_seconds: int = 300
    minimum_face_size: int = 80
    liveness_enabled: bool = True
    silentface_threshold: float = 0.50
    tracking_enabled: bool = False
    # Require employee to be "engaged" (looking at camera + waving) before recording attendance.
    # When False, any quality-passed face triggers attendance (legacy behavior).
    require_engagement: bool = True

    # --- Engagement (CCTV-optimized) -----------------------------------------
    # Minimum face width as a fraction of frame width.
    # High-mounted CCTV captures very small faces (1.5-5% of frame width). Default 1.5%.
    engagement_min_face_ratio: float = 0.015
    # Required continuous engagement time in seconds (looking + hand visible).
    # Wall-clock time, independent of frame_skip. Default 1.0 second.
    engagement_required_seconds: float = 1.0

    # --- Face quality gates --------------------------------------------------
    quality_min_blur: float = 100.0
    quality_min_lighting: float = 40.0
    quality_max_lighting: float = 220.0
    quality_max_roll_deg: float = 30.0

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

    # --- Attendance log sync to the existing ERP DB (cron) -----------------------
    # Replicates the INSERT the C# attendance software performs into
    # ct_hr_employee_attendance_log. Disabled unless erp_sync_enabled is true; the
    # sync runs on a background interval scheduler and/or on demand via the /sync API.
    erp_sync_enabled: bool = False
    erp_db_host: str = "localhost"
    erp_db_port: int = 3306
    erp_db_name: str = "attendance"
    erp_db_user: str = "root"
    erp_db_password: str = ""
    # JSON map of camera_id -> {"device_id": <int>, "branch_id": <int>}.
    erp_camera_mapping: dict[str, dict[str, int]] = {}
    # verify_mode string column value (C# maps 0/1->FINGERPRINT, 2->PIN, 3->PASSWORD).
    erp_verify_mode: str = "FACE"
    erp_created_by: str = "system"
    # in_out_mode: a literal int (e.g. "1" = always check-in), or "toggle" for the
    # proper per-employee-per-day rule: first punch = check-in(1), second = check-out(2),
    # any further punches that day are skipped (no duplicate check-ins/outs). The rule
    # is seeded from rows already present in ct_hr_employee_attendance_log for that day.
    erp_in_out_mode: str = "toggle"
    erp_sync_interval_seconds: int = 300
    erp_sync_batch_size: int = 500
    # Employee code -> ERP attendance id lookup. The C# report selects
    # ct_hr_employee_master.emp_id / attendance_thumb_id_no and uses the id as the
    # attendance_id_no written to the log; table + columns are configurable here.
    erp_employee_table: str = "ct_hr_employee_master"
    erp_employee_code_column: str = "emp_code"
    erp_employee_id_column: str = "emp_id"
    # Optional extra WHERE clause applied to the employee lookup (the C# software
    # filters on `_status`, e.g. `_status='employed'`).
    erp_employee_active_filter: str = "_status"

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

    @field_validator("erp_camera_mapping", mode="before")
    @classmethod
    def _parse_erp_camera_mapping(cls, v: object) -> object:
        if isinstance(v, str):
            import json

            parsed = json.loads(v)
            if not isinstance(parsed, dict):
                raise ValueError("erp_camera_mapping must be a JSON object")
            return {
                camera_id: {str(k): int(val) for k, val in val_map.items()}
                for camera_id, val_map in parsed.items()
            }
        return v


settings = Settings()
