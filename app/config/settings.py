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
    # Silence the onnxruntime and core-OpenCV native loggers, which write to stderr
    # and ignore LOG_LEVEL entirely. Turn off to diagnose ONNX problems. OpenCV's
    # RTSP/FFmpeg chatter is NOT covered -- that needs OPENCV_LOG_LEVEL exported
    # before the process starts; see app/core/logging.py.
    quiet_native_logs: bool = True

    # --- Camera --------------------------------------------------------------
    camera_id: str = "cam-01"
    rtsp_url: str = ""
    camera_source: Literal["rtsp", "device"] = "rtsp"
    camera_device_index: int = 0
    camera_autostart: bool = False
    max_frame_width: int = 1280
    # Run the webcam pipeline on every Nth frame of the /camera/ws stream. Detection
    # runs concurrently with streaming, so this caps CPU, not the frame rate. At ~30fps
    # 10 is a scan every ~330ms, which leaves a real gap either side of a
    # PALM_SCAN_GRID=2 scan instead of pegging a core continuously.
    # The camera runner is time-driven, not frame-counted: counting frames at 30fps
    # implies "scan every 330ms" while a frame reaching recognition costs over a
    # second. Start a scan at most this often, and never while one is in flight.
    camera_scan_interval_ms: int = 1000

    # --- Palm detection (the /webcam/ws and /camera/ws pipeline) --------------
    palm_score_threshold: float = 0.50
    # NxN overlapping crops scanned per frame on the *camera* path, where the subject
    # is far enough that a palm does not survive BlazePalm's 192x192 input. The browser
    # webcam keeps the whole-frame scan: a palm held up to a laptop is already large.
    # Costs up to N*N+1 forward passes, and one cv2.dnn pass measured ~72ms on a 4-core
    # CPU: grid 2 is ~280ms/scan, grid 3 ~720ms. Raise CAMERA_DETECT_EVERY alongside it.
    palm_scan_grid: int = 2
    palm_scan_overlap: float = 0.20

    # --- Looking gate --------------------------------------------------------
    # Nothing past face detection runs unless a face is turned toward the camera.
    # Yaw is the nose's offset along the eye axis over the eye span: measured 0.03 or
    # less on frontal enrolment photos and 0.76 on a turned head, so 0.35 sits well
    # clear of both. Raise it to accept more angled faces. Pitch is not gated -- a
    # high-mounted camera sees every face pitched. See app/core/face_processing/gaze.py.
    looking_max_yaw_ratio: float = 0.35
    looking_max_roll_degrees: float = 25.0
    # How far around a looking face to hunt for a raised hand, in multiples of that
    # face's width (sideways) and height (below). Widen it if people wave far out to
    # the side; narrowing it makes the palm gate stricter.
    palm_search_margin: float = 0.6

    # --- Recognition ---------------------------------------------------------
    recognition_threshold: float = 0.60
    detect_thresh: float = 0.40
    duplicate_timeout_seconds: int = 300
    # Require employee to be "engaged" (looking at camera + waving) before recording attendance.
    # When False, any quality-passed face triggers attendance (legacy behavior).

    # --- Engagement (CCTV-optimized) -----------------------------------------
    # Minimum face width as a fraction of frame width.
    # High-mounted CCTV captures very small faces (1.5-5% of frame width). Default 1.5%.
    # Required continuous engagement time in seconds (looking + hand visible).
    # Wall-clock time, independent of frame_skip. Default 1.0 second.

    # --- Face quality gates --------------------------------------------------

    # --- AI models -----------------------------------------------------------
    models_dir: Path = APP_ROOT / "models"
    detect_model: str = "det_10g.onnx"
    recognize_model: str = "w600k_r50.onnx"
    # Fetch missing SCRFD/ArcFace/hand-landmarker files into models_dir on first
    # boot. Turn off for air-gapped hosts or images that bake the models in.
    models_auto_download: bool = True
    onnx_providers: Annotated[list[str], NoDecode] = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    # --- Employee photo source -----------------------------------------------
    # Comma-separated, and each entry is either a local directory or an https:// URL
    # of a folder laid out the same way (one subdirectory per employee_code). Held as
    # strings, not Paths: Path("https://host/x") collapses the double slash and would
    # silently mangle every URL.
    employee_photos_source: Annotated[list[str], NoDecode] = [
        str(APP_ROOT / "uploads" / "employees")
    ]
    # Sent with every remote photo request, as "Header-Name: value".
    employee_photos_auth_header: str = ""
    employee_photos_timeout_seconds: float = 15.0
    employee_photos_manifest: str = "manifest.json"
    # Re-enumerate and re-embed on this interval so a new hire appears without a
    # restart. 0 disables it.
    gallery_refresh_seconds: int = 3600

    # --- Attendance reporting (message queue to the existing system) ----------
    attendance_broker: Literal["rabbitmq", "null"] = "rabbitmq"
    attendance_mq_url: str = "amqp://guest:guest@localhost:5672/"
    attendance_exchange: str = "attendance.events"
    attendance_routing_key: str = "checkin"
    attendance_queue: str = "attendance.checkin"
    attendance_publish_retries: int = 3
    # Where the consumer sends messages it can never process: a malformed payload, an
    # unknown employee, an unmapped camera. Requeueing those would spin a core at
    # 100% forever, so they are parked here for an operator instead.
    attendance_dead_letter_exchange: str = "attendance.dlx"
    attendance_dead_letter_queue: str = "attendance.dead"
    # The consumer writes into the attendance table, so it stays off until switched
    # on deliberately -- nothing should start writing to an ERP database by default.
    attendance_consumer_enabled: bool = False
    # In-process daemon thread by default: one deployment unit, and the consumer is
    # I/O-bound on MySQL while ONNX releases the GIL. Set false and run
    # `python -m app.services.attendance_consumer` to give it its own process.
    attendance_consumer_inproc: bool = True
    # One at a time keeps queue order, and the first message of a day has to be the
    # check-in. Raising this breaks that guarantee.
    attendance_consumer_prefetch: int = 1
    # The zone the attendance table is read in. Events travel as UTC; log_date_time
    # and log_date_only are both derived from the local value. Setting this to UTC
    # would store a 09:15 arrival as 03:30 and put a 22:00 punch on the previous day.
    attendance_timezone: str = "Asia/Kathmandu"
    # A punch within this many seconds of an existing row for the same employee and
    # day is ignored. Does double duty: it makes an at-least-once redelivery a no-op
    # without any dedup store, and stops someone lingering in front of the camera
    # from turning one arrival into several punches.
    attendance_min_punch_gap_seconds: int = 60

# --- Attendance log sync to an external ERP/DB (cron) -----------------------
    # Writes attendance punches to an external MySQL table (ct_hr_employee_attendance_log).
    # Disabled unless erp_sync_enabled is true; runs on a background interval
    # scheduler and/or on demand via the /sync API.
    erp_db_host: str = "localhost"
    erp_db_port: int = 3306
    erp_db_name: str = "attendance"
    erp_db_user: str = "root"
    erp_db_password: str = ""
    # JSON map of camera_id -> {"device_id": <int>, "branch_id": <int>}.
    erp_camera_mapping: dict[str, dict[str, int]] = {}
    # verify_mode string column value.
    erp_verify_mode: str = "FACE"
    erp_created_by: str = "system"
    # in_out_mode: a literal int (e.g. "1" = always check-in), or "toggle" for the
    # proper per-employee-per-day rule: first punch = check-in(1), second = check-out(2),
    # any further punches that day are skipped (no duplicate check-ins/outs). The rule
    # is seeded from rows already present in the target attendance table for that day.
    # Employee code -> ERP attendance id lookup.
    # The external system's employee table (emp_code -> emp_id) provides the
    # attendance_id_no written to the log; table + columns are configurable here.
    erp_employee_table: str = "ct_hr_employee_master"
    erp_employee_code_column: str = "emp_code"
    erp_employee_id_column: str = "emp_id"
    # Optional extra WHERE clause applied to the employee lookup
    # (e.g. `_status='employed'`).
    erp_employee_active_filter: str = "_status"

    # --- This service's own database -----------------------------------------

    # --- Storage ---------------------------------------------------------------
    storage_path: Path = APP_ROOT / "storage"

    # --- Security ---------------------------------------------------------------
    jwt_secret_key: str = "insecure-dev-secret-change-me-please-use-a-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    rate_limit_max_requests: int = 20
    rate_limit_window_seconds: int = 60

    # --- API --------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Derived paths ------------------------------------------------------------

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
    def _parse_photo_sources(cls, v: object) -> object:
        """Split on commas, keeping raw strings so URLs survive intact."""
        if isinstance(v, str):
            return [part.strip() for part in v.split(",") if part.strip()]
        if isinstance(v, list):
            return [str(part).strip() for part in v if str(part).strip()]
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
