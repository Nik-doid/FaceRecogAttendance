# Face Recognition Attendance Service

Production-ready face recognition service for RTSP/IP cameras. Detects and
recognizes employees at a camera, reports check-in events to an **existing**
attendance system over a message queue, and exposes a small FastAPI control plane.

- Python 3.12, FastAPI, SQLAlchemy 2.0 (PostgreSQL), Alembic
- InsightFace SCRFD detector + ArcFace (w600k_r50) recognizer, ONNX Runtime
- FAISS cosine-similarity embedding index (atomic background rebuilds)
- Optional SilentFace liveness (anti-spoofing) and ByteTrack-style tracking
- RabbitMQ attendance reporting (contract below)
- Structured JSON logging, Prometheus metrics, JWT-protected admin endpoints

## Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │                 this service                        │
   RTSP/IP          │  CameraReader ── RecognitionPipeline ── Recognizer  │
   camera ─────────▶│       │              │                    (ArcFace) │
                    │       ▼              ▼                               │
                    │  (frame)      Detector (SCRFD) ── Quality gates     │
                    │                    │      └─ Liveness (SilentFace)  │
                    │                    ▼                                │
                    │              FAISS index ◀── IndexService (worker)  │
                    │                    │                ▲               │
                    │                    ▼                │ employee      │
                    │            RecognitionLoop          │ photos tree   │
                    │              │  │  │                │               │
                    └──────────────┼──┼──┼────────────────┴───────────────┘
                                   │  │  └─ recognition_logs / unknown_faces
                                   │  └──── snapshots (JPEG evidence)
                                   └─────── RabbitMQ  ──▶  existing attendance system
```

The **recognition worker** is a single daemon thread (not inside the API). The
FastAPI app only starts/stops it (`POST /camera/start|stop`). All heavy work
(detect → quality → liveness → embed → search → decide → report) is synchronous;
the async API dispatches its short DB calls with `asyncio.to_thread`.

## Quick start (local dev)

Requirements: Windows/macOS/Linux, `uv` installed. Python 3.12 is recommended
(the project requires `>=3.12`).

```bash
uv sync                 # install base deps (control plane + tests)
cp .env.example .env    # adjust values
uv run pytest           # 51 tests, no AI models or camera needed (fakes)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

To also install the heavy AI inference stack locally:

```bash
uv sync --extra ai
```

> **Windows note:** the `ai` extra requires `insightface` which needs the
> Microsoft C++ Build Tools to compile. The Linux Docker image installs it
> without issue. On Windows without the toolchain, run the control plane only
> (tests and API) or use Docker.

## Test with your webcam (quick local demo)

The workflow is: **enroll photos → boot → build index → start camera → watch events**.

1. **Install the AI deps** (one-time). On Windows you must first install
   [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   (free, needed to compile `insightface`), then:

   ```bash
   uv sync --extra ai
   ```

   `onnxruntime` + `insightface` install and the SCRFD/ArcFace models
   auto-download into `./models` on first run.

2. **Start a Postgres** for the service's own tables (it uses Postgres via
   asyncpg/psycopg; RabbitMQ is optional — use `ATTENDANCE_BROKER=null`):

   ```bash
   docker compose up -d postgres        # RabbitMQ optional if broker=null
   cp .env.example .env
   ```

   In `.env` set at minimum:

   ```dotenv
   CAMERA_SOURCE=device
   CAMERA_DEVICE_INDEX=0
   LIVENESS_ENABLED=false
   ATTENDANCE_BROKER=null
   SNAPSHOT_ENABLED=true
   ```

3. **Create the schema and enroll faces**:

   ```bash
   uv run alembic upgrade head
   mkdir -p uploads/employees/EMP1
   # put 1-3 clear, frontal photos of your face at uploads/employees/EMP1/EMP1.jpg
   ```

   Subdirectory name = `employee_code`; skip blurry/backlit photos (quality
   gates reject them and log the reason).

4. **Boot and rebuild the index**:

   ```bash
   uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
   # in another terminal:
   curl http://localhost:8000/api/v1/health
   curl http://localhost:8000/api/v1/index/status   # repeat until size > 0
   ```

   `POST /index/rebuild` rebuilds later when you add photos, but startup already
   builds from `uploads/employees`.

5. **Generate a control token** (admin endpoints require `Authorization: Bearer <jwt>`):

   ```bash
   uv run python -c "from app.core.security import create_access_token as t; print(t('admin'))"
   ```

   Use it to start the camera:

   ```bash
   curl -X POST http://localhost:8000/api/v1/camera/start -H "Authorization: Bearer <token>"
   ```

6. **Face the camera.** Watch it work:

   - `GET /api/v1/camera/status` → `"running"`
   - `GET /api/v1/recognition/logs` → matched events with confidence + `reported`
   - `GET /api/v1/unknown-faces` → unrecognized faces (each snapshotted)
   - `storage/snapshots/attendance_*.jpg` → evidence images
   - `GET /api/v1/metrics` → `face_recognitions_total`, `face_unknown_faces_total`, …
   - `POST /api/v1/camera/stop` to halt.

   Terminal logs are JSON with `event: recognition|unknown_face` markers.
   `GET /api/v1/recognition/logs?employee_code=EMP1` filters one person. Each
   employee is reported at most once per `DUPLICATE_TIMEOUT_SECONDS` window.

> If the webcam won't open, try `CAMERA_DEVICE_INDEX=1`, or swap
> `opencv-python-headless` for `opencv-python` (DirectShow capture) in
> `pyproject.toml`. For a headless box without a camera, use an RTSP/IP cam with
> `CAMERA_SOURCE=rtsp`.

## Debug console (Streamlit UI)

A browser UI that wraps the whole API for quick debugging — no curl or JWT
handling by hand.

```bash
uv sync --extra frontend
uv run streamlit run frontend/app.py
# open http://localhost:8501
```

The console automatically signs a JWT with the same `JWT_SECRET_KEY` from `.env`,
so every admin action works out of the box against `http://localhost:8000`.

What it can do (sidebar → page radio):

- **Dashboard** — live health/index/camera cards, a live camera view
  (`/api/v1/camera/stream`), a recent-activity panel (last recognized + unknown
  faces), start/stop the camera and trigger an index rebuild with one click
  (auto-refresh while building)
- **Enrollment** — pick an employee code + upload a face photo; it saves the file
  into the right `uploads/employees/<code>/` folder and lets you rebuild the index
  from the same page (previews of current enrollments included)
- **Recognition Logs / Unknown Faces** — filterable tables plus JPEG snapshot
  previews pulled straight from the server's `storage/snapshots/`
- **Metrics** — key Prometheus counters as cards + the raw exposition text
- **API Console** — arbitrary `GET/POST/...` requests against `/api/v1` with the
  generated bearer token, for poking at anything the UI doesn't cover

If the service runs on another host, change the **API base URL** in the sidebar
and paste a token issued by that deployment (the console can't sign tokens for a
different secret).

## Employee photos

The existing system's employee photos are read from a local folder tree. Each
subdirectory name is the `employee_code` and must contain at least one face photo:

```
uploads/employees/
├── EMP1001/
│   ├── EMP1001.jpg
│   └── back.jpg
└── EMP1002/
    └── front.png
```

On startup and on `POST /index/rebuild` the `IndexService` scans the source
root(s), detects one face per photo, runs quality gates (blur, lighting, size,
face count, readability), embeds the best face with ArcFace, persists the
embeddings to the `face_embeddings` table and swaps the in-memory FAISS index
atomically. Photos that fail quality gates are skipped and logged.

## Attendance integration (message queue)

The integration seam is `app/services/attendance_reporter/`. The default broker
is RabbitMQ; `ATTENDANCE_BROKER=null` disables publishing (used by tests).
To integrate with a REST/Kafka/etc. system, implement the `AttendanceReporter`
ABC in that package and register it in `factory.py` — nothing else changes.

### RabbitMQ contract

| Setting              | Default                |
|----------------------|------------------------|
| `ATTENDANCE_MQ_URL`  | `amqp://guest:guest@rabbitmq:5672/` |
| `ATTENDANCE_EXCHANGE`| `attendance.events` (topic) |
| `ATTENDANCE_ROUTING_KEY` | `checkin`          |
| `ATTENDANCE_QUEUE`   | `attendance.checkin` (bound to the key) |

Published event payload (one per recognized, non-duplicate face):

```json
{
  "employee_code": "EMP1001",
  "camera_id": "cam-01",
  "timestamp": "2026-01-01T12:00:00Z",
  "confidence": 0.912,
  "snapshot_path": "storage/snapshots/attendance_EMP1001_t3_20260101_120000_123456.jpg",
  "track_id": 3,
  "schema_version": 1
}
```

Duplicate suppression: the same employee is only reported once per
`DUPLICATE_TIMEOUT_SECONDS` window (default 300s). Suppressed and failed
deliveries are recorded in `recognition_logs` with the response detail.

## Attendance log sync to the existing ERP DB

The C# attendance software writes punches into `ct_hr_employee_attendance_log`
(`AttendanceLog.cs`). This service can write the same table directly, so the
face-recognition events become a drop-in source of attendance rows without any
change to the existing system.

The sync is **disabled by default**. Enable it and point it at the ERP MySQL:

| Setting                  | Default                | Meaning                                      |
|--------------------------|------------------------|----------------------------------------------|
| `ERP_SYNC_ENABLED`       | `false`                | Run the background sync + allow `/sync` API  |
| `ERP_DB_HOST` / `_PORT`  | `localhost` / `3306`   | ERP MySQL connection                         |
| `ERP_DB_NAME` / `_USER` / `_PASSWORD` | `attendance` / `root` | ERP credentials                    |
| `ERP_CAMERA_MAPPING`     | `{}`                   | JSON `camera_id -> {device_id, branch_id}`   |
| `ERP_VERIFY_MODE`        | `FACE`                 | `verify_mode` column value                   |
| `ERP_CREATED_BY`         | `system`               | `created_by` column value                    |
| `ERP_IN_OUT_MODE`        | `1`                    | `in_out_mode`: literal int, or `"toggle"`    |
| `ERP_SYNC_INTERVAL_SECONDS` | `300`               | Background pass interval                     |
| `ERP_SYNC_BATCH_SIZE`    | `500`                  | Max rows written per pass                    |

```bash
ERP_SYNC_ENABLED=true
ERP_DB_HOST=192.168.1.50
ERP_DB_NAME=attendance
ERP_DB_USER=root
ERP_DB_PASSWORD=secret
ERP_CAMERA_MAPPING='{"cam-01":{"device_id":1,"branch_id":1},"cam-02":{"device_id":2,"branch_id":1}}'
```

Behavior mirrors the C# INSERT (`attendance_id_no`, `in_out_mode`, `verify_mode`,
`log_date_time`, `device_id`, `branch_id`, `created_by`, `created_date`,
`log_date_only`):

- Only events with an `employee_code` that were successfully reported are
  candidates.
- Each camera maps to the physical attendance `device_id`/`branch_id` its
  punches belong to; events from unmapped cameras are skipped and logged.
- With `ERP_IN_OUT_MODE=toggle`, check-in (1) and check-out (2) alternate per
  employee per calendar day so both a first and a later appearance are written.
- Successful rows are stamped `erp_synced_at` in `recognition_logs`, making the
  sync idempotent (never duplicates a punch, and the ERP insert's
  `ON DUPLICATE KEY` upsert is mirrored).
- The scheduler runs on a daemon thread; runs can also be triggered on demand
  via `POST /api/v1/sync/attendance-log` (admin + rate-limited, audited).

## Configuration

Full reference in [`.env.example`](.env.example). Key groups:

- **Camera**: `CAMERA_SOURCE` (`rtsp` or `device` for a local webcam),
  `RTSP_URL`, `CAMERA_DEVICE_INDEX`, `CAMERA_AUTOSTART`, `FRAME_SKIP`,
  `MAX_FRAME_WIDTH`
- **Recognition**: `RECOGNITION_THRESHOLD`, `DUPLICATE_TIMEOUT_SECONDS`,
  `MINIMUM_FACE_SIZE`, `LIVENESS_ENABLED`, `SILENTFACE_THRESHOLD`,
  `TRACKING_ENABLED`
- **AI models**: `MODELS_DIR`, `DETECT_MODEL`, `RECOGNIZE_MODEL`,
  `SILENTFACE_MODEL_PATH`, `ONNX_PROVIDERS` (insightface auto-downloads into
  `MODELS_DIR` on first use)
- **Photos**: `EMPLOYEE_PHOTOS_SOURCE` (comma-separated roots)
- **Attendance**: `ATTENDANCE_BROKER`, `ATTENDANCE_MQ_URL`, `ATTENDANCE_EXCHANGE`,
  `ATTENDANCE_ROUTING_KEY`, `ATTENDANCE_QUEUE`
- **ERP sync**: `ERP_SYNC_ENABLED`, `ERP_DB_HOST/_PORT/_NAME/_USER/_PASSWORD`,
  `ERP_CAMERA_MAPPING`, `ERP_VERIFY_MODE`, `ERP_CREATED_BY`, `ERP_IN_OUT_MODE`,
  `ERP_SYNC_INTERVAL_SECONDS`, `ERP_SYNC_BATCH_SIZE`
- **DB**: `DATABASE_URL` (async) and `DATABASE_SYNC_URL` (sync psycopg, same DB —
  used by the background worker)
- **Storage**: `STORAGE_PATH`, `SNAPSHOTS_DIR`, `UNKNOWN_FACES_DIR`,
  `SNAPSHOT_ENABLED`
- **Security**: `JWT_SECRET_KEY` (>=32 chars), `JWT_EXPIRE_MINUTES`,
  `RATE_LIMIT_MAX_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`

## API

| Method | Path                           | Auth  | Description                                    |
|--------|--------------------------------|-------|------------------------------------------------|
| GET    | `/api/v1/health`               | –     | Service/db/camera/index liveness               |
| GET    | `/api/v1/metrics`              | –     | Prometheus metrics                             |
| GET    | `/api/v1/index/status`         | –     | Index size, employees, last build              |
| POST   | `/api/v1/index/rebuild`        | JWT   | Rebuild embeddings from photo tree (background)|
| GET    | `/api/v1/camera/status`        | –     | Camera worker status                           |
| POST   | `/api/v1/camera/start`         | JWT   | Start recognition loop (rate-limited)          |
| POST   | `/api/v1/camera/stop`          | JWT   | Stop recognition loop (rate-limited)           |
| GET    | `/api/v1/recognition/logs`     | –     | Recent recognition events (filter by employee) |
| GET    | `/api/v1/unknown-faces`        | –     | Recent unknown-face events                     |
| GET    | `/api/v1/sync/attendance-log/status` | – | ERP sync enabled / pending backlog        |
| POST   | `/api/v1/sync/attendance-log`  | JWT   | Run one ERP sync pass now (rate-limited, audited) |

Admin endpoints require `Authorization: Bearer <jwt>` where the JWT is issued by
the ops/attendance tooling with `sub = <operator>` and signed with
`JWT_SECRET_KEY`. Interactive docs at `/docs`.

## Development

```bash
uv run pytest              # tests (sqlite + fakes, no models/camera needed)
uv run ruff check app tests alembic
uv run mypy app
uv run alembic upgrade head   # create/upgrade the service's own schema
```

## Docker deployment

`docker-compose.yml` starts PostgreSQL 16, RabbitMQ 3.13, and the service. The
image installs the `ai` extra and runs Alembic migrations on entry.

```bash
docker compose up -d --build
```

Mount a volume with the employee photos tree at `./uploads/employees`
(`EMPLOYEE_PHOTOS_SOURCE`), set a strong `JWT_SECRET_KEY` and the camera URL in
`.env`, then `POST /api/v1/camera/start` with a valid token to begin recognition.

## Own database

The service manages its own schema (migrations `alembic/versions/0001_initial.py`
and `0002_erp_sync.py`):
`face_embeddings`, `recognition_logs`, `unknown_faces`, `cameras`, `settings`
(runtime overrides for threshold/timeout/etc., editable without redeploy), and
`audit_logs`. The existing attendance system's tables are never touched — the
ERP sync only ever writes `ct_hr_employee_attendance_log`.
