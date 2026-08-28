# Face Recognition Attendance Service — Project Blueprint

## Overview

A production-ready face recognition service that runs on RTSP/IP cameras (or local webcams), detects and recognizes enrolled employees, and reports attendance check-in events to an external attendance system via message queue or direct database writes.

## Core Purpose

- **Input**: Live video stream from CCTV/IP cameras (RTSP) or USB webcams
- **Processing**: Detect faces → quality filter → liveness check (optional) → embed with ArcFace → match against enrolled employee gallery
- **Output**: Attendance events sent to external system (RabbitMQ + optional MySQL sync)

## Architecture

```
RTSP Camera ──▶ Frame Reader ──▶ Detection (SCRFD) ──▶ Quality Gates
                                                    │
                                                    ▼
                                            Liveness (SilentFace, optional)
                                                    │
                                                    ▼
                                            Embedding (ArcFace)
                                                    │
                                                    ▼
                                            FAISS Index Search
                                                    │
                                                    ▼
                                            Engagement Check (1s gaze + hand raise)
                                                    │
                                                    ▼
                                            Attendance Reporter ──▶ RabbitMQ / MySQL
```

## Key Components

### AI Pipeline
- **Detector**: SCRFD 10G (ONNX, InsightFace) — 640×640 input, up to 10 faces/frame
- **Recognizer**: ArcFace ResNet-50 w600k_r50 (512-d embeddings, ONNX)
- **Quality Gates**: Blur (Laplacian), lighting, head roll, min face size, single-face enforcement
- **Liveness**: SilentFace anti-spoofing (MiniFASNet ONNX) — optional, disabled by default
- **Tracking**: ByteTrack (optional, requires C++ toolchain) — fallback: spatial hashing

### Engagement Logic
Employee must:
1. **Look at camera** ≥ 1 second (face width ≥ 1% of frame width)
2. **Raise hand** near face (within 50% frame width horizontally)

Only then is attendance recorded. Prevents walk-by false positives.

### Multi-Embedding per Employee
Each employee folder (`uploads/employees/{CODE}/`) can contain multiple photos (`{CODE}_1.jpg`, `{CODE}_2.jpg`, etc.). Each produces a separate embedding vector tagged with the same `employee_code`. Matching searches all vectors and returns best score per employee.

### Index Management
- FAISS IndexFlatIP (cosine similarity via inner product on L2-normalized vectors)
- Atomic background rebuilds — swap index on completion, never serves half-built state
- Persisted to `face_embeddings` table + `storage/index.faiss`

### Attendance Reporting
- **Primary**: RabbitMQ topic exchange (`attendance.events` / `checkin` routing key)
- **Payload**: employee_code, camera_id, timestamp, confidence, snapshot_path, track_id
- **Deduplication**: Per-employee cooldown window (default 300s)

### External DB Sync (Optional)
- Writes directly to target MySQL table (`ct_hr_employee_attendance_log`)
- Maps camera → device_id/branch_id
- Resolves employee_code → attendance_id_no via lookup table
- Toggle in/out mode: first punch = check-in, second = check-out, further skipped
- Idempotent: stamps `erp_synced_at`, uses ON DUPLICATE KEY upsert
- At most one snapshot per employee per day retained

## Configuration (Environment Variables)

| Category | Key Variables |
|----------|---------------|
| **Camera** | `CAMERA_SOURCE` (rtsp/device), `RTSP_URL`, `CAMERA_DEVICE_INDEX`, `CAMERA_AUTOSTART`, `FRAME_SKIP`, `MAX_FRAME_WIDTH` |
| **Recognition** | `RECOGNITION_THRESHOLD` (0.45), `DETECT_THRESH` (0.40), `MINIMUM_FACE_SIZE` (15), `LIVENESS_ENABLED`, `SILENTFACE_THRESHOLD`, `TRACKING_ENABLED` |
| **Engagement** | `ENGAGEMENT_MIN_FACE_RATIO` (0.01), `ENGAGEMENT_REQUIRED_SECONDS` (1.0) |
| **Quality** | `QUALITY_MIN_BLUR` (15), `QUALITY_MIN_LIGHTING` (25), `QUALITY_MAX_ROLL_DEG` (40) |
| **Attendance** | `ATTENDANCE_BROKER` (rabbitmq/null), MQ connection settings |
| **ERP Sync** | `ERP_SYNC_ENABLED`, DB credentials, `ERP_CAMERA_MAPPING`, `ERP_IN_OUT_MODE` (toggle), interval/batch size |
| **Database** | `DATABASE_URL` (async), `DATABASE_SYNC_URL` (sync psycopg) |
| **Security** | `JWT_SECRET_KEY`, `JWT_EXPIRE_MINUTES`, rate limits |

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/health` | — | Service/DB/camera/index status |
| GET | `/api/v1/metrics` | — | Prometheus metrics |
| GET | `/api/v1/index/status` | — | Index size, employees, last build |
| POST | `/api/v1/index/rebuild` | JWT | Rebuild embeddings from photo tree |
| GET | `/api/v1/camera/status` | — | Camera worker status |
| POST | `/api/v1/camera/start` | JWT | Start recognition loop |
| POST | `/api/v1/camera/stop` | JWT | Stop recognition loop |
| GET | `/api/v1/camera/stream` | — | MJPEG live feed |
| GET | `/api/v1/recognition/logs` | — | Recent recognition events |
| GET | `/api/v1/unknown-faces` | — | Recent unknown-face events |
| GET | `/api/v1/sync/attendance-log/status` | — | ERP sync status |
| POST | `/api/v1/sync/attendance-log` | JWT | Trigger ERP sync pass |

## Debug Dashboard (Streamlit)

```bash
uv sync --extra frontend
uv run streamlit run frontend/app.py
# http://localhost:8501
```

Features: live camera view with bounding boxes, health/index/camera cards, enrollment UI, recognition/unknown logs with snapshots, metrics cards, raw API console.

## Deployment

### Local Dev
```bash
uv sync --extra ai
cp .env.example .env
# edit .env
uv run alembic upgrade head
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker compose up -d --build
# postgres + rabbitmq + service
```

## Database Schema (Managed by Alembic)

- `face_embeddings` — employee_code, embedding (JSON), quality_score, source_image_path
- `recognition_logs` — employee_code, camera_id, timestamp, confidence, snapshot_path, track_id, reported, response, erp_synced_at
- `unknown_faces` — camera_id, timestamp, snapshot_path, track_id
- `cameras` — camera_id, status, last_connected_at, last_error
- `settings` — runtime overrides (threshold, timeout, etc.)
- `audit_logs` — admin actions (index rebuild, camera start/stop, sync runs)

## Tuning for High-Angle CCTV

Current optimized defaults for ceiling-mounted cameras:
```
RECOGNITION_THRESHOLD=0.45      # lowered from 0.60
DETECT_THRESH=0.40              # lowered from 0.50
MINIMUM_FACE_SIZE=15            # lowered from 80
ENGAGEMENT_MIN_FACE_RATIO=0.01  # 1% of frame width
QUALITY_MAX_ROLL_DEG=40         # raised from 30
QUALITY_MIN_BLUR=15             # lowered from 20
QUALITY_MIN_LIGHTING=25         # lowered from 40
```

Add multiple enrollment photos per employee at different angles for best results.

## Monitoring

- **Prometheus**: `/api/v1/metrics` — counters for detections, quality rejections (by reason), recognitions, unknown faces, ERP sync rows
- **Structured JSON logs**: stdout with `event` field (`recognition`, `unknown_face`, `index_rebuilt`, `camera_connected`, etc.)
- **Health endpoint**: camera status (`running`/`stopped`/`error`/`connecting`)

## Security

- JWT HS256 for admin endpoints (camera control, index rebuild, ERP sync)
- Rate limiting on control endpoints (20 req/60s default)
- API token fallback for read endpoints