# Face Recognition Attendance Service — Project Blueprint

## What it does

Watches one RTSP camera. When somebody looks at it and raises a hand, it recognises
them and records a punch in the existing attendance system.

```
RTSP frame
  ├─ 1. face detection        SCRFD, always runs — it decides step 2
  ├─ 2. looking gate          landmark arithmetic, no model
  ├─ 3. palm detection        BlazePalm, per face, in that face's own search box
  ├─ 4. face recognition      ArcFace, only for the people who raised a hand
  └─ 5. publish               RabbitMQ
                                 └─ consumer ─▶ ct_hr_employee_attendance_log
```

There is no local database, no enrollment UI, and no attendance log kept here. The
only durable state this service owns is a cache of enrolment embeddings, which exists
purely so a restart is not a six-minute outage.

## The two halves

**The camera runner** (`app/camera/runner.py`) owns the reader and the pipeline for
the life of the process, on a daemon thread with its own event loop. It is *not* tied
to a WebSocket connection: attendance keeps being recorded with nobody watching. It
publishes frames and detection results into a `FrameHub`, and `app/api/v1/endpoints/
websocket.py` is a viewer that forwards whatever is in there.

**The consumer** (`app/services/attendance_consumer/`) drains the queue and writes
MySQL. It is split so the rules can be tested without a broker or a database:
`policy.py` decides (pure functions), `mysql.py` writes, `writer.py` joins them,
`consumer.py` only routes acks.

## Gates, in order

Each one exists to stop work that would otherwise be wasted or wrong.

| Gate | Effect |
|---|---|
| **Looking** (`gaze.py`) | Nothing past face detection runs unless a face is turned toward the camera. Yaw is the nose's offset along the eye axis over the eye span: frontal faces measure ≤0.03, a turned head 0.76, and the threshold is 0.35. Pitch is deliberately *not* gated — a high-mounted camera sees every face pitched. |
| **Palm, per face** | Each looking face gets its own search box (`PALM_SEARCH_MARGIN` face-widths). Only people whose own box contains a hand are recognised, so a bystander standing beside someone who waves is detected and reported but never identified. |
| **Publish rate** | One hand-raise spans several scans; `DuplicateSuppressor` publishes once per employee per `DUPLICATE_TIMEOUT_SECONDS`. |
| **Punch gap** | A punch within `ATTENDANCE_MIN_PUNCH_GAP_SECONDS` of an existing row is skipped. Makes an at-least-once redelivery a no-op with no dedup store, and stops lingering in frame from creating several punches. |

## The attendance row rule

Two rows per employee per day.

- 0 rows → **INSERT** (check-in)
- 1 row → **INSERT** (check-out)
- 2+ rows → **UPDATE the second row**, so it holds the latest punch

The check-in row is written once and never touched again, so no sequence of punches
can destroy the arrival time. `in_out_mode` is the constant **255** on every row; in
and out are told apart by row order, not by that value.

Day state is re-read from the table for every message, which is what makes a
redelivered message take the same branch as the original.

Columns written: `attendance_id_no, in_out_mode, verify_mode, log_date_time,
device_id, branch_id, created_by, created_date, log_date_only`. `employee_code` is
resolved to `attendance_id_no` through `ct_hr_employee_master` (table and columns are
configurable).

`log_date_time` and `log_date_only` are both derived from the local value.
`ATTENDANCE_TIMEZONE` defaults to **`Asia/Kathmandu`** (UTC+05:45). Setting it to UTC
would store a 09:15 arrival as 03:30 and put a 22:00 punch on the previous day.

## Employee photos

`EMPLOYEE_PHOTOS_SOURCE` is comma-separated and each entry is either a local directory
or an `https://` URL. Both use one subdirectory per `employee_code`. An employee
present in two sources gains reference vectors rather than conflicting.

A remote folder is read from `{BASE}/manifest.json` when one exists:

```json
{"employees": {"EMP1023": [{"file": "front.jpg", "etag": "W/\"a1b2\""}]}}
```

The plain `{"EMP1023": ["front.jpg"]}` form works too. Without a manifest the server's
directory listing is scraped, which works on nginx/Apache autoindex but not on object
stores (S3/MinIO answer with XML). Publish a manifest if you can — the ETags also let
a restart skip re-downloading.

Embeddings are cached under `STORAGE_PATH/gallery`, keyed by photo identity and
version. Measured: five photos over HTTP took 12.5 s cold, 0.2 s warm.

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/` | — | The debug page (WebSocket client, camera start/stop) |
| GET | `/api/v1/health` | — | Service, camera and gallery state |
| GET | `/api/v1/metrics` | — | Prometheus |
| GET | `/api/v1/camera/status` | — | Runner state: frames, scans, reconnects, last error |
| GET | `/api/v1/camera/frame` | — | Latest JPEG |
| GET | `/api/v1/camera/stream` | — | MJPEG |
| POST | `/api/v1/camera/start` | JWT | Start the runner |
| POST | `/api/v1/camera/stop` | JWT | Stop the runner |
| GET | `/api/v1/gallery/status` | — | Enrolled employees, photos, vectors, sources |
| POST | `/api/v1/gallery/reload` | JWT | Re-enumerate photos and swap the index in |
| WS | `/api/v1/camera/ws` | — | Viewer: frames + detections |
| WS | `/api/v1/webcam/ws` | — | Debug: browser webcam, **never records attendance** |

Mint a token with:

```bash
uv run python -c "from app.core.security import create_access_token as t; print(t('admin'))"
```

## Running it

```bash
uv sync --extra ai
cp .env.example .env          # then edit
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On an RTSP camera, export these before launching or OpenCV logs a line per dropped
packet — the cv2 DLL reads them at process start, so setting them from Python is too
late:

```powershell
$env:OPENCV_LOG_LEVEL="ERROR"; $env:OPENCV_FFMPEG_LOGLEVEL="8"
```

`docker compose up -d --build` runs rabbitmq + app and sets both already.

To run the consumer as its own process: `ATTENDANCE_CONSUMER_INPROC=false` and
`python -m app.services.attendance_consumer`.

## Measured performance

Intel i3-7020U, 2 cores / 4 threads, **CPU only** — CUDA is requested by default but
absent, and `filter_providers` silently drops it. The startup log reports the
providers onnxruntime actually resolved.

| | |
|---|---|
| SCRFD | ~363 ms, constant (input is always letterboxed to 640×640, so frame width is irrelevant) |
| ArcFace | ~350 ms per face |
| FAISS search | 0.06 ms |
| Pipeline, no face | ~407 ms |
| Pipeline, not looking | ~498 ms |
| Pipeline, recognised | ~1168 ms |

So roughly one scan per second. An employee has to hold a palm up for a second or two.

Accuracy on the five enrolled identities: impostor pairs max +0.150, genuine
different-capture probes 0.579–0.763, threshold 0.45. Clean separation, but far too
small a sample to quote a false-accept rate.

## Known limits

- **Palm attribution degrades when people stand closer together than
  `PALM_SEARCH_MARGIN`.** BlazePalm is decoded for its score only, so there is no hand
  *location* to attribute; fixing it properly means decoding the box head.
- **BlazePalm's score is noisy under reframing** — the same hand measured 0.12 to 0.77
  as the search box changed. Tune `PALM_SEARCH_MARGIN` with `PALM_SCORE_THRESHOLD` on
  real footage.
- **Neither WebSocket is authenticated.** `/camera/ws` serves an always-running feed to
  anyone who can reach the port. Put it behind the JWT or a network ACL.
- **The consumer is correct for exactly one instance at `prefetch=1`.** Two would need
  `SELECT ... FOR UPDATE` around the read-then-write.
- **A dead-lettered message and a failed enrolment exist only in the logs and the
  `attendance.dead` queue.** With no local database there is no other trace.
