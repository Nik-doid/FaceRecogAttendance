# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read this first: the repo does not currently run

`app/models/` and `app/storage/` are **absent from the working tree and were never
committed** — the `.gitignore` patterns `models/` (line 22, meant for the downloaded
ONNX zoo) and `storage/` (line 18, meant for runtime snapshot output) are unanchored,
so git silently excluded the source packages of the same name. Verify with
`git check-ignore -v app/models/audit_log.py app/storage/snapshot.py`.

Consequence: `import app` fails, so `pytest`, `alembic upgrade head`, and `uvicorn`
all fail immediately. `ruff` and `mypy` still pass (mypy has
`ignore_missing_imports = true`), so green lint/type checks prove nothing here.

What is missing, reconstructible from its call sites:

| Module | Exports | Shape recoverable from |
|---|---|---|
| `app/models/__init__.py` | must import every model module — `alembic/env.py` does `from app import models` to populate `Base.metadata` | `alembic/env.py:14` |
| `app/models/{face_embedding,recognition_log,unknown_face,camera,setting,audit_log}.py` | `FaceEmbedding`, `RecognitionLog`, `UnknownFace`, `Camera`, `Setting`, `AuditLog` on `app.database.base.Base` | `alembic/versions/0001_initial.py` (+ `erp_synced_at` from `0002`, `erp_skip_reason` from `0003`), and the matching `app/repositories/*.py` |
| `app/storage/snapshot.py` | `SnapshotStorage(dir, enabled=...)` with `.enabled`, `.save(frame, prefix=, employee_code=, track_id=) -> Path \| None`, `.remove(path)` | `app/workers/recognition_loop/loop.py`, `app/services/erp_sync/service.py`, `tests/test_snapshot.py` |

Fix the ignore rules before recreating the files (anchor them: `/models/`, `/storage/`),
otherwise the new sources will be invisible to git again.

## Commands

```bash
uv sync                      # control plane + tests (no ONNX/camera needed)
uv sync --extra ai           # + onnxruntime/insightface (needs a C toolchain on Windows)
uv sync --extra frontend     # + streamlit for the debug console

uv run pytest                # 87 tests (README's "51" is stale)
uv run pytest tests/test_erp_sync.py::test_name   # single test
uv run ruff check app tests alembic --fix         # 3 pre-existing I001 errors in tests/
uv run mypy app                                   # strict; currently clean
uv run alembic upgrade head
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
uv run streamlit run frontend/app.py              # debug UI on :8501
docker compose up -d --build                      # postgres + rabbitmq + app
```

Admin endpoints need a bearer token:
`uv run python -c "from app.core.security import create_access_token as t; print(t('admin'))"`.
The service only *verifies* JWTs — it issues none, and has no user CRUD.

## Architecture

Two planes over one Postgres database, and the split is load-bearing:

- **Control plane** — async FastAPI (`app/api/`), asyncpg engine. Repositories are
  written once against the *sync* `Session`, so endpoints call them through
  `await run_db(repo.method, session, ...)` (`app/api/deps.py`) to keep the event loop
  free. Never add an async repository; extend the sync one and dispatch it.
- **Recognition worker** — a single daemon thread (`RecognitionLoop`), psycopg engine.
  OpenCV/ONNX are synchronous, so everything from frame read to MQ publish runs there.
  The API only starts/stops the thread; it never processes frames.

`app/container.py` is the composition root. It constructs every singleton once
(FAISS index, reporters, repos, snapshot storage, frame buffer, ERP scheduler) and is
handed to the API via the module-global in `app/api/deps.py:set_container`. Tests build
a `Container(settings=..., ai=build_ai(...))` with fakes instead of patching internals.

Per-frame flow: `CameraReader` → frame-skip → optional `ByteTrackTracker` →
`RecognitionPipeline.process_frame` (quality gate → liveness → ArcFace embed →
FAISS search) → `RecognitionLoop._handle_events` (engagement gate → duplicate
suppression → `AttendanceReporter.report` → `recognition_logs` row) → `annotate_frame`
into `FrameBuffer` for `/camera/stream`. The pipeline is pure per-frame computation;
all persistence and reporting live in the loop, which is why the pipeline is trivially
testable.

### Seams to extend rather than edit around

- **`app/services/attendance_reporter/`** — the only module that knows how events leave
  the service. Add a class implementing the `AttendanceReporter` ABC and a branch in
  `factory.py`; nothing upstream changes. `ATTENDANCE_BROKER=null` is the test path.
- **`app/ai/components.py`** — `AIComponents` is the frozen bundle of detector,
  recognizer, liveness, quality, hand detector, wave tracker. Swap members here (or
  inject a fake dataclass) rather than reaching into the pipeline.
- **`app/ai/_loader.py`** — `import_optional` / `try_import` defer the heavy `ai`-extra
  imports so the control plane installs and tests without them. Anything importing
  `insightface`, `onnxruntime`, `faiss`, `pika`, or `pymysql` at module scope breaks
  that guarantee; import them lazily inside `__init__`.

### Index lifecycle

Embeddings live in two places: the `face_embeddings` table (durable) and the in-memory
FAISS index (searched). `IndexService` walks `EMPLOYEE_PHOTOS_SOURCE` (each subdirectory
name *is* the `employee_code`), quality-gates each photo, embeds it, `replace_all`s the
table, then swaps the FAISS index atomically — live recognition never sees a half-built
index. Rebuilds run on a background thread at startup and on `POST /index/rebuild`.
`FaceIndex.search` over-fetches `k*10` candidates and keeps the best score per employee,
since one employee has several reference embeddings.

### Configuration precedence

`app/config/settings.py` is a pydantic-settings model read from env/`.env` with no
prefix, validated at import time. Five keys (`recognition_threshold`,
`duplicate_timeout_seconds`, `minimum_face_size`, `frame_skip`, `silentface_threshold`)
are additionally overridable at runtime from the `settings` DB table — the loop re-reads
them every `TUNING_REFRESH_SECONDS` (30s), so tuning those does not need a redeploy.
Everything else requires a restart. Sequence matters in `app/main.py`: logging → container
(models load eagerly, fail fast) → seed DB rows → background index build → optional camera
autostart → ERP scheduler.

### ERP attendance sync

Optional (`ERP_SYNC_ENABLED`, default off). Writes recognised+reported events into an
external MySQL `ct_hr_employee_attendance_log` via PyMySQL, resolving `employee_code` →
`emp_id` through `ct_hr_employee_master`. Idempotency comes from stamping `erp_synced_at`
(success) or `erp_skip_reason` (unmapped camera / no employee id / duplicate punch) on the
`recognition_logs` row, so a skipped event is never retried forever. This is the only code
that touches a database this service does not own — it never writes to its own tables from
the ERP path except for those two stamps.

## Gotchas

- **The `ai` extra only installs on Python 3.12.** `onnx==1.16.2` and `ml-dtypes==0.4.1`
  — the pair pinned in `pyproject.toml` for numpy-2.x compatibility — publish cp312
  wheels and nothing newer. On 3.13 uv falls back to building `onnx` from its sdist,
  which fails in CMake 4.x (`third_party/pybind11` still declares
  `cmake_minimum_required(VERSION <3.5)`). `.python-version` pins 3.12 for this reason;
  if the venv was created on 3.13, delete `.venv` and re-run `uv sync --extra ai`.
- **`REQUIRE_ENGAGEMENT` defaults to `true`** and gates *all* attendance reporting: a
  recognised employee is logged with `response="not_engaged"` and nothing is published
  unless their face has held ≥`ENGAGEMENT_MIN_FACE_RATIO` of frame width for
  ≥`ENGAGEMENT_REQUIRED_SECONDS` *and* a hand is detected within half a frame width
  horizontally. None of these keys are in `.env.example` (nor are `detect_thresh`,
  `attendance_publish_retries`, `app_name`) — check `settings.py`, not the example file.
- **`models/hand_landmarker.task` is required to boot.** `HandDetector.__init__` raises
  `FileNotFoundError` if it is missing, which fails `load_ai_components` and therefore
  startup. Unlike the SCRFD/ArcFace models, insightface does not download it; fetch it
  from the MediaPipe URL in the error message. Neither README nor `.env.example` says so.
- `WaveTracker` (`app/ai/gesture/wave.py`) is constructed and threaded through
  `AIComponents` but **never called** in production code. The engagement "wave" flag is
  set purely by hand-to-face horizontal proximity in `RecognitionLoop._update_engagement`.
- That same method is full of bare `print("[ENGAGE] ...")` debug output that goes to
  stdout on every processed frame, bypassing the JSON logger.
- Without a tracker (`TRACKING_ENABLED=false`, the default), engagement state is keyed by
  a 100px spatial hash of the face centre, not a real identity — it drifts when people move.
- README's ERP `in_out_mode` description (toggle 1=in / 2=out) is stale: `InOutResolver`
  writes the constant `255` and distinguishes punches by INSERT (first of day) vs UPDATE
  (second), skipping the third onward.
- README's endpoint table omits `/camera/frame`, `/camera/stream`, and `/index/build-info`.
- The rate limiter (`app/core/security.py`) is per-process and in-memory; the documented
  deployment is a single instance. Multi-replica needs a shared backend.
- `frontend/app.py` imports `client`/`jwtgen` as top-level modules, so it only runs via
  `streamlit run frontend/app.py` (Streamlit puts the script's directory on `sys.path`).
- Tests swap the whole DB to SQLite by monkeypatching the engine globals in
  `app.database.session` (`tests/conftest.py`); both planes then share one SQLite file.
  Add new engines/factories to that fixture or tests will silently hit Postgres.
