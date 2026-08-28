# CLAUDE.md

Guidance for Claude Code working in this repository.

## Commands

```bash
uv sync --extra ai                      # onnxruntime + insightface (needs a C toolchain)
uv run pytest                           # 155 tests, ~27s
uv run pytest tests/test_attendance_policy.py::test_third_punch_updates_the_second_row
uv run ruff check app tests --fix
uv run mypy app                         # strict
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
python -m app.services.attendance_consumer   # consumer as its own process
docker compose up -d --build            # rabbitmq + app
```

Admin endpoints need a bearer token:
`uv run python -c "from app.core.security import create_access_token as t; print(t('admin'))"`.
The service only *verifies* JWTs — it issues none and has no user CRUD.

`mypy` occasionally reports a stale `Unused "type: ignore"` from its incremental
cache; `mypy app --no-incremental` is the truth.

## Architecture

One flow, and the shape of it is load-bearing:

```
RTSP ─▶ CameraRunner ─▶ FaceRecognitionProcess ─▶ AttendanceReporter ─▶ RabbitMQ
                │                                                          │
                └─▶ FrameHub ─▶ /camera/ws (viewer)      AttendanceConsumer ┘
                                                                  │
                                                    ct_hr_employee_attendance_log
```

**`app/camera/runner.py` owns the camera, not the WebSocket.** This is the one
invariant most worth preserving. The reader, the pipeline and the loop used to live
inside the `/camera/ws` handler, so closing a browser tab stopped attendance. The
runner now holds them for the life of the process and `/camera/ws` is a viewer over
`FrameHub`. `tests/test_camera_runner.py::test_runs_with_no_websocket_client_connected`
is the regression test — do not let that inversion come back.

It runs on a daemon thread with a **private event loop**, not a task on the FastAPI
loop, because `CameraReader.read` is blocking and would stall `/health` and every
viewer on a two-core box.

**`app/services/face_recognition/process.py`** is the pipeline: detect → looking gate
→ per-face palm → recognise → mark attendance. Steps dispatch through
`app/core/face_processing/dispatchers.py`, so swapping an implementation is an enum
value in `FaceProcessConfig`, never an edit to `process_frames`.

The looking gate is the exception to that rule: it is arithmetic over landmarks step 1
already produced, with no model to swap, so it is a plain function in `gaze.py`.

**`app/runtime.py` is the only place SCRFD and ArcFace are constructed.** Both are
onnxruntime sessions (`Run` is thread-safe) and are shared by the gallery, the runner
and every viewer. The BlazePalm net is `cv2.dnn`, which has no such guarantee, so it
stays private to each `FaceRecognitionProcess`.

**`app/services/attendance_consumer/`** is split four ways on purpose: `policy.py`
decides (pure, no I/O), `mysql.py` writes, `writer.py` joins them, `consumer.py` only
routes acks. The punch rules are therefore testable without a broker or a database,
and `tests/test_attendance_policy.py` is where the correctness of the flow lives.

## Rules that are easy to break

- **`process_frames(frame, ctx=None)` records attendance only when `ctx` is given.**
  `/webcam/ws` (browser webcam) passes none, so a developer holding a palm up to a
  laptop cannot punch a colleague in. `FaceProcessConfig.attendance_sink` defaults to
  `NULL` for the same reason. Only the camera runner passes `RABBITMQ`.
- **`captured_at` is stamped when the frame is read**, not when it is published, so a
  broker outage cannot move everyone's punch time.
- **`to_local()` returns a *naive* datetime.** MySQL `DATETIME` carries no zone, so
  the rows read back are naive local; returning an aware value makes the gap
  comparison raise on the first punch of a day that already had one.
- **The attendance handlers must never raise.** The ABC says so and
  `BrokerMarkAttendance` enforces it — a failed publish must not take the camera loop
  down.
- **Never requeue a poison message.** At `prefetch=1` it spins a core and blocks the
  queue head forever. Dead-letter and ack.
- **`EMPLOYEE_PHOTOS_SOURCE` is `list[str]`, not `list[Path]`.**
  `Path("https://h/x")` collapses the double slash and silently mangles every URL.
- **`Container._lock` is an `RLock`.** The `gallery` property builds under it and
  reaches through `models`, which takes it again.

## Gotchas

- **The `ai` extra only installs on Python 3.12.** `onnx==1.16.2` and
  `ml-dtypes==0.4.1` — pinned for numpy-2.x compatibility — publish cp312 wheels and
  nothing newer. On 3.13 uv builds `onnx` from sdist and CMake 4.x fails on
  pybind11's `cmake_minimum_required(VERSION <3.5)`. `.python-version` pins 3.12; if
  the venv was made on 3.13, delete `.venv` and re-run `uv sync --extra ai`.
- **OpenCV's RTSP chatter cannot be silenced from Python.** The cv2 DLL snapshots its
  environment at process start, so neither `os.environ` nor Win32
  `SetEnvironmentVariableW` reaches it, and its FFmpeg backend is a separate plugin
  that `cv2.utils.logging.setLogLevel` does not touch. Export `OPENCV_LOG_LEVEL=ERROR`
  and `OPENCV_FFMPEG_LOGLEVEL=8` before launching; docker-compose already does.
- **CUDA is requested by default and is absent on the deployment box.**
  `filter_providers` drops it silently; `app/runtime.py` logs the providers
  onnxruntime actually resolved, which is the only way to notice.
- **SCRFD's cost is constant in frame width** — input is always letterboxed to
  640×640, so `MAX_FRAME_WIDTH` is not a speed or accuracy lever. What matters is face
  pixels at the sensor.
- **Recognition costs ~1.2 s per frame on the target i3**, so roughly one scan per
  second. `CAMERA_SCAN_INTERVAL_MS` is time-based for this reason; frame-counting at
  30fps would claim a cadence the hardware cannot meet.
- **The gallery is empty until the background build finishes.** That is not an error
  path: `ArcFaceRecognition._match` returns faces unchanged when `recognizer is None`,
  so detection, gaze and palm all work meanwhile and `/health` reports `degraded`.
- **`tests/conftest.py` injects the container into `create_app(container)`** so the
  lifespan does not enrol photos or open a camera behind a test.
