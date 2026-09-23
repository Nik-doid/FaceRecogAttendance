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
- **`ATTENDANCE_TIMEZONE` is `Asia/Kathmandu` (UTC+05:45), not a whole-hour offset.**
  A zone that is merely close is fifteen minutes wrong on every row.
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
- **`MAX_FRAME_WIDTH` and `DETECT_INPUT_SIZE` are orthogonal, and both are needed.**
  A face reaches SCRFD at `frame_fraction × DETECT_INPUT_SIZE` pixels *whatever*
  `MAX_FRAME_WIDTH` is — the reader's resize and insightface's letterbox cancel
  exactly. So `DETECT_INPUT_SIZE` is the only **detection** lever (at 640 a face 3% of
  frame width reaches the network as 19px, below what the stride-8 anchors resolve
  with landmarks good enough for the gaze gate or ArcFace alignment), while
  `MAX_FRAME_WIDTH` is the only **crop** lever, because `norm_crop` samples the
  delivered frame. Raising one without the other is half a fix. Must be a multiple
  of 32.
- **`DETECT_INPUT_SIZE` cost grows faster than the area.** Measured with
  `uv run python -m app.services.evaluation` on a 4-core dev box: **640 is ~0.69s per
  pass, 1280 is ~4.9s** — a 7× jump for 4× the pixels. Every earlier estimate in this
  repo said 4×, and they were all guesses; this one is the harness's. At ~4.9s of
  detection plus ~1.1s of ArcFace, a single scan already outlasts the few seconds
  someone will stand still, so 1280 is not simply a knob to turn — it needs the frame
  cropped to the band people occupy, and `CAMERA_SCAN_INTERVAL_MS` raised to match.
  Re-measure on the deployment box; do not trust the numbers in comments, including
  these.
- **Raising `MAX_FRAME_WIDTH` changes what `MIN_FACE_PIXELS` means.** It is measured on
  the delivered frame, so the same person in the same spot goes 40px → 60px and
  `faces_too_small_total` falls for reasons that have nothing to do with the camera.
  Accuracy figures do not compare across a change to it.
- **`PREVIEW_FRAME_WIDTH` exists because the JPEG encode is per-frame, not per-scan.**
  `cv2.imencode` runs ~30 times a second against one scan per second, so at 1080p it
  is ~0.4 of a core spent on viewers who may not be connected — `FrameHub` keeps no
  registry, so the runner cannot tell. Downscale the preview, never the frame handed
  to `process_frames`. Box alignment survives because `detection_payload` reports the
  *detection* frame's dimensions and the page scales by those.
- **`MIN_FRAME_WIDTH` upscaling adds no information.** It exists so a 640×360 CCTV
  substream is resampled once from a known size rather than twice, and because
  insightface would upscale it anyway when `DETECT_INPUT_SIZE` exceeds the source.
  A feed that recognises nothing needs the main stream or a closer camera, not this.
- **`CameraRunner` must follow `GalleryHandle`, not snapshot it.** `app/main.py` starts
  the enrolment thread and the runner back to back, so `gallery.current` is
  `EMPTY_GALLERY` for the first minutes of every boot — and a pipeline built once
  against that holds `recognizer is None`, which makes
  `ArcFaceRecognition._match` return every face with `confidence` exactly 0.00 for the
  life of the process, while `/webcam/ws` (which builds per connection) recognises
  fine. `_loop` rebuilds the process when the handle swaps;
  `tests/test_camera_runner.py::test_a_gallery_built_after_start_reaches_the_pipeline`
  is the regression test.
- **`FaceIndex.search` must snapshot `_codes` under the same lock as the search.** The
  index returns row numbers, and `rebuild` replaces index and codes together — reading
  codes after releasing the lock maps this search's rows onto the next gallery's names,
  i.e. a punch against the wrong employee, roughly once per `GALLERY_REFRESH_SECONDS`.
- **`SCRFDDetector` passes `max_num=0`, and it must stay 0.** When more detections than
  `max_num` survive, insightface ranks by `area - 2·offset_dist²` in *pixel* units: on
  a 1920-wide frame a face 400px off centre contributes 320_000 against a 60px face's
  area of ~4_500, so the area term is noise and the cap is purely "keep the most
  central". It discards small off-centre faces first — exactly what a ceiling camera
  sees. Not exposed as a setting, because 0 is the only correct value.
- **The embedding cache key covers the whole enrolment path, not just ArcFace.**
  `enrolment_fingerprint` folds in the detector file, `DETECT_INPUT_SIZE` and
  `PREPROCESS_VERSION` because a cached vector is the output of detect → align → embed.
  Keying on the recognition model alone meant a changed detector or input size reported
  a cache hit and returned vectors from a path that no longer existed — and the symptom
  is a real improvement measuring as no change at all. **Bump `PREPROCESS_VERSION`
  whenever `gallery._embed_ref` changes what a vector means.**
- **Face width in source pixels is the diagnostic.** `process_frames` logs
  `face_widths_px` at DEBUG on every scan, and `yaw_ratios` when the looking gate
  rejects everyone, and `face_centres` so a fixed camera's usable band can be found.
  Small faces make recognition less certain, but "certainly too small" is a measured
  curve, not a round number -- get it from rank-1 by width bucket in
  `app/services/evaluation.py` rather than asserting a threshold here.
- **Recognition costs ~1.2 s per frame on the target i3**, so roughly one scan per
  second. `CAMERA_SCAN_INTERVAL_MS` is time-based for this reason; frame-counting at
  30fps would claim a cadence the hardware cannot meet.
- **A `MIN_FACE_PIXELS` above the widths a site actually produces switches
  recognition off silently.** It shipped at 60 and a real deployment measured ~44px at
  punch-in distance, so every face was gated and the symptom was identical to the
  gallery being broken — `confidence` exactly 0.00 either way. The default is 32 now:
  that is where ArcFace genuinely has nothing (~4px across an eye), and between 32 and
  60 the score is allowed to speak for itself. Raise it only from a measured
  rank-1-by-width-bucket curve, never from an argument about 112×112.
- **`MIN_FACE_PIXELS` exists so "too far" and "not enrolled" stay different
  answers.** ArcFace aligns every crop to 112×112, so a 27×35 box carries a handful
  of pixels across each eye and the nose bridge and scores 0.00 against everyone.
  Under the floor, recognition is skipped and the face comes back `too_small=True`
  with `employee_code=None` — the gallery was never searched. Reporting those as
  unknown is what makes an accuracy figure lie, and it is also what sends people
  looking for a threshold fix to a problem only a closer camera solves.
  `faces_too_small_total` counts them and the `face_too_small` log line carries the
  measured widths.
- **Upscaling never recovers a small face.** Interpolating 27×35 to 112×112 adds no
  facial detail, so neither `MIN_FRAME_WIDTH` nor a bigger `DETECT_INPUT_SIZE` helps
  once the pixels are not at the sensor. `DETECT_INPUT_SIZE` only stops SCRFD from
  *throwing away* detail the frame already has; past that it is camera placement.
- **The tracker is injected, never owned.** Track identity belongs to one camera
  stream, so `CameraRunner` builds the `FaceTracker` and passes it in;
  `/webcam/ws` passes none, and `FaceRecognitionProcess._tracker` defaults to `None`
  at class level so an instance built past `__init__` (the `__new__` trick in
  `tests/test_min_face_pixels.py`) still works. `None` means the per-frame behaviour
  the pipeline had before tracking existed, which is what every test gets.
  `tracker.update` runs **after step 1 and before the gaze gate**: a track has to
  survive the scan where somebody glanced away.
- **A track votes on the employee *code*, and the two kinds of disagreement differ.**
  A scan that matched nobody is an absence of evidence and must not reset the run --
  people blink. A scan that named *someone else* is evidence the track is not one
  person, and it does reset. Without that distinction a track alternating EMP1/EMP2
  accumulates two votes for EMP1 and punches them in, which is a coin flip presented
  as agreement.
- **`TRACK_CONFIRM_SCANS` does not give you p^K.** Consecutive scans of a stationary
  person under fixed lighting through a fixed lens are nearly the same vector, so a
  wrong match repeats for the reason it happened the first time — they are one error
  observed K times, not K samples. What agreement removes is the uncorrelated part
  (one landmark glitch, one blurred scan). Worth having, not exponential, and its size
  is a property of this room: measure per-track FAR in
  `app/services/evaluation.py`, and never publish a figure derived from p^K.
- **`RECOGNITION_MARGIN` is differential where the threshold is absolute.** As detail
  is lost, every cosine against the gallery shrinks toward the gallery mean *together*
  — a common-mode shift, which an absolute floor is the wrong instrument for. The
  first-to-second gap cancels that shift. It ships at 0.0 because a margin tuned on
  four employees does not transfer to five hundred, where the runner-up is far
  stronger. A single-employee gallery has no runner-up, and that case is treated as an
  infinite margin — treating it as zero would reject the only person enrolled.
- **`build_gallery(degrade_to=...)` must never be used with a cache.** It low-passes
  each aligned enrolment crop so the gallery carries the same loss of detail the live
  faces already have — an experiment about *symmetry*, not quality, since a live crop
  has lost its high frequencies to distance and JPEG while a portrait has not. The
  cache fingerprint does not cover it, so degraded and undegraded vectors would mix
  silently under one key; `app/services/evaluation.py` passes `cache=None` for exactly
  this reason. Judge it on rank-1 and EER, never on the mean genuine score — that
  rises whenever both sides get blurrier, which is not the same as telling people
  apart. Applied to the *aligned* crop, not the source photo, so the landmarks that
  drive alignment are still found at full resolution.
- **The gallery is empty until the background build finishes.** That is not an error
  path: `ArcFaceRecognition._match` returns faces unchanged when `recognizer is None`,
  so detection, gaze and palm all work meanwhile and `/health` reports `degraded`.
- **`tests/conftest.py` injects the container into `create_app(container)`** so the
  lifespan does not enrol photos or open a camera behind a test.
