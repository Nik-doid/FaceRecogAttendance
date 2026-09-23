"""The always-on camera runner.

The headline test here is ``test_runs_with_no_websocket_client_connected``: that is
the regression test for the bug this whole module exists to fix, where the camera was
owned by a WebSocket connection and attendance stopped when the browser tab closed.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import cv2
import numpy as np
import pytest

from app.ai.faiss.index import FaceIndex
from app.camera.hub import FrameHub
from app.camera.runner import CameraRunner, CameraRunnerAlreadyRunningError, encode_jpeg
from app.config.settings import Settings
from app.core.face_processing.gallery import EMBEDDING_DIM, Gallery, GalleryHandle
from app.schemas.face_processing import FrameContext, FrameResult, PalmResult

NO_PALM = PalmResult(detected=False, score=0.0)


def _gallery(employees: int = 1) -> Gallery:
    """A gallery that is a distinct object from EMPTY_GALLERY; contents do not matter."""
    return Gallery(
        index=FaceIndex(dim=EMBEDDING_DIM),
        recognizer=None,
        employees=employees,
        photos=employees,
    )


class FakeReader:
    """A camera that yields blank frames, and can be told to fail on demand."""

    def __init__(
        self,
        *,
        opens: bool = True,
        fail_after: int | None = None,
        raise_after: int | None = None,
        width: int = 160,
        height: int = 120,
    ) -> None:
        self._opens = opens
        self._fail_after = fail_after
        self._raise_after = raise_after
        self._width = width
        self._height = height
        self.reads = 0
        self.opened = 0
        self.closed = 0

    def open(self) -> bool:
        self.opened += 1
        return self._opens

    def read(self) -> np.ndarray | None:
        self.reads += 1
        if self._raise_after is not None and self.reads > self._raise_after:
            raise OSError("rtsp socket died")
        if self._fail_after is not None and self.reads > self._fail_after:
            return None
        return np.zeros((self._height, self._width, 3), np.uint8)

    def close(self) -> None:
        self.closed += 1


class FakeProcess:
    """Records every FrameContext it is handed, so we can assert on capture time."""

    def __init__(self, *, raises: bool = False) -> None:
        self.contexts: list[FrameContext | None] = []
        self.frames: list[np.ndarray] = []
        self.calls = 0
        self._raises = raises
        self.entered = threading.Event()

    async def process_frames(
        self, frame: np.ndarray, ctx: FrameContext | None = None
    ) -> FrameResult:
        self.calls += 1
        self.contexts.append(ctx)
        self.frames.append(frame)
        self.entered.set()
        if self._raises:
            raise RuntimeError("bad frame")
        return FrameResult(palm=NO_PALM)


def _runner(
    reader: FakeReader,
    process: FakeProcess,
    hub: FrameHub | None = None,
    **overrides: object,
) -> CameraRunner:
    settings = Settings(
        _env_file=None,
        camera_id="cam-test",
        camera_scan_interval_ms=50,
        **overrides,  # type: ignore[arg-type]
    )
    return CameraRunner(
        settings,
        models=lambda: None,  # type: ignore[arg-type,return-value]
        gallery=GalleryHandle(),
        hub=hub or FrameHub(),
        reader_factory=lambda: reader,
        process_factory=lambda: process,
    )


def _wait(predicate: object, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(0.02)
    return False


# --- the blocker -------------------------------------------------------------


def test_runs_with_no_websocket_client_connected() -> None:
    """Capture and recognition must not depend on anyone watching.

    This is the whole reason the loop moved out of the /camera/ws handler.
    """
    reader, process, hub = FakeReader(), FakeProcess(), FrameHub()
    runner = _runner(reader, process, hub)
    try:
        runner.start()
        assert _wait(lambda: process.calls > 0), "pipeline never ran"
        assert _wait(lambda: hub.snapshot().jpeg is not None), "no frame published"
        # Checked before stop(), which deliberately clears the hub.
        assert _wait(lambda: hub.snapshot().detection is not None), "no detection published"
        assert reader.reads > 0
    finally:
        runner.stop()


def test_stop_halts_capture_and_releases_the_camera() -> None:
    reader, process = FakeReader(), FakeProcess()
    runner = _runner(reader, process)
    runner.start()
    assert _wait(lambda: reader.reads > 2)
    runner.stop()

    assert runner.running is False
    assert reader.closed >= 1, "the camera must be released on stop"
    settled = reader.reads
    time.sleep(0.15)
    assert reader.reads == settled, "the loop kept reading after stop"


def test_stop_clears_the_hub_so_viewers_stop_seeing_a_frozen_frame() -> None:
    reader, process, hub = FakeReader(), FakeProcess(), FrameHub()
    runner = _runner(reader, process, hub)
    runner.start()
    assert _wait(lambda: hub.snapshot().jpeg is not None)
    runner.stop()
    assert hub.snapshot().jpeg is None


def test_starting_twice_is_refused() -> None:
    runner = _runner(FakeReader(), FakeProcess())
    runner.start()
    try:
        with pytest.raises(CameraRunnerAlreadyRunningError):
            runner.start()
    finally:
        runner.stop()


def test_stop_is_safe_when_never_started() -> None:
    runner = _runner(FakeReader(), FakeProcess())
    assert runner.stop().running is False


def test_restart_after_stop() -> None:
    reader, process = FakeReader(), FakeProcess()
    runner = _runner(reader, process)
    runner.start()
    assert _wait(lambda: process.calls > 0)
    runner.stop()

    process.entered.clear()
    before = process.calls
    runner.start()
    try:
        assert _wait(lambda: process.calls > before), "runner did not resume"
    finally:
        runner.stop()


# --- resilience --------------------------------------------------------------


def test_reconnects_when_a_read_fails() -> None:
    """Unattended, a dropped RTSP stream must be retried, not treated as fatal."""
    reader, process = FakeReader(fail_after=2), FakeProcess()
    runner = _runner(reader, process)
    try:
        runner.start()
        assert _wait(lambda: reader.opened > 1), "never reopened the camera"
    finally:
        runner.stop()
    assert runner.state.reconnects > 0


def test_a_read_that_raises_is_survived() -> None:
    """OpenCV can throw on a half-open RTSP socket, not merely return None."""
    reader, process = FakeReader(raise_after=2), FakeProcess()
    runner = _runner(reader, process)
    try:
        runner.start()
        assert _wait(lambda: reader.opened > 1), "never recovered from a raising read"
        assert runner.running, "an exception from the camera killed the runner"
    finally:
        runner.stop()
    assert "rtsp socket died" in (runner.state.last_error or "")


def test_a_camera_that_will_not_open_keeps_retrying() -> None:
    reader, process = FakeReader(opens=False), FakeProcess()
    runner = _runner(reader, process)
    try:
        runner.start()
        assert _wait(lambda: reader.opened > 0)
        assert runner.running, "the runner gave up instead of waiting for the camera"
    finally:
        runner.stop()
    assert runner.state.connected is False
    assert runner.state.last_error


def test_one_bad_frame_does_not_kill_the_loop() -> None:
    reader, process = FakeReader(), FakeProcess(raises=True)
    runner = _runner(reader, process)
    try:
        runner.start()
        assert _wait(lambda: process.calls >= 2), "loop stopped after the first failure"
        assert runner.running
    finally:
        runner.stop()
    assert runner.state.last_error == "bad frame"


# --- cadence and context -----------------------------------------------------


def test_scans_are_never_concurrent() -> None:
    """A scan costs over a second on this hardware; overlapping them would thrash."""
    in_flight = 0
    peak = 0

    class SlowProcess(FakeProcess):
        async def process_frames(self, frame, ctx=None):  # type: ignore[no-untyped-def]
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                import asyncio

                await asyncio.sleep(0.08)
                return await super().process_frames(frame, ctx)
            finally:
                in_flight -= 1

    process = SlowProcess()
    runner = _runner(FakeReader(), process)
    try:
        runner.start()
        assert _wait(lambda: process.calls >= 3)
    finally:
        runner.stop()
    assert peak == 1, f"{peak} scans ran at once"


def test_capture_time_is_stamped_at_read_not_at_publish() -> None:
    """A broker outage must not move everyone's punch time."""
    process = FakeProcess()
    runner = _runner(FakeReader(), process)
    before = datetime.now(UTC)
    try:
        runner.start()
        assert _wait(lambda: process.calls > 0)
    finally:
        runner.stop()
    after = datetime.now(UTC)

    ctx = process.contexts[0]
    assert ctx is not None
    assert ctx.camera_id == "cam-test"
    assert before <= ctx.captured_at <= after


def test_state_reports_progress() -> None:
    reader, process = FakeReader(), FakeProcess()
    runner = _runner(reader, process)
    try:
        runner.start()
        assert _wait(lambda: runner.state.scans > 0)
    finally:
        runner.stop()

    state = runner.state.as_dict()
    assert state["frames"] > 0
    assert state["scans"] > 0
    assert state["started_at"] is not None
    assert state["running"] is False


# --- the gallery the pipeline recognises against -----------------------------
def test_a_gallery_built_after_start_reaches_the_pipeline() -> None:
    """The runner must follow ``GalleryHandle``, not snapshot it once at boot.

    ``app/main.py`` starts the enrolment thread and this runner back to back, so the
    gallery is empty for the first few minutes of every boot. A process built once
    against that snapshot holds ``recognizer is None``, and
    ``ArcFaceRecognition._match`` then returns every face unmatched -- confidence
    exactly 0.00, for the life of the process -- while ``/webcam/ws``, which builds
    its pipeline per connection, recognises perfectly. That asymmetry is the bug.
    """
    reader = FakeReader()
    handle = GalleryHandle()
    built: list[FakeProcess] = []

    def factory() -> FakeProcess:
        process = FakeProcess()
        built.append(process)
        return process

    settings = Settings(_env_file=None, camera_id="cam-test", camera_scan_interval_ms=50)
    runner = CameraRunner(
        settings,
        models=lambda: None,  # type: ignore[arg-type,return-value]
        gallery=handle,
        hub=FrameHub(),
        reader_factory=lambda: reader,
        process_factory=factory,
    )
    runner.start()
    try:
        assert _wait(lambda: len(built) == 1)
        assert _wait(lambda: built[0].calls > 0)

        handle.swap(_gallery(employees=3))
        assert _wait(lambda: len(built) == 2), "the swapped-in gallery never reached the pipeline"
        assert _wait(lambda: built[1].calls > 0)
    finally:
        runner.stop(timeout=5)


def test_the_pipeline_is_not_rebuilt_while_the_gallery_is_unchanged() -> None:
    """Rebuilding per frame would reload a 3.7 MiB palm net thirty times a second."""
    reader = FakeReader()
    built: list[FakeProcess] = []

    def factory() -> FakeProcess:
        process = FakeProcess()
        built.append(process)
        return process

    settings = Settings(_env_file=None, camera_id="cam-test", camera_scan_interval_ms=50)
    runner = CameraRunner(
        settings,
        models=lambda: None,  # type: ignore[arg-type,return-value]
        gallery=GalleryHandle(),
        hub=FrameHub(),
        reader_factory=lambda: reader,
        process_factory=factory,
    )
    runner.start()
    try:
        assert _wait(lambda: reader.reads > 10)
        assert len(built) == 1
    finally:
        runner.stop(timeout=5)


# --- the preview is not the pipeline -----------------------------------------
def test_the_preview_is_downscaled_while_detection_keeps_the_full_frame() -> None:
    """The encode is per-frame at ~30fps; detection is per-scan at ~1Hz.

    So the preview is where the cost lives and the pipeline is where the pixels
    matter. ArcFace crops from the frame `process_frames` is handed, so downscaling
    that to save encode time would throw away the face pixels this whole exercise is
    about.
    """
    reader = FakeReader(width=1920, height=1080)
    process = FakeProcess()
    hub = FrameHub()
    runner = _runner(reader, process, hub, preview_frame_width=640)
    runner.start()
    try:
        assert _wait(lambda: process.calls > 0)
        assert process.frames[0].shape[:2] == (1080, 1920), "detection lost resolution"

        jpeg = hub.snapshot().jpeg
        assert jpeg is not None
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape[:2] == (360, 640)
    finally:
        runner.stop(timeout=5)


def test_the_reported_dimensions_are_the_detection_frames() -> None:
    """The page scales boxes by these, so they must describe the frame boxes came from."""
    reader = FakeReader(width=1920, height=1080)
    process = FakeProcess()
    hub = FrameHub()
    runner = _runner(reader, process, hub, preview_frame_width=640)
    runner.start()
    try:
        assert _wait(lambda: hub.snapshot().detection is not None)
        payload = hub.snapshot().detection
        assert payload is not None
        assert (payload["width"], payload["height"]) == (1920, 1080)
    finally:
        runner.stop(timeout=5)


def test_encode_jpeg_only_shrinks_what_is_too_wide() -> None:
    tall = np.zeros((1080, 1920, 3), np.uint8)
    small = np.zeros((240, 320, 3), np.uint8)

    wide = cv2.imdecode(
        np.frombuffer(encode_jpeg(tall, max_width=640) or b"", np.uint8), cv2.IMREAD_COLOR
    )
    assert wide.shape[:2] == (360, 640)

    # Already inside the limit: untouched, never upscaled.
    left = cv2.imdecode(
        np.frombuffer(encode_jpeg(small, max_width=640) or b"", np.uint8), cv2.IMREAD_COLOR
    )
    assert left.shape[:2] == (240, 320)

    # 0 disables the downscale entirely.
    full = cv2.imdecode(
        np.frombuffer(encode_jpeg(tall, max_width=0) or b"", np.uint8), cv2.IMREAD_COLOR
    )
    assert full.shape[:2] == (1080, 1920)
