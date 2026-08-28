"""The always-on camera runner.

The headline test here is ``test_runs_with_no_websocket_client_connected``: that is
the regression test for the bug this whole module exists to fix, where the camera was
owned by a WebSocket connection and attendance stopped when the browser tab closed.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import numpy as np
import pytest

from app.camera.hub import FrameHub
from app.camera.runner import CameraRunner, CameraRunnerAlreadyRunningError
from app.config.settings import Settings
from app.core.face_processing.gallery import GalleryHandle
from app.schemas.face_processing import FrameContext, FrameResult, PalmResult

NO_PALM = PalmResult(detected=False, score=0.0)


class FakeReader:
    """A camera that yields blank frames, and can be told to fail on demand."""

    def __init__(
        self,
        *,
        opens: bool = True,
        fail_after: int | None = None,
        raise_after: int | None = None,
    ) -> None:
        self._opens = opens
        self._fail_after = fail_after
        self._raise_after = raise_after
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
        return np.zeros((120, 160, 3), np.uint8)

    def close(self) -> None:
        self.closed += 1


class FakeProcess:
    """Records every FrameContext it is handed, so we can assert on capture time."""

    def __init__(self, *, raises: bool = False) -> None:
        self.contexts: list[FrameContext | None] = []
        self.calls = 0
        self._raises = raises
        self.entered = threading.Event()

    async def process_frames(
        self, frame: np.ndarray, ctx: FrameContext | None = None
    ) -> FrameResult:
        self.calls += 1
        self.contexts.append(ctx)
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
        models=None,  # type: ignore[arg-type]
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
