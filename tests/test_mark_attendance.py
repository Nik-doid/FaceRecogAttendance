"""Step 5: turning recognised faces into published attendance events."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.face_processing.attendance_handlers import (
    BrokerMarkAttendance,
    NullMarkAttendance,
)
from app.schemas.face_processing import FaceResult, FrameContext
from app.services.attendance_reporter.base import (
    AttendanceEvent,
    AttendanceReporter,
    ReportResult,
)
from app.services.duplicate_suppressor import DuplicateSuppressor

CTX = FrameContext(camera_id="cam-test", captured_at=datetime(2026, 3, 1, 9, 15, tzinfo=UTC))


def _face(code: str | None, confidence: float = 0.8) -> FaceResult:
    return FaceResult(
        bbox=(0.0, 0.0, 10.0, 10.0),
        score=0.9,
        looking=True,
        palm=True,
        employee_code=code,
        confidence=confidence,
    )


class RecordingReporter(AttendanceReporter):
    def __init__(self, *, fail: bool = False, raises: bool = False) -> None:
        self.events: list[AttendanceEvent] = []
        self._fail = fail
        self._raises = raises

    def report(self, event: AttendanceEvent) -> ReportResult:
        if self._raises:
            raise RuntimeError("broker exploded")
        self.events.append(event)
        if self._fail:
            return ReportResult(success=False, detail="nope")
        return ReportResult(success=True, detail="published")


def _broker(**kwargs: bool) -> tuple[BrokerMarkAttendance, RecordingReporter]:
    reporter = RecordingReporter(**kwargs)
    return BrokerMarkAttendance(reporter, DuplicateSuppressor(300)), reporter


# --- the null sink -----------------------------------------------------------


async def test_null_sink_records_but_publishes_nothing() -> None:
    handler = NullMarkAttendance()
    await handler.mark([_face("EMP1")], CTX)
    assert [event.employee_code for event in handler.events] == ["EMP1"]


# --- the broker sink ---------------------------------------------------------


async def test_publishes_one_event_per_identified_face() -> None:
    handler, reporter = _broker()
    await handler.mark([_face("EMP1"), _face("EMP4")], CTX)
    assert [event.employee_code for event in reporter.events] == ["EMP1", "EMP4"]


async def test_unidentified_faces_are_never_published() -> None:
    """A near-miss still carries a confidence, so filtering on score would leak it."""
    handler, reporter = _broker()
    await handler.mark([_face(None, confidence=0.59)], CTX)
    assert reporter.events == []


async def test_event_carries_the_frame_context() -> None:
    handler, reporter = _broker()
    await handler.mark([_face("EMP1", confidence=0.71)], CTX)
    (event,) = reporter.events
    assert event.camera_id == "cam-test"
    assert event.timestamp == CTX.captured_at
    assert event.confidence == pytest.approx(0.71)


async def test_one_wave_publishes_once() -> None:
    """A hand-raise spans several scans; without the limiter each one publishes."""
    handler, reporter = _broker()
    for _ in range(5):
        await handler.mark([_face("EMP1")], CTX)
    assert len(reporter.events) == 1


async def test_the_limit_is_per_employee() -> None:
    handler, reporter = _broker()
    await handler.mark([_face("EMP1")], CTX)
    await handler.mark([_face("EMP1")], CTX)
    await handler.mark([_face("EMP4")], CTX)
    assert [event.employee_code for event in reporter.events] == ["EMP1", "EMP4"]


async def test_the_window_expires() -> None:
    reporter = RecordingReporter()
    handler = BrokerMarkAttendance(reporter, DuplicateSuppressor(0))
    await handler.mark([_face("EMP1")], CTX)
    await handler.mark([_face("EMP1")], CTX)
    assert len(reporter.events) == 2


# --- failure must never reach the camera loop --------------------------------


async def test_a_failed_publish_does_not_raise() -> None:
    handler, reporter = _broker(fail=True)
    await handler.mark([_face("EMP1")], CTX)
    assert len(reporter.events) == 1


async def test_a_raising_reporter_does_not_take_the_loop_down() -> None:
    """The ABC promises 'must not raise'; that has to be enforced, not documented."""
    handler, _ = _broker(raises=True)
    await handler.mark([_face("EMP1")], CTX)  # must not propagate
