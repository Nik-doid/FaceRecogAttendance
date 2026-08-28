"""Step 5: record attendance for the people who raised a hand.

This step delegates to the existing ``AttendanceReporter`` seam
(``app/services/attendance_reporter/``) rather than talking to a broker itself, so
swapping RabbitMQ for something else stays a change of reporter, not of pipeline.

Two suppression layers exist and they answer different questions. The one here is a
publish-rate limiter: a hand-raise spans several scans, and without it one wave emits
an event per scan. The *business* rule -- how many punches a day is worth recording --
lives in the consumer, where the attendance table is the source of truth. Doing it
here instead would be wrong the moment the process restarts.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from app.core.logging import get_logger
from app.schemas.face_processing import FaceResult, FrameContext
from app.services.attendance_reporter.base import AttendanceEvent, AttendanceReporter
from app.services.duplicate_suppressor import DuplicateSuppressor

log = get_logger(__name__)


class BaseMarkAttendance(ABC):
    """Records an attendance event for each recognised face."""

    @abstractmethod
    async def mark(self, faces: list[FaceResult], ctx: FrameContext) -> None:
        """Best-effort recording of attendance. Must not raise."""


class NullMarkAttendance(BaseMarkAttendance):
    """Records nowhere, remembers everything. The test double and the safe default."""

    def __init__(self) -> None:
        self.events: list[AttendanceEvent] = []

    async def mark(self, faces: list[FaceResult], ctx: FrameContext) -> None:
        self.events.extend(_events(faces, ctx))


class BrokerMarkAttendance(BaseMarkAttendance):
    """Publishes one event per recognised employee, rate-limited per employee."""

    def __init__(self, reporter: AttendanceReporter, suppressor: DuplicateSuppressor) -> None:
        self._reporter = reporter
        self._suppressor = suppressor

    async def mark(self, faces: list[FaceResult], ctx: FrameContext) -> None:
        try:
            for event in _events(faces, ctx):
                if not self._suppressor.check_and_record(event.employee_code):
                    continue
                # pika's BlockingConnection is synchronous and lock-guarded; calling
                # it inline would stall the loop for the length of a broker reconnect.
                result = await asyncio.to_thread(self._reporter.report, event)
                if result.success:
                    log.info(
                        "attendance published",
                        extra={
                            "event": "attendance_published",
                            "employee_code": event.employee_code,
                            "camera_id": event.camera_id,
                            "confidence": event.confidence,
                        },
                    )
                else:
                    log.warning(
                        "attendance publish failed",
                        extra={
                            "event": "attendance_publish_failed",
                            "employee_code": event.employee_code,
                            "detail": result.detail,
                        },
                    )
        except Exception:  # noqa: BLE001 - the contract says this must not raise
            # A failure to record attendance must never take the camera loop down
            # with it; the frame is lost, the next one is not.
            log.exception("marking attendance failed")


def _events(faces: list[FaceResult], ctx: FrameContext) -> list[AttendanceEvent]:
    """One event per identified face.

    Filtered on ``employee_code`` rather than on ``confidence``: a face that lost to
    the threshold still carries its best score, so testing the score would publish
    near-misses as if they were matches.
    """
    return [
        AttendanceEvent(
            employee_code=face.employee_code,
            camera_id=ctx.camera_id,
            timestamp=ctx.captured_at,
            confidence=face.confidence,
        )
        for face in faces
        if face.employee_code
    ]
