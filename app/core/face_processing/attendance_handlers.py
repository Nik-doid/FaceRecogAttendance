"""Step 4 interface and handlers: record attendance for recognised faces.

Not implemented yet. When it is, this is the step that should delegate to the existing
``AttendanceReporter`` seam (``app/services/attendance_reporter/``) rather than talk to
a broker itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.face_processing import FaceResult


class BaseMarkAttendance(ABC):
    """Records an attendance event for each recognised face."""

    @abstractmethod
    async def mark(self, faces: list[FaceResult]) -> None:
        """Best-effort recording of attendance. Must not raise."""


class BrokerMarkAttendance(BaseMarkAttendance): ...


class NullMarkAttendance(BaseMarkAttendance): ...
