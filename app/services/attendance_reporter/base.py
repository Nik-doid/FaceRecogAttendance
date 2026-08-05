"""Attendance event contract and reporter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

DEFAULT_EVENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AttendanceEvent:
    """Payload published for every recognized attendance event.

    Field names intentionally mirror the REST contract in the original spec so a
    future HTTP reporter can reuse the same payload without reshaping.
    """

    employee_code: str
    camera_id: str
    timestamp: datetime
    confidence: float
    snapshot_path: str | None = None
    track_id: int | None = None
    schema_version: int = DEFAULT_EVENT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.astimezone(UTC).isoformat()
        return payload


@dataclass(frozen=True)
class ReportResult:
    """Outcome of one report attempt; persisted on the recognition_log row."""

    success: bool
    detail: str = ""


@dataclass
class _ReporterState:
    """Bookkeeping shared by reporters (not part of the public contract)."""

    published: int = field(default=0)
    failed: int = field(default=0)


class AttendanceReporter(ABC):
    """Interface every integration with the existing attendance system implements."""

    @abstractmethod
    def report(self, event: AttendanceEvent) -> ReportResult:
        """Best-effort delivery of one attendance event. Must not raise."""

    def close(self) -> None:  # noqa: B027
        """Release any underlying connections/resources."""

    def flush(self) -> None:  # noqa: B027
        """No-op default; override for buffered/batched reporters."""
