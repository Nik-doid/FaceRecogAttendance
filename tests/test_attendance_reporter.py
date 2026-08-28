"""Attendance reporter tests: null + RabbitMQ (with a fake pika module)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import pytest

from app.services.attendance_reporter.base import AttendanceEvent
from app.services.attendance_reporter.null import NullAttendanceReporter
from app.services.attendance_reporter.rabbitmq import RabbitMQAttendanceReporter


def make_event() -> AttendanceEvent:
    return AttendanceEvent(
        employee_code="EMP1023",
        camera_id="cam-01",
        timestamp=datetime(2026, 8, 4, 9, 12, 33, tzinfo=UTC),
        confidence=0.97,
        snapshot_path="/storage/snapshots/x.jpg",
    )


def test_null_reporter_records() -> None:
    reporter = NullAttendanceReporter()
    result = reporter.report(make_event())
    assert result.success
    assert len(reporter.events) == 1
    assert reporter.events[0].employee_code == "EMP1023"


def test_event_serializes_as_iso_utc() -> None:
    event = make_event()
    payload = event.as_dict()
    assert payload["timestamp"].endswith("+00:00")
    assert payload["employee_code"] == "EMP1023"
    assert payload["schema_version"] == 1


class _FakePikaModule:
    def __init__(self, fail_publish: int = 0) -> None:
        self.fail_publish = fail_publish  # number of publishes to fail
        self.publish_attempts = 0
        self.closed = False

    class BasicProperties:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.kwargs = kwargs

    class URLParameters:
        def __init__(self, url: str) -> None:
            self.url = url

    class BlockingConnection:
        def __init__(self, params) -> None:  # type: ignore[no-untyped-def]
            self.params = params
            self.is_open = True
            self._channel = None

        def channel(self):
            if self._channel is None:
                self._channel = _FakePikaModule._Channel()
            return self._channel

        def close(self) -> None:
            self.is_open = False

    class _Channel:
        confirms = False

        def __init__(self) -> None:
            self.is_open = True

        def confirm_delivery(self) -> None:
            self.confirms = True

        def exchange_declare(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def queue_declare(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def queue_bind(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def basic_publish(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def close(self) -> None:
            self.is_open = False


@pytest.fixture
def fake_pika(monkeypatch):
    module = _FakePikaModule()
    monkeypatch.setitem(sys.modules, "pika", module)
    return module


def test_rabbitmq_publishes_and_declares_topology(fake_pika) -> None:
    reporter = RabbitMQAttendanceReporter(
        url="amqp://localhost", exchange="att.events", routing_key="checkin", queue="att.q"
    )
    result = reporter.report(make_event())
    assert result.success
    reporter.close()


def test_rabbitmq_retries_on_transport_failure(fake_pika) -> None:
    # Force every publish to fail so all retries are consumed.
    def boom(**kwargs):  # type: ignore[no-untyped-def]
        raise ConnectionError("broken pipe")


    # Patch the channel class used by the fake connection.
    fake_pika._Channel.basic_publish = staticmethod(boom)

    reporter = RabbitMQAttendanceReporter(
        url="amqp://localhost", exchange="att.events", routing_key="checkin", retries=2
    )
    result = reporter.report(make_event())
    assert result.success is False
    assert "failed after retries" in result.detail
    reporter.close()
