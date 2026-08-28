"""The consumer: writing one message into the attendance table, and acking it.

Split the way the code is: the writer decides and writes (against a fake MySQL), and
the consumer only routes acks and nacks (against a fake channel). Neither test needs
a broker or a database.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.services.attendance_consumer.consumer import MAX_ATTEMPTS, AttendanceConsumer
from app.services.attendance_consumer.mysql import AttendanceRow, TransientDatabaseError
from app.services.attendance_consumer.policy import DayRow
from app.services.attendance_consumer.writer import (
    REASON_UNKNOWN_EMPLOYEE,
    REASON_UNMAPPED_CAMERA,
    AttendanceLogWriter,
    PermanentFailure,
    WriteConfig,
)

MAPPING = {"cam-01": {"device_id": 7, "branch_id": 3}}


def _message(**overrides: Any) -> dict[str, Any]:
    message = {
        "employee_code": "EMP1",
        "camera_id": "cam-01",
        "timestamp": "2026-03-02T03:45:00+00:00",
        "confidence": 0.71,
    }
    message.update(overrides)
    return message


class FakeMysql:
    """In-memory stand-in that records every statement the writer issues."""

    def __init__(
        self,
        *,
        rows: list[DayRow] | None = None,
        employee_id: str | None = "42",
        raises: Exception | None = None,
        update_hits: bool = True,
    ) -> None:
        # `rows or []` would silently copy an empty list, so a caller that keeps a
        # reference to mutate it (as the whole-day test does) would never be seen.
        self._rows = rows if rows is not None else []
        self._employee_id = employee_id
        self._raises = raises
        self._update_hits = update_hits
        self.inserted: list[AttendanceRow] = []
        self.updated: list[tuple[int, AttendanceRow]] = []
        self.lookups = 0

    def lookup_employee_id(self, employee_code: str) -> str | None:
        self.lookups += 1
        if self._raises:
            raise self._raises
        return self._employee_id

    def day_rows(self, attendance_id_no: str, log_date_only: str) -> list[DayRow]:
        if self._raises:
            raise self._raises
        return list(self._rows)

    def insert(self, row: AttendanceRow) -> None:
        if self._raises:
            raise self._raises
        self.inserted.append(row)

    def update(self, row_id: int, row: AttendanceRow) -> bool:
        if self._raises:
            raise self._raises
        self.updated.append((row_id, row))
        return self._update_hits


def _writer(client: FakeMysql, **config: Any) -> AttendanceLogWriter:
    defaults = {
        "camera_mapping": MAPPING,
        "timezone": "Asia/Kolkata",
        "min_punch_gap_seconds": 60,
    }
    defaults.update(config)
    return AttendanceLogWriter(client, WriteConfig(**defaults))  # type: ignore[arg-type]


# --- the happy paths ---------------------------------------------------------


def test_first_punch_inserts_with_the_mapped_camera() -> None:
    client = FakeMysql()
    outcome = _writer(client).handle(_message())

    assert outcome.action == "inserted"
    (row,) = client.inserted
    assert row.attendance_id_no == "42"
    assert row.in_out_mode == 255
    assert row.device_id == 7
    assert row.branch_id == 3
    assert row.verify_mode == "FACE"


def test_the_written_time_is_local_not_utc() -> None:
    """03:45 UTC is 09:15 in Asia/Kolkata; storing the UTC value was the old bug."""
    client = FakeMysql()
    _writer(client).handle(_message())
    (row,) = client.inserted
    assert row.log_date_time == "2026-03-02 09:15:00"
    assert row.log_date_only == "2026-03-02"


def test_second_punch_inserts_a_second_row() -> None:
    client = FakeMysql(rows=[DayRow(1, datetime(2026, 3, 2, 9, 15))])
    outcome = _writer(client).handle(_message(timestamp="2026-03-02T11:45:00+00:00"))
    assert outcome.action == "inserted"
    assert client.updated == []


def test_third_punch_updates_the_second_row() -> None:
    client = FakeMysql(
        rows=[DayRow(1, datetime(2026, 3, 2, 9, 15)), DayRow(2, datetime(2026, 3, 2, 17, 15))]
    )
    outcome = _writer(client).handle(_message(timestamp="2026-03-02T13:00:00+00:00"))

    assert outcome.action == "updated"
    (row_id, row) = client.updated[0]
    assert row_id == 2, "the check-in row must never be the update target"
    assert row.log_date_time == "2026-03-02 18:30:00"
    assert client.inserted == []


def test_a_redelivered_message_writes_nothing() -> None:
    """At-least-once delivery has to converge without any local dedup store."""
    client = FakeMysql(rows=[DayRow(1, datetime(2026, 3, 2, 9, 15))])
    outcome = _writer(client).handle(_message())

    assert outcome.action == "skipped"
    assert client.inserted == []
    assert client.updated == []


# --- permanent failures ------------------------------------------------------


def test_an_unmapped_camera_is_permanent() -> None:
    with pytest.raises(PermanentFailure) as caught:
        _writer(FakeMysql()).handle(_message(camera_id="cam-99"))
    assert caught.value.reason == REASON_UNMAPPED_CAMERA


def test_an_unknown_employee_is_permanent() -> None:
    """Retrying will not conjure the employee; the dead-letter queue is the worklist."""
    with pytest.raises(PermanentFailure) as caught:
        _writer(FakeMysql(employee_id=None)).handle(_message())
    assert caught.value.reason == REASON_UNKNOWN_EMPLOYEE


@pytest.mark.parametrize(
    "message",
    [
        _message(employee_code=""),
        _message(camera_id=""),
        _message(timestamp="not-a-time"),
        {"employee_code": "EMP1", "camera_id": "cam-01"},
    ],
)
def test_malformed_messages_are_permanent(message: dict[str, Any]) -> None:
    with pytest.raises(PermanentFailure):
        _writer(FakeMysql()).handle(message)


def test_a_vanished_update_target_is_permanent() -> None:
    client = FakeMysql(
        rows=[DayRow(1, datetime(2026, 3, 2, 9, 15)), DayRow(2, datetime(2026, 3, 2, 17, 15))],
        update_hits=False,
    )
    with pytest.raises(PermanentFailure):
        _writer(client).handle(_message(timestamp="2026-03-02T13:00:00+00:00"))


def test_a_naive_timestamp_is_read_as_utc() -> None:
    """Treating it as local would silently shift the punch by the offset."""
    client = FakeMysql()
    _writer(client).handle(_message(timestamp="2026-03-02T03:45:00"))
    assert client.inserted[0].log_date_time == "2026-03-02 09:15:00"


# --- transient failures ------------------------------------------------------


def test_a_database_outage_is_transient() -> None:
    client = FakeMysql(raises=TransientDatabaseError("connection refused"))
    with pytest.raises(TransientDatabaseError):
        _writer(client).handle(_message())


# --- the ack matrix ----------------------------------------------------------


class FakeChannel:
    def __init__(self) -> None:
        self.acks: list[int] = []
        self.nacks: list[tuple[int, bool]] = []
        self.published: list[dict[str, Any]] = []

    def basic_ack(self, delivery_tag: int) -> None:
        self.acks.append(delivery_tag)

    def basic_nack(self, delivery_tag: int, requeue: bool) -> None:
        self.nacks.append((delivery_tag, requeue))

    def basic_publish(self, **kwargs: Any) -> None:
        self.published.append(kwargs)


class FakeWriter:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.calls = 0

    def handle(self, message: dict[str, Any]) -> Any:
        self.calls += 1
        if self._error:
            raise self._error

        class _Outcome:
            action = "inserted"

        return _Outcome()


def _consumer(writer: Any, channel: FakeChannel) -> AttendanceConsumer:
    consumer = AttendanceConsumer(
        writer, url="amqp://localhost", queue="q", dead_letter_exchange="dlx"
    )
    consumer._channel = channel  # noqa: SLF001 - the broker is what we are faking
    return consumer


class _Method:
    delivery_tag = 7


class _Props:
    message_id = "cam-01-EMP1-2026-03-02T03:45:00+00:00"


def _body(**overrides: Any) -> bytes:
    return json.dumps(_message(**overrides)).encode("utf-8")


def test_a_written_message_is_acked() -> None:
    channel = FakeChannel()
    _consumer(FakeWriter(), channel)._dispatch(_Method(), _Props(), _body())
    assert channel.acks == [7]
    assert channel.nacks == []


def test_a_permanent_failure_is_dead_lettered_then_acked() -> None:
    """Never requeue a poison message: at prefetch=1 it blocks everything behind it."""
    channel = FakeChannel()
    writer = FakeWriter(PermanentFailure(REASON_UNKNOWN_EMPLOYEE, "no such employee"))
    _consumer(writer, channel)._dispatch(_Method(), _Props(), _body())

    assert channel.acks == [7], "a poison message must leave the queue"
    assert channel.nacks == []
    assert channel.published[0]["routing_key"] == REASON_UNKNOWN_EMPLOYEE


def test_unparseable_bytes_are_dead_lettered_without_reaching_the_writer() -> None:
    channel = FakeChannel()
    writer = FakeWriter()
    _consumer(writer, channel)._dispatch(_Method(), _Props(), b"{not json")

    assert writer.calls == 0
    assert channel.acks == [7]
    assert channel.published


def test_a_transient_failure_is_requeued() -> None:
    channel = FakeChannel()
    writer = FakeWriter(TransientDatabaseError("mysql is down"))
    consumer = _consumer(writer, channel)
    consumer._dispatch(_Method(), _Props(), _body())

    assert channel.nacks == [(7, True)]
    assert channel.acks == []


def test_a_transient_failure_gives_up_eventually() -> None:
    """One unlucky row must not block the queue head forever."""
    channel = FakeChannel()
    writer = FakeWriter(TransientDatabaseError("mysql is down"))
    consumer = _consumer(writer, channel)
    # Pretend the earlier attempts already happened, so the test does not sleep
    # through the real backoff.
    consumer._attempts[_Props.message_id] = MAX_ATTEMPTS - 1
    consumer._dispatch(_Method(), _Props(), _body())

    assert channel.acks == [7], "gave up but never removed the message"
    assert channel.published[0]["routing_key"] == "database_unavailable"


def test_an_unexpected_bug_does_not_stall_the_queue() -> None:
    channel = FakeChannel()
    writer = FakeWriter(ZeroDivisionError("a genuine bug"))
    _consumer(writer, channel)._dispatch(_Method(), _Props(), _body())
    assert channel.acks == [7]
    assert channel.published


def test_a_successful_retry_clears_the_attempt_counter() -> None:
    channel = FakeChannel()
    consumer = _consumer(FakeWriter(), channel)
    consumer._attempts[_Props.message_id] = 2
    consumer._dispatch(_Method(), _Props(), _body())
    assert _Props.message_id not in consumer._attempts


def test_gap_skips_are_acked_like_successes() -> None:
    """A skip is a decision, not a failure: the message is done with."""
    channel = FakeChannel()
    client = FakeMysql(rows=[DayRow(1, datetime(2026, 3, 2, 9, 15))])
    _consumer(_writer(client), channel)._dispatch(_Method(), _Props(), _body())
    assert channel.acks == [7]
    assert channel.published == []


def test_the_whole_day_through_the_consumer() -> None:
    """Four punches, one day: insert, insert, update, update -- two rows, ever."""
    channel = FakeChannel()
    rows: list[DayRow] = []
    client = FakeMysql(rows=rows)

    def sync_rows() -> None:
        rows.clear()
        rows.extend(
            DayRow(index + 1, datetime.fromisoformat(row.log_date_time))
            for index, row in enumerate(client.inserted)
        )

    consumer = _consumer(_writer(client), channel)
    base = datetime(2026, 3, 2, 3, 45, tzinfo=UTC)
    for hours in (0, 8, 9, 10):
        consumer._dispatch(
            _Method(), _Props(), _body(timestamp=(base + timedelta(hours=hours)).isoformat())
        )
        sync_rows()

    assert len(client.inserted) == 2, "more than two rows were created for one day"
    assert len(client.updated) == 2
    assert client.inserted[0].log_date_time == "2026-03-02 09:15:00", "check-in moved"
    assert all(row_id == 2 for row_id, _ in client.updated)
