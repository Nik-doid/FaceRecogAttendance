"""Drains the attendance queue into the attendance table.

This closes the flow: the camera pipeline publishes, this consumes and writes. It
owns its **own** AMQP connection -- ``pika.BlockingConnection`` is not thread-safe and
must never share a channel with the publisher running in the same process.

Ack strategy, and why each branch is what it is:

* **success** -- ack.
* **permanent failure** (malformed payload, unknown employee, unmapped camera) --
  dead-letter and ack. Requeueing a poison message with ``prefetch_count=1`` spins a
  core at 100% forever and blocks the queue head behind a message that can never
  succeed. The dead-letter queue is the operator's worklist.
* **transient failure** (MySQL unreachable, deadlock) -- sleep, then nack with
  requeue. The sleep is not optional: without it a MySQL outage becomes a hot loop.
  After ``MAX_ATTEMPTS`` the message is dead-lettered anyway, so one unlucky row
  cannot block everything behind it.

``prefetch_count=1`` also keeps ordering, which matters here: the first message of the
day has to be the check-in.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from typing import Any

from app.core.logging import get_logger
from app.core.metrics import ATTENDANCE_DEAD_LETTERED
from app.services.attendance_consumer.mysql import TransientDatabaseError
from app.services.attendance_consumer.writer import (
    REASON_MALFORMED,
    AttendanceLogWriter,
    PermanentFailure,
)

log = get_logger(__name__)

MAX_ATTEMPTS = 5
MAX_BACKOFF_SECONDS = 30.0


class AttendanceConsumer:
    """Blocking consume loop. Run it on a thread, or as its own process."""

    def __init__(
        self,
        writer: AttendanceLogWriter,
        *,
        url: str,
        queue: str,
        dead_letter_exchange: str = "",
        prefetch: int = 1,
    ) -> None:
        try:
            import pika  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pika is not installed") from exc
        self._pika = pika
        self._writer = writer
        self._url = url
        self._queue = queue
        self._dead_letter_exchange = dead_letter_exchange
        self._prefetch = max(1, prefetch)
        self._stop = threading.Event()
        self._attempts: dict[str, int] = {}
        self._connection: Any = None
        self._channel: Any = None

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        """Consume until stopped, reconnecting to the broker with backoff."""
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._consume()
                backoff = 1.0
            except Exception as exc:  # noqa: BLE001 - the loop must outlive the broker
                if self._stop.is_set():
                    break
                log.warning("attendance consumer disconnected", extra={"error": str(exc)})
                self._teardown()
                time.sleep(backoff)
                backoff = min(MAX_BACKOFF_SECONDS, backoff * 2)
        self._teardown()
        log.info("attendance consumer stopped")

    # -- the loop -------------------------------------------------------------
    def _consume(self) -> None:
        self._connection = self._pika.BlockingConnection(self._pika.URLParameters(self._url))
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue=self._queue, durable=True, passive=True)
        self._channel.basic_qos(prefetch_count=self._prefetch)
        log.info("attendance consumer connected", extra={"queue": self._queue})

        for method, properties, body in self._channel.consume(
            queue=self._queue, auto_ack=False, inactivity_timeout=1.0
        ):
            if self._stop.is_set():
                break
            if method is None:
                continue  # inactivity tick, so the stop flag is checked promptly
            self._dispatch(method, properties, body)
        self._channel.cancel()

    def _dispatch(self, method: Any, properties: Any, body: bytes) -> None:
        tag = method.delivery_tag
        key = getattr(properties, "message_id", None) or str(tag)
        try:
            message = json.loads(body.decode("utf-8"))
            if not isinstance(message, dict):
                raise PermanentFailure(REASON_MALFORMED, "payload is not an object")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._dead_letter(tag, body, REASON_MALFORMED, str(exc))
            return
        except PermanentFailure as exc:
            self._dead_letter(tag, body, exc.reason, str(exc))
            return

        try:
            outcome = self._writer.handle(message)
        except PermanentFailure as exc:
            self._dead_letter(tag, body, exc.reason, str(exc))
            return
        except TransientDatabaseError as exc:
            self._retry(tag, body, key, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - an unexpected bug is not the queue's
            self._dead_letter(tag, body, REASON_MALFORMED, f"unhandled: {exc}")
            return

        self._attempts.pop(key, None)
        self._channel.basic_ack(delivery_tag=tag)
        log.debug("attendance message acked", extra={"action": outcome.action})

    def _retry(self, tag: int, body: bytes, key: str, detail: str) -> None:
        attempts = self._attempts.get(key, 0) + 1
        self._attempts[key] = attempts
        if attempts >= MAX_ATTEMPTS:
            log.error(
                "attendance message failed repeatedly; dead-lettering",
                extra={"attempts": attempts, "error": detail},
            )
            self._attempts.pop(key, None)
            self._dead_letter(tag, body, "database_unavailable", detail)
            return

        # Without this pause a MySQL outage becomes a hot requeue loop.
        delay = min(MAX_BACKOFF_SECONDS, 2.0**attempts)
        log.warning(
            "attendance write failed; requeueing",
            extra={"attempt": attempts, "retry_in": delay, "error": detail},
        )
        time.sleep(delay)
        self._channel.basic_nack(delivery_tag=tag, requeue=True)

    def _dead_letter(self, tag: int, body: bytes, reason: str, detail: str) -> None:
        ATTENDANCE_DEAD_LETTERED.labels(reason=reason).inc()
        log.error(
            "attendance message dead-lettered",
            extra={"event": "attendance_dead_lettered", "reason": reason, "detail": detail},
        )
        if self._dead_letter_exchange:
            try:
                self._channel.basic_publish(
                    exchange=self._dead_letter_exchange,
                    routing_key=reason,
                    body=body,
                    properties=self._pika.BasicProperties(
                        content_type="application/json",
                        delivery_mode=2,
                        headers={"x-dead-letter-reason": reason, "x-detail": detail[:512]},
                    ),
                )
            except Exception:  # noqa: BLE001 - never let this block the ack below
                log.exception("could not publish to the dead-letter exchange")
        # Acked either way: a message that can never be processed must leave the
        # queue, or it blocks every message behind it at prefetch=1.
        self._channel.basic_ack(delivery_tag=tag)

    def _teardown(self) -> None:
        for closeable in (self._channel, self._connection):
            with contextlib.suppress(Exception):  # best-effort close
                if closeable is not None and closeable.is_open:
                    closeable.close()
        self._channel = None
        self._connection = None
