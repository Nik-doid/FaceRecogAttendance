"""RabbitMQ attendance reporter.

Publishes each event to a topic exchange with routing key ``checkin``. Declaring the
exchange and (optionally) queue + binding at startup means the broker topology is
created idempotently without operator action. Delivery is fire-and-forget with a
publish-confirm handshake? No — we use mandatory=False and rely on the exchange
declaration; retries happen on transport errors with linear backoff.

Connection health: pika's BlockingConnection is opened lazily on first use and
re-opened on failure, so a broker restart does not kill the worker.
"""

from __future__ import annotations

import json
import threading
import time

from app.core.logging import get_logger
from app.services.attendance_reporter.base import (
    AttendanceEvent,
    AttendanceReporter,
    ReportResult,
)

log = get_logger(__name__)


class RabbitMQAttendanceReporter(AttendanceReporter):
    def __init__(
        self,
        url: str,
        exchange: str,
        routing_key: str,
        queue: str | None = None,
        retries: int = 3,
        dead_letter_exchange: str = "",
        dead_letter_queue: str = "",
    ) -> None:
        try:
            import pika  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pika is not installed") from exc
        self._pika = pika
        self._url = url
        self._exchange = exchange
        self._routing_key = routing_key
        self._queue = queue
        self._dead_letter_exchange = dead_letter_exchange
        self._dead_letter_queue = dead_letter_queue
        self._retries = max(1, retries)
        self._lock = threading.Lock()
        self._connection: pika.BlockingConnection | None = None
        self._channel: pika.channel.Channel | None = None
        self._ensure_connected()

    # -- connection management ------------------------------------------------
    def _ensure_connected(self) -> None:
        if self._connection is not None and self._connection.is_open:
            return
        self._connection = self._pika.BlockingConnection(
            self._pika.URLParameters(self._url)
        )
        self._channel = self._connection.channel()
        self._channel.exchange_declare(
            exchange=self._exchange, exchange_type="topic", durable=True
        )
        # Without confirms, basic_publish only means "handed to the socket": a broker
        # that dies mid-flush loses the message and reports success. There is no local
        # outbox to replay from, so the acknowledgement is the only safety net.
        self._channel.confirm_delivery()

        arguments: dict[str, object] = {}
        if self._dead_letter_exchange:
            self._channel.exchange_declare(
                exchange=self._dead_letter_exchange, exchange_type="topic", durable=True
            )
            if self._dead_letter_queue:
                self._channel.queue_declare(queue=self._dead_letter_queue, durable=True)
                self._channel.queue_bind(
                    queue=self._dead_letter_queue,
                    exchange=self._dead_letter_exchange,
                    routing_key="#",
                )
            arguments["x-dead-letter-exchange"] = self._dead_letter_exchange

        if self._queue:
            # NOTE: an existing queue cannot gain x-dead-letter-exchange by
            # redeclaration -- the broker answers PRECONDITION_FAILED. A queue created
            # before this argument existed must be deleted once before first deploy.
            self._channel.queue_declare(
                queue=self._queue, durable=True, arguments=arguments or None
            )
            self._channel.queue_bind(
                queue=self._queue, exchange=self._exchange, routing_key=self._routing_key
            )

    def _teardown(self) -> None:
        try:
            if self._channel is not None and self._channel.is_open:
                self._channel.close()
        except Exception:  # noqa: BLE001  - best-effort cleanup
            pass
        try:
            if self._connection is not None and self._connection.is_open:
                self._connection.close()
        except Exception:  # noqa: BLE001
            pass
        self._channel = None
        self._connection = None

    # -- reporter contract ------------------------------------------------------
    def report(self, event: AttendanceEvent) -> ReportResult:
        body = json.dumps(event.as_dict()).encode("utf-8")
        properties = self._pika.BasicProperties(
            content_type="application/json",
            delivery_mode=2,  # persistent message
            message_id=f"{event.camera_id}-{event.employee_code}-{event.timestamp.isoformat()}",
        )
        for attempt in range(self._retries):
            try:
                with self._lock:
                    self._ensure_connected()
                    assert self._channel is not None
                    self._channel.basic_publish(
                        exchange=self._exchange,
                        routing_key=self._routing_key,
                        body=body,
                        properties=properties,
                    )
                return ReportResult(success=True, detail="published")
            except Exception as exc:  # noqa: BLE001 - transport can fail for many reasons
                log.warning(
                    "MQ publish failed (attempt %s/%s): %s",
                    attempt + 1,
                    self._retries,
                    exc,
                )
                with self._lock:
                    self._teardown()
                if attempt < self._retries - 1:
                    time.sleep(0.25 * (attempt + 1))
        return ReportResult(success=False, detail="publish failed after retries")

    def close(self) -> None:
        with self._lock:
            self._teardown()
