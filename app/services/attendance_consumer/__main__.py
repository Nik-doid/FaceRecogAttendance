"""Run the consumer as its own process:  python -m app.services.attendance_consumer

Available so the consumer can be split into a separate container without a rewrite --
it imports no FastAPI and no container. In-process on a daemon thread is the default,
which suits a single small box; move it out when MySQL and the camera should fail
independently.
"""

from __future__ import annotations

import signal
from types import FrameType

from app.config.settings import settings
from app.core.logging import get_logger, setup_logging
from app.services.attendance_consumer.factory import build_attendance_consumer


def main() -> None:
    setup_logging(settings.log_level, quiet_native=settings.quiet_native_logs)
    log = get_logger(__name__)
    consumer = build_attendance_consumer(settings)

    def handle_signal(signum: int, frame: FrameType | None) -> None:
        log.info("attendance consumer received a signal; stopping", extra={"signal": signum})
        consumer.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    consumer.run_forever()


if __name__ == "__main__":
    main()
