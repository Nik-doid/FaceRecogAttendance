"""Structured JSON logging.

Every log record is emitted as a single JSON object on stdout so it can be shipped
straight to CloudWatch, Loki, or any JSON log aggregator without a sidecar parser.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_CONFIGURED = False

# Everything logging puts on a record itself. Anything else came from an ``extra=``
# and belongs in the JSON payload -- carrying them all beats an allowlist, which
# silently swallowed the numbers you turn DEBUG on to read.
_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}

# Third-party loggers that report routine startup detail at INFO. faiss announces
# every CPU extension it could not load; onnxruntime narrates provider selection.
_NOISY_LOGGERS = ("faiss", "faiss.loader", "onnxruntime", "insightface", "matplotlib")

# OpenCV's videoio chatter CANNOT be silenced from here, or from anywhere in Python:
# the cv2 DLL snapshots its environment when the process starts, so neither
# ``os.environ`` nor Win32 ``SetEnvironmentVariableW`` reaches it, and its FFmpeg
# backend is a separate plugin with its own logging state that
# ``cv2.utils.logging.setLogLevel`` does not touch. Export OPENCV_LOG_LEVEL=ERROR and
# OPENCV_FFMPEG_LOGLEVEL=8 in the real environment before launching -- see README and
# docker-compose.yml.


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED and value is not None:
                payload[key] = value
        return json.dumps(payload, default=str)


def quiet_native_libraries() -> None:
    """Silence the C/C++ loggers that write to stderr without going through logging.

    Python log levels cannot reach these: OpenCV and onnxruntime each carry their own
    native logger. Both are chatty about conditions this service already handles and
    reports itself -- a camera that will not open is answered with an error message on
    the socket, not left to a stderr warning nobody is reading.

    This reaches the main cv2 module and onnxruntime, whose levels are settable at
    runtime. It does NOT reach OpenCV's videoio FFmpeg plugin -- that one needs
    environment variables set before the process starts; see the note above.

    Imports are local and optional so a control-plane-only install still starts.
    """
    try:
        import cv2  # noqa: PLC0415

        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except (ImportError, AttributeError):  # pragma: no cover - depends on the build
        pass
    try:
        import onnxruntime  # noqa: PLC0415

        onnxruntime.set_default_logger_severity(3)  # 3 = ERROR
    except (ImportError, AttributeError):  # pragma: no cover - optional ai extra
        pass


def setup_logging(
    level: str = "INFO", json_output: bool = True, quiet_native: bool = True
) -> None:
    """Configure root logging exactly once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    root.addHandler(handler)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    if quiet_native:
        quiet_native_libraries()
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Module-level logger factory; guarantees logging is configured first."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
