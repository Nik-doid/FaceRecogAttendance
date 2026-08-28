"""The single slot the camera runner publishes into and viewers read from.

Frames arrive from the runner at roughly 30/s and detections at roughly 1/s, so the
two are tracked separately: a viewer that has already seen the current detection
should still receive new video, and a slow detection should not hold up the stream.

Deliberately a latest-value slot rather than a queue or a subscriber registry. There
is nothing useful about a stale frame, so there is nothing to buffer, and with no
registry there is no fan-out bookkeeping and no way for one slow client to apply
backpressure to the runner -- it simply misses frames and picks up the newest one.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HubSnapshot:
    """What the hub held at one instant."""

    frame_seq: int
    detection_seq: int
    jpeg: bytes | None
    detection: dict[str, Any] | None


class FrameHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._detection: dict[str, Any] | None = None
        self._frame_seq = 0
        self._detection_seq = 0

    def publish_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._frame_seq += 1

    def publish_detection(self, detection: dict[str, Any]) -> None:
        with self._lock:
            self._detection = detection
            self._detection_seq += 1

    def snapshot(self) -> HubSnapshot:
        with self._lock:
            return HubSnapshot(
                frame_seq=self._frame_seq,
                detection_seq=self._detection_seq,
                jpeg=self._jpeg,
                detection=self._detection,
            )

    def clear(self) -> None:
        """Drop the held frame so a stopped camera stops serving a frozen image."""
        with self._lock:
            self._jpeg = None
            self._detection = None
            self._frame_seq += 1
            self._detection_seq += 1
