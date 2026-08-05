"""Thread-safe holder for the most recent camera frame (as JPEG bytes).

The recognition loop publishes the latest decoded frame here (throttled); the
debug camera endpoints read it on demand. Keeping the stored form as JPEG bytes
means the loop only pays the encode cost on a timer, not for every consumer, and
the bytes can be handed straight to a browser or to ``st.image``.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

DEFAULT_INTERVAL = 0.15  # ~6 fps ceiling for the debug feed


class FrameBuffer:
    def __init__(self, interval: float = DEFAULT_INTERVAL) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._updated = 0.0
        self._interval = interval

    def publish(self, frame_bgr: np.ndarray) -> None:
        """Encode and store the frame if the throttle interval has elapsed."""
        now = time.monotonic()
        with self._lock:
            if now - self._updated < self._interval:
                return
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        with self._lock:
            self._jpeg = encoded.tobytes()
            self._updated = now

    def latest(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def clear(self) -> None:
        with self._lock:
            self._jpeg = None
            self._updated = 0.0
