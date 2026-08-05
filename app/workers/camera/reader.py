"""OpenCV camera reader with resize and connection state tracking.

Supports two sources:
- ``rtsp``: an RTSP/IP camera stream opened with FFmpeg (production default).
- ``device``: a local webcam opened by device index (0 = first camera) for
  local testing, using DirectShow on Windows.

The loop drives reconnection (backoff + metrics) and calls this class for raw
frames only. A failed ``read()`` marks the reader disconnected; the loop then
reopens it.
"""

from __future__ import annotations

import sys
import threading

import cv2
import numpy as np

_BACKEND_DEVICE = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY


class CameraReader:
    def __init__(
        self,
        rtsp_url: str = "",
        *,
        source: str = "rtsp",
        device_index: int = 0,
        max_width: int = 1280,
    ) -> None:
        self._url = rtsp_url
        self._source = source
        self._device_index = device_index
        self._max_width = max_width
        self._lock = threading.Lock()
        self._cap: cv2.VideoCapture | None = None
        self.connected = False

    def open(self) -> bool:
        """Open the camera stream. Returns False if it cannot connect."""
        with self._lock:
            if self._source == "device":
                cap = cv2.VideoCapture(self._device_index, _BACKEND_DEVICE)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            else:
                cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            if not cap.isOpened():
                cap.release()
                self.connected = False
                return False
            self._cap = cap
            self.connected = True
            return True

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame (resized if wider than max_width) or None."""
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                self.connected = False
                return None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.connected = False
                return None
            return self._resize(frame)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._max_width and w > self._max_width:
            scale = self._max_width / float(w)
            frame = cv2.resize(frame, (self._max_width, int(round(h * scale))))
        return frame

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self.connected = False
