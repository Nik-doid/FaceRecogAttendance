"""ByteTrack wrapper (optional).

``bytetracker`` is NOT a declared dependency: it pulls ``lap`` which needs a C
compiler on some platforms. It is only used when ``TRACKING_ENABLED=true`` and the
package is present; otherwise a clear error explains how to enable it.

Mapping ByteTrack's output back to our detections uses greedy IoU matching because
ByteTrack does not return per-input ids in a stable order.
"""

from __future__ import annotations

import numpy as np

from app.ai._loader import try_import
from app.ai.tracker.base import Tracker
from app.ai.types import DetectedFace, TrackedFace


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return float(inter / (area_a + area_b - inter))


class ByteTrackTracker(Tracker):
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ) -> None:
        bytetracker = try_import("bytetracker")
        if bytetracker is None:
            raise RuntimeError(
                "bytetracker is not installed. Install it with: uv pip install bytetracker "
                "(note: it requires a C toolchain on Windows for the 'lap' dependency)."
            )
        self._tracker = bytetracker.BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
        )

    def update(self, faces: list[DetectedFace]) -> list[TrackedFace]:
        if not faces:
            self._tracker.update(np.empty((0, 6), dtype="float32"), frame_size=(1080, 1920))
            return []
        dets = np.array(
            [[f.bbox[0], f.bbox[1], f.bbox[2], f.bbox[3], f.score, 0.0] for f in faces],
            dtype="float32",
        )
        tracks = self._tracker.update(dets, frame_size=(1080, 1920))
        tracked: list[TrackedFace] = []
        for track in tracks:
            tlbr = track.tlbr  # x1, y1, x2, y2
            best_idx, best_iou = -1, 0.0
            for i, face in enumerate(faces):
                iou = _iou(
                    np.asarray(face.bbox, dtype="float32"),
                    np.asarray(tlbr, dtype="float32"),
                )
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx >= 0:
                tracked.append(TrackedFace(face=faces[best_idx], track_id=int(track.track_id)))
        return tracked
