"""Draw bounding boxes + labels onto frames for the live preview feed.

The recognition loop runs expensive detection only on frames that pass the frame-skip
policy, so the annotated copy is built from that frame's events and handed to the
frame buffer for the debug endpoints. The original frame is never mutated.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.workers.recognition_loop.pipeline import FaceEvent

RECOGNIZED_COLOR = (0, 200, 0)  # BGR green
UNKNOWN_COLOR = (0, 0, 255)  # BGR red
LOW_QUALITY_COLOR = (128, 128, 128)  # BGR gray
SPOOF_COLOR = (0, 165, 255)  # BGR orange


def annotate_frame(frame_bgr: np.ndarray, events: list[FaceEvent]) -> np.ndarray:
    """Return a copy of ``frame_bgr`` with one box + label per face event."""
    annotated = frame_bgr.copy()
    for ev in events:
        x1, y1, x2, y2 = (int(v) for v in ev.face.bbox)
        if ev.employee_code:
            label, color = ev.employee_code, RECOGNIZED_COLOR
        elif not ev.quality_passed:
            label, color = "quality", LOW_QUALITY_COLOR
        elif not ev.live:
            label, color = "spoof", SPOOF_COLOR
        else:
            label, color = "unknown", UNKNOWN_COLOR

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        pad = 4
        top_gap = text_h + 2 * pad
        above = y1 - top_gap - 4
        label_top = above if above >= 0 else y2 + 4
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (x1 + text_w + 2 * pad, label_top + top_gap),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + pad, label_top + text_h + pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated
