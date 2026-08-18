"""Draw bounding boxes + labels + hand landmarks onto frames for the live preview feed.

The recognition loop runs expensive detection only on frames that pass the frame-skip
policy, so the annotated copy is built from that frame's events and handed to the
frame buffer for the debug endpoints. The original frame is never mutated.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.detector.hand import HAND_CONNECTIONS, HandLandmarks
from app.workers.recognition_loop.pipeline import FaceEvent

RECOGNIZED_COLOR = (0, 200, 0)  # BGR green
UNKNOWN_COLOR = (0, 0, 255)  # BGR red
LOW_QUALITY_COLOR = (128, 128, 128)  # BGR gray
SPOOF_COLOR = (0, 165, 255)  # BGR orange
HAND_COLOR = (0, 255, 255)  # BGR yellow
HAND_POINT_COLOR = (0, 255, 0)  # BGR green
HAND_CONNECTION_COLOR = (0, 255, 255)  # BGR yellow
ENGAGEMENT_COLOR = (255, 255, 0)  # BGR cyan


def annotate_frame(
    frame_bgr: np.ndarray,
    events: list[FaceEvent],
    hands: list[HandLandmarks] | None = None,
    engagement_confirmed: dict[int, bool] | None = None,
    looking_frames: dict[int, int] | None = None,
    wave_detected: dict[int, bool] | None = None,
    required_frames: int = 60,
) -> np.ndarray:
    """Return a copy of ``frame_bgr`` with boxes, labels, and hand landmarks."""
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    # Draw hand landmarks and connections
    if hands:
        for hand in hands:
            _draw_hand_landmarks(annotated, hand, w, h)

    # Draw face boxes and labels
    for ev in events:
        x1, y1, x2, y2 = (int(v) for v in ev.face.bbox)
        track_id = ev.track_id

        # Determine engagement status (track_id can be None)
        engaged = (
            engagement_confirmed.get(track_id, False)
            if engagement_confirmed and track_id is not None
            else False
        )

        # Determine label and color
        if ev.employee_code:
            if engaged:
                label = f"{ev.employee_code} ✓"
                color = RECOGNIZED_COLOR
            else:
                # Show engagement progress instead of name
                looking_val = (
                    looking_frames.get(track_id, 0)
                    if looking_frames and track_id is not None
                    else 0
                )
                progress = min(looking_val / 60 * 100, 100)
                wave_icon = (
                    "w"
                    if wave_detected and track_id is not None and wave_detected.get(track_id, False)
                    else "..."
                )
                label = f"{ev.employee_code} {progress:.0f}% {wave_icon}"
                color = ENGAGEMENT_COLOR
        elif not ev.quality_passed:
            label, color = "quality", (128, 128, 128)
        elif not ev.live:
            label, color = "spoof", (0, 165, 255)
        else:
            label, color = "unknown", (0, 0, 255)

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

        # Show engagement progress bar below box
        if ev.employee_code and not engaged and looking_frames and track_id is not None:
            looking_f = looking_frames.get(track_id, 0)
            progress = min(looking_f / 60, 1.0)
            bar_w = int((x2 - x1) * progress)
            cv2.rectangle(annotated, (x1, y2 + 4), (x1 + bar_w, y2 + 10), (0, 255, 255), -1)
            cv2.rectangle(annotated, (x1, y2 + 4), (x2, y2 + 10), (255, 255, 255), 1)

    return annotated


def _draw_hand_landmarks(image: np.ndarray, hand: HandLandmarks, w: int, h: int) -> None:
    """Draw MediaPipe hand landmarks and connections on the image."""
    points_px = (hand.points * np.array([w, h])).astype(int)

    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        start_pt = tuple(points_px[start_idx])
        end_pt = tuple(points_px[end_idx])
        cv2.line(image, start_pt, end_pt, HAND_CONNECTION_COLOR, 2)

    # Draw landmarks
    for pt in points_px:
        cv2.circle(image, tuple(pt), 4, HAND_POINT_COLOR, -1)

    # Draw handedness label at wrist
    if len(points_px) > 0:
        wrist = tuple(points_px[0])
        cv2.putText(
            image,
            hand.handedness,
            (wrist[0] + 10, wrist[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )