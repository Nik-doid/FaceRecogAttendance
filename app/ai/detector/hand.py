"""MediaPipe Hands detector for hand landmarks and gesture recognition.

Uses the MediaPipe Tasks API (v1.x). The hand_landmarker.task model file
must be present in ``models/`` at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@dataclass(frozen=True)
class HandLandmarks:
    """21 hand landmarks in normalised coordinates (0-1)."""

    points: np.ndarray  # shape (21, 2) float32
    handedness: str  # "Left" or "Right"
    score: float


class HandDetector:
    """MediaPipe HandLandmarker wrapper for real-time hand detection."""

    def __init__(
        self,
        *,
        model_path: str | Path = "models/hand_landmarker.task",
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. "
                "Download from https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = HandLandmarker.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> list[HandLandmarks]:
        """Detect hands in a BGR image. Returns list of HandLandmarks."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)

        result = self._detector.detect_for_video(mp_image, timestamp_ms=0)

        hands: list[HandLandmarks] = []
        if result.landmarks and result.handednesses:
            for idx, lm_list in enumerate(result.landmarks):
                points = np.array(
                    [[lm.x, lm.y] for lm in lm_list],
                    dtype="float32",
                )
                handedness = "Right"
                if idx < len(result.handednesses) and result.handednesses[idx]:
                    handedness = result.handednesses[idx][0].category_name or "Right"
                hands.append(HandLandmarks(points=points, handedness=handedness, score=1.0))

        return hands

    def close(self) -> None:
        self._detector.close()
