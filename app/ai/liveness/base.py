"""Liveness / anti-spoofing interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.ai.types import DetectedFace


class LivenessChecker(ABC):
    """Decides whether a detected face is a live person vs. a photo/video spoof."""

    @abstractmethod
    def is_live(self, image_bgr: np.ndarray, face: DetectedFace) -> tuple[bool, float]:
        """Return ``(live, score)`` where score ~ probability of being live."""
