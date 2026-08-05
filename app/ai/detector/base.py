"""Detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.ai.types import DetectedFace


class Detector(ABC):
    """Detects faces in a BGR frame. Thread-safe per loaded model."""

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Return zero or more detected faces, sorted by detection score."""
