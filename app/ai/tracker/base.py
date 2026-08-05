"""Tracker interface and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.ai.types import DetectedFace, TrackedFace


class Tracker(ABC):
    """Assigns stable track ids to detections across consecutive frames.

    Tracking is optional (ByteTrack). When disabled, ``NoopTracker`` assigns a
    monotonically increasing id per frame, which still satisfies the interface.
    """

    @abstractmethod
    def update(self, faces: list[DetectedFace]) -> list[TrackedFace]:
        """Associate detections with track ids for the current frame."""
