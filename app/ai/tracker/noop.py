"""Tracker that assigns a unique id per detection per frame (no temporal logic)."""

from __future__ import annotations

import itertools

from app.ai.tracker.base import Tracker
from app.ai.types import DetectedFace, TrackedFace


class NoopTracker(Tracker):
    def __init__(self) -> None:
        self._ids = itertools.count(1)

    def update(self, faces: list[DetectedFace]) -> list[TrackedFace]:
        return [TrackedFace(face=face, track_id=next(self._ids)) for face in faces]
