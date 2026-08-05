"""Shared domain types for the AI pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KP_SHAPE = (5, 2)


@dataclass(frozen=True)
class DetectedFace:
    """A face as produced by a detector: bounding box + optional landmarks.

    bbox is (x1, y1, x2, y2) in absolute pixel coordinates. kps is a (5, 2) array of
    [left eye, right eye, nose, mouth left, mouth right] when available (SCRFD provides
    these; they are required for ArcFace alignment).
    """

    bbox: tuple[float, float, float, float]
    score: float
    kps: np.ndarray | None = None

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class QualityReport:
    """Result of the quality gates: passed iff no gate flagged the face."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class TrackedFace:
    """A detection carrying an identity track id (from the tracker)."""

    face: DetectedFace
    track_id: int


@dataclass(frozen=True)
class IndexResult:
    """Nearest-neighbour result from the FAISS index."""

    employee_code: str
    score: float
    distance: float


@dataclass
class RecognitionResult:
    """Outcome of processing a single face through the pipeline."""

    face: DetectedFace
    track_id: int | None
    embedding: np.ndarray
    matched: bool
    employee_code: str | None = None
    confidence: float = 0.0
    quality: QualityReport | None = None
    liveness_score: float | None = None
