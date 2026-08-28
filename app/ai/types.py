"""Shared domain types for the AI pipeline."""

from __future__ import annotations

from dataclasses import dataclass

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




@dataclass(frozen=True)
class IndexResult:
    """Nearest-neighbour result from the FAISS index."""

    employee_code: str
    score: float
    distance: float
