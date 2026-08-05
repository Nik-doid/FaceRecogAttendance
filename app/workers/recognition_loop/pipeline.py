"""Per-frame recognition processing (stateless wrt. reporting).

Turns one camera frame into a list of ``FaceEvent``s. All the expensive work
(detect -> quality -> liveness -> embed -> search) happens here; reporting and
persistence belong to the loop, keeping this class trivially testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.ai.components import AIComponents
from app.ai.faiss.index import FaceIndex
from app.ai.types import DetectedFace
from app.core.metrics import FACES_DETECTED, QUALITY_REJECTED


@dataclass
class FaceEvent:
    """Everything the loop needs to decide what to do with one face."""

    face: DetectedFace
    track_id: int | None
    embedding: np.ndarray | None
    employee_code: str | None
    confidence: float
    best_score: float
    quality_passed: bool
    quality_reasons: list[str]
    live: bool
    liveness_score: float


class RecognitionPipeline:
    def __init__(
        self,
        ai: AIComponents,
        face_index: FaceIndex,
        *,
        recognition_threshold: float,
    ) -> None:
        self._ai = ai
        self._index = face_index
        # Public so the loop can apply runtime (DB) tuning without rebuilding.
        self.recognition_threshold = recognition_threshold

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        """Run the detector and update the detection metric."""
        detections = self._ai.detector.detect(frame_bgr)
        FACES_DETECTED.inc(len(detections))
        return detections

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        tracking: Sequence[tuple[int | None, DetectedFace]] | None = None,
    ) -> list[FaceEvent]:
        """Run the full pipeline on one frame and return events per face.

        ``tracking`` is an optional list of ``(track_id, face)`` pairs produced by a
        tracker after calling :meth:`detect`. When None the raw detections are used
        with no track ids.
        """
        if tracking is None:
            detections = self.detect(frame_bgr)
            faces: Sequence[tuple[int | None, DetectedFace]] = [
                (None, f) for f in detections
            ]
        else:
            faces = tracking

        events: list[FaceEvent] = []
        for track_id, face in faces:
            quality = self._ai.quality.assess(
                frame_bgr, face, face_count=len(faces)
            )
            if not quality.passed:
                for reason in quality.reasons:
                    QUALITY_REJECTED.labels(reason=reason).inc()
                events.append(
                    FaceEvent(
                        face=face,
                        track_id=track_id,
                        embedding=None,
                        employee_code=None,
                        confidence=0.0,
                        best_score=0.0,
                        quality_passed=False,
                        quality_reasons=quality.reasons,
                        live=False,
                        liveness_score=0.0,
                    )
                )
                continue

            live, liveness_score = self._ai.liveness.is_live(frame_bgr, face)
            if not live:
                events.append(
                    FaceEvent(
                        face=face,
                        track_id=track_id,
                        embedding=None,
                        employee_code=None,
                        confidence=0.0,
                        best_score=0.0,
                        quality_passed=True,
                        quality_reasons=[],
                        live=False,
                        liveness_score=liveness_score,
                    )
                )
                continue

            if face.kps is None:
                events.append(
                    FaceEvent(
                        face=face,
                        track_id=track_id,
                        embedding=None,
                        employee_code=None,
                        confidence=0.0,
                        best_score=0.0,
                        quality_passed=True,
                        quality_reasons=[],
                        live=True,
                        liveness_score=liveness_score,
                    )
                )
                continue

            embedding = self._ai.recognizer.embed(frame_bgr, face.kps)
            matches = self._index.search(embedding, k=1)
            best = matches[0] if matches else None

            if best is None or best.score < self.recognition_threshold:
                events.append(
                    FaceEvent(
                        face=face,
                        track_id=track_id,
                        embedding=embedding,
                        employee_code=None,
                        confidence=0.0,
                        best_score=best.score if best else 0.0,
                        quality_passed=True,
                        quality_reasons=[],
                        live=True,
                        liveness_score=liveness_score,
                    )
                )
                continue

            events.append(
                FaceEvent(
                    face=face,
                    track_id=track_id,
                    embedding=embedding,
                    employee_code=best.employee_code,
                    confidence=best.score,
                    best_score=best.score,
                    quality_passed=True,
                    quality_reasons=[],
                    live=True,
                    liveness_score=liveness_score,
                )
            )

        return events
