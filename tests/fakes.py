"""Shared test doubles for the AI pipeline.

These let unit/integration tests exercise the full worker pipeline (detect ->
quality -> liveness -> embed -> search -> report) without downloading ONNX models.
The real quality checker is exercised separately in test_quality.py.
"""

from __future__ import annotations

import numpy as np

from app.ai.components import AIComponents
from app.ai.liveness.base import LivenessChecker
from app.ai.quality.quality import FaceQualityChecker, QualityReport
from app.ai.recognizer.base import Recognizer
from app.ai.types import DetectedFace


class FakeDetector:
    """Returns a scripted list of faces, one 'scene' per call.

    When no scenes are configured and ``default_face`` is set, a centered face is
    returned for every image (useful for index-build tests where we don't care about
    detection itself).
    """

    def __init__(
        self,
        scenes: list[list[DetectedFace]] | None = None,
        default_face: DetectedFace | None = None,
    ) -> None:
        self.scenes = scenes or []
        self.default_face = default_face
        self.calls = 0

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        if self.calls < len(self.scenes):
            faces = self.scenes[self.calls]
        elif self.scenes:
            faces = self.scenes[-1]
        elif self.default_face is not None:
            faces = [self.default_face]
        else:
            faces = []
        self.calls += 1
        return list(faces)


class FakeRecognizer(Recognizer):
    """Returns one of two embeddings depending on the face's x-position in kps."""

    def __init__(self, left_embedding: np.ndarray, right_embedding: np.ndarray) -> None:
        self.left = left_embedding
        self.right = right_embedding
        self.calls: list[np.ndarray] = []

    def embed(self, image_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        self.calls.append(kps)
        if kps[0][0] < 300:
            return self.left
        return self.right


class FakeLiveness(LivenessChecker):
    def __init__(self, live: bool = True, score: float = 0.99) -> None:
        self._live = live
        self._score = score

    def is_live(self, image_bgr: np.ndarray, face: DetectedFace) -> tuple[bool, float]:
        return self._live, self._score


class PermissiveQuality(FaceQualityChecker):
    """Quality gate that accepts every face (for pipeline tests)."""

    def __init__(self) -> None:
        super().__init__(minimum_face_size=1, blur_threshold=0, min_lighting=0, max_lighting=255)

    def assess(self, image, face, *, face_count: int = 1) -> QualityReport:
        return QualityReport(passed=True)


def face_at(x1: int, y1: int, x2: int, y2: int, eye_x: float) -> DetectedFace:
    """Build a DetectedFace whose left eye is at ``eye_x`` (used by FakeRecognizer)."""
    kps = np.array(
        [
            [eye_x, (y1 + y2) / 2.0],
            [x2 - 10.0, (y1 + y2) / 2.0],
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0 + 10],
            [x1 + 10, y2 - 10],
            [x2 - 10, y2 - 10],
        ],
        dtype="float32",
    )
    return DetectedFace(bbox=(float(x1), float(y1), float(x2), float(y2)), score=0.99, kps=kps)


def build_ai(embedding_left: np.ndarray, embedding_right: np.ndarray) -> AIComponents:
    return AIComponents(
        detector=FakeDetector(),
        recognizer=FakeRecognizer(embedding_left, embedding_right),
        liveness=FakeLiveness(),
        quality=PermissiveQuality(),
    )
