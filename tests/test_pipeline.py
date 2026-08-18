"""Recognition pipeline tests (detect -> quality -> liveness -> embed -> search)."""

from __future__ import annotations

import numpy as np

from app.ai.components import AIComponents
from app.ai.faiss.index import FaceIndex
from app.ai.gesture.wave import WaveTracker
from app.ai.liveness.base import LivenessChecker
from app.workers.recognition_loop.pipeline import RecognitionPipeline
from tests.fakes import FakeDetector, FakeHandDetector, FakeRecognizer, PermissiveQuality, face_at

E1 = np.full(8, 0.5, dtype="float32")
E2 = np.full(8, -0.5, dtype="float32")


def build_pipeline(
    index: FaceIndex | None = None,
    liveness: LivenessChecker | None = None,
) -> tuple[RecognitionPipeline, FakeDetector]:
    detector = FakeDetector()
    ai = AIComponents(
        detector=detector,
        recognizer=FakeRecognizer(E1, E2),
        liveness=liveness or _live(),
        quality=PermissiveQuality(),
        hand_detector=FakeHandDetector(),  # type: ignore[arg-type]
        wave_tracker=WaveTracker(),
    )
    idx = index or FaceIndex(dim=8)
    if index is None:
        idx.add(E1, "EMP1")
    return RecognitionPipeline(ai, idx, recognition_threshold=0.8), detector


class _AlwaysLive(LivenessChecker):
    def is_live(self, image_bgr, face):  # type: ignore[no-untyped-def]
        return True, 0.99


def _live() -> LivenessChecker:
    return _AlwaysLive()


def test_matches_known_employee() -> None:
    pipeline, detector = build_pipeline()
    detector.scenes = [[face_at(20, 20, 120, 140, 50)]]  # left-eye x=50 < 300 -> E1
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    events = pipeline.process_frame(frame)
    assert len(events) == 1
    ev = events[0]
    assert ev.quality_passed and ev.live
    assert ev.employee_code == "EMP1"
    assert ev.confidence > 0.8
    assert ev.embedding is not None


def test_unknown_face_when_below_threshold() -> None:
    pipeline, detector = build_pipeline()
    detector.scenes = [[face_at(200, 30, 300, 150, 350)]]  # right-eye x=350 -> E2
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    events = pipeline.process_frame(frame)
    assert events[0].employee_code is None
    assert events[0].best_score < 0.8


def test_liveness_failure_drops_face() -> None:
    class FakeDead(LivenessChecker):
        def is_live(self, image_bgr, face):  # type: ignore[no-untyped-def]
            return False, 0.1

    pipeline, detector = build_pipeline(liveness=FakeDead())
    detector.scenes = [[face_at(20, 20, 120, 140, 50)]]
    events = pipeline.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))
    assert events[0].live is False
    assert events[0].employee_code is None


def test_quality_rejection_never_embeds() -> None:
    class RejectQuality(PermissiveQuality):
        def assess(self, image, face, *, face_count: int = 1):
            from app.ai.quality.quality import QualityReport

            return QualityReport(passed=False, reasons=["blurry"])

    detector = FakeDetector()
    ai = AIComponents(
        detector=detector,
        recognizer=FakeRecognizer(E1, E2),
        liveness=_live(),
        quality=RejectQuality(),
        hand_detector=FakeHandDetector(),  # type: ignore[arg-type]
        wave_tracker=WaveTracker(),
    )
    idx = FaceIndex(dim=8)
    idx.add(E1, "EMP1")
    pipeline = RecognitionPipeline(ai, idx, recognition_threshold=0.8)
    detector.scenes = [[face_at(20, 20, 120, 140, 50)]]

    events = pipeline.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))
    assert events[0].quality_passed is False
    assert events[0].embedding is None
