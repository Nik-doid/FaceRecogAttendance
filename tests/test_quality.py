"""Face quality gate tests against synthetic images."""

from __future__ import annotations

import numpy as np

from app.ai.quality.quality import FaceQualityChecker, estimate_blur
from app.ai.types import DetectedFace

CHECKER = FaceQualityChecker(minimum_face_size=20)


def _image_with_face(face_region: np.ndarray) -> tuple[np.ndarray, DetectedFace]:
    """Place a generated face region into a neutral 100x100 background."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    face = np.ascontiguousarray(face_region)
    h, w = face.shape[:2]
    img[10 : 10 + h, 10 : 10 + w] = face
    det = DetectedFace(bbox=(10.0, 10.0, 10.0 + w, 10.0 + h), score=0.99, kps=None)
    return img, det


def _sharp_face() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)


def test_sharp_face_passes() -> None:
    img, det = _image_with_face(_sharp_face())
    report = CHECKER.assess(img, det)
    assert report.passed is True
    assert report.scores["blur"] > CHECKER._blur_threshold  # noqa: SLF001


def test_blurry_face_rejected() -> None:
    img, det = _image_with_face(np.full((40, 40, 3), 128, dtype=np.uint8))
    report = CHECKER.assess(img, det)
    assert report.passed is False
    assert "blurry" in report.reasons


def test_face_too_small_rejected() -> None:
    img, det = _image_with_face(_sharp_face()[:10, :10])
    report = CHECKER.assess(img, det)
    assert report.passed is False
    assert "face_too_small" in report.reasons


def test_multiple_faces_allowed() -> None:
    """Multi-face frames are accepted (surveillance cameras see many faces)."""
    img, det = _image_with_face(_sharp_face())
    report = CHECKER.assess(img, det, face_count=3)
    assert report.passed is True
    assert "multiple_faces" not in report.reasons


def test_poor_lighting_rejected() -> None:
    img, det = _image_with_face(np.full((40, 40, 3), 2, dtype=np.uint8))
    report = CHECKER.assess(img, det)
    assert "poor_lighting" in report.reasons


def test_head_roll_rejected() -> None:
    img, det = _image_with_face(_sharp_face())
    kps = np.array([[20.0, 15.0], [20.0, 25.0], [30.0, 20.0], [40.0, 30.0], [40.0, 40.0]])
    tilted = DetectedFace(bbox=det.bbox, score=0.99, kps=kps)
    report = CHECKER.assess(img, tilted)
    assert "bad_angle" in report.reasons


def test_estimate_blur_distinguishes() -> None:
    sharp = np.random.default_rng(0).integers(0, 255, (30, 30), dtype=np.uint8)
    flat = np.full((30, 30), 128, dtype=np.uint8)
    assert estimate_blur(sharp) > estimate_blur(flat)
