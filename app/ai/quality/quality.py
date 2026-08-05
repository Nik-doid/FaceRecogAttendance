"""Face quality checks: blur, size, angle, lighting, multiple faces.

Heuristic gates applied BEFORE recognition so garbage input never reaches the
expensive embedding step. Thresholds are deliberately conservative; a face that
fails quality is skipped for that frame and will almost certainly be picked up on
a later frame, so false rejects here are cheap.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.ai.types import DetectedFace, QualityReport

# Laplacian variance below this => blurry (in px^2 of the Laplacian response).
DEFAULT_BLUR_THRESHOLD = 100.0
# Mean pixel luminance outside [min, max] => too dark / overexposed.
DEFAULT_MIN_LIGHTING = 40.0
DEFAULT_MAX_LIGHTING = 220.0
# Absolute head roll (degrees) beyond which ArcFace alignment degrades.
DEFAULT_MAX_ROLL_DEG = 30.0


def _roll_degrees(kps: np.ndarray) -> float | None:
    """Estimate head roll from the two eye landmarks, in degrees."""
    if kps is None or kps.shape[0] < 2:
        return None
    left_eye, right_eye = kps[0], kps[1]
    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    if abs(dx) < 1e-6:
        return 90.0
    return math.degrees(math.atan2(dy, dx))


def estimate_blur(crop_gray: np.ndarray) -> float:
    """Variance of the Laplacian; lower = blurrier."""
    return float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())


class FaceQualityChecker:
    """Composable quality gate used both at index-build time and in the live pipeline."""

    def __init__(
        self,
        minimum_face_size: int,
        *,
        blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
        min_lighting: float = DEFAULT_MIN_LIGHTING,
        max_lighting: float = DEFAULT_MAX_LIGHTING,
        max_roll_deg: float = DEFAULT_MAX_ROLL_DEG,
    ) -> None:
        self._min_size = minimum_face_size
        self._blur_threshold = blur_threshold
        self._min_lighting = min_lighting
        self._max_lighting = max_lighting
        self._max_roll_deg = max_roll_deg

    def assess(
        self,
        image: np.ndarray,
        face: DetectedFace,
        *,
        face_count: int = 1,
    ) -> QualityReport:
        """Evaluate all gates against a single detected face.

        ``image`` is the full BGR frame; the face crop is taken from it.
        """
        report = QualityReport(passed=True)
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if face.width < self._min_size or face.height < self._min_size:
            report.passed = False
            report.reasons.append("face_too_small")
            report.scores["size"] = min(face.width, face.height)

        if face_count > 1:
            report.passed = False
            report.reasons.append("multiple_faces")

        if x2 <= x1 or y2 <= y1:
            report.passed = False
            report.reasons.append("invalid_bbox")
            return report

        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

        blur = estimate_blur(gray)
        report.scores["blur"] = blur
        if blur < self._blur_threshold:
            report.passed = False
            report.reasons.append("blurry")

        lighting = float(gray.mean())
        report.scores["lighting"] = lighting
        if lighting < self._min_lighting or lighting > self._max_lighting:
            report.passed = False
            report.reasons.append("poor_lighting")

        roll = _roll_degrees(face.kps) if face.kps is not None else None
        if roll is not None:
            report.scores["roll"] = roll
            if abs(roll) > self._max_roll_deg:
                report.passed = False
                report.reasons.append("bad_angle")

        return report
