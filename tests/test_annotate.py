"""Tests for the live-feed frame annotation (bounding boxes + labels)."""

from __future__ import annotations

import numpy as np

from app.ai.types import DetectedFace
from app.workers.recognition_loop.annotate import annotate_frame
from app.workers.recognition_loop.pipeline import FaceEvent


def _face(x1: int, y1: int, x2: int, y2: int) -> DetectedFace:
    return DetectedFace(bbox=(float(x1), float(y1), float(x2), float(y2)), score=0.99)


def _event(
    *, employee_code: str | None, quality_passed: bool = True, live: bool = True
) -> FaceEvent:
    return FaceEvent(
        face=_face(10, 10, 60, 70),
        track_id=None,
        embedding=None,
        employee_code=employee_code,
        confidence=0.9 if employee_code else 0.0,
        best_score=0.9 if employee_code else 0.3,
        quality_passed=quality_passed,
        quality_reasons=[],
        live=live,
        liveness_score=0.9 if live else 0.1,
    )


def test_annotate_draws_recognized_label() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    out = annotate_frame(frame, [_event(employee_code="EMP1")])
    assert out.shape == frame.shape
    # The green box (0,200,0) must be present somewhere in the output.
    assert (out == np.array([0, 200, 0], dtype=np.uint8)).any()


def test_annotate_draws_unknown_label() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    out = annotate_frame(frame, [_event(employee_code=None)])
    assert (out == np.array([0, 0, 255], dtype=np.uint8)).any()


def test_annotate_no_events_returns_identical_copy() -> None:
    frame = np.full((50, 80, 3), 127, dtype=np.uint8)
    out = annotate_frame(frame, [])
    np.testing.assert_array_equal(out, frame)


def test_annotate_marks_quality_and_spoof() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    quality = annotate_frame(frame, [_event(employee_code=None, quality_passed=False)])
    assert (quality == np.array([128, 128, 128], dtype=np.uint8)).any()
    spoof = annotate_frame(frame, [_event(employee_code=None, live=False)])
    assert (spoof == np.array([0, 165, 255], dtype=np.uint8)).any()
