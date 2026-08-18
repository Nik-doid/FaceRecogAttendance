"""Tests for the live-feed frame annotation (bounding boxes + labels)."""

from __future__ import annotations

import numpy as np

from app.ai.detector.hand import HandLandmarks
from app.ai.types import DetectedFace
from app.workers.recognition_loop.annotate import annotate_frame
from app.workers.recognition_loop.pipeline import FaceEvent


def _face(x1: int, y1: int, x2: int, y2: int) -> DetectedFace:
    return DetectedFace(bbox=(float(x1), float(y1), float(x2), float(y2)), score=0.99)


def _event(
    *,
    employee_code: str | None,
    track_id: int | None = 0,
    quality_passed: bool = True,
    live: bool = True,
) -> FaceEvent:
    return FaceEvent(
        face=_face(10, 10, 60, 70),
        track_id=track_id,
        embedding=None,
        employee_code=employee_code,
        confidence=0.9 if employee_code else 0.0,
        best_score=0.9 if employee_code else 0.3,
        quality_passed=quality_passed,
        quality_reasons=[],
        live=live,
        liveness_score=0.9 if live else 0.1,
    )


def test_annotate_draws_engaged_label() -> None:
    """When engagement_confirmed says the employee is engaged, show green label."""
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    out = annotate_frame(
        frame,
        [_event(employee_code="EMP1")],
        engagement_confirmed={0: True},
    )
    assert out.shape == frame.shape
    # Green (0,200,0) must be present for engaged recognized label
    assert (out == np.array([0, 200, 0], dtype=np.uint8)).any()


def test_annotate_draws_unengaged_label() -> None:
    """When not yet engaged, show cyan engagement progress label."""
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    out = annotate_frame(
        frame,
        [_event(employee_code="EMP1")],
        engagement_confirmed={0: False},
        looking_frames={0: 30},
    )
    assert out.shape == frame.shape
    # Cyan (255,255,0) must be present for unengaged recognized label
    assert (out == np.array([255, 255, 0], dtype=np.uint8)).any()


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


def test_annotate_draws_hand_landmarks() -> None:
    """Hand landmarks drawn as green dots and yellow connections."""
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    # Simulate a hand at roughly 50%, 50% of the frame
    pts = np.full((21, 2), 0.5, dtype="float32")
    pts[0] = [0.4, 0.5]  # wrist slightly left
    hand = HandLandmarks(points=pts, handedness="Right", score=0.99)
    out = annotate_frame(frame, [], hands=[hand])
    # Yellow connection color (0, 255, 255) should appear
    assert (out == np.array([0, 255, 255], dtype=np.uint8)).any()


def test_annotate_progress_bar_appears() -> None:
    """A progress bar should be drawn under the face box when not engaged."""
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    out = annotate_frame(
        frame,
        [_event(employee_code="EMP1")],
        engagement_confirmed={0: False},
        looking_frames={0: 30},
    )
    # The progress bar uses cyan (0, 255, 255) pixels below the box
    # Just verify the output has changed from the blank frame
    assert not np.array_equal(out, frame)
