"""``MIN_FACE_PIXELS``: a face too small to embed must not be reported as unknown.

At 27x35px ArcFace has a handful of pixels across each eye and the nose bridge, so it
scores 0.00 against every enrolled employee. Counting those as "unknown person" is
what makes an accuracy figure lie -- they are "too far from the camera", which is a
different problem with a different fix.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from app.schemas.face_processing import (
    FaceProcessConfig,
    FaceResult,
    FrameContext,
    PalmResult,
)
from app.services.face_recognition.process import FaceRecognitionProcess
from app.services.face_recognition.tracker import FaceTracker


class _Detector:
    def __init__(self, faces: list[FaceResult]) -> None:
        self._faces = faces

    async def detect(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        return list(self._faces)


class _Palm:
    async def detect(self, frame_bgr: np.ndarray, regions: object = None) -> list[PalmResult]:
        count = len(regions) if regions is not None else 1  # type: ignore[arg-type]
        return [PalmResult(detected=True, score=0.9) for _ in range(count)]


class _Recognizer:
    """Records what it was asked to embed; matches nothing, like an empty gallery."""

    def __init__(self) -> None:
        self.seen: list[FaceResult] = []
        self.code: str | None = None

    async def recognize(
        self, frame_bgr: np.ndarray, faces: list[FaceResult]
    ) -> list[FaceResult]:
        self.seen.extend(faces)
        if self.code is None:
            return faces
        return [
            face.model_copy(update={"employee_code": self.code, "confidence": 0.9})
            for face in faces
        ]


class _Sink:
    def __init__(self) -> None:
        self.marked: list[FaceResult] = []

    async def mark(self, faces: list[FaceResult], ctx: FrameContext) -> None:
        self.marked.extend(faces)


def _looking_face(width: float) -> FaceResult:
    """A frontal face ``width`` px wide, with landmarks the gaze gate accepts."""
    height = width * 1.25
    x1, y1 = 100.0, 100.0
    x2, y2 = x1 + width, y1 + height
    mid_y = y1 + height * 0.4
    return FaceResult(
        bbox=(x1, y1, x2, y2),
        score=0.9,
        kps=[
            (x1 + width * 0.3, mid_y),
            (x1 + width * 0.7, mid_y),
            (x1 + width * 0.5, mid_y + height * 0.15),
            (x1 + width * 0.35, y2 - height * 0.15),
            (x1 + width * 0.65, y2 - height * 0.15),
        ],
    )


def _process(faces: list[FaceResult], min_face_pixels: int) -> tuple[
    FaceRecognitionProcess, _Recognizer, _Sink
]:
    process = FaceRecognitionProcess.__new__(FaceRecognitionProcess)
    process.config = FaceProcessConfig(min_face_pixels=min_face_pixels)
    process.models_dir = Path(".")
    process.face_detection = _Detector(faces)  # type: ignore[assignment]
    process.palm_detection = _Palm()  # type: ignore[assignment]
    recognizer = _Recognizer()
    process.face_recognition = recognizer  # type: ignore[assignment]
    sink = _Sink()
    process.mark_attendance = sink  # type: ignore[assignment]
    return process, recognizer, sink


def test_a_face_under_the_floor_is_never_embedded() -> None:
    process, recognizer, _ = _process([_looking_face(30.0)], min_face_pixels=60)
    result = asyncio.run(process.process_frames(np.zeros((480, 640, 3), np.uint8)))
    assert recognizer.seen == []
    (face,) = result.faces
    assert face.too_small is True
    # Not "unknown": the gallery was never searched.
    assert face.employee_code is None
    assert face.confidence == 0.0


def test_a_face_over_the_floor_is_embedded() -> None:
    process, recognizer, _ = _process([_looking_face(90.0)], min_face_pixels=60)
    result = asyncio.run(process.process_frames(np.zeros((480, 640, 3), np.uint8)))
    assert len(recognizer.seen) == 1
    assert result.faces[0].too_small is False


def test_a_floor_of_zero_embeds_everything() -> None:
    process, recognizer, _ = _process([_looking_face(20.0)], min_face_pixels=0)
    asyncio.run(process.process_frames(np.zeros((480, 640, 3), np.uint8)))
    assert len(recognizer.seen) == 1


def test_a_big_face_is_still_recognised_beside_a_tiny_one() -> None:
    """One person at the desk must not be dropped because someone stood at the back."""
    small = _looking_face(30.0)
    big = _looking_face(90.0).model_copy(
        update={"bbox": (300.0, 100.0, 390.0, 212.5)}
    )
    process, recognizer, _ = _process([small, big], min_face_pixels=60)
    result = asyncio.run(process.process_frames(np.zeros((480, 640, 3), np.uint8)))
    assert len(recognizer.seen) == 1
    assert [face.too_small for face in result.faces] == [True, False]


@pytest.mark.parametrize("width", [59.0, 60.0, 61.0])
def test_the_floor_is_inclusive_of_the_configured_width(width: float) -> None:
    process, recognizer, _ = _process([_looking_face(width)], min_face_pixels=60)
    asyncio.run(process.process_frames(np.zeros((480, 640, 3), np.uint8)))
    assert len(recognizer.seen) == (0 if width < 60.0 else 1)


# --- the tracker gating publication ------------------------------------------
def test_a_confirming_tracker_holds_the_punch_until_scans_agree() -> None:
    """End to end through `process_frames`: one person, two scans, one event.

    Before tracking, every scan that cleared the threshold was its own attempt at the
    broker. This is the change that makes a punch a decision about a *person*.
    """
    face = _looking_face(90.0)
    process, recognizer, sink = _process([face], min_face_pixels=0)
    process._tracker = FaceTracker(confirm_scans=2)
    recognizer.code = "EMP7"
    frame = np.zeros((480, 640, 3), np.uint8)
    ctx = FrameContext(camera_id="cam-test", captured_at=datetime(2026, 9, 7, tzinfo=UTC))

    asyncio.run(process.process_frames(frame, ctx))
    assert sink.marked == [], "published on the first scan despite confirm_scans=2"

    asyncio.run(process.process_frames(frame, ctx))
    assert [f.employee_code for f in sink.marked] == ["EMP7"]

    # Standing there longer must not punch them in again.
    for _ in range(3):
        asyncio.run(process.process_frames(frame, ctx))
    assert len(sink.marked) == 1


def test_without_a_tracker_every_accepted_scan_still_publishes() -> None:
    """The default path has to stay exactly what it was."""
    process, recognizer, sink = _process([_looking_face(90.0)], min_face_pixels=0)
    recognizer.code = "EMP7"
    frame = np.zeros((480, 640, 3), np.uint8)
    ctx = FrameContext(camera_id="cam-test", captured_at=datetime(2026, 9, 7, tzinfo=UTC))

    asyncio.run(process.process_frames(frame, ctx))
    asyncio.run(process.process_frames(frame, ctx))
    assert len(sink.marked) == 2


def test_a_tracker_assigns_a_track_id_to_every_detected_face() -> None:
    """Including faces that never reach recognition -- the id is how they are followed."""
    process, _, _ = _process([_looking_face(30.0)], min_face_pixels=60)
    process._tracker = FaceTracker(confirm_scans=1)
    result = asyncio.run(process.process_frames(np.zeros((480, 640, 3), np.uint8)))
    assert result.faces[0].track_id is not None


def test_the_track_id_reaches_the_attendance_event() -> None:
    """`AttendanceEvent.track_id` sat declared and unassigned until now."""
    process, recognizer, sink = _process([_looking_face(90.0)], min_face_pixels=0)
    process._tracker = FaceTracker(confirm_scans=1)
    recognizer.code = "EMP7"
    ctx = FrameContext(camera_id="cam-test", captured_at=datetime(2026, 9, 7, tzinfo=UTC))
    asyncio.run(process.process_frames(np.zeros((480, 640, 3), np.uint8), ctx))
    assert sink.marked[0].track_id is not None
