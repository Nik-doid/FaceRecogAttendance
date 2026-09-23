"""The rank-margin gate: rejecting the failure a threshold cannot see.

A confident-looking score against a gallery where two people score alike is the
dangerous case, because the absolute number says nothing about whether the winner
actually won. These tests use a stub index so the arithmetic is visible; the real
FAISS behaviour is covered in ``tests/test_faiss_index.py``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

from app.ai.types import IndexResult
from app.core.face_processing.face_recognition_handlers import ArcFaceRecognition
from app.schemas.face_processing import FaceResult

KPS = [(10.0, 10.0), (30.0, 10.0), (20.0, 20.0), (12.0, 30.0), (28.0, 30.0)]


class _Index:
    """Returns a scripted ranking, and records the k it was asked for."""

    def __init__(self, ranking: list[tuple[str, float]]) -> None:
        self._ranking = ranking
        self.k_requested: int | None = None

    def search(self, embedding: np.ndarray, k: int = 1) -> list[IndexResult]:
        self.k_requested = k
        return [
            IndexResult(employee_code=code, score=score, distance=1.0 - score)
            for code, score in self._ranking[:k]
        ]


class _Recognizer:
    def embed(self, image: np.ndarray, kps: np.ndarray) -> np.ndarray:
        return np.ones(512, dtype="float32")


@dataclass
class _Gallery:
    index: _Index
    recognizer: _Recognizer | None
    employees: int = 2
    photos: int = 2


def _recognise(
    ranking: list[tuple[str, float]],
    *,
    threshold: float = 0.45,
    margin: float = 0.0,
) -> FaceResult:
    gallery = _Gallery(index=_Index(ranking), recognizer=_Recognizer())
    handler = ArcFaceRecognition(gallery, threshold, margin)  # type: ignore[arg-type]
    face = FaceResult(bbox=(0.0, 0.0, 100.0, 120.0), score=0.9, kps=KPS)
    frame = np.zeros((240, 320, 3), np.uint8)
    (result,) = asyncio.run(handler.recognize(frame, [face]))
    return result


def test_the_runner_up_is_actually_requested() -> None:
    """Without k=2 there is no margin to compute."""
    gallery = _Gallery(index=_Index([("EMP1", 0.9), ("EMP2", 0.2)]), recognizer=_Recognizer())
    handler = ArcFaceRecognition(gallery, 0.45, 0.0)  # type: ignore[arg-type]
    face = FaceResult(bbox=(0.0, 0.0, 100.0, 120.0), score=0.9, kps=KPS)
    asyncio.run(handler.recognize(np.zeros((240, 320, 3), np.uint8), [face]))
    assert gallery.index.k_requested == 2


def test_a_clear_winner_is_accepted() -> None:
    result = _recognise([("EMP1", 0.80), ("EMP2", 0.20)], margin=0.15)
    assert result.employee_code == "EMP1"
    assert result.confidence == pytest.approx(0.80)
    assert result.margin == pytest.approx(0.60)


def test_a_near_tie_is_rejected_even_well_above_the_threshold() -> None:
    """The whole point: 0.80 looks decisive until you see 0.78 behind it."""
    result = _recognise([("EMP1", 0.80), ("EMP2", 0.78)], threshold=0.45, margin=0.15)
    assert result.employee_code is None
    # The scores survive the rejection -- they are what the gate gets tuned against.
    assert result.confidence == pytest.approx(0.80)
    assert result.margin == pytest.approx(0.02)


def test_a_margin_of_zero_leaves_the_old_behaviour_exactly() -> None:
    """The shipped default must change nothing until a sweep picks a value."""
    result = _recognise([("EMP1", 0.80), ("EMP2", 0.7999)], margin=0.0)
    assert result.employee_code == "EMP1"


def test_the_threshold_still_applies_on_its_own() -> None:
    """A huge margin does not rescue a score that is simply too low."""
    result = _recognise([("EMP1", 0.20), ("EMP2", 0.01)], threshold=0.45, margin=0.05)
    assert result.employee_code is None
    assert result.confidence == pytest.approx(0.20)


def test_the_gate_is_inclusive_of_the_configured_margin() -> None:
    exact = _recognise([("EMP1", 0.80), ("EMP2", 0.65)], margin=0.15)
    assert exact.employee_code == "EMP1"
    under = _recognise([("EMP1", 0.80), ("EMP2", 0.6501)], margin=0.15)
    assert under.employee_code is None


def test_a_single_employee_gallery_still_matches() -> None:
    """No runner-up means no gap to measure, and infinity is the honest value.

    Treating a missing second place as a zero margin would reject the only person in
    the gallery -- which is exactly the state anyone testing a fresh install is in.
    """
    result = _recognise([("EMP1", 0.80)], margin=0.30)
    assert result.employee_code == "EMP1"
    # Reported as 0.0 rather than inf: the field is serialised, and "not measured" and
    # "measured as zero" are both honest readings of an absent runner-up.
    assert result.margin == 0.0


def test_an_empty_gallery_leaves_the_face_untouched() -> None:
    gallery = _Gallery(index=_Index([]), recognizer=_Recognizer(), employees=0, photos=0)
    handler = ArcFaceRecognition(gallery, 0.45, 0.15)  # type: ignore[arg-type]
    face = FaceResult(bbox=(0.0, 0.0, 100.0, 120.0), score=0.9, kps=KPS)
    (result,) = asyncio.run(handler.recognize(np.zeros((240, 320, 3), np.uint8), [face]))
    assert result.employee_code is None
    assert result.confidence == 0.0
    assert result.margin == 0.0


def test_a_gallery_that_is_still_building_leaves_the_face_untouched() -> None:
    """`recognizer is None` until enrolment finishes; that is not an error path."""
    gallery = _Gallery(index=_Index([("EMP1", 0.9)]), recognizer=None)
    handler = ArcFaceRecognition(gallery, 0.45, 0.15)  # type: ignore[arg-type]
    face = FaceResult(bbox=(0.0, 0.0, 100.0, 120.0), score=0.9, kps=KPS)
    (result,) = asyncio.run(handler.recognize(np.zeros((240, 320, 3), np.uint8), [face]))
    assert result.employee_code is None
    assert result.confidence == 0.0


def test_a_face_without_landmarks_is_never_embedded() -> None:
    gallery = _Gallery(index=_Index([("EMP1", 0.9), ("EMP2", 0.1)]), recognizer=_Recognizer())
    handler = ArcFaceRecognition(gallery, 0.45, 0.0)  # type: ignore[arg-type]
    face = FaceResult(bbox=(0.0, 0.0, 100.0, 120.0), score=0.9, kps=None)
    (result,) = asyncio.run(handler.recognize(np.zeros((240, 320, 3), np.uint8), [face]))
    assert result.employee_code is None
    assert gallery.index.k_requested is None
