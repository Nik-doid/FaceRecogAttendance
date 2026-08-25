"""Frame-processing step keys and results.

The ``*Type`` enums are the dispatch keys: one per step of
:class:`~app.services.face_recognition.process.FaceRecognitionProcess`. Selecting a
different implementation of a step is a change of enum value, never a change to
``process_frames``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class PalmDetectorType(StrEnum):
    BLAZEPALM = "blazepalm"
    MEDIAPIPE = "mediapipe"


class FaceDetectorType(StrEnum):
    SCRFD = "scrfd"


class FaceRecognizerType(StrEnum):
    ARCFACE = "arcface"


class AttendanceSinkType(StrEnum):
    RABBITMQ = "rabbitmq"
    NULL = "null"


class PalmResult(BaseModel):
    """Outcome of step 1. ``score`` is the best palm confidence in the frame."""

    detected: bool
    score: float


class FaceResult(BaseModel):
    """Outcome of step 2. Placeholder until face detection is implemented."""

    bbox: tuple[float, float, float, float]
    score: float


class FrameResult(BaseModel):
    """What one frame produced. Steps that did not run leave their field empty."""

    palm: PalmResult
    faces: list[FaceResult] = Field(default_factory=list)


class FaceProcessConfig(BaseModel):
    """Which implementation each step dispatches to."""

    palm_detector: PalmDetectorType = PalmDetectorType.BLAZEPALM
    face_detector: FaceDetectorType = FaceDetectorType.SCRFD
    face_recognizer: FaceRecognizerType = FaceRecognizerType.ARCFACE
    attendance_sink: AttendanceSinkType = AttendanceSinkType.NULL
    palm_score_threshold: float = 0.5
