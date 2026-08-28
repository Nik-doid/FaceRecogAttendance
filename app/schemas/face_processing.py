"""Frame-processing step keys and results.

The ``*Type`` enums are the dispatch keys: one per step of
:class:`~app.services.face_recognition.process.FaceRecognitionProcess`. Selecting a
different implementation of a step is a change of enum value, never a change to
``process_frames``.
"""

from __future__ import annotations

from datetime import datetime
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
    """One palm verdict: for a single face's search box, or for a whole frame."""

    detected: bool
    score: float


class FaceResult(BaseModel):
    """One face, annotated by whichever steps have run.

    Step 1 fills ``bbox`` (x1, y1, x2, y2 in frame pixels), ``score`` and ``kps``.
    Step 2 fills ``looking``, ``yaw_ratio`` and ``roll_degrees``.
    Step 3 fills ``palm`` and ``palm_score``.
    Step 4 fills ``employee_code`` and ``confidence``.
    """

    bbox: tuple[float, float, float, float]
    score: float
    # The 5-point landmarks ArcFace aligns on, in frame pixels. Detectors that do not
    # produce them leave this None, which makes the face unrecognisable but still drawable.
    kps: list[tuple[float, float]] | None = None
    # False also covers "could not tell" (no landmarks): the gate fails closed.
    looking: bool = False
    # Kept for the same reason as ``confidence`` -- these are the numbers you tune
    # LOOKING_MAX_YAW_RATIO against when the gate is too strict or too loose.
    yaw_ratio: float = 0.0
    roll_degrees: float = 0.0
    # Whether a palm was found in *this* face's own search box. Recognition runs only
    # for faces where this is True, so a bystander standing next to someone who waves
    # is never identified.
    palm: bool = False
    palm_score: float = 0.0
    # None means "no gallery match cleared the threshold" -- not "not searched".
    employee_code: str | None = None
    # Best cosine similarity found, kept even when it lost to the threshold: an
    # under-threshold near-miss is the useful number when tuning that threshold.
    confidence: float = 0.0


class FrameContext(BaseModel):
    """Per-frame facts the pipeline needs for step 5 but a face does not carry.

    Kept off ``FaceResult`` on purpose: that model is serialised to the browser on
    every frame, and neither field means anything to the renderer. ``captured_at`` is
    stamped when the frame is read, not when it is published, so a broker outage
    cannot move everyone's punch time.
    """

    camera_id: str
    captured_at: datetime


class FrameResult(BaseModel):
    """What one frame produced. Steps that did not run leave their field empty.

    ``palm`` is the frame-wide summary -- the best verdict across every face -- and
    drives the page badge. Which *person* raised a hand is on ``FaceResult.palm``.
    """

    palm: PalmResult
    faces: list[FaceResult] = Field(default_factory=list)


class FaceProcessConfig(BaseModel):
    """Which implementation each step dispatches to."""

    palm_detector: PalmDetectorType = PalmDetectorType.BLAZEPALM
    face_detector: FaceDetectorType = FaceDetectorType.SCRFD
    face_recognizer: FaceRecognizerType = FaceRecognizerType.ARCFACE
    attendance_sink: AttendanceSinkType = AttendanceSinkType.NULL
    palm_score_threshold: float = 0.5
    face_score_threshold: float = 0.5
    recognition_threshold: float = 0.6
    # The looking gate. See ``app/core/face_processing/gaze.py``; pitch is deliberately
    # not gated, because a high-mounted camera sees every face pitched.
    looking_max_yaw_ratio: float = 0.35
    looking_max_roll_degrees: float = 25.0
    # How far either side of a looking face to search for a raised hand, in multiples
    # of that face's width/height.
    palm_search_margin: float = 0.6
    # 1 scans the whole frame only. Raise it for wide/distant views, where a palm is
    # too few pixels to survive BlazePalm's 192x192 input -- see ``scan_regions``.
    palm_scan_grid: int = 1
    palm_scan_overlap: float = 0.2
