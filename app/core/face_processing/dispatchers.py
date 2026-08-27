"""Step dispatchers -- the single place that decides WHICH implementation runs.

One Factory + Dispatcher pair per step of ``FaceRecognitionProcess``. To swap an
implementation, add a handler class in the matching ``*_handlers`` module and a branch
in its factory here; ``process_frames`` does not change.

Steps 1-3 are implemented so far. The remaining factory raises
``StepNotImplementedError`` rather than returning a half-working handler, so an
unimplemented step fails loudly at dispatch time instead of silently returning nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.core.face_processing.attendance_handlers import (
    BaseMarkAttendance,
    BrokerMarkAttendance,  # noqa: F401 - registered here once implemented
    NullMarkAttendance,  # noqa: F401 - registered here once implemented
)
from app.core.face_processing.face_detection_handlers import (
    BaseFaceDetection,
    ScrfdFaceDetection,
)
from app.core.face_processing.face_recognition_handlers import (
    ArcFaceRecognition,
    BaseFaceRecognition,
)
from app.core.face_processing.palm_detection_handlers import (
    BasePalmDetection,
    BlazePalmDetection,
    MediaPipePalmDetection,  # noqa: F401 - registered here once implemented
)
from app.schemas.face_processing import (
    AttendanceSinkType,
    FaceDetectorType,
    FaceRecognizerType,
    PalmDetectorType,
)
from app.services.face_recognition.exceptions import StepNotImplementedError


# palm detection handlers and dispatchers
class PalmDetectionHandlerFactory:
    @staticmethod
    def create_handler(
        palm_detector: PalmDetectorType,
        models_dir: Path,
        score_threshold: float,
        scan_grid: int = 1,
        scan_overlap: float = 0.2,
    ) -> BasePalmDetection:
        if palm_detector == PalmDetectorType.BLAZEPALM:
            return BlazePalmDetection(
                model_path=models_dir / "palm_detection_mediapipe_2023feb.onnx",
                score_threshold=score_threshold,
                scan_grid=scan_grid,
                scan_overlap=scan_overlap,
            )

        if palm_detector == PalmDetectorType.MEDIAPIPE:
            raise StepNotImplementedError(
                "MediaPipe palm detection is not implemented; it costs ~52ms/frame "
                "against BlazePalm's ~10ms. Use PalmDetectorType.BLAZEPALM."
            )

        raise StepNotImplementedError(f"Unknown palm detector: {palm_detector}")


class PalmDetectionDispatcher:
    handler = PalmDetectionHandlerFactory

    @classmethod
    def dispatch(
        cls,
        palm_detector: PalmDetectorType,
        models_dir: Path,
        score_threshold: float,
        scan_grid: int = 1,
        scan_overlap: float = 0.2,
    ) -> BasePalmDetection:
        """Dispatch the palm detection handler for the given detector type.

        Args:
            palm_detector (PalmDetectorType)
            models_dir (Path): where the model file lives.
            score_threshold (float): minimum confidence to call it a palm.
            scan_grid (int): NxN region scan; 1 means the whole frame only.
            scan_overlap (float): tile overlap as a fraction of one tile.
        Returns:
            BasePalmDetection: palm detection handler instance
        """
        palm_inst = cls.handler.create_handler(
            palm_detector, models_dir, score_threshold, scan_grid, scan_overlap
        )
        return palm_inst


# face detection handlers and dispatchers
class FaceDetectionHandlerFactory:
    @staticmethod
    def create_handler(
        face_detector: FaceDetectorType,
        models_dir: Path,
        score_threshold: float,
    ) -> BaseFaceDetection:
        if face_detector == FaceDetectorType.SCRFD:
            return ScrfdFaceDetection(
                models_dir=models_dir,
                score_threshold=score_threshold,
            )

        raise StepNotImplementedError(f"Unknown face detector: {face_detector}")


class FaceDetectionDispatcher:
    handler = FaceDetectionHandlerFactory

    @classmethod
    def dispatch(
        cls,
        face_detector: FaceDetectorType,
        models_dir: Path,
        score_threshold: float,
    ) -> BaseFaceDetection:
        """Dispatch the face detection handler for the given detector type.

        Args:
            face_detector (FaceDetectorType)
            models_dir (Path): where the model file lives.
            score_threshold (float): minimum confidence to keep a detection.
        Returns:
            BaseFaceDetection: face detection handler instance
        """
        face_inst = cls.handler.create_handler(face_detector, models_dir, score_threshold)
        return face_inst


# face recognition handlers and dispatchers
class FaceRecognitionHandlerFactory:
    @staticmethod
    def create_handler(
        face_recognizer: FaceRecognizerType,
        models_dir: Path,
        photo_sources: Sequence[str | Path],
        score_threshold: float,
    ) -> BaseFaceRecognition:
        if face_recognizer == FaceRecognizerType.ARCFACE:
            return ArcFaceRecognition(
                models_dir=models_dir,
                photo_sources=photo_sources,
                score_threshold=score_threshold,
            )

        raise StepNotImplementedError(f"Unknown face recognizer: {face_recognizer}")


class FaceRecognitionDispatcher:
    handler = FaceRecognitionHandlerFactory

    @classmethod
    def dispatch(
        cls,
        face_recognizer: FaceRecognizerType,
        models_dir: Path,
        photo_sources: Sequence[str | Path],
        score_threshold: float,
    ) -> BaseFaceRecognition:
        """Dispatch the face recognition handler for the given recognizer type.

        Args:
            face_recognizer (FaceRecognizerType)
            models_dir (Path): where the model file lives.
            photo_sources (Sequence[str | Path]): enrolment roots, one subdirectory
                per employee_code.
            score_threshold (float): minimum cosine similarity to accept a match.
        Returns:
            BaseFaceRecognition: face recognition handler instance
        """
        recognize_inst = cls.handler.create_handler(
            face_recognizer, models_dir, photo_sources, score_threshold
        )
        return recognize_inst


# mark attendance handlers and dispatchers
class MarkAttendanceHandlerFactory:
    @staticmethod
    def create_handler(attendance_sink: AttendanceSinkType) -> BaseMarkAttendance:
        if attendance_sink in (AttendanceSinkType.NULL, AttendanceSinkType.RABBITMQ):
            raise StepNotImplementedError(
                f"Attendance sink '{attendance_sink}' is not implemented; when it is, "
                "delegate to app/services/attendance_reporter/ rather than reimplement it"
            )

        raise StepNotImplementedError(f"Unknown attendance sink: {attendance_sink}")


class MarkAttendanceDispatcher:
    handler = MarkAttendanceHandlerFactory

    @classmethod
    def dispatch(cls, attendance_sink: AttendanceSinkType) -> BaseMarkAttendance:
        """Dispatch the mark-attendance handler for the given sink type."""
        attendance_inst = cls.handler.create_handler(attendance_sink)
        return attendance_inst
