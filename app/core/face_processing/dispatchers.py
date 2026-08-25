"""Step dispatchers -- the single place that decides WHICH implementation runs.

One Factory + Dispatcher pair per step of ``FaceRecognitionProcess``. To swap an
implementation, add a handler class in the matching ``*_handlers`` module and a branch
in its factory here; ``process_frames`` does not change.

Only palm detection is implemented so far. The remaining factories raise
``StepNotImplementedError`` rather than returning a half-working handler, so an
unimplemented step fails loudly at dispatch time instead of silently returning nothing.
"""

from __future__ import annotations

from pathlib import Path

from app.core.face_processing.attendance_handlers import (
    BaseMarkAttendance,
    BrokerMarkAttendance,  # noqa: F401 - registered here once implemented
    NullMarkAttendance,  # noqa: F401 - registered here once implemented
)
from app.core.face_processing.face_detection_handlers import (
    BaseFaceDetection,
    ScrfdFaceDetection,  # noqa: F401 - registered here once implemented
)
from app.core.face_processing.face_recognition_handlers import (
    ArcFaceRecognition,  # noqa: F401 - registered here once implemented
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
    ) -> BasePalmDetection:
        if palm_detector == PalmDetectorType.BLAZEPALM:
            return BlazePalmDetection(
                model_path=models_dir / "palm_detection_mediapipe_2023feb.onnx",
                score_threshold=score_threshold,
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
    ) -> BasePalmDetection:
        """Dispatch the palm detection handler for the given detector type.

        Args:
            palm_detector (PalmDetectorType)
            models_dir (Path): where the model file lives.
            score_threshold (float): minimum confidence to call it a palm.
        Returns:
            BasePalmDetection: palm detection handler instance
        """
        palm_inst = cls.handler.create_handler(palm_detector, models_dir, score_threshold)
        return palm_inst


# face detection handlers and dispatchers
class FaceDetectionHandlerFactory:
    @staticmethod
    def create_handler(face_detector: FaceDetectorType) -> BaseFaceDetection:
        if face_detector == FaceDetectorType.SCRFD:
            raise StepNotImplementedError("SCRFD face detection is not implemented")

        raise StepNotImplementedError(f"Unknown face detector: {face_detector}")


class FaceDetectionDispatcher:
    handler = FaceDetectionHandlerFactory

    @classmethod
    def dispatch(cls, face_detector: FaceDetectorType) -> BaseFaceDetection:
        """Dispatch the face detection handler for the given detector type."""
        face_inst = cls.handler.create_handler(face_detector)
        return face_inst


# face recognition handlers and dispatchers
class FaceRecognitionHandlerFactory:
    @staticmethod
    def create_handler(face_recognizer: FaceRecognizerType) -> BaseFaceRecognition:
        if face_recognizer == FaceRecognizerType.ARCFACE:
            raise StepNotImplementedError("ArcFace recognition is not implemented")

        raise StepNotImplementedError(f"Unknown face recognizer: {face_recognizer}")


class FaceRecognitionDispatcher:
    handler = FaceRecognitionHandlerFactory

    @classmethod
    def dispatch(cls, face_recognizer: FaceRecognizerType) -> BaseFaceRecognition:
        """Dispatch the face recognition handler for the given recognizer type."""
        recognize_inst = cls.handler.create_handler(face_recognizer)
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
