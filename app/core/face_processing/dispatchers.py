"""Step dispatchers -- the single place that decides WHICH implementation runs.

One Factory + Dispatcher pair per step of ``FaceRecognitionProcess``. To swap an
implementation, add a handler class in the matching ``*_handlers`` module and a branch
in its factory here; ``process_frames`` does not change.

Every step is implemented. A factory still raises ``StepNotImplementedError`` for an
unknown enum value rather than returning a half-working handler, so a typo fails loudly
at dispatch time instead of silently recording nothing.
"""

from __future__ import annotations

from pathlib import Path

from app.ai.detector.scrfd import SCRFDDetector
from app.core.face_processing.attendance_handlers import (
    BaseMarkAttendance,
    BrokerMarkAttendance,
    NullMarkAttendance,
)
from app.core.face_processing.face_detection_handlers import (
    BaseFaceDetection,
    ScrfdFaceDetection,
)
from app.core.face_processing.face_recognition_handlers import (
    ArcFaceRecognition,
    BaseFaceRecognition,
)
from app.core.face_processing.gallery import Gallery
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
from app.services.attendance_reporter.base import AttendanceReporter
from app.services.duplicate_suppressor import DuplicateSuppressor
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
        detector: SCRFDDetector,
    ) -> BaseFaceDetection:
        if face_detector == FaceDetectorType.SCRFD:
            return ScrfdFaceDetection(detector)

        raise StepNotImplementedError(f"Unknown face detector: {face_detector}")


class FaceDetectionDispatcher:
    handler = FaceDetectionHandlerFactory

    @classmethod
    def dispatch(
        cls,
        face_detector: FaceDetectorType,
        detector: SCRFDDetector,
    ) -> BaseFaceDetection:
        """Dispatch the face detection handler for the given detector type.

        Args:
            face_detector (FaceDetectorType)
            detector (SCRFDDetector): the already-loaded session from app.runtime.
                The detection threshold is a property of that session, set when it
                was built, so it is no longer passed here.
        Returns:
            BaseFaceDetection: face detection handler instance
        """
        face_inst = cls.handler.create_handler(face_detector, detector)
        return face_inst


# face recognition handlers and dispatchers
class FaceRecognitionHandlerFactory:
    @staticmethod
    def create_handler(
        face_recognizer: FaceRecognizerType,
        gallery: Gallery,
        score_threshold: float,
    ) -> BaseFaceRecognition:
        if face_recognizer == FaceRecognizerType.ARCFACE:
            return ArcFaceRecognition(gallery, score_threshold)

        raise StepNotImplementedError(f"Unknown face recognizer: {face_recognizer}")


class FaceRecognitionDispatcher:
    handler = FaceRecognitionHandlerFactory

    @classmethod
    def dispatch(
        cls,
        face_recognizer: FaceRecognizerType,
        gallery: Gallery,
        score_threshold: float,
    ) -> BaseFaceRecognition:
        """Dispatch the face recognition handler for the given recognizer type.

        Args:
            face_recognizer (FaceRecognizerType)
            gallery (Gallery): the enrolled employees, already embedded.
            score_threshold (float): minimum cosine similarity to accept a match.
        Returns:
            BaseFaceRecognition: face recognition handler instance
        """
        recognize_inst = cls.handler.create_handler(face_recognizer, gallery, score_threshold)
        return recognize_inst


# mark attendance handlers and dispatchers
class MarkAttendanceHandlerFactory:
    @staticmethod
    def create_handler(
        attendance_sink: AttendanceSinkType,
        reporter: AttendanceReporter | None = None,
        suppressor: DuplicateSuppressor | None = None,
    ) -> BaseMarkAttendance:
        if attendance_sink == AttendanceSinkType.NULL:
            return NullMarkAttendance()

        if attendance_sink == AttendanceSinkType.RABBITMQ:
            if reporter is None or suppressor is None:
                raise StepNotImplementedError(
                    "the rabbitmq sink needs a reporter and a suppressor; they are "
                    "injected because the reporter owns one AMQP connection per "
                    "process, while a pipeline is built per consumer"
                )
            return BrokerMarkAttendance(reporter, suppressor)

        raise StepNotImplementedError(f"Unknown attendance sink: {attendance_sink}")


class MarkAttendanceDispatcher:
    handler = MarkAttendanceHandlerFactory

    @classmethod
    def dispatch(
        cls,
        attendance_sink: AttendanceSinkType,
        reporter: AttendanceReporter | None = None,
        suppressor: DuplicateSuppressor | None = None,
    ) -> BaseMarkAttendance:
        """Dispatch the mark-attendance handler for the given sink type.

        Args:
            attendance_sink (AttendanceSinkType)
            reporter (AttendanceReporter | None): required by the rabbitmq sink.
            suppressor (DuplicateSuppressor | None): per-employee publish rate limit.
        Returns:
            BaseMarkAttendance: mark attendance handler instance
        """
        attendance_inst = cls.handler.create_handler(attendance_sink, reporter, suppressor)
        return attendance_inst
