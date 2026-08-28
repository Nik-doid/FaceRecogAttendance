"""The loaded inference models, built once per process.

Before this module existed, ArcFace (174 MiB) and SCRFD were each constructed three
times over: once in ``AIComponents`` for the recognition worker, again inside
``gallery._build`` for enrolment, and again for every WebSocket connection. Nothing
required that -- both are onnxruntime sessions, whose ``Run`` is thread-safe, so one
instance can serve the enrolment thread, the camera loop and every viewer at once.

The BlazePalm net is deliberately NOT here. It runs on ``cv2.dnn``, which offers no
such guarantee, so it stays private to each ``FaceRecognitionProcess`` -- at 3.7 MiB
that is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ai.detector.scrfd import SCRFDDetector
from app.ai.recognizer.arcface import ArcFaceRecognizer
from app.config.settings import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Models:
    """The two ONNX sessions shared across every consumer in the process."""

    detector: SCRFDDetector
    recognizer: ArcFaceRecognizer


def load_models(settings: Settings) -> Models:
    """Construct both sessions. Slow (~seconds) and called exactly once."""
    models_dir = str(settings.models_dir)
    providers = list(settings.onnx_providers)
    detector = SCRFDDetector(
        model_name=settings.detect_model,
        providers=providers,
        det_thresh=settings.detect_thresh,
        models_dir=models_dir,
    )
    recognizer = ArcFaceRecognizer(
        model_name=settings.recognize_model,
        providers=providers,
        models_dir=models_dir,
    )
    # Log what onnxruntime actually resolved, not what was requested: asking for
    # CUDA on a CPU-only box silently falls through, and the difference is a 20x
    # throughput change that should never be a mystery.
    log.info(
        "inference models loaded",
        extra={
            "requested_providers": providers,
            "detector_providers": list(detector.providers),
            "recognizer_providers": list(recognizer.providers),
        },
    )
    return Models(detector=detector, recognizer=recognizer)
