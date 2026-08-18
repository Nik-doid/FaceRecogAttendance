"""Factory that wires the AI pipeline components together from settings.

Kept separate from settings/DI so components can be swapped in tests (e.g. a
``FakeDetector``) without touching the production wiring. Model loading is
intentionally NOT lazy here — startup fails fast if a configured model can't load,
and the loaded ONNX sessions are reused for every frame (models are never reloaded
per-frame).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ai.detector.base import Detector
from app.ai.detector.hand import HandDetector
from app.ai.detector.scrfd import SCRFDDetector
from app.ai.gesture.wave import WaveTracker
from app.ai.liveness.base import LivenessChecker
from app.ai.liveness.silentface import SilentFaceLiveness
from app.ai.quality.quality import FaceQualityChecker
from app.ai.recognizer.arcface import ArcFaceRecognizer
from app.ai.recognizer.base import Recognizer
from app.config.settings import Settings


@dataclass(frozen=True)
class AIComponents:
    detector: Detector
    recognizer: Recognizer
    liveness: LivenessChecker
    quality: FaceQualityChecker
    hand_detector: HandDetector
    wave_tracker: WaveTracker


def load_ai_components(settings: Settings) -> AIComponents:
    """Instantiate every AI model against the configured providers."""
    providers = settings.onnx_providers
    models_dir = str(settings.models_dir)

    detector = SCRFDDetector(
        model_name=settings.detect_model,
        providers=providers,
        models_dir=models_dir,
    )
    recognizer = ArcFaceRecognizer(
        model_name=settings.recognize_model,
        providers=providers,
        models_dir=models_dir,
    )

    if settings.liveness_enabled:
        if not settings.silentface_model_path:
            raise RuntimeError(
                "LIVENESS_ENABLED=true but SILENTFACE_MODEL_PATH is not set. "
                "Provide an exported MiniFASNet ONNX model or set LIVENESS_ENABLED=false."
            )
        liveness: LivenessChecker = SilentFaceLiveness(
            model_path=str(settings.silentface_model_path),
            threshold=settings.silentface_threshold,
            providers=providers,
        )
    else:
        liveness = _PassThroughLiveness()

    quality = FaceQualityChecker(
        settings.minimum_face_size,
        blur_threshold=settings.quality_min_blur,
        min_lighting=settings.quality_min_lighting,
        max_lighting=settings.quality_max_lighting,
        max_roll_deg=settings.quality_max_roll_deg,
    )

    hand_detector = HandDetector(
        model_path=settings.models_dir / "hand_landmarker.task",
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    wave_tracker = WaveTracker()

    return AIComponents(
        detector=detector,
        recognizer=recognizer,
        liveness=liveness,
        quality=quality,
        hand_detector=hand_detector,
        wave_tracker=wave_tracker,
    )


class _PassThroughLiveness(LivenessChecker):
    """Used only when liveness is explicitly disabled via config."""

    def is_live(self, image_bgr, face):  # type: ignore[no-untyped-def]
        return True, 1.0
