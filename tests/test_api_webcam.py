"""The browser-webcam WebSocket: frame echo plus dispatched palm detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config.settings import settings
from app.core.face_processing.dispatchers import (
    FaceDetectionDispatcher,
    FaceRecognitionDispatcher,
    MarkAttendanceDispatcher,
    PalmDetectionDispatcher,
)
from app.core.face_processing.palm_detection_handlers import BlazePalmDetection
from app.schemas.face_processing import (
    AttendanceSinkType,
    FaceDetectorType,
    FaceProcessConfig,
    FaceRecognizerType,
    PalmDetectorType,
)
from app.services.face_recognition.exceptions import StepNotImplementedError
from app.services.face_recognition.process import FaceRecognitionProcess

PALM_MODEL = settings.models_dir / "palm_detection_mediapipe_2023feb.onnx"
needs_model = pytest.mark.skipif(
    not PALM_MODEL.is_file(),
    reason="palm detector model absent; run `python -m app.ai._download`",
)


def _jpeg(width: int = 64, height: int = 48) -> bytes:
    _, encoded = cv2.imencode(".jpg", np.zeros((height, width, 3), np.uint8))
    return bytes(encoded.tobytes())


@needs_model
def test_webcam_ws_echoes_a_jpeg_frame(client: TestClient) -> None:
    with client.websocket_connect("/api/v1/webcam/ws") as ws:
        ws.send_bytes(_jpeg())
        assert ws.receive_bytes()[:2] == b"\xff\xd8"  # JPEG start-of-image marker


@needs_model
def test_webcam_ws_ignores_undecodable_bytes(client: TestClient) -> None:
    with client.websocket_connect("/api/v1/webcam/ws") as ws:
        ws.send_bytes(b"not a jpeg")
        ws.send_bytes(_jpeg())
        assert ws.receive_bytes()[:2] == b"\xff\xd8"


@needs_model
def test_palm_dispatcher_returns_blazepalm() -> None:
    handler = PalmDetectionDispatcher.dispatch(
        PalmDetectorType.BLAZEPALM, settings.models_dir, 0.5
    )
    assert isinstance(handler, BlazePalmDetection)


@pytest.mark.parametrize("detector", [PalmDetectorType.MEDIAPIPE, "nope"])
def test_palm_dispatcher_rejects_unimplemented(detector: PalmDetectorType) -> None:
    with pytest.raises(StepNotImplementedError):
        PalmDetectionDispatcher.dispatch(detector, settings.models_dir, 0.5)


def test_later_steps_are_not_implemented_yet() -> None:
    """Pins the stub contract: wiring steps 2-4 must be a deliberate change."""
    with pytest.raises(StepNotImplementedError):
        FaceDetectionDispatcher.dispatch(FaceDetectorType.SCRFD)
    with pytest.raises(StepNotImplementedError):
        FaceRecognitionDispatcher.dispatch(FaceRecognizerType.ARCFACE)
    with pytest.raises(StepNotImplementedError):
        MarkAttendanceDispatcher.dispatch(AttendanceSinkType.NULL)


@needs_model
async def test_process_frames_finds_no_palm_in_a_blank_frame() -> None:
    process = FaceRecognitionProcess(FaceProcessConfig(), settings.models_dir)
    result = await process.process_frames(np.zeros((240, 320, 3), np.uint8))
    assert result.palm.detected is False
    assert result.faces == []
