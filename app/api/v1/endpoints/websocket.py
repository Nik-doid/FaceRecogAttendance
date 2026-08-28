"""The two WebSocket surfaces. Both emit the same JSON shape, so one page renders both.

- ``/camera/ws`` is a **viewer**. It owns no camera and triggers no processing: the
  always-on :class:`~app.camera.runner.CameraRunner` captures and recognises
  regardless, and this forwards whatever it last published. Closing the tab stops
  nothing, which is the entire point -- attendance used to die with the connection.
- ``/webcam/ws`` is a **debug surface**. It runs the pipeline inline on frames the
  browser sends, so a developer can check detection against a laptop camera. It
  passes no ``FrameContext``, so it can never record attendance.
"""

from __future__ import annotations

import asyncio

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.deps import ContainerDep
from app.camera.runner import detection_payload
from app.config.settings import Settings
from app.container import Container
from app.core.logging import get_logger
from app.schemas.face_processing import FaceProcessConfig
from app.services.face_recognition.process import FaceRecognitionProcess

router = APIRouter(tags=["webcam"])
log = get_logger(__name__)

JPEG_QUALITY = 70
DETECT_EVERY = 2
CONFIRM_FRAMES = 2
VIEWER_INTERVAL = 0.033  # ~30 fps poll of the hub


class PalmDebounce:
    """Holds the reported palm state until enough detections disagree with it.

    Palm scores sit near the threshold for a frame or two as a hand enters, which
    would otherwise flip the page's badge on and off several times per second.
    """

    def __init__(self, confirm_frames: int = CONFIRM_FRAMES) -> None:
        self._confirm_frames = confirm_frames
        self._reported = False
        self._pending = 0

    def update(self, detected: bool) -> bool:
        if detected == self._reported:
            self._pending = 0
            return self._reported
        self._pending += 1
        if self._pending >= self._confirm_frames:
            self._reported = detected
            self._pending = 0
        return self._reported


def _create_process(
    container: Container, config: FaceProcessConfig
) -> FaceRecognitionProcess:
    # Per-connection, because the process owns a cv2.dnn palm net that must not be
    # shared. The SCRFD/ArcFace sessions and the gallery behind it are process-wide
    # and injected, so a new connection no longer reloads 174 MiB of ArcFace.
    return FaceRecognitionProcess(
        config, container.models, container.gallery, container.settings.models_dir
    )


@router.websocket("/webcam/ws")
async def webcam_ws(websocket: WebSocket, container: ContainerDep) -> None:
    """Echo the client's webcam frames back, with palm state and face boxes alongside.

    One JSON message per *detected* frame (every ``DETECT_EVERY`` frames), so the page
    can move the face boxes as the subject moves. Only the palm badge is debounced;
    boxes are whatever the latest detection found.
    """
    await websocket.accept()
    settings: Settings = container.settings
    # Whole-frame palm scan: a hand held up to a laptop already fills enough of it.
    process = _create_process(
        container,
        FaceProcessConfig(
            palm_score_threshold=settings.palm_score_threshold,
            looking_max_yaw_ratio=settings.looking_max_yaw_ratio,
            looking_max_roll_degrees=settings.looking_max_roll_degrees,
            palm_search_margin=settings.palm_search_margin,
        ),
    )
    debounce = PalmDebounce()

    frame_index = 0
    try:
        while True:
            frame = cv2.imdecode(
                np.frombuffer(await websocket.receive_bytes(), np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is None:
                continue

            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                await websocket.send_bytes(jpeg.tobytes())

            frame_index += 1
            if frame_index % DETECT_EVERY:
                continue

            # No FrameContext: this route is a debug surface and must never punch
            # anyone in. Attendance is recorded only by the camera runner.
            result = await process.process_frames(frame)
            payload = detection_payload(result, frame.shape[:2])
            payload["palm"] = debounce.update(result.palm.detected)
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        return


@router.websocket("/camera/ws")
async def camera_ws(websocket: WebSocket, container: ContainerDep) -> None:
    """Watch what the camera runner is doing. Read-only.

    This socket no longer owns a camera. The runner captures and recognises whether
    or not anyone is connected, and this simply forwards whatever it last published:
    new video as it arrives, and a new detection roughly once a second. Closing the
    tab stops nothing.
    """
    await websocket.accept()
    hub = container.frame_hub
    last_frame = -1
    last_detection = -1
    try:
        if not container.camera_runner.running:
            await websocket.send_json(
                {"error": "camera runner is stopped -- POST /api/v1/camera/start"}
            )
        while True:
            snapshot = hub.snapshot()
            if snapshot.jpeg is not None and snapshot.frame_seq != last_frame:
                last_frame = snapshot.frame_seq
                await websocket.send_bytes(snapshot.jpeg)
            if snapshot.detection is not None and snapshot.detection_seq != last_detection:
                last_detection = snapshot.detection_seq
                await websocket.send_json(snapshot.detection)
            await asyncio.sleep(VIEWER_INTERVAL)
    except WebSocketDisconnect:
        return
