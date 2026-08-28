"""Browser-webcam WebSocket and Camera WebSocket streaming.

Both sockets run the same pipeline and emit the same JSON shape, so one page renderer
serves both. They differ in how frames arrive and in what that costs:

- ``/webcam/ws`` receives frames from the browser and answers each one, so detection
  runs inline -- the client is already paced by its own send interval.
- ``/camera/ws`` pulls from RTSP at full rate, so detection runs as a background task
  and the stream never waits on it. Most frames stop at face detection (nobody is
  looking at the camera), but the ones that do not can run SCRFD, a palm scan and an
  ArcFace pass back to back -- far too much to sit in the send loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.deps import ContainerDep
from app.config.settings import Settings, settings
from app.core.logging import get_logger
from app.schemas.face_processing import FaceProcessConfig, FrameResult
from app.services.face_recognition.process import FaceRecognitionProcess
from app.workers.camera.reader import CameraReader

router = APIRouter(tags=["webcam"])
log = get_logger(__name__)

JPEG_QUALITY = 70
DETECT_EVERY = 2
CONFIRM_FRAMES = 2
CAMERA_FRAME_INTERVAL = 0.033  # ~30 fps


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


def _detection_payload(
    result: FrameResult, palm: bool, shape: tuple[int, int]
) -> dict[str, Any]:
    """The one message shape both sockets send. ``shape`` is the detected frame's (h, w)."""
    height, width = shape
    return {
        "palm": palm,
        "score": round(result.palm.score, 3),
        # Boxes are in frame pixels; the page scales by these dimensions.
        "width": width,
        "height": height,
        # kps are deliberately omitted: the page draws boxes, not landmarks.
        "faces": [
            {
                "bbox": [round(v, 1) for v in face.bbox],
                "score": round(face.score, 3),
                "looking": face.looking,
                "yaw": face.yaw_ratio,
                "palm": face.palm,
                "employee_code": face.employee_code,
                "confidence": round(face.confidence, 3),
            }
            for face in result.faces
        ],
    }


def _create_camera_reader(settings: Settings) -> CameraReader:
    return CameraReader(
        rtsp_url=settings.rtsp_url,
        source=settings.camera_source,
        device_index=settings.camera_device_index,
        max_width=settings.max_frame_width,
    )


def _create_process(
    settings: Settings, config: FaceProcessConfig
) -> FaceRecognitionProcess:
    # Per-connection: the process owns a cv2.dnn net that must not be shared. The
    # recognition gallery behind it is process-wide and cached (see gallery.py).
    return FaceRecognitionProcess(
        config, settings.models_dir, settings.employee_photos_source
    )


@router.websocket("/webcam/ws")
async def webcam_ws(websocket: WebSocket) -> None:
    """Echo the client's webcam frames back, with palm state and face boxes alongside.

    One JSON message per *detected* frame (every ``DETECT_EVERY`` frames), so the page
    can move the face boxes as the subject moves. Only the palm badge is debounced;
    boxes are whatever the latest detection found.
    """
    await websocket.accept()
    # Whole-frame palm scan: a hand held up to a laptop already fills enough of it.
    process = _create_process(
        settings,
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

            result = await process.process_frames(frame)
            await websocket.send_json(
                _detection_payload(
                    result, debounce.update(result.palm.detected), frame.shape[:2]
                )
            )
    except WebSocketDisconnect:
        return


def _harvest(
    task: asyncio.Task[FrameResult], debounce: PalmDebounce, shape: tuple[int, int]
) -> dict[str, Any] | None:
    """Turn a finished detection task into a payload, or None if it failed.

    A frame that blows up in the pipeline must not take the stream down with it.
    """
    try:
        result = task.result()
    except asyncio.CancelledError:
        return None
    except Exception:  # noqa: BLE001 - one bad frame must not end the stream
        log.exception("camera frame detection failed")
        return None
    return _detection_payload(result, debounce.update(result.palm.detected), shape)


@router.websocket("/camera/ws")
async def camera_ws(websocket: WebSocket, container: ContainerDep) -> None:
    """Stream camera frames (RTSP or device) as JPEG over WebSocket, with detection.

    Source is determined by CAMERA_SOURCE in .env:
    - "rtsp": uses RTSP_URL
    - "device": uses CAMERA_DEVICE_INDEX

    Detection runs on every ``CAMERA_DETECT_EVERY``-th frame as a background task and
    is harvested whenever it finishes, so a slow scan delays the boxes, never the video.
    """
    await websocket.accept()
    settings: Settings = container.settings
    reader = _create_camera_reader(settings)

    if not reader.open():
        await websocket.send_json({"error": "failed to open camera stream"})
        await websocket.close()
        return

    process = _create_process(
        settings,
        FaceProcessConfig(
            palm_score_threshold=settings.palm_score_threshold,
            # Only reached if the grid scan is needed, i.e. a face with no landmarks
            # to anchor the palm search to. The looking gate makes that rare.
            palm_scan_grid=settings.palm_scan_grid,
            palm_scan_overlap=settings.palm_scan_overlap,
            looking_max_yaw_ratio=settings.looking_max_yaw_ratio,
            looking_max_roll_degrees=settings.looking_max_roll_degrees,
            palm_search_margin=settings.palm_search_margin,
        ),
    )
    debounce = PalmDebounce()
    detecting: asyncio.Task[FrameResult] | None = None
    detected_shape = (0, 0)
    frame_index = 0

    try:
        while True:
            frame = reader.read()
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                await websocket.send_bytes(jpeg.tobytes())

            if detecting is not None and detecting.done():
                payload = _harvest(detecting, debounce, detected_shape)
                detecting = None
                if payload is not None:
                    await websocket.send_json(payload)

            frame_index += 1
            # Skip the frame entirely while a scan is in flight rather than queueing
            # work the stream has already moved past.
            if detecting is None and frame_index % settings.camera_detect_every == 0:
                detected_shape = frame.shape[:2]
                # reader.read() hands back a fresh array per call, so no copy is needed.
                detecting = asyncio.create_task(process.process_frames(frame))

            await asyncio.sleep(CAMERA_FRAME_INTERVAL)
    except WebSocketDisconnect:
        return
    finally:
        if detecting is not None:
            detecting.cancel()
        reader.close()
