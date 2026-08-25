"""Browser-webcam WebSocket: the page sends JPEG frames, the server sends them back.

Frames are echoed first and only then processed, so palm detection never delays the
video. The detection result is pushed as a JSON *text* frame on the same socket, and
only when the palm state changes -- the page distinguishes the two by message type.
"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config.settings import settings
from app.schemas.face_processing import FaceProcessConfig
from app.services.face_recognition.process import FaceRecognitionProcess

router = APIRouter(tags=["webcam"])

JPEG_QUALITY = 70
DETECT_EVERY = 2  # process every 2nd frame (~7 Hz at the page's 15 fps)
CONFIRM_FRAMES = 2  # consecutive agreeing results before flipping the reported state


@router.websocket("/webcam/ws")
async def webcam_ws(websocket: WebSocket) -> None:
    """Echo the client's webcam frames back, and report palm presence as it changes."""
    await websocket.accept()
    # Per-connection: the process owns a cv2.dnn net that must not be shared.
    process = FaceRecognitionProcess(FaceProcessConfig(), settings.models_dir)

    frame_index = 0
    reported = False  # palm state last sent to the page
    pending = 0  # consecutive detections disagreeing with `reported`
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
            if result.palm.detected == reported:
                pending = 0
                continue
            # Require agreement across frames so the badge does not flicker.
            pending += 1
            if pending >= CONFIRM_FRAMES:
                reported = result.palm.detected
                pending = 0
                await websocket.send_json(
                    {"palm": reported, "score": round(result.palm.score, 3)}
                )
    except WebSocketDisconnect:
        return
