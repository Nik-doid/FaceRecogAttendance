"""The frame-processing domain service.

Each step is a class resolved through its dispatcher, so swapping an implementation is
a change of enum value in ``FaceProcessConfig`` -- never an edit to ``process_frames``.

Unlike the dispatchers this is modelled on, handlers are dispatched once in ``__init__``
rather than per call: the palm handler owns a loaded 3.7 MiB ``cv2.dnn`` net, which is
far too expensive to rebuild per frame and is not safe to share across threads. Build one
process per consumer (e.g. per WebSocket connection) and the net stays private to it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.face_processing.dispatchers import PalmDetectionDispatcher
from app.core.logging import get_logger
from app.schemas.face_processing import FaceProcessConfig, FrameResult

log = get_logger(__name__)


class FaceRecognitionProcess:
    def __init__(
        self,
        config: FaceProcessConfig,
        models_dir: Path,
    ) -> None:
        self.config = config
        self.models_dir = models_dir

        self.palm_detection = PalmDetectionDispatcher.dispatch(
            config.palm_detector,
            models_dir,
            config.palm_score_threshold,
        )

    async def process_frames(self, frame_bgr: np.ndarray) -> FrameResult:
        """
        process each webcam frame into the following steps:
            1. palm detection
            2. face detection      (not implemented)
            3. face recognition    (not implemented)
            4. mark attendance     (not implemented)

        Step 1 gates the rest: no palm in frame means steps 2-4 never run.

        Args:
            frame_bgr (np.ndarray): one decoded BGR frame.
        """
        # 1. palm detection
        palm = await self.palm_detection.detect(frame_bgr)
        if not palm.detected:
            return FrameResult(palm=palm)
        log.debug("palm detection successful", extra={"score": palm.score})

        # Steps 2-4 land here. Until then a detected palm is the whole result.
        return FrameResult(palm=palm)
