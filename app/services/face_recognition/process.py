"""The frame-processing domain service.

Each step is a class resolved through its dispatcher, so swapping an implementation is
a change of enum value in ``FaceProcessConfig`` -- never an edit to ``process_frames``.

Unlike the dispatchers this is modelled on, handlers are dispatched once in ``__init__``
rather than per call: the palm handler owns a loaded 3.7 MiB ``cv2.dnn`` net, which is
far too expensive to rebuild per frame and is not safe to share across threads. Build one
process per consumer (e.g. per WebSocket connection) and the net stays private to it.

The looking gate between steps 1 and 3 is the exception to the dispatcher rule: it is
arithmetic over landmarks step 1 already produced, with no model behind it and nothing
to swap, so it lives as a plain function in ``app/core/face_processing/gaze.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from app.core.face_processing.dispatchers import (
    FaceDetectionDispatcher,
    FaceRecognitionDispatcher,
    PalmDetectionDispatcher,
)
from app.core.face_processing.gaze import estimate_gaze
from app.core.face_processing.palm_detection_handlers import palm_search_box
from app.core.logging import get_logger
from app.schemas.face_processing import (
    FaceProcessConfig,
    FaceResult,
    FrameResult,
    PalmResult,
)

log = get_logger(__name__)

NO_PALM = PalmResult(detected=False, score=0.0)


class FaceRecognitionProcess:
    def __init__(
        self,
        config: FaceProcessConfig,
        models_dir: Path,
        photo_sources: Sequence[str | Path] = (),
    ) -> None:
        self.config = config
        self.models_dir = models_dir
        self.photo_sources = photo_sources

        self.palm_detection = PalmDetectionDispatcher.dispatch(
            config.palm_detector,
            models_dir,
            config.palm_score_threshold,
            config.palm_scan_grid,
            config.palm_scan_overlap,
        )
        self.face_detection = FaceDetectionDispatcher.dispatch(
            config.face_detector,
            models_dir,
            config.face_score_threshold,
        )
        self.face_recognition = FaceRecognitionDispatcher.dispatch(
            config.face_recognizer,
            models_dir,
            photo_sources,
            config.recognition_threshold,
        )

    async def process_frames(self, frame_bgr: np.ndarray) -> FrameResult:
        """
        process each webcam frame into the following steps:
            1. face detection
            2. looking gate
            3. palm detection
            4. face recognition
            5. mark attendance     (not implemented)

        Step 2 gates the rest: nobody facing the camera means steps 3-5 never run.
        Face detection is what *decides* step 2, so it is the one step that always
        runs -- there is no cheaper way to know whether anyone is looking. That makes
        an empty room cost one SCRFD pass per frame and nothing else.

        Step 3 is scored per face, against that face's own search box, and step 4 runs
        only for the faces that scored. Someone standing beside a colleague who raises
        a hand is detected and reported, but never embedded or identified.

        Args:
            frame_bgr (np.ndarray): one decoded BGR frame.
        """
        # 1. face detection
        faces = await self.face_detection.detect(frame_bgr)
        if not faces:
            return FrameResult(palm=NO_PALM)

        # 2. looking gate -- free, it only reads the landmarks step 1 returned.
        faces = [self._with_gaze(face) for face in faces]
        looking = [index for index, face in enumerate(faces) if face.looking]
        if not looking:
            # Faces are still returned: the page draws who is present, unengaged.
            log.debug("no face is looking at the camera", extra={"faces": len(faces)})
            return FrameResult(palm=NO_PALM, faces=faces)

        # 3. palm detection -- one search box per looking face, so the verdict is
        # attributable to a person rather than to the frame.
        regions = [
            palm_search_box(
                faces[index].bbox, frame_bgr.shape[:2], self.config.palm_search_margin
            )
            for index in looking
        ]
        palms = await self.palm_detection.detect(frame_bgr, regions)
        for index, palm in zip(looking, palms, strict=True):
            faces[index] = faces[index].model_copy(
                update={"palm": palm.detected, "palm_score": round(palm.score, 3)}
            )

        frame_palm = max(palms, key=lambda result: result.score, default=NO_PALM)
        raised = [index for index in looking if faces[index].palm]
        if not raised:
            return FrameResult(palm=frame_palm, faces=faces)
        log.debug("palm detected", extra={"hands": len(raised)})

        # 4. face recognition -- only the people who raised a hand.
        recognized = await self.face_recognition.recognize(
            frame_bgr, [faces[index] for index in raised]
        )
        for index, face in zip(raised, recognized, strict=True):
            faces[index] = face

        # Step 5 lands here. Until then the recognised faces are the whole result.
        return FrameResult(palm=frame_palm, faces=faces)

    def _with_gaze(self, face: FaceResult) -> FaceResult:
        gaze = estimate_gaze(
            face.kps,
            self.config.looking_max_yaw_ratio,
            self.config.looking_max_roll_degrees,
        )
        return face.model_copy(
            update={
                "looking": gaze.looking,
                "yaw_ratio": round(gaze.yaw_ratio, 3),
                "roll_degrees": round(gaze.roll_degrees, 1),
            }
        )
