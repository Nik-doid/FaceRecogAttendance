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

from pathlib import Path

import numpy as np

from app.core.face_processing.dispatchers import (
    FaceDetectionDispatcher,
    FaceRecognitionDispatcher,
    MarkAttendanceDispatcher,
    PalmDetectionDispatcher,
)
from app.core.face_processing.gallery import Gallery
from app.core.face_processing.gaze import estimate_gaze
from app.core.face_processing.palm_detection_handlers import palm_search_box
from app.core.logging import get_logger
from app.core.metrics import (
    FACES_TOO_SMALL,
    TRACK_SCANS_TO_CONFIRM,
    TRACKS_CONFIRMED,
    TRACKS_UNRESOLVED,
)
from app.runtime import Models
from app.schemas.face_processing import (
    FaceProcessConfig,
    FaceResult,
    FrameContext,
    FrameResult,
    PalmResult,
)
from app.services.attendance_reporter.base import AttendanceReporter
from app.services.duplicate_suppressor import DuplicateSuppressor
from app.services.face_recognition.tracker import FaceTracker

log = get_logger(__name__)

NO_PALM = PalmResult(detected=False, score=0.0)


class FaceRecognitionProcess:
    # Class-level default so an instance built past __init__ still has it. Tests
    # assemble this class with `__new__` plus attribute assignment to avoid loading
    # models (see tests/test_min_face_pixels.py), and "no tracker" is the correct
    # default state anyway: per-frame decisions, exactly as before tracking existed.
    _tracker: FaceTracker | None = None

    def __init__(
        self,
        config: FaceProcessConfig,
        models: Models,
        gallery: Gallery,
        models_dir: Path,
        reporter: AttendanceReporter | None = None,
        suppressor: DuplicateSuppressor | None = None,
        tracker: FaceTracker | None = None,
    ) -> None:
        self.config = config
        self.models_dir = models_dir
        # Injected, never owned. Track identity belongs to one camera stream, and two
        # browser viewers must not share tracks with each other or with the RTSP feed
        # -- so the object with camera-stream lifetime creates it. None means the
        # per-frame behaviour this class had before tracking existed, which is what
        # every test and the browser debug route get.
        self._tracker = tracker

        # Palm is the one step that still builds its own net: cv2.dnn is not
        # thread-safe, so it must stay private to this process instance.
        self.palm_detection = PalmDetectionDispatcher.dispatch(
            config.palm_detector,
            models_dir,
            config.palm_score_threshold,
            config.palm_scan_grid,
            config.palm_scan_overlap,
        )
        self.face_detection = FaceDetectionDispatcher.dispatch(
            config.face_detector, models.detector
        )
        self.face_recognition = FaceRecognitionDispatcher.dispatch(
            config.face_recognizer,
            gallery,
            config.recognition_threshold,
            config.recognition_margin,
        )
        self.mark_attendance = MarkAttendanceDispatcher.dispatch(
            config.attendance_sink, reporter, suppressor
        )

    async def process_frames(
        self, frame_bgr: np.ndarray, ctx: FrameContext | None = None
    ) -> FrameResult:
        """
        process each webcam frame into the following steps:
            1. face detection
            2. looking gate
            3. palm detection
            4. face recognition
            5. mark attendance

        With a tracker injected the pipeline is no longer stateless across frames --
        that is the one piece of state it may hold, and it exists so a decision is
        made once per *person* rather than once per frame.

        Step 2 gates the rest: nobody facing the camera means steps 3-5 never run.
        Face detection is what *decides* step 2, so it is the one step that always
        runs -- there is no cheaper way to know whether anyone is looking. That makes
        an empty room cost one SCRFD pass per frame and nothing else.

        Step 3 is scored per face, against that face's own search box, and step 4 runs
        only for the faces that scored. Someone standing beside a colleague who raises
        a hand is detected and reported, but never embedded or identified.

        Args:
            frame_bgr (np.ndarray): one decoded BGR frame.
            ctx (FrameContext | None): camera and capture time. Step 5 needs both and
                a face carries neither. None means "do not record attendance" -- the
                browser-webcam debug route passes nothing, so nobody can punch a
                colleague in by holding a palm up to a laptop.
        """
        # 1. face detection
        faces = await self.face_detection.detect(frame_bgr)
        if not faces:
            log.debug(
                "no face detected", extra={"frame_width": int(frame_bgr.shape[1])}
            )
            return FrameResult(palm=NO_PALM)
        # Face width in source pixels is the number that decides whether an RTSP feed
        # can work at all: SCRFD sees it scaled by DETECT_INPUT_SIZE/frame_width, and
        # ArcFace resamples it to 112x112. Under ~60px here means recognition is
        # guessing no matter how the thresholds are set -- move the camera or raise
        # DETECT_INPUT_SIZE rather than lowering RECOGNITION_THRESHOLD.
        log.debug(
            "faces detected",
            extra={
                "frame_width": int(frame_bgr.shape[1]),
                "face_widths_px": [round(f.bbox[2] - f.bbox[0]) for f in faces],
                # Where in the frame people actually stand. The camera is fixed, so if
                # these cluster in a band, detection can be cropped to it -- which
                # buys the resolution a bigger DETECT_INPUT_SIZE buys, at a fraction
                # of the cost, and drops the ceiling and floor regions that can only
                # produce false positives. Collect a day of these before designing it.
                "face_centres": [
                    (round((f.bbox[0] + f.bbox[2]) / 2), round((f.bbox[1] + f.bbox[3]) / 2))
                    for f in faces
                ],
            },
        )

        # 1b. track people across scans. Before the gaze gate on purpose: a track has
        # to survive the scan where somebody glanced away, and matching boxes needs no
        # idea whether they were looking.
        if self._tracker is not None:
            ids = self._tracker.update([face.bbox for face in faces])
            faces = [
                face.model_copy(update={"track_id": track_id})
                for face, track_id in zip(faces, ids, strict=True)
            ]

        # 2. looking gate -- free, it only reads the landmarks step 1 returned.
        faces = [self._with_gaze(face) for face in faces]
        looking = [index for index, face in enumerate(faces) if face.looking]
        if not looking:
            # Faces are still returned: the page draws who is present, unengaged.
            # Landmark precision falls with face size, so a small RTSP face can be
            # frontal and still measure a jittery yaw. The numbers are here so the
            # gate can be tuned against real footage instead of by feel.
            log.debug(
                "no face is looking at the camera",
                extra={
                    "faces": len(faces),
                    "yaw_ratios": [f.yaw_ratio for f in faces],
                    "roll_degrees": [f.roll_degrees for f in faces],
                },
            )
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

        # 4. face recognition -- only the people who raised a hand, and only those
        # whose face is big enough for the embedding to mean anything. A 30px box
        # aligned up to ArcFace's 112x112 scores 0.00 against everyone, which reads
        # as "not enrolled" and is really "too far from the camera".
        too_small = [index for index in raised if self._too_small(faces[index])]
        for index in too_small:
            faces[index] = faces[index].model_copy(update={"too_small": True})
        if too_small:
            FACES_TOO_SMALL.inc(len(too_small))
            log.info(
                "face too small to recognise",
                extra={
                    "event": "face_too_small",
                    "min_face_pixels": self.config.min_face_pixels,
                    "face_widths_px": [
                        round(faces[i].bbox[2] - faces[i].bbox[0]) for i in too_small
                    ],
                },
            )
        recognisable = [index for index in raised if not faces[index].too_small]
        recognized = await self.face_recognition.recognize(
            frame_bgr, [faces[index] for index in recognisable]
        )
        for index, face in zip(recognisable, recognized, strict=True):
            faces[index] = face

        # 4c. decide once per person. Without a tracker every raised hand is its own
        # verdict, which is the old behaviour; with one, a run of scans has to agree on
        # the same employee before anybody is punched in.
        publishable = [faces[index] for index in raised]
        if self._tracker is not None:
            publishable = self._confirmed(publishable)

        # 5. mark attendance -- only when a context says which camera and when. The
        # browser-webcam debug route passes none, so it can never punch anyone in.
        if ctx is not None:
            await self.mark_attendance.mark(publishable, ctx)

        if self._tracker is not None:
            self._retire_tracks()
        return FrameResult(palm=frame_palm, faces=faces)

    def _confirmed(self, faces: list[FaceResult]) -> list[FaceResult]:
        """The faces whose track just reached agreement, at most once per track."""
        assert self._tracker is not None
        confirmed: list[FaceResult] = []
        for face in faces:
            if face.track_id is None:
                continue
            code = self._tracker.vote(face.track_id, face.employee_code, face.confidence)
            if code is None:
                continue
            TRACKS_CONFIRMED.inc()
            # How long somebody had to stand there. The only honest check on whether
            # the confirmation depth fits the few seconds people actually give it.
            TRACK_SCANS_TO_CONFIRM.observe(self._tracker.scans_of(face.track_id))
            confirmed.append(face)
        return confirmed

    def _retire_tracks(self) -> None:
        """Report every person who stood here and left without being identified.

        The four reasons are the diagnosis this service otherwise cannot give: a face
        too small to embed, a gallery with no candidate, a candidate that never cleared
        the threshold, and scans that never agreed all look identical from outside, and
        each has a different fix.
        """
        assert self._tracker is not None
        for summary in self._tracker.expire():
            TRACKS_UNRESOLVED.labels(reason=summary.reason).inc()
            log.info(
                "track ended unidentified",
                extra={
                    "event": "track_unresolved",
                    "track_id": summary.track_id,
                    "scans": summary.scans,
                    "reason": summary.reason,
                    "best_code": summary.best_code,
                    "best_score": round(summary.best_score, 3),
                    "max_width_px": round(summary.max_width),
                },
            )

    def _too_small(self, face: FaceResult) -> bool:
        minimum = self.config.min_face_pixels
        return bool(minimum) and (face.bbox[2] - face.bbox[0]) < minimum

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
