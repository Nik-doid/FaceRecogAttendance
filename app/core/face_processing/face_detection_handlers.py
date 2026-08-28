"""Step 2 interface and handlers: locate faces in the frame.

Imports stay free of the ``ai`` extra at module scope: ``SCRFDDetector`` pulls
``insightface`` in lazily from its own ``__init__``, so importing this module (and
therefore the webcam route) still works in a control-plane-only install. Nothing here
may import ``app.ai.components``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import numpy as np

from app.ai.detector.scrfd import SCRFDDetector
from app.schemas.face_processing import FaceResult

# SCRFD's letterboxed square input. 640 is the size det_10g was trained at.
SCRFD_INPUT_SIZE = 640
SCRFD_MAX_FACES = 10


class BaseFaceDetection(ABC):
    """Detects faces in a BGR frame."""

    @abstractmethod
    async def detect(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        """Return zero or more detected faces, best score first."""


class ScrfdFaceDetection(BaseFaceDetection):
    """SCRFD-10g (``det_10g.onnx``) through the existing :class:`SCRFDDetector`.

    Delegates rather than re-decoding the model: SCRFD's output needs a per-stride
    anchor table and NMS, and ``app/ai/detector/scrfd.py`` already does that against
    the insightface reference implementation. The 5-point landmarks are carried through
    on the result: step 3 needs them to align the crop before embedding.
    """

    def __init__(self, detector: SCRFDDetector) -> None:
        self._detector = detector

    async def detect(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        # The ONNX session is blocking; keep it off the event loop like BlazePalm.
        faces = await asyncio.to_thread(self._detector.detect, frame_bgr)
        # SCRFDDetector already sorts by score descending and applies det_thresh.
        return [
            FaceResult(
                bbox=face.bbox,
                score=face.score,
                kps=None
                if face.kps is None
                else [(float(x), float(y)) for x, y in face.kps],
            )
            for face in faces
        ]
