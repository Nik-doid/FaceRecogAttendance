"""Step 2 interface and handlers: locate faces in the frame.

Not implemented yet -- the ABC and handler exist so wiring step 2 into
``process_frames`` later is an implementation, not a redesign.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.schemas.face_processing import FaceResult


class BaseFaceDetection(ABC):
    """Detects faces in a BGR frame."""

    @abstractmethod
    async def detect(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        """Return zero or more detected faces, best score first."""


class ScrfdFaceDetection(BaseFaceDetection): ...
