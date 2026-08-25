"""Step 3 interface and handlers: put a name to each detected face.

Not implemented yet -- see ``face_detection_handlers`` for the rationale.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.schemas.face_processing import FaceResult


class BaseFaceRecognition(ABC):
    """Matches detected faces against the enrolled employee index."""

    @abstractmethod
    async def recognize(
        self, frame_bgr: np.ndarray, faces: list[FaceResult]
    ) -> list[FaceResult]:
        """Return the faces annotated with whatever identity was matched."""


class ArcFaceRecognition(BaseFaceRecognition): ...
