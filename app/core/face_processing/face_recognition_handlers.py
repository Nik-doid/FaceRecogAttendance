"""Step 3 interface and handlers: put a name to each detected face.

Imports stay free of the ``ai`` extra at module scope for the same reason as
``face_detection_handlers``: everything heavy is pulled in lazily by the gallery.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import numpy as np

from app.core.face_processing.gallery import Gallery
from app.schemas.face_processing import FaceResult


class BaseFaceRecognition(ABC):
    """Matches detected faces against the enrolled employee index."""

    @abstractmethod
    async def recognize(
        self, frame_bgr: np.ndarray, faces: list[FaceResult]
    ) -> list[FaceResult]:
        """Return the faces annotated with whatever identity was matched."""


class ArcFaceRecognition(BaseFaceRecognition):
    """ArcFace (``w600k_r50.onnx``) embeddings searched against the employee gallery.

    Embeds each detected face from the *live* frame using the landmarks step 2 found,
    then takes the nearest enrolled employee by cosine similarity. Faces below
    ``score_threshold`` keep their score but get no ``employee_code``.
    """

    def __init__(self, gallery: Gallery, score_threshold: float = 0.6) -> None:
        self._gallery = gallery
        self._score_threshold = score_threshold

    @property
    def enrolled_employees(self) -> int:
        return self._gallery.employees

    async def recognize(
        self, frame_bgr: np.ndarray, faces: list[FaceResult]
    ) -> list[FaceResult]:
        if not faces:
            return faces
        # One ArcFace forward pass per face; keep them off the event loop together.
        return await asyncio.to_thread(self._match_all, frame_bgr, faces)

    def _match_all(self, frame_bgr: np.ndarray, faces: list[FaceResult]) -> list[FaceResult]:
        return [self._match(frame_bgr, face) for face in faces]

    def _match(self, frame_bgr: np.ndarray, face: FaceResult) -> FaceResult:
        if face.kps is None:
            # ArcFace aligns on the 5 landmarks; without them there is nothing to embed.
            return face
        if self._gallery.recognizer is None:
            return face  # Nothing enrolled, so nothing to match against.

        embedding = self._gallery.recognizer.embed(
            frame_bgr, np.asarray(face.kps, dtype="float32")
        )
        best = self._gallery.index.search(embedding, k=1)
        if not best:
            return face  # Empty gallery, or no candidate survived the search.

        top = best[0]
        matched = top.score >= self._score_threshold
        return face.model_copy(
            update={
                "employee_code": top.employee_code if matched else None,
                "confidence": top.score,
            }
        )
