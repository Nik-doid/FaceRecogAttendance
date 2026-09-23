"""Step 3 interface and handlers: put a name to each detected face.

Imports stay free of the ``ai`` extra at module scope for the same reason as
``face_detection_handlers``: everything heavy is pulled in lazily by the gallery.
"""

from __future__ import annotations

import asyncio
import math
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

    Two gates, not one. The threshold is absolute; ``margin_threshold`` is the gap
    between the best employee and the runner-up, and the two fail differently. As a
    face loses detail every cosine against the gallery shrinks toward the gallery mean
    *together*, because the detail that vanishes first is the high-frequency structure
    common to all identities. An absolute floor is the wrong instrument for a
    common-mode shift like that -- it rejects genuine matches wholesale, and lowering
    it to compensate raises the false-accept rate across the whole gallery at once.
    The first-to-second gap is differential: the shared shift cancels, so it still
    says something when the absolute score has stopped meaning much.

    The margin costs one extra index row, which is microseconds and no extra ONNX
    pass. It ships at 0.0 (inert) because the value has to come from a FAR/FRR sweep
    over real footage -- see ``app/services/evaluation.py`` -- and a margin tuned
    against four employees does not transfer to five hundred, where the runner-up is
    a much stronger competitor.
    """

    def __init__(
        self,
        gallery: Gallery,
        score_threshold: float = 0.6,
        margin_threshold: float = 0.0,
    ) -> None:
        self._gallery = gallery
        self._score_threshold = score_threshold
        self._margin_threshold = margin_threshold

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
        # k=2 so the runner-up is available for the margin. `FaceIndex.search` groups
        # by employee keeping each one's best photo, so the second row is a different
        # person rather than the same person's second portrait.
        best = self._gallery.index.search(embedding, k=2)
        if not best:
            return face  # Empty gallery, or no candidate survived the search.

        top = best[0]
        # A single enrolled employee has no runner-up, so there is no gap to measure
        # and infinity is the honest value -- the alternative reads as a zero margin
        # and rejects the only person in the gallery.
        margin = top.score - best[1].score if len(best) >= 2 else math.inf
        matched = top.score >= self._score_threshold and margin >= self._margin_threshold
        return face.model_copy(
            update={
                "employee_code": top.employee_code if matched else None,
                "confidence": top.score,
                # Kept whether or not it passed, for the same reason `confidence` is:
                # the near-miss is the number you tune the gate against.
                "margin": 0.0 if math.isinf(margin) else margin,
            }
        )
