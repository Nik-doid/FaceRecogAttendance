"""Recognizer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Recognizer(ABC):
    """Embeds an aligned face region into a fixed-size feature vector."""

    @abstractmethod
    def embed(self, image_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """Return a normalized embedding for the face in ``image_bgr``.

        ``kps`` are the 5-point landmarks from the detector, used for geometric
        alignment so embeddings are comparable across poses/angles.
        """
