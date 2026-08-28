"""ArcFace embedding via the InsightFace ONNX implementation.

Uses ``w600k_r50.onnx`` (ArcFace, 512-d) which is the recognition component of the
``buffalo_l`` pack. Landmark-based alignment (``norm_crop``) is applied before the
forward pass, which is what makes the embeddings robust to head pose within the
quality-gate limits.

Loaded through the concrete ``ArcFaceONNX`` class rather than
``model_zoo.get_model``, which can silently return ``None``.
"""

from __future__ import annotations

import numpy as np

from app.ai._loader import filter_providers, import_optional, resolve_model_file
from app.ai.recognizer.base import Recognizer


class ArcFaceRecognizer(Recognizer):
    def __init__(
        self,
        model_name: str = "w600k_r50.onnx",
        providers: list[str] | None = None,
        image_size: int = 112,
        models_dir: str | None = None,
    ) -> None:
        insightface = import_optional("insightface")
        model_file = resolve_model_file(model_name, models_dir)
        self._model = insightface.model_zoo.ArcFaceONNX(model_file=model_file)
        available = filter_providers(providers)
        if available:
            self._model.session.set_providers(available)
        self._image_size = image_size
        self._face_align = insightface.utils.face_align

    @property
    def providers(self) -> list[str]:
        """Execution providers onnxruntime actually resolved for this session."""
        return list(self._model.session.get_providers())

    def embed(self, image_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        aimg = self._face_align.norm_crop(image_bgr, landmark=kps, image_size=self._image_size)
        feat = self._model.get_feat(aimg)
        return np.asarray(feat, dtype="float32").flatten()
