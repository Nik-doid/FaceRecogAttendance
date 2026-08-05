"""ArcFace embedding via the InsightFace model zoo (ONNX).

Uses ``w600k_r50`` (ArcFace, 512-d) which is the recognition component of the
``buffalo_l`` pack. Landmark-based alignment (``norm_crop``) is applied before the
forward pass, which is what makes the embeddings robust to head pose within the
quality-gate limits.
"""

from __future__ import annotations

import numpy as np

from app.ai._loader import import_optional
from app.ai.recognizer.base import Recognizer


class ArcFaceRecognizer(Recognizer):
    def __init__(
        self,
        model_name: str = "w600k_r50",
        providers: list[str] | None = None,
        image_size: int = 112,
        models_dir: str | None = None,
    ) -> None:
        insightface = import_optional("insightface")
        kwargs: dict[str, object] = {"providers": providers or []}
        if models_dir:
            kwargs["root"] = models_dir
        self._model = insightface.model_zoo.get_model(model_name, **kwargs)
        self._image_size = image_size
        self._face_align = insightface.utils.face_align

    def embed(self, image_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        aimg = self._face_align.norm_crop(image_bgr, landmark=kps, image_size=self._image_size)
        feat = self._model.get_feat(aimg)
        return np.asarray(feat, dtype="float32").flatten()
