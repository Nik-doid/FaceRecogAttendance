"""ArcFace embedding via the InsightFace ONNX implementation.

Uses ``w600k_r50.onnx`` (ArcFace, 512-d) which is the recognition component of the
``buffalo_l`` pack. Landmark-based alignment (``norm_crop``) is applied before the
forward pass, which is what makes the embeddings robust to head pose within the
quality-gate limits.

Loaded through the concrete ``ArcFaceONNX`` class rather than
``model_zoo.get_model``, which can silently return ``None``.
"""

from __future__ import annotations

import cv2
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

    def embed(
        self, image_bgr: np.ndarray, kps: np.ndarray, degrade_to: int = 0
    ) -> np.ndarray:
        """Embed the aligned face. ``degrade_to`` low-passes the crop first.

        ``degrade_to`` exists for one experiment, and it is an experiment about
        *symmetry* rather than about quality. A live face at the camera and an
        enrolment portrait are compared in the same 512-d space, but they do not
        arrive through the same optics: the live crop has lost its high-frequency
        detail to distance and to JPEG, and the portrait has not. Resampling the
        aligned enrolment crop down to the live faces' typical width and back up puts
        the same loss on both sides, so the cosine measures identity rather than
        measuring which camera took the picture.

        It throws real detail away, so it is only ever worth it if measurement says
        the mismatch costs more than the detail does -- run it through
        ``app/services/evaluation.py`` and read rank-1 and EER, never the mean genuine
        score, which rises for uninteresting reasons whenever both sides get blurrier.
        0 disables it, which is production.

        Deliberately applied to the *aligned* crop and not the source photo: alignment
        is driven by landmarks found at full resolution, so degrading first would
        degrade the landmarks too and confound the experiment with a second effect.
        """
        aimg = self._face_align.norm_crop(image_bgr, landmark=kps, image_size=self._image_size)
        if degrade_to and degrade_to < self._image_size:
            small = cv2.resize(
                aimg, (degrade_to, degrade_to), interpolation=cv2.INTER_AREA
            )
            # Back up with the same kernel insightface's own alignment would use, so
            # the only difference from an undegraded crop is the detail that is gone.
            aimg = cv2.resize(
                small,
                (self._image_size, self._image_size),
                interpolation=cv2.INTER_CUBIC,
            )
        feat = self._model.get_feat(aimg)
        return np.asarray(feat, dtype="float32").flatten()
