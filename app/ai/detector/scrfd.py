"""SCRFD face detector via the InsightFace ONNX implementation.

Uses ``det_10g.onnx`` (SCRFD-10g from the ``buffalo_l`` pack) which returns
bounding boxes + 5-point landmarks in one pass. Detection is decoupled from
recognition: this class never computes embeddings, so the recognition model stays
a separate loaded artifact.

The model is loaded through the concrete ``SCRFD`` class (not
``model_zoo.get_model``, which can silently return ``None`` and routes SCRFD
architectures to the wrong wrapper).
"""

from __future__ import annotations

import numpy as np

from app.ai._loader import filter_providers, import_optional, resolve_model_file
from app.ai.detector.base import Detector
from app.ai.types import DetectedFace


class SCRFDDetector(Detector):
    def __init__(
        self,
        model_name: str = "det_10g.onnx",
        providers: list[str] | None = None,
        input_size: int = 640,
        max_num: int = 10,
        det_thresh: float = 0.5,
        models_dir: str | None = None,
    ) -> None:
        insightface = import_optional("insightface")
        model_file = resolve_model_file(model_name, models_dir)
        self._model = insightface.model_zoo.SCRFD(model_file=model_file)
        self._model.det_thresh = det_thresh
        available = filter_providers(providers)
        if available:
            self._model.session.set_providers(available)
        self._input_size = input_size
        self._max_num = max_num
        self._det_thresh = det_thresh

    @property
    def providers(self) -> list[str]:
        """Execution providers onnxruntime actually resolved for this session."""
        return list(self._model.session.get_providers())

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        try:
            bboxes, kpss = self._model.detect(
                image_bgr,
                input_size=(self._input_size, self._input_size),
                max_num=self._max_num,
                metric="default",
            )
        except ValueError:
            # SCRFD raises when no detection survives the threshold (empty
            # scores/bboxes lists in forward); treat as a frame with no faces.
            return []
        faces: list[DetectedFace] = []
        if bboxes is None or len(bboxes) == 0:
            return faces
        for i in range(len(bboxes)):
            x1, y1, x2, y2, score = (float(v) for v in bboxes[i])
            kps = kpss[i] if kpss is not None and i < len(kpss) else None
            faces.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    kps=np.asarray(kps, dtype="float32") if kps is not None else None,
                )
            )
        faces.sort(key=lambda f: f.score, reverse=True)
        return faces
