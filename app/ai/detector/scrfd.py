"""SCRFD face detector via the InsightFace model zoo.

Uses ``scrfd_10g_bnkps`` (ONNX) which returns bounding boxes + 5-point landmarks
in one pass. Detection is decoupled from recognition: this class never computes
embeddings, so the recognition model stays a separate loaded artifact.
"""

from __future__ import annotations

import numpy as np

from app.ai._loader import import_optional
from app.ai.detector.base import Detector
from app.ai.types import DetectedFace


class SCRFDDetector(Detector):
    def __init__(
        self,
        model_name: str = "scrfd_10g_bnkps",
        providers: list[str] | None = None,
        input_size: int = 640,
        max_num: int = 10,
        det_thresh: float = 0.5,
        models_dir: str | None = None,
    ) -> None:
        insightface = import_optional("insightface")
        kwargs: dict[str, object] = {"providers": providers or []}
        if models_dir:
            kwargs["root"] = models_dir
        self._model = insightface.model_zoo.get_model(model_name, **kwargs)
        self._input_size = input_size
        self._max_num = max_num
        self._det_thresh = det_thresh

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        bboxes, kpss = self._model.detect(
            image_bgr,
            input_size=self._input_size,
            max_num=self._max_num,
            metric="default",
            det_thresh=self._det_thresh,
        )
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
