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
        # 0 = uncapped, and it must stay that way on a wide room shot. When more
        # detections than this survive, insightface ranks them by
        # `area - 2 * offset_dist_squared` in *pixel* units and keeps the top N: on a
        # 1920-wide frame a face 400px off centre contributes 2*400**2 = 320_000
        # against a 60px face's area of ~4_500, so the area term is noise and the
        # ranking is purely "closest to frame centre". The cap therefore discards
        # small off-centre faces first -- exactly the population a ceiling camera
        # sees. SCRFD's own per-pass NMS and det_thresh already bound the output to
        # real faces, so there is nothing left for the cap to do.
        max_num: int = 0,
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

    def detect(
        self, image_bgr: np.ndarray, input_size: int | None = None
    ) -> list[DetectedFace]:
        """Detect faces. ``input_size`` overrides the configured letterbox for one call.

        The override exists for the evaluation harness, which sweeps input sizes to
        find where this camera's faces stop being resolvable. insightface takes the
        size per call, so sweeping needs no second session -- and production passes
        nothing, keeping the configured value.
        """
        size = self._input_size if input_size is None else input_size
        try:
            bboxes, kpss = self._model.detect(
                image_bgr,
                input_size=(size, size),
                max_num=self._max_num,
                # Unreachable at max_num=0, which is the default and the only sane
                # value here; kept so the call still matches insightface's signature.
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
