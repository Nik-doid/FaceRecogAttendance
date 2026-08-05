"""Silent-Face-Anti-Spoofing via an exported ONNX MiniFASNet model.

The model is a binary classifier (live vs. spoof) operating on a 224x224 RGB crop.
This implementation is deliberately generic over the ONNX I/O shape: it supports
both 2-class logit outputs (softmax, class 1 = live) and scalar sigmoid outputs.
The model file is provided via ``SILENTFACE_MODEL_PATH``. Exporting MiniFASNet to
ONNX is a one-time ops task; the runtime only needs the ``.onnx`` file.

If the model file is missing at construction, we fail fast rather than silently
disabling anti-spoofing.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.ai._loader import import_optional
from app.ai.liveness.base import LivenessChecker
from app.ai.types import DetectedFace

_IMAGE_SIZE = 224
_MEAN = np.array([0.5, 0.5, 0.5], dtype="float32")
_STD = np.array([0.5, 0.5, 0.5], dtype="float32")


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return np.asarray(exp / exp.sum(axis=-1, keepdims=True))


class SilentFaceLiveness(LivenessChecker):
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.50,
        providers: list[str] | None = None,
    ) -> None:
        onnxruntime = import_optional("onnxruntime")
        self._threshold = threshold
        self._session = onnxruntime.InferenceSession(
            model_path, providers=providers or None
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_count = len(self._session.get_outputs())

    def is_live(self, image_bgr: np.ndarray, face: DetectedFace) -> tuple[bool, float]:
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return False, 0.0

        crop = image_bgr[y1:y2, x1:x2]
        crop = cv2.resize(crop, (_IMAGE_SIZE, _IMAGE_SIZE))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = crop.astype("float32") / 255.0
        tensor = (tensor - _MEAN) / _STD
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        outputs = self._session.run(None, {self._input_name: tensor})
        raw = outputs[0][0]

        if self._output_count == 1 and raw.ndim == 1 and raw.shape[0] == 1:
            score = float(raw[0])
            score = 1.0 / (1.0 + float(np.exp(-score)))
        elif raw.ndim >= 1 and raw.shape[-1] >= 2:
            probs = _softmax(np.asarray(raw))
            score = float(probs[1])
        else:
            score = float(raw.reshape(-1)[0])

        return score >= self._threshold, score
