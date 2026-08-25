"""Step 1 interface and handlers: is a palm present in the frame?

Imports stay limited to cv2/numpy so this module (and therefore the webcam route)
works in a control-plane-only install, where ``insightface``/``onnxruntime`` are
absent. Nothing here may import ``app.ai.components``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from app.schemas.face_processing import PalmResult

# BlazePalm takes a square 192x192 NHWC input normalised to 0..1.
BLAZEPALM_INPUT_SIZE = 192


class BasePalmDetection(ABC):
    """Detects whether a hand/palm is present in a BGR frame."""

    @abstractmethod
    async def detect(self, frame_bgr: np.ndarray) -> PalmResult:
        """Return the best palm score in the frame and whether it clears the threshold."""


class BlazePalmDetection(BasePalmDetection):
    """MediaPipe BlazePalm (ONNX) run through ``cv2.dnn``.

    Only the score head is decoded. The model also emits box and landmark deltas, but
    turning those into coordinates needs a 2016-row SSD anchor table, and presence does
    not: score filtering is what decides whether a detection exists at all, so
    ``max(sigmoid(scores)) >= threshold`` is equivalent to "at least one palm survives".
    """

    def __init__(
        self,
        *,
        model_path: str | Path = "models/palm_detection_mediapipe_2023feb.onnx",
        score_threshold: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Palm detector model not found at {model_path}. "
                "Download it with `python -m app.ai._download`, or manually from "
                "https://huggingface.co/opencv/palm_detection_mediapipe"
            )
        self._net = cv2.dnn.readNet(str(model_path))
        self._output_names = self._net.getUnconnectedOutLayersNames()
        self._score_threshold = score_threshold

    async def detect(self, frame_bgr: np.ndarray) -> PalmResult:
        # cv2.dnn.forward is blocking and releases the GIL; keep it off the event loop.
        score = await asyncio.to_thread(self._best_score, frame_bgr)
        return PalmResult(detected=score >= self._score_threshold, score=score)

    def _best_score(self, frame_bgr: np.ndarray) -> float:
        self._net.setInput(self._letterbox(frame_bgr)[np.newaxis])
        outputs = self._net.forward(self._output_names)
        # outputs: [(1, 2016, 18) box+landmark deltas, (1, 2016, 1) score logits]
        logits = outputs[1][0, :, 0].astype(np.float64)
        return float(1.0 / (1.0 + np.exp(-logits.max())))

    def _letterbox(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Resize preserving aspect ratio, pad to square, to RGB, scale to 0..1.

        Built by hand rather than with ``cv2.dnn.blobFromImage`` because that emits
        NCHW and this model expects NHWC.
        """
        size = BLAZEPALM_INPUT_SIZE
        height, width = frame_bgr.shape[:2]
        ratio = min(size / height, size / width)
        resized = cv2.resize(frame_bgr, (round(width * ratio), round(height * ratio)))

        pad_h = size - resized.shape[0]
        pad_w = size - resized.shape[1]
        top, left = pad_h // 2, pad_w // 2
        padded = cv2.copyMakeBorder(
            resized,
            top,
            pad_h - top,
            left,
            pad_w - left,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        return np.asarray(rgb, dtype=np.float32) / 255.0


class MediaPipePalmDetection(BasePalmDetection): ...
