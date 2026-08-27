"""Step 1 interface and handlers: is a palm present in the frame?

Imports stay limited to cv2/numpy so this module (and therefore the webcam route)
works in a control-plane-only install, where ``insightface``/``onnxruntime`` are
absent. Nothing here may import ``app.ai.components``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from app.schemas.face_processing import PalmResult

# BlazePalm takes a square 192x192 NHWC input normalised to 0..1.
BLAZEPALM_INPUT_SIZE = 192


def scan_regions(
    frame_bgr: np.ndarray, grid: int = 1, overlap: float = 0.2
) -> Iterator[np.ndarray]:
    """Yield the whole frame, then ``grid``x``grid`` overlapping crops of it.

    A palm is only a few pixels wide once a wide CCTV frame is squashed into
    BlazePalm's 192x192 input -- a 60px hand in a 1280px frame survives as 9px, well
    under anything the model fires on. Each crop is letterboxed to the same 192x192,
    so scanning an NxN grid multiplies the palm's effective resolution by N.

    Crops overlap by ``overlap`` of a tile on each side so a hand straddling a tile
    boundary still lands whole inside at least one region. The full frame comes first
    because a close-up palm scores there immediately, letting the caller stop early.
    """
    yield frame_bgr
    if grid <= 1:
        return

    height, width = frame_bgr.shape[:2]
    step_y, step_x = height / grid, width / grid
    pad_y, pad_x = step_y * overlap, step_x * overlap
    for row in range(grid):
        for col in range(grid):
            y1 = max(0, int(row * step_y - pad_y))
            x1 = max(0, int(col * step_x - pad_x))
            y2 = min(height, int((row + 1) * step_y + pad_y))
            x2 = min(width, int((col + 1) * step_x + pad_x))
            if y2 > y1 and x2 > x1:
                yield frame_bgr[y1:y2, x1:x2]


def palm_search_box(
    bbox: tuple[float, float, float, float],
    frame_shape: tuple[int, int],
    margin: float = 2.0,
) -> tuple[int, int, int, int]:
    """The region around one face where *that person's* raised hand plausibly is.

    Anchoring the search to a face beats scanning the frame blind twice over: it is
    one forward pass instead of ``grid**2 + 1``, and the crop is tight enough that a
    distant hand still fills a usable share of BlazePalm's 192x192 input.

    The box reaches ``margin`` face-widths to each side and from one face-height above
    to ``margin`` below -- asymmetric because a hand raised beside the head sits level
    with or below it far more often than above.

    Keeping it *narrow* is what makes the palm attributable to a person. A square or
    generously-wide box around one face reaches into the next person's space, and two
    colleagues standing together then both register the same hand: measured on a real
    two-person capture, ``margin=0.4`` scored 0.14 against 0.94 across the two faces,
    while ``margin=1.0`` scored 0.98 and 0.96 and identified both.
    """
    height, width = frame_shape
    x1, y1, x2, y2 = bbox
    face_w, face_h = x2 - x1, y2 - y1
    return (
        max(0, int(x1 - margin * face_w)),
        max(0, int(y1 - face_h)),
        min(width, int(x2 + margin * face_w)),
        min(height, int(y2 + margin * face_h)),
    )


class BasePalmDetection(ABC):
    """Detects whether a hand/palm is present in a BGR frame."""

    @abstractmethod
    async def detect(
        self,
        frame_bgr: np.ndarray,
        regions: Sequence[tuple[int, int, int, int]] | None = None,
    ) -> list[PalmResult]:
        """Score each region for a palm, in the order given.

        ``regions`` is normally one box per looking face, from :func:`palm_search_box`,
        and the result is one verdict per face -- which is what lets the caller tell
        *whose* hand is up rather than only that somebody's is. Without regions the
        whole frame is scanned and a single result comes back.
        """


class BlazePalmDetection(BasePalmDetection):
    """MediaPipe BlazePalm (ONNX) run through ``cv2.dnn``.

    Only the score head is decoded. The model also emits box and landmark deltas, but
    turning those into coordinates needs a 2016-row SSD anchor table, and presence does
    not: score filtering is what decides whether a detection exists at all, so
    ``max(sigmoid(scores)) >= threshold`` is equivalent to "at least one palm survives".

    ``scan_grid`` > 1 trades inference time for reach: see :func:`scan_regions`.
    """

    def __init__(
        self,
        *,
        model_path: str | Path = "models/palm_detection_mediapipe_2023feb.onnx",
        score_threshold: float = 0.5,
        scan_grid: int = 1,
        scan_overlap: float = 0.2,
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
        self._scan_grid = scan_grid
        self._scan_overlap = scan_overlap

    async def detect(
        self,
        frame_bgr: np.ndarray,
        regions: Sequence[tuple[int, int, int, int]] | None = None,
    ) -> list[PalmResult]:
        # cv2.dnn.forward is blocking and releases the GIL; keep it off the event loop.
        scores = await asyncio.to_thread(self._scores, frame_bgr, regions)
        return [
            PalmResult(detected=score >= self._score_threshold, score=score)
            for score in scores
        ]

    def _scores(
        self,
        frame_bgr: np.ndarray,
        boxes: Sequence[tuple[int, int, int, int]] | None,
    ) -> list[float]:
        """One score per box, or a single best-of-frame score when there are none."""
        if not boxes:
            return [self._whole_frame_score(frame_bgr)]
        # No early exit here: every box belongs to a different face and each needs its
        # own answer, so a clear on one says nothing about the next.
        return [self._box_score(frame_bgr, box) for box in boxes]

    def _box_score(
        self, frame_bgr: np.ndarray, box: tuple[int, int, int, int]
    ) -> float:
        x1, y1, x2, y2 = box
        crop = frame_bgr[y1:y2, x1:x2]
        return self._region_score(crop) if crop.size else 0.0

    def _whole_frame_score(self, frame_bgr: np.ndarray) -> float:
        """Best score over the grid scan, stopping at the first region that clears.

        Here the question really is only "is a palm present anywhere", so once any
        region clears the threshold the rest cannot change the answer -- worth
        skipping, since a 3x3 grid is ten forward passes.
        """
        best = 0.0
        for region in scan_regions(frame_bgr, self._scan_grid, self._scan_overlap):
            best = max(best, self._region_score(region))
            if best >= self._score_threshold:
                break
        return best

    def _region_score(self, region_bgr: np.ndarray) -> float:
        self._net.setInput(self._letterbox(region_bgr)[np.newaxis])
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
