"""``CameraReader`` frame normalisation, tested without any camera hardware."""

from __future__ import annotations

import numpy as np

from app.workers.camera.reader import CameraReader


def _frame(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_downscales_when_narrower_than_max_width() -> None:
    reader = CameraReader(max_width=1280, min_width=0)
    out = reader._resize(_frame(1920, 1080))
    assert out.shape[:2] == (720, 1280)


def test_leaves_frames_inside_the_window_untouched() -> None:
    reader = CameraReader(max_width=1280, min_width=960)
    out = reader._resize(_frame(1280, 720))
    assert out.shape[:2] == (720, 1280)


def test_upscales_rtsp_substreams_to_min_width() -> None:
    """A 640x360 CCTV substream must not reach recognition as-is."""
    reader = CameraReader(max_width=1280, min_width=1280)
    out = reader._resize(_frame(640, 360))
    assert out.shape[:2] == (720, 1280)


def test_min_width_zero_disables_upscaling() -> None:
    reader = CameraReader(max_width=0, min_width=0)
    out = reader._resize(_frame(320, 240))
    assert out.shape[:2] == (240, 320)


def test_upscaling_keeps_aspect_ratio_and_pixel_count() -> None:
    reader = CameraReader(max_width=1280, min_width=1280)
    out = reader._resize(_frame(720, 576))
    h, w = out.shape[:2]
    assert w == 1280
    # 720x576 -> 1280x1024
    assert (h, w) == (1024, 1280)