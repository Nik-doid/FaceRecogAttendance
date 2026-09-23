"""``DETECT_INPUT_SIZE`` is the lever that decides whether small RTSP faces survive.

Nothing here loads a model: the point is that the configured value actually reaches
the SCRFD session, because a value that silently stays at 640 looks identical to a
camera that is simply too far away.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config.settings import Settings


def test_defaults_to_the_size_det_10g_was_trained_at() -> None:
    assert Settings().detect_input_size == 640


def test_rejects_a_size_the_strides_do_not_divide() -> None:
    # SCRFD's feature maps are the input over strides 8/16/32; 700 has no stride-32
    # map, so onnxruntime would fail deep inside forward() instead of at boot.
    with pytest.raises(ValidationError):
        Settings(detect_input_size=700)


def test_rejects_a_size_below_the_smallest_useful_input() -> None:
    with pytest.raises(ValidationError):
        Settings(detect_input_size=288)


def test_reaches_the_detector_session() -> None:
    from app.ai.detector import scrfd

    captured: dict[str, object] = {}

    class _FakeSCRFD:
        def __init__(self, model_file: str) -> None:
            self.det_thresh = 0.5
            self.session = _FakeSession()

    class _FakeSession:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

        def set_providers(self, providers: list[str]) -> None:
            captured["providers"] = providers

    class _FakeModelZoo:
        SCRFD = _FakeSCRFD

    class _FakeInsightface:
        model_zoo = _FakeModelZoo

    original_import = scrfd.import_optional
    original_resolve = scrfd.resolve_model_file
    scrfd.import_optional = lambda name: _FakeInsightface  # type: ignore[assignment]
    scrfd.resolve_model_file = lambda name, d: "det_10g.onnx"  # type: ignore[assignment]
    try:
        detector = scrfd.SCRFDDetector(input_size=1280, providers=[])
    finally:
        scrfd.import_optional = original_import  # type: ignore[assignment]
        scrfd.resolve_model_file = original_resolve  # type: ignore[assignment]

    assert detector._input_size == 1280
