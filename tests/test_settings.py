"""Config parsing tests."""

from __future__ import annotations

from pathlib import Path

from app.config.settings import Settings


def test_comma_separated_lists_parsed() -> None:
    s = Settings(
        onnx_providers="CUDAExecutionProvider, CPUExecutionProvider",
        employee_photos_source="a,b",
    )
    assert s.onnx_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert s.employee_photos_source == [Path("a"), Path("b")]


def test_defaults() -> None:
    s = Settings()
    assert s.frame_skip == 2
    assert s.recognition_threshold == 0.60
    assert s.camera_id == "cam-01"
    assert s.attendance_broker == "rabbitmq"


def test_path_derived() -> None:
    s = Settings(storage_path=Path("/tmp/storage"))
    assert s.face_index_dump_path == Path("/tmp/storage") / "face_index"
