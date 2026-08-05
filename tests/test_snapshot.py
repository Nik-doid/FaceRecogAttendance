"""Snapshot storage tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.storage.snapshot import SnapshotStorage


def test_save_writes_jpg(tmp_path: Path) -> None:
    storage = SnapshotStorage(tmp_path / "snapshots", enabled=True)
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    path = storage.save(image, "attendance", employee_code="EMP1", track_id=7)
    assert path is not None
    assert path.exists()
    assert path.suffix == ".jpg"
    assert "EMP1" in path.name
    assert "_t7" in path.name


def test_disabled_returns_none(tmp_path: Path) -> None:
    storage = SnapshotStorage(tmp_path / "snapshots", enabled=False)
    assert storage.save(np.zeros((10, 10, 3), dtype=np.uint8), "unknown") is None
