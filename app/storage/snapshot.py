"""Snapshot persistence for recognized and unknown faces.

Snapshots are written as timestamped JPEGs. The path is reported to the attendance
system (for recognized faces) and recorded in ``unknown_faces`` (for unknown faces).
The attendance system and ops can then inspect the evidence offline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from app.core.logging import get_logger

log = get_logger(__name__)


class SnapshotStorage:
    def __init__(self, base_dir: Path, enabled: bool = True) -> None:
        self._enabled = enabled
        self._dir = base_dir
        if enabled:
            base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def save(
        self,
        image_bgr: np.ndarray,
        prefix: str,
        *,
        employee_code: str | None = None,
        track_id: int | None = None,
    ) -> Path | None:
        """Persist the frame crop/label; returns the absolute path or None."""
        if not self._enabled:
            return None
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        identity = employee_code or "unknown"
        track = f"_t{track_id}" if track_id is not None else ""
        filename = f"{prefix}_{identity}{track}_{ts}.jpg"
        path = self._dir / filename
        ok = cv2.imwrite(str(path), image_bgr)
        return path if ok else None

    def remove(self, path: Path | str | None) -> bool:
        """Delete a snapshot file. Returns True when the file existed and was removed."""
        if not path:
            return False
        try:
            file = Path(path)
            if file.exists():
                file.unlink()
                return True
        except OSError:
            log.warning("failed to remove snapshot", extra={"path": str(path)})
        return False
