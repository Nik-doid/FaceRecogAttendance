"""Face index building / rebuilding.

Owns the two places the index lives:
1. ``face_embeddings`` table (durable cache, one row per usable employee photo);
2. the in-memory FAISS index the recognition worker searches.

Rebuilds run in a background thread (embedding thousands of photos takes seconds to
minutes) and swap the FAISS index atomically when done, so a rebuild never produces
a half-built index for live recognition.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.ai.detector.base import Detector
from app.ai.faiss.index import FaceIndex, IndexItem
from app.ai.quality.quality import FaceQualityChecker
from app.ai.recognizer.base import Recognizer
from app.config.settings import Settings
from app.core.logging import get_logger
from app.models.face_embedding import FaceEmbedding
from app.repositories.face_embedding_repo import FaceEmbeddingRepository

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class IndexBuildStats:
    processed: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    employees: int = 0
    took_seconds: float = 0.0


class IndexService:
    def __init__(
        self,
        settings: Settings,
        face_index: FaceIndex,
        detector: Detector,
        recognizer: Recognizer,
        quality: FaceQualityChecker,
        repo: FaceEmbeddingRepository,
    ) -> None:
        self._settings = settings
        self._face_index = face_index
        self._detector = detector
        self._recognizer = recognizer
        self._quality = quality
        self._repo = repo
        self._log = get_logger(__name__)
        self._lock = threading.Lock()
        self._building = False
        self._last_built_at: datetime | None = None
        self._last_error: str | None = None
        self._last_stats: IndexBuildStats | None = None

    # -- public control surface ------------------------------------------------
    def start_rebuild(self, session_factory: Callable[[], Session]) -> bool:
        """Kick off a rebuild in the background. Returns False if one is already running."""
        with self._lock:
            if self._building:
                return False
            self._building = True
            self._last_error = None
            thread = threading.Thread(
                target=self._run_build,
                args=(session_factory,),
                name="face-index-rebuild",
                daemon=True,
            )
            thread.start()
            return True

    def rebuild_now(self, session_factory: Callable[[], Session]) -> None:
        """Run a rebuild synchronously (used by tests and CLI tooling)."""
        self._run_build(session_factory)

    @property
    def building(self) -> bool:
        with self._lock:
            return self._building

    @property
    def last_built_at(self) -> datetime | None:
        with self._lock:
            return self._last_built_at

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def last_stats(self) -> IndexBuildStats | None:
        with self._lock:
            return self._last_stats

    @property
    def size(self) -> int:
        return self._face_index.size

    @property
    def employee_count(self) -> int:
        return len(self._face_index.employee_codes)

    # -- build logic ------------------------------------------------------------
    def _run_build(self, session_factory: Callable[[], Session]) -> None:
        started = time.monotonic()
        stats = IndexBuildStats()
        items: list[IndexItem] = []
        rows: list[FaceEmbedding] = []
        try:
            for root in self._settings.employee_photos_source:
                self._collect_from_root(Path(root), items, rows, stats)

            if items:
                with session_factory() as session:
                    self._persist(session, rows)
                self._face_index.rebuild(items)
            stats.employees = len(self._face_index.employee_codes)
            stats.took_seconds = time.monotonic() - started
            self._log.info(
                "face index rebuilt",
                extra={
                    "event": "index_rebuilt",
                    "processed": stats.processed,
                    "skipped": dict(stats.skipped),
                    "employees": stats.employees,
                    "size": self._face_index.size,
                    "took_seconds": round(stats.took_seconds, 2),
                },
            )
        except Exception as exc:  # noqa: BLE001 - the thread must never die silently
            self._log.exception("face index rebuild failed")
            self._last_error = str(exc)
        finally:
            self._last_stats = stats
            self._last_built_at = datetime.now(UTC)
            with self._lock:
                self._building = False

    def _collect_from_root(
        self,
        root: Path,
        items: list[IndexItem],
        rows: list[FaceEmbedding],
        stats: IndexBuildStats,
    ) -> None:
        """Walk one source root: each subdirectory name is an employee_code."""
        if not root.is_dir():
            self._log.warning(
                "employee photos source does not exist, skipping",
                extra={"path": str(root)},
            )
            return
        for employee_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            code = employee_dir.name
            images = sorted(
                p for p in employee_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                stats.skipped["no_images_for_employee"] += 1
                self._log.warning("no images for employee", extra={"employee_code": code})
                continue
            for img_path in images:
                self._process_photo(img_path, code, items, rows, stats)

    def _process_photo(
        self,
        img_path: Path,
        code: str,
        items: list[IndexItem],
        rows: list[FaceEmbedding],
        stats: IndexBuildStats,
    ) -> None:
        image = cv2.imread(str(img_path))
        if image is None:
            stats.skipped["unreadable_image"] += 1
            return

        faces = self._detector.detect(image)
        if not faces:
            stats.skipped["no_face"] += 1
            return

        best = faces[0]
        quality = self._quality.assess(image, best, face_count=len(faces))
        if not quality.passed:
            stats.skipped[quality.reasons[0]] += 1
            self._log.info(
                "skipping unusable enrollment photo",
                extra={"employee_code": code, "image": str(img_path), "reasons": quality.reasons},
            )
            return

        if best.kps is None:
            stats.skipped["no_landmarks"] += 1
            return

        embedding = self._recognizer.embed(image, best.kps)
        if not bool(np.isfinite(embedding).all()):
            stats.skipped["non_finite_embedding"] += 1
            return

        items.append(
            IndexItem(employee_code=code, embedding=embedding)
        )
        rows.append(
            FaceEmbedding(
                employee_code=code,
                embedding=json.dumps([float(v) for v in embedding.tolist()]),
                quality_score=best.score,
                source_image_path=str(img_path),
            )
        )
        stats.processed += 1

    def _persist(self, session: Session, rows: list[FaceEmbedding]) -> None:
        # Reuse the injected repo so the session commit lifecycle stays consistent.
        self._repo.replace_all(session, rows)
