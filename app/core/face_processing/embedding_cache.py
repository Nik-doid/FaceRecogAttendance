"""Persisted enrolment embeddings, so a restart is not a six-minute outage.

The ``face_embeddings`` table being removed with the rest of the local database had
exactly one job worth keeping: not re-running SCRFD and ArcFace over every enrolment
photo on every boot. Measured on the deployment machine that is ~0.7 s per photo, so
500 employees is roughly six minutes before anyone can be recognised. This file is the
replacement, and it is deliberately a *cache*, not a log -- deleting it costs time and
nothing else.

Entries are keyed by :attr:`PhotoRef.key`, which folds in the photo's version (mtime
and size locally, ETag remotely), so an edited photo misses the cache and re-embeds
while an untouched one does not. The whole cache is discarded when the recognition
model changes, since embeddings from a different model are not comparable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.logging import get_logger

log = get_logger(__name__)

CACHE_VERSION = 1
_VECTORS_NAME = "embeddings.npy"
_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class CachedEmbedding:
    employee_code: str
    key: str
    embedding: np.ndarray


class EmbeddingCache:
    """A directory holding one ``.npy`` matrix plus a JSON manifest describing it.

    Two files rather than one ``.npz`` so the vectors can be memory-mapped later and
    so a corrupt manifest never costs the vectors. Both are written to a temp name and
    ``os.replace``d, which is atomic on POSIX and Windows alike -- a crash mid-write
    leaves the previous cache intact rather than a truncated one.
    """

    def __init__(self, directory: str | Path, model_fingerprint: str) -> None:
        self._dir = Path(directory)
        self._fingerprint = model_fingerprint

    def load(self) -> dict[str, CachedEmbedding]:
        """Return cached entries by key, or an empty dict if unusable for any reason."""
        manifest_path = self._dir / _MANIFEST_NAME
        vectors_path = self._dir / _VECTORS_NAME
        if not manifest_path.is_file() or not vectors_path.is_file():
            return {}

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("embedding cache manifest unreadable", extra={"error": str(exc)})
            return {}

        if manifest.get("version") != CACHE_VERSION:
            log.info("embedding cache discarded: format changed")
            return {}
        if manifest.get("model") != self._fingerprint:
            # Embeddings from a different recognition model share no coordinate
            # system, so mixing them would silently wreck every similarity score.
            log.info("embedding cache discarded: recognition model changed")
            return {}

        entries = manifest.get("entries")
        if not isinstance(entries, list):
            return {}

        try:
            vectors = np.load(vectors_path)
        except (OSError, ValueError) as exc:
            log.warning("embedding cache vectors unreadable", extra={"error": str(exc)})
            return {}

        if vectors.ndim != 2 or len(vectors) != len(entries):
            log.warning(
                "embedding cache is inconsistent; discarding",
                extra={"vectors": len(vectors), "entries": len(entries)},
            )
            return {}

        cached: dict[str, CachedEmbedding] = {}
        for row, entry in zip(vectors, entries, strict=True):
            key = entry.get("key")
            code = entry.get("employee_code")
            if not key or not code:
                continue
            cached[key] = CachedEmbedding(
                employee_code=code, key=key, embedding=np.asarray(row, dtype="float32")
            )
        log.info("embedding cache loaded", extra={"entries": len(cached)})
        return cached

    def save(self, entries: list[CachedEmbedding]) -> None:
        """Replace the cache with exactly these entries. Best effort; never raises."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            vectors = (
                np.vstack([entry.embedding for entry in entries])
                if entries
                else np.empty((0, 0), dtype="float32")
            )
            manifest = {
                "version": CACHE_VERSION,
                "model": self._fingerprint,
                "entries": [
                    {"employee_code": entry.employee_code, "key": entry.key}
                    for entry in entries
                ],
            }
            _atomic_write_bytes(self._dir / _VECTORS_NAME, _to_npy_bytes(vectors))
            _atomic_write_bytes(
                self._dir / _MANIFEST_NAME,
                json.dumps(manifest, indent=None).encode("utf-8"),
            )
            log.info("embedding cache written", extra={"entries": len(entries)})
        except OSError as exc:
            # A cache that cannot be written is a slow startup, not a broken service.
            log.warning("embedding cache could not be written", extra={"error": str(exc)})


def model_fingerprint(model_path: Path) -> str:
    """Identify the recognition model by name and mtime, without hashing 174 MiB."""
    try:
        stat = model_path.stat()
        return f"{model_path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        return model_path.name


def _to_npy_bytes(vectors: np.ndarray) -> bytes:
    import io  # noqa: PLC0415 - only needed on the write path

    buffer = io.BytesIO()
    np.save(buffer, vectors, allow_pickle=False)
    return buffer.getvalue()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)
