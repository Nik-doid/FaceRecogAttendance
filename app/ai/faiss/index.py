"""In-memory FAISS embedding index for similarity search.

The index holds L2-normalized embeddings so inner-product == cosine similarity.
It is rebuilt wholesale on ``POST /index/rebuild`` and only read by the recognition
worker, so a simple RLock around mutation is sufficient (FAISS itself is not
guaranteed thread-safe for concurrent add/search).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ai._loader import import_optional
from app.ai.types import IndexResult


@dataclass
class IndexItem:
    employee_code: str
    embedding: np.ndarray


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return np.asarray(vec / norm)


class FaceIndex:
    """Thread-safe cosine-similarity index mapping embeddings to employee codes."""

    def __init__(self, dim: int = 512) -> None:
        faiss = import_optional("faiss")
        self._faiss = faiss
        self._dim = dim
        self._lock = __import__("threading").RLock()
        self._index = faiss.IndexFlatIP(dim)
        self._codes: list[str] = []

    # -- mutation ------------------------------------------------------------
    def add(self, embedding: np.ndarray, employee_code: str) -> None:
        vec = l2_normalize(embedding).astype("float32").reshape(1, -1)
        if vec.shape[1] != self._dim:
            raise ValueError(f"embedding dim {vec.shape[1]} != index dim {self._dim}")
        with self._lock:
            self._index.add(vec)
            self._codes.append(employee_code)

    def rebuild(self, items: list[IndexItem]) -> None:
        """Atomically replace the whole index contents."""
        faiss = self._faiss
        new_index = faiss.IndexFlatIP(self._dim)
        codes: list[str] = []
        if items:
            matrix = np.vstack([l2_normalize(item.embedding).astype("float32") for item in items])
            new_index.add(matrix)
            codes = [item.employee_code for item in items]
        with self._lock:
            self._index = new_index
            self._codes = codes

    def clear(self) -> None:
        with self._lock:
            self._index = self._faiss.IndexFlatIP(self._dim)
            self._codes = []

    # -- reads ---------------------------------------------------------------
    def search(self, embedding: np.ndarray, k: int = 1) -> list[IndexResult]:
        vec = l2_normalize(embedding).astype("float32").reshape(1, -1)
        with self._lock:
            if self._index.ntotal == 0:
                return []
            k = min(k, self._index.ntotal)
            scores, idxs = self._index.search(vec, k)
        results: list[IndexResult] = []
        for score, idx in zip(scores[0], idxs[0], strict=False):
            if idx < 0 or idx >= len(self._codes):
                continue
            score = float(score)
            results.append(
                IndexResult(
                    employee_code=self._codes[int(idx)],
                    score=score,
                    distance=float(1.0 - score),
                )
            )
        return results

    @property
    def size(self) -> int:
        with self._lock:
            return int(self._index.ntotal)

    @property
    def employee_codes(self) -> set[str]:
        with self._lock:
            return set(self._codes)
