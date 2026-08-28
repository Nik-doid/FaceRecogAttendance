"""The embedding cache: what stops a restart from being a six-minute outage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.core.face_processing.embedding_cache import (
    CachedEmbedding,
    EmbeddingCache,
    model_fingerprint,
)
from app.core.face_processing.gallery import build_gallery
from app.core.face_processing.photos import LocalPhotoSource
from app.runtime import Models

FINGERPRINT = "w600k_r50.onnx:1:2"


def _entries(count: int = 3) -> list[CachedEmbedding]:
    return [
        CachedEmbedding(
            employee_code=f"EMP{i}",
            key=f"local:/photos/EMP{i}/a.jpg:1:2",
            embedding=np.full(512, float(i), dtype="float32"),
        )
        for i in range(count)
    ]


# --- round trip --------------------------------------------------------------


def test_saved_entries_come_back(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, FINGERPRINT)
    cache.save(_entries())

    loaded = EmbeddingCache(tmp_path, FINGERPRINT).load()
    assert sorted(loaded) == sorted(e.key for e in _entries())
    assert loaded[_entries()[1].key].employee_code == "EMP1"
    np.testing.assert_array_equal(loaded[_entries()[1].key].embedding, np.full(512, 1.0))


def test_missing_cache_is_empty_not_an_error(tmp_path: Path) -> None:
    assert EmbeddingCache(tmp_path / "nothing-here", FINGERPRINT).load() == {}


def test_saving_nothing_is_survivable(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, FINGERPRINT)
    cache.save([])
    assert cache.load() == {}


# --- invalidation ------------------------------------------------------------


def test_a_different_recognition_model_discards_the_cache(tmp_path: Path) -> None:
    """Embeddings from another model share no coordinate system with these."""
    EmbeddingCache(tmp_path, FINGERPRINT).save(_entries())
    assert EmbeddingCache(tmp_path, "w600k_mbf.onnx:9:9").load() == {}


def test_corrupt_manifest_discards_rather_than_raises(tmp_path: Path) -> None:
    EmbeddingCache(tmp_path, FINGERPRINT).save(_entries())
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    assert EmbeddingCache(tmp_path, FINGERPRINT).load() == {}


def test_row_count_mismatch_discards(tmp_path: Path) -> None:
    """A manifest describing more rows than exist would mis-attribute embeddings."""
    cache = EmbeddingCache(tmp_path, FINGERPRINT)
    cache.save(_entries(3))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["entries"].append({"employee_code": "EMP9", "key": "invented"})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert cache.load() == {}


def test_fingerprint_tracks_the_model_file(tmp_path: Path) -> None:
    model = tmp_path / "w600k_r50.onnx"
    model.write_bytes(b"weights")
    first = model_fingerprint(model)

    model.write_bytes(b"different weights entirely")
    assert model_fingerprint(model) != first
    # A missing file still yields something usable rather than raising.
    assert model_fingerprint(tmp_path / "gone.onnx") == "gone.onnx"


# --- the point of all this ---------------------------------------------------


class _CountingModels:
    """Wraps real models to count how many photos actually reach ArcFace."""

    def __init__(self, models: Models) -> None:
        self.detector = models.detector
        self.recognizer = models.recognizer
        self.embeds = 0
        self._inner = models.recognizer.embed

    def __enter__(self) -> _CountingModels:
        def counted(image, kps):  # type: ignore[no-untyped-def]
            self.embeds += 1
            return self._inner(image, kps)

        self.recognizer.embed = counted  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: object) -> None:
        self.recognizer.embed = self._inner  # type: ignore[method-assign]


def test_warm_start_embeds_nothing(tmp_path: Path, models: Models) -> None:
    """The whole reason this module exists: a second build must skip both passes."""
    photos = tmp_path / "photos"
    (photos / "EMP1").mkdir(parents=True)
    source_image = Path("uploads/employees/EMP1/EMP1.jpg")
    if not source_image.is_file():
        import pytest

        pytest.skip("no enrolment photo available")
    (photos / "EMP1" / "a.jpg").write_bytes(source_image.read_bytes())

    cache = EmbeddingCache(tmp_path / "cache", model_fingerprint(Path("w600k_r50.onnx")))

    with _CountingModels(models) as counting:
        cold = build_gallery(counting, [LocalPhotoSource(photos)], cache=cache)  # type: ignore[arg-type]
        assert cold.photos == 1
        assert counting.embeds == 1

    with _CountingModels(models) as counting:
        warm = build_gallery(counting, [LocalPhotoSource(photos)], cache=cache)  # type: ignore[arg-type]
        assert warm.photos == 1
        assert counting.embeds == 0, "a cached photo must not reach ArcFace again"

    # And the warm gallery still recognises: the vectors survived the round trip.
    assert warm.index.size == cold.index.size
    assert warm.employees == 1
