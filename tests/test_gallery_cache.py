"""The embedding cache: what stops a restart from being a six-minute outage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from app.core.face_processing.embedding_cache import (
    CachedEmbedding,
    EmbeddingCache,
    enrolment_fingerprint,
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
        # Mirrors the real signature including keywords, so this double does not have
        # to be revisited every time an optional argument is added to embed().
        def counted(image, kps, **kwargs):  # type: ignore[no-untyped-def]
            self.embeds += 1
            return self._inner(image, kps, **kwargs)

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


# --- what invalidates a cached vector ----------------------------------------
# A cached entry is the output of detect -> align -> embed, not of ArcFace alone, so
# the key has to cover all three. It used to cover only the recognition model, which
# meant a changed detector or DETECT_INPUT_SIZE reported a cache hit and handed back
# vectors from a path that no longer existed -- and the symptom of that is a genuine
# improvement measuring as no change at all.


def _models(tmp_path: Path) -> tuple[Path, Path]:
    recognize = tmp_path / "w600k_r50.onnx"
    detect = tmp_path / "det_10g.onnx"
    recognize.write_bytes(b"recognize")
    detect.write_bytes(b"detect")
    return recognize, detect


def test_the_detector_input_size_is_part_of_the_key(tmp_path: Path) -> None:
    """It decides landmark precision, and the landmarks are the whole alignment."""
    recognize, detect = _models(tmp_path)
    assert enrolment_fingerprint(recognize, detect, 640) != enrolment_fingerprint(
        recognize, detect, 1280
    )


def test_the_detector_model_is_part_of_the_key(tmp_path: Path) -> None:
    recognize, detect = _models(tmp_path)
    before = enrolment_fingerprint(recognize, detect, 640)
    detect.write_bytes(b"a different detector entirely")
    os.utime(detect, (0, 0))
    assert enrolment_fingerprint(recognize, detect, 640) != before


def test_the_recognition_model_is_still_part_of_the_key(tmp_path: Path) -> None:
    recognize, detect = _models(tmp_path)
    before = enrolment_fingerprint(recognize, detect, 640)
    recognize.write_bytes(b"a different arcface")
    os.utime(recognize, (0, 0))
    assert enrolment_fingerprint(recognize, detect, 640) != before


def test_an_unchanged_enrolment_path_keeps_the_cache(tmp_path: Path) -> None:
    """Otherwise every restart is a six-minute outage instead of a two-second one."""
    recognize, detect = _models(tmp_path)
    first = enrolment_fingerprint(recognize, detect, 640)
    assert enrolment_fingerprint(recognize, detect, 640) == first


def test_a_cache_written_under_one_path_is_discarded_under_another(tmp_path: Path) -> None:
    recognize, detect = _models(tmp_path)
    entry = CachedEmbedding(
        employee_code="EMP1", key="k1", embedding=np.ones(4, dtype="float32")
    )

    EmbeddingCache(
        tmp_path / "cache", enrolment_fingerprint(recognize, detect, 640)
    ).save([entry])

    same = EmbeddingCache(tmp_path / "cache", enrolment_fingerprint(recognize, detect, 640))
    assert set(same.load()) == {"k1"}

    resized = EmbeddingCache(
        tmp_path / "cache", enrolment_fingerprint(recognize, detect, 1280)
    )
    assert resized.load() == {}
