"""Index build: embeds usable photos, skips unusable ones, persists + rebuilds FAISS."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ai.faiss.index import FaceIndex
from app.ai.quality.quality import FaceQualityChecker
from app.ai.types import DetectedFace  # noqa: F401  (re-exported for clarity)
from app.database.session import sync_session
from app.repositories.face_embedding_repo import FaceEmbeddingRepository
from app.services.index_service import IndexService
from tests.fakes import FakeDetector, FakeRecognizer, face_at

EMB_DIM = 8


def _make_photo(path: Path, *, noise: bool) -> None:
    if noise:
        img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    else:
        img = np.full((64, 64, 3), 128, dtype=np.uint8)  # constant -> blurry
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _build_service(test_settings) -> tuple[IndexService, FaceIndex, FakeDetector]:
    # A detector that always finds one face so the quality gate is the only filter.
    detector = FakeDetector(default_face=face_at(5, 5, 55, 55, 25.0))
    recognizer = FakeRecognizer(
        np.full(EMB_DIM, 0.5, dtype="float32"), np.full(EMB_DIM, -0.5, dtype="float32")
    )
    quality = FaceQualityChecker(
        minimum_face_size=10, blur_threshold=50, min_lighting=0, max_lighting=255
    )
    idx = FaceIndex(dim=EMB_DIM)
    service = IndexService(
        test_settings, idx, detector, recognizer, quality, FaceEmbeddingRepository()
    )
    return service, idx, detector


@pytest.fixture
def photo_store(tmp_path) -> Path:
    good = tmp_path / "photos" / "EMP1"
    good.mkdir(parents=True)
    _make_photo(good / "face.jpg", noise=True)  # sharp -> accepted
    bad = tmp_path / "photos" / "EMP2"
    bad.mkdir(parents=True)
    _make_photo(bad / "blurry.jpg", noise=False)  # constant -> rejected as blurry
    empty = tmp_path / "photos" / "EMP3"
    empty.mkdir(parents=True)  # no images at all
    return tmp_path / "photos"


def test_rebuild_skips_unusable_and_persists(db, test_settings, photo_store) -> None:
    test_settings = test_settings.model_copy(update={"employee_photos_source": [photo_store]})
    service, idx, _ = _build_service(test_settings)

    service.rebuild_now(sync_session)

    assert idx.size == 1
    assert idx.employee_codes == {"EMP1"}
    assert service.last_stats is not None
    assert service.last_stats.skipped["blurry"] == 1
    assert service.last_stats.skipped["no_images_for_employee"] == 1

    with sync_session() as session:
        rows = FaceEmbeddingRepository().list_all_rows(session)
    assert len(rows) == 1
    assert rows[0].employee_code == "EMP1"


def test_multiple_employees_build_complete_index(db, test_settings, tmp_path) -> None:
    store = tmp_path / "photos"
    for code in ("E1", "E2"):
        _make_photo(store / code / "a.jpg", noise=True)

    test_settings = test_settings.model_copy(update={"employee_photos_source": [store]})
    service, idx, _ = _build_service(test_settings)
    service.rebuild_now(sync_session)

    assert idx.size == 2
    assert idx.employee_codes == {"E1", "E2"}
