"""The enrolled-employee gallery step 3 searches.

``app/services/index_service.py`` is the durable enrolment path: it embeds the same
photos, persists them to ``face_embeddings`` and feeds the recognition worker. This is
the pipeline-local equivalent with no database attached -- it walks
``EMPLOYEE_PHOTOS_SOURCE`` straight into an in-memory :class:`FaceIndex`, which is what
lets the webcam route recognise faces without a DB session in the request path.

Built once per (models, sources) and cached for the process: enrolment costs one SCRFD
pass plus one ArcFace pass per photo, far too slow to repeat per WebSocket connection.
Sharing across connections is safe because both models are onnxruntime sessions --
whose ``Run`` is thread-safe -- and :class:`FaceIndex` guards itself with an RLock. The
``cv2.dnn`` palm net has neither guarantee, which is why that one stays per-connection.

The sessions arrive already loaded, from :mod:`app.runtime`. Building them here would
mean a second SCRFD and a second 174 MiB ArcFace alongside the ones the process already
has.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.ai.detector.scrfd import SCRFDDetector
from app.ai.faiss.index import FaceIndex, IndexItem
from app.ai.recognizer.arcface import ArcFaceRecognizer
from app.core.logging import get_logger
from app.runtime import Models

log = get_logger(__name__)

# w600k_r50 emits 512-d embeddings.
EMBEDDING_DIM = 512
# Kept local rather than imported from index_service: that module pulls SQLAlchemy and
# app.models in behind it, and nothing on the webcam path should need a DB import.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


@dataclass(frozen=True)
class Gallery:
    """A searchable set of enrolled employees plus the recognizer that embedded them.

    ``recognizer`` is None when nothing was enrolled: an empty index can never match,
    so there is no point handing a probe embedder to a caller that cannot use it.
    ``ArcFaceRecognition._match`` already returns the face untouched in that case.
    """

    index: FaceIndex
    recognizer: ArcFaceRecognizer | None
    employees: int
    photos: int


_cache: dict[tuple[int, tuple[str, ...]], Gallery] = {}
_lock = threading.Lock()


def load_gallery(models: Models, sources: Sequence[str | Path]) -> Gallery:
    """Return the cached gallery for these sources, building it on first use.

    The lock is held across the build so two connections opening at once enrol once
    rather than twice. Keyed on the identity of the shared ``Models`` bundle, so a
    process with one bundle -- the normal case -- has exactly one gallery.
    """
    key = (id(models), tuple(str(source) for source in sources))
    with _lock:
        cached = _cache.get(key)
        if cached is None:
            cached = _build(models, sources)
            _cache[key] = cached
        return cached


def _iter_photos(sources: Sequence[str | Path]) -> Iterator[tuple[Path, str]]:
    """Yield (photo, employee_code) pairs. Each subdirectory name *is* the code."""
    for source in sources:
        root = Path(source)
        if not root.is_dir():
            log.warning("employee photo source missing", extra={"path": str(root)})
            continue
        for employee_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            photos = sorted(
                p for p in employee_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not photos:
                log.warning("no photos for employee", extra={"employee_code": employee_dir.name})
            for photo in photos:
                yield photo, employee_dir.name


def _build(models: Models, sources: Sequence[str | Path]) -> Gallery:
    photos = list(_iter_photos(sources))
    index = FaceIndex(dim=EMBEDDING_DIM)
    if not photos:
        log.warning("recognition gallery is empty", extra={"sources": [str(s) for s in sources]})
        return Gallery(index=index, recognizer=None, employees=0, photos=0)

    recognizer = models.recognizer
    detector = models.detector
    items: list[IndexItem] = []
    codes: set[str] = set()
    for photo, code in photos:
        embedding = _embed_photo(photo, detector, recognizer)
        if embedding is None:
            continue
        items.append(IndexItem(employee_code=code, embedding=embedding))
        codes.add(code)

    index.rebuild(items)
    log.info(
        "recognition gallery built",
        extra={"employees": len(codes), "photos": len(items), "skipped": len(photos) - len(items)},
    )
    return Gallery(index=index, recognizer=recognizer, employees=len(codes), photos=len(items))


def _embed_photo(
    photo: Path, detector: SCRFDDetector, recognizer: ArcFaceRecognizer
) -> np.ndarray | None:
    """Embed the largest-scoring face in one enrolment photo, or None if unusable."""
    image = cv2.imread(str(photo))
    if image is None:
        log.warning("unreadable enrolment photo", extra={"photo": str(photo)})
        return None

    faces = detector.detect(image)
    if not faces:
        log.warning("no face in enrolment photo", extra={"photo": str(photo)})
        return None

    best = faces[0]  # SCRFDDetector sorts by score descending.
    if best.kps is None:
        log.warning("no landmarks in enrolment photo", extra={"photo": str(photo)})
        return None

    embedding = recognizer.embed(image, best.kps)
    if not bool(np.isfinite(embedding).all()):
        log.warning("non-finite enrolment embedding", extra={"photo": str(photo)})
        return None
    return embedding
