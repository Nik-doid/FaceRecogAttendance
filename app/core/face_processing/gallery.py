"""The enrolled-employee gallery step 4 searches.

Photos are enumerated by :mod:`app.core.face_processing.photos`, which makes a local
``uploads/`` directory and an HTTPS folder on an HR server interchangeable, and
embeddings are reused across restarts by
:mod:`app.core.face_processing.embedding_cache`. Everything here works on bytes, so
neither concern leaks into the detect/embed/index path.

The sessions arrive already loaded, from :mod:`app.runtime`. Building them here would
mean a second SCRFD and a second 174 MiB ArcFace alongside the ones the process already
has. Sharing is safe: both are onnxruntime sessions, whose ``Run`` is thread-safe, and
:class:`FaceIndex` guards itself with an RLock. The ``cv2.dnn`` palm net has neither
guarantee, which is why that one stays per-pipeline.

:class:`GalleryHandle` exists so a rebuild can swap the whole gallery atomically while
recognition is running: an in-flight match keeps the object it started with, and the
next one picks up the new index. That is what lets a new hire appear without a restart,
which the old process-wide dict cache -- never invalidated -- could not do.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.ai.detector.scrfd import SCRFDDetector
from app.ai.faiss.index import FaceIndex, IndexItem
from app.ai.recognizer.arcface import ArcFaceRecognizer
from app.core.face_processing.embedding_cache import (
    CachedEmbedding,
    EmbeddingCache,
    model_fingerprint,
)
from app.core.face_processing.photos import PhotoRef, PhotoSource, collect
from app.core.logging import get_logger
from app.core.metrics import GALLERY_SIZE
from app.runtime import Models

log = get_logger(__name__)

# w600k_r50 emits 512-d embeddings.
EMBEDDING_DIM = 512
# Enrolment can run for minutes on a slow box; say something periodically so the
# silence is never mistaken for a hang.
PROGRESS_EVERY = 25


@dataclass(frozen=True)
class Gallery:
    """A searchable set of enrolled employees plus the recognizer that embedded them.

    ``recognizer`` is None when nothing was enrolled: an empty index can never match,
    so there is no point handing a probe embedder to a caller that cannot use it.
    ``ArcFaceRecognition._match`` already returns the face untouched in that case,
    which is also what makes a not-yet-built gallery safe to serve.
    """

    index: FaceIndex
    recognizer: ArcFaceRecognizer | None
    employees: int
    photos: int


EMPTY_GALLERY = Gallery(index=FaceIndex(dim=EMBEDDING_DIM), recognizer=None, employees=0, photos=0)


class GalleryHandle:
    """A swappable reference to the current gallery.

    Recognition reads ``.current`` once per frame; a rebuild calls ``swap`` when it
    finishes. No lock is held across a match, so a slow rebuild never stalls the
    camera loop.
    """

    def __init__(self, gallery: Gallery | None = None) -> None:
        self._gallery = gallery if gallery is not None else EMPTY_GALLERY
        self._lock = threading.Lock()
        self.ready = threading.Event()
        if gallery is not None:
            self.ready.set()

    @property
    def current(self) -> Gallery:
        with self._lock:
            return self._gallery

    def swap(self, gallery: Gallery) -> None:
        with self._lock:
            self._gallery = gallery
        self.ready.set()


def build_gallery(
    models: Models,
    sources: list[PhotoSource],
    *,
    cache: EmbeddingCache | None = None,
) -> Gallery:
    """Enumerate every source, embed what is new, and return a searchable gallery.

    Photos already in ``cache`` under the same key skip both model passes entirely,
    which is the difference between a six-minute restart and a two-second one.
    """
    refs = collect(sources)
    index = FaceIndex(dim=EMBEDDING_DIM)
    if not refs:
        log.warning("recognition gallery is empty: no enrolment photos found")
        return Gallery(index=index, recognizer=None, employees=0, photos=0)

    cached = cache.load() if cache is not None else {}
    entries: list[CachedEmbedding] = []
    reused = 0
    for position, ref in enumerate(refs, start=1):
        hit = cached.get(ref.key)
        if hit is not None:
            entries.append(CachedEmbedding(ref.employee_code, ref.key, hit.embedding))
            reused += 1
        else:
            embedding = _embed_ref(ref, models.detector, models.recognizer)
            if embedding is not None:
                entries.append(CachedEmbedding(ref.employee_code, ref.key, embedding))
        if position % PROGRESS_EVERY == 0:
            log.info("enrolling", extra={"done": position, "total": len(refs)})

    index.rebuild(
        [
            IndexItem(employee_code=entry.employee_code, embedding=entry.embedding)
            for entry in entries
        ]
    )
    if cache is not None:
        cache.save(entries)

    codes = {entry.employee_code for entry in entries}
    GALLERY_SIZE.set(len(codes))
    log.info(
        "recognition gallery built",
        extra={
            "employees": len(codes),
            "photos": len(entries),
            "reused_from_cache": reused,
            "skipped": len(refs) - len(entries),
        },
    )
    return Gallery(
        index=index,
        recognizer=models.recognizer if entries else None,
        employees=len(codes),
        photos=len(entries),
    )


def _embed_ref(
    ref: PhotoRef, detector: SCRFDDetector, recognizer: ArcFaceRecognizer
) -> np.ndarray | None:
    """Embed the best-scoring face in one enrolment photo, or None if unusable."""
    data = ref.read()
    if not data:
        return None
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        log.warning("undecodable enrolment photo", extra={"photo": ref.label})
        return None

    faces = detector.detect(image)
    if not faces:
        log.warning("no face in enrolment photo", extra={"photo": ref.label})
        return None

    best = faces[0]  # SCRFDDetector sorts by score descending.
    if best.kps is None:
        log.warning("no landmarks in enrolment photo", extra={"photo": ref.label})
        return None

    embedding = recognizer.embed(image, best.kps)
    if not bool(np.isfinite(embedding).all()):
        log.warning("non-finite enrolment embedding", extra={"photo": ref.label})
        return None
    return embedding


def build_cache(settings_storage_path: Path, recognize_model: Path) -> EmbeddingCache:
    """The cache directory for this deployment, fingerprinted by recognition model."""
    return EmbeddingCache(
        Path(settings_storage_path) / "gallery", model_fingerprint(recognize_model)
    )
