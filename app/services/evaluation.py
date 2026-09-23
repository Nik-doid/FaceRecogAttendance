"""Offline recognition evaluation: the numbers every accuracy claim has to come from.

Every latency and threshold figure in this repository used to be a hand-measured
comment with no way to reproduce it. That is fine until someone has to decide whether
a change helped, at which point "the confidences look better" is indistinguishable
from a coincidence. This module is the arbiter.

It lives under ``app/`` rather than in a ``scripts/`` directory on purpose. ``mypy``
is configured with ``files = ["app"]`` and pytest with ``testpaths = ["tests"]``, so a
file outside ``app/`` would be the one piece of arithmetic in the project that nobody
type-checks -- and it is the piece that threshold decisions rest on.

Two deliberate differences from the production path, both load-bearing:

* **The embedding cache is never used.** An experiment that changes the detector or
  ``detect_input_size`` produces vectors that are wrong for production, and writing
  them into ``storage/gallery/`` would poison the live service.
* **``FaceRecognitionProcess`` is never called.** Its gaze and palm gates would
  confound every number here -- a turned head would score as a recognition failure
  rather than as a gate rejection. Detection, embedding and search are driven
  directly, and the gaze verdict is *reported* alongside rather than applied.

Input layout reuses the convention :mod:`app.core.face_processing.photos` already
established, so labelling a capture means dropping a JPEG in a folder:

.. code-block:: text

    eval_frames/
      EMP1/            the subject is EMP1
        t01_000.jpg    the `t<NN>_` prefix groups consecutive frames into one track
        t01_001.jpg
      EMP1+EMP4/       two enrolled people in shot; a face is right if it matches either
      UNKNOWN/         nobody enrolled -- these supply the impostor scores

Usage::

    uv run python -m app.services.evaluation --frames eval_frames --json baseline.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.ai.faiss.index import l2_normalize
from app.config.settings import Settings
from app.config.settings import settings as env_settings
from app.core.face_processing.gallery import Gallery, build_gallery
from app.core.face_processing.gaze import estimate_gaze
from app.core.face_processing.photos import build_sources
from app.runtime import Models, load_models

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
UNKNOWN_LABEL = "UNKNOWN"
_TRACK_PREFIX = re.compile(r"^t(\d+)_")

# Straddling MIN_FACE_PIXELS=60, because the question these buckets answer is "where
# does the embedding stop carrying identity", and that gate is the current guess at it.
WIDTH_BUCKETS: tuple[int, ...] = (30, 40, 50, 60, 80, 120)
# A production box counts as finding a reference face at this overlap. Deliberately
# loose: the two passes letterbox differently, so boxes for the same face differ by a
# pixel or two even when both are correct.
MATCH_IOU = 0.4


# --------------------------------------------------------------------------- labels
def parse_label(directory_name: str) -> frozenset[str]:
    """The employee codes present in frames under this directory.

    ``UNKNOWN`` returns the empty set, which is what makes a frame an impostor sample
    rather than an unlabelled one.
    """
    if directory_name.strip().upper() == UNKNOWN_LABEL:
        return frozenset()
    return frozenset(part.strip() for part in directory_name.split("+") if part.strip())


def parse_track(file_name: str) -> int | None:
    """The track a frame belongs to, from a ``t<NN>_`` prefix. None means its own."""
    match = _TRACK_PREFIX.match(file_name)
    return int(match.group(1)) if match else None


def bucket_of(width: float) -> str:
    """The width bucket a face falls in, as a label for reports."""
    edges = WIDTH_BUCKETS
    if width < edges[0]:
        return f"<{edges[0]}"
    for low, high in zip(edges, edges[1:], strict=False):
        if width < high:
            return f"{low}-{high - 1}"
    return f"{edges[-1]}+"


def bucket_labels() -> list[str]:
    """Every bucket label, in ascending order, so empty buckets still print."""
    edges = WIDTH_BUCKETS
    labels = [f"<{edges[0]}"]
    labels += [f"{low}-{high - 1}" for low, high in zip(edges, edges[1:], strict=False)]
    labels.append(f"{edges[-1]}+")
    return labels


# ----------------------------------------------------------------------- geometry
def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


# --------------------------------------------------------------------------- sweep
@dataclass(frozen=True)
class SweepRow:
    threshold: float
    far: float
    frr: float


@dataclass(frozen=True)
class Sweep:
    """A threshold sweep, plus the operating points anyone actually asks for."""

    rows: list[SweepRow]
    eer: float
    eer_threshold: float
    genuine: int
    impostor: int

    def threshold_at_far(self, target: float) -> SweepRow | None:
        """The lowest threshold whose FAR is at or under ``target`` (lowest FRR)."""
        allowed = [row for row in self.rows if row.far <= target]
        return min(allowed, key=lambda row: row.frr) if allowed else None

    def row_at(self, threshold: float) -> SweepRow | None:
        if not self.rows:
            return None
        return min(self.rows, key=lambda row: abs(row.threshold - threshold))


def far_frr_sweep(
    genuine: Sequence[float],
    impostor: Sequence[float],
    start: float = 0.05,
    stop: float = 0.90,
    step: float = 0.01,
) -> Sweep:
    """FAR and FRR at every threshold, from scores already computed.

    No inference happens here -- the arrays come from one pass over the frames, which
    is why the whole sweep is free and why the threshold should be read off this table
    rather than guessed. FAR is impostor scores accepted; FRR is genuine scores
    rejected. Accept is ``score >= threshold``, matching ``ArcFaceRecognition._match``.
    """
    rows: list[SweepRow] = []
    n_genuine, n_impostor = len(genuine), len(impostor)
    steps = int(round((stop - start) / step)) + 1
    for index in range(max(0, steps)):
        threshold = round(start + index * step, 4)
        far = (
            sum(1 for score in impostor if score >= threshold) / n_impostor
            if n_impostor
            else 0.0
        )
        frr = (
            sum(1 for score in genuine if score < threshold) / n_genuine
            if n_genuine
            else 0.0
        )
        rows.append(SweepRow(threshold=threshold, far=far, frr=frr))

    if rows:
        crossing = min(rows, key=lambda row: abs(row.far - row.frr))
        eer = (crossing.far + crossing.frr) / 2.0
        eer_threshold = crossing.threshold
    else:  # pragma: no cover - only when start > stop
        eer, eer_threshold = float("nan"), float("nan")
    return Sweep(
        rows=rows,
        eer=eer,
        eer_threshold=eer_threshold,
        genuine=n_genuine,
        impostor=n_impostor,
    )


def describe(scores: Sequence[float]) -> dict[str, float]:
    """n / mean / sd / percentiles, the shape every score distribution gets reported in."""
    if not scores:
        return {"n": 0.0}
    ordered = sorted(scores)

    def percentile(fraction: float) -> float:
        position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[position]

    return {
        "n": float(len(ordered)),
        "mean": statistics.fmean(ordered),
        "sd": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "p5": percentile(0.05),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def histogram(
    genuine: Sequence[float], impostor: Sequence[float], bins: int = 20
) -> list[str]:
    """Both distributions on one axis, because the gap between them is the whole story."""
    if not genuine and not impostor:
        return ["  (no scores)"]
    low = min([*genuine, *impostor])
    high = max([*genuine, *impostor])
    if high <= low:
        high = low + 1e-6
    width = (high - low) / bins

    def counts(scores: Sequence[float]) -> list[int]:
        buckets = [0] * bins
        for score in scores:
            index = min(bins - 1, int((score - low) / width))
            buckets[index] += 1
        return buckets

    g_counts, i_counts = counts(genuine), counts(impostor)
    peak = max([*g_counts, *i_counts, 1])
    lines = [f"  {'range':>13}  {'genuine':<22} {'impostor':<22}"]
    for index in range(bins):
        start = low + index * width
        g_bar = "#" * round(20 * g_counts[index] / peak)
        i_bar = "." * round(20 * i_counts[index] / peak)
        lines.append(
            f"  {start:6.3f}-{start + width:6.3f}  "
            f"{g_bar:<20} {g_counts[index]:>4}  {i_bar:<20} {i_counts[index]:>4}"
        )
    return lines


# ---------------------------------------------------------------------- the frames
@dataclass(frozen=True)
class EvalFrame:
    path: Path
    labels: frozenset[str]
    track: str

    @property
    def is_impostor(self) -> bool:
        return not self.labels


def find_frames(root: Path) -> list[EvalFrame]:
    """Every labelled image under ``root``, one directory level deep.

    Frames sit in per-label directories rather than carrying a manifest, so adding a
    subject is a folder and adding a frame is a file. Track ids are namespaced by
    directory: ``EMP1/t01`` and ``EMP4/t01`` are different people, not one track.
    """
    frames: list[EvalFrame] = []
    for directory in sorted(entry for entry in root.iterdir() if entry.is_dir()):
        labels = parse_label(directory.name)
        for image in sorted(directory.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            track = parse_track(image.name)
            frames.append(
                EvalFrame(
                    path=image,
                    labels=labels,
                    track=f"{directory.name}/t{track:02d}" if track is not None
                    else f"{directory.name}/{image.stem}",
                )
            )
    return frames


# ------------------------------------------------------------------- accumulators
@dataclass
class Accumulator:
    """Everything one pass over the frames collects."""

    frames: int = 0
    unreadable: int = 0
    reference_faces: int = 0
    # detection: input size -> bucket -> found / total
    found: dict[int, dict[str, int]] = field(default_factory=dict)
    reference_by_bucket: dict[str, int] = field(default_factory=dict)
    face_widths: list[float] = field(default_factory=list)
    genuine: list[float] = field(default_factory=list)
    impostor: list[float] = field(default_factory=list)
    rank1_hits: int = 0
    rank1_total: int = 0
    rank1_by_bucket: dict[str, list[int]] = field(default_factory=dict)
    margins: list[float] = field(default_factory=list)
    not_looking: int = 0
    looking_total: int = 0
    # track -> embeddings, for the consecutive-frame correlation
    track_embeddings: dict[str, list[np.ndarray]] = field(default_factory=dict)
    detect_seconds: dict[int, list[float]] = field(default_factory=dict)
    embed_seconds: list[float] = field(default_factory=list)
    search_seconds: list[float] = field(default_factory=list)

    def note_rank1(self, bucket: str, hit: bool) -> None:
        self.rank1_total += 1
        self.rank1_hits += int(hit)
        tally = self.rank1_by_bucket.setdefault(bucket, [0, 0])
        tally[0] += int(hit)
        tally[1] += 1


def consecutive_cosines(embeddings: Sequence[np.ndarray]) -> list[float]:
    """Cosine between each pair of consecutive embeddings in one track.

    The number that decides whether multi-frame averaging or K-of-N confirmation can
    possibly help. Near 1.0 means consecutive frames carry the same error, so
    averaging cancels nothing and K-of-N confirms a wrong match as readily as a right
    one -- and no amount of temporal logic recovers information that is not there.
    """
    pairs: list[float] = []
    for first, second in zip(embeddings, embeddings[1:], strict=False):
        a, b = l2_normalize(first), l2_normalize(second)
        pairs.append(float(np.dot(a, b)))
    return pairs


# ------------------------------------------------------------------------ the run
def evaluate(
    frames: Sequence[EvalFrame],
    models: Models,
    gallery: Gallery,
    *,
    detect_sizes: Sequence[int],
    reference_size: int,
    settings: Settings,
) -> Accumulator:
    """One pass over every frame, filling in every report's inputs."""
    acc = Accumulator()
    enrolled = gallery.index.employee_codes
    for size in detect_sizes:
        acc.found[size] = {}
        acc.detect_seconds[size] = []

    for frame in frames:
        image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
        if image is None:
            acc.unreadable += 1
            continue
        acc.frames += 1

        # The reference pass stands in for ground truth. Without it, a face the
        # production pass missed has no known width, and detection recall by width
        # would be measured only over the faces detection already found.
        reference = models.detector.detect(image, input_size=reference_size)
        acc.reference_faces += len(reference)
        for face in reference:
            bucket = bucket_of(face.bbox[2] - face.bbox[0])
            acc.reference_by_bucket[bucket] = acc.reference_by_bucket.get(bucket, 0) + 1
            acc.face_widths.append(face.bbox[2] - face.bbox[0])

        for size in detect_sizes:
            started = time.perf_counter()
            found = models.detector.detect(image, input_size=size)
            acc.detect_seconds[size].append(time.perf_counter() - started)
            for ref_face in reference:
                bucket = bucket_of(ref_face.bbox[2] - ref_face.bbox[0])
                matched = any(iou(ref_face.bbox, got.bbox) >= MATCH_IOU for got in found)
                if matched:
                    acc.found[size][bucket] = acc.found[size].get(bucket, 0) + 1

        # Recognition is scored on the production input size, which is the first of
        # the swept sizes -- the others exist only for the recall table.
        production = models.detector.detect(image, input_size=detect_sizes[0])
        _score_faces(
            acc, frame, image, production, gallery, enrolled, settings=settings
        )
    return acc


def _score_faces(
    acc: Accumulator,
    frame: EvalFrame,
    image: np.ndarray,
    faces: Sequence[Any],
    gallery: Gallery,
    enrolled: set[str],
    *,
    settings: Settings,
) -> None:
    recognizer = gallery.recognizer
    for face in faces:
        if face.kps is None:
            continue
        # Reported, never applied: a gate rejection and a recognition failure are
        # different diagnoses, and conflating them is what makes an accuracy figure
        # lie about which one to fix.
        gaze = estimate_gaze(
            [(float(x), float(y)) for x, y in face.kps],
            settings.looking_max_yaw_ratio,
            settings.looking_max_roll_degrees,
        )
        acc.looking_total += 1
        acc.not_looking += int(not gaze.looking)

        if recognizer is None:
            continue
        started = time.perf_counter()
        embedding = recognizer.embed(image, np.asarray(face.kps, dtype="float32"))
        acc.embed_seconds.append(time.perf_counter() - started)
        if not bool(np.isfinite(embedding).all()):
            continue
        acc.track_embeddings.setdefault(frame.track, []).append(embedding)

        started = time.perf_counter()
        results = gallery.index.search(embedding, k=max(2, len(enrolled)))
        acc.search_seconds.append(time.perf_counter() - started)
        if not results:
            continue
        if len(results) >= 2:
            acc.margins.append(results[0].score - results[1].score)

        bucket = bucket_of(face.bbox[2] - face.bbox[0])
        by_code = {result.employee_code: result.score for result in results}
        if frame.is_impostor:
            # Nobody in this frame is enrolled, so every score is an impostor score.
            acc.impostor.extend(by_code.values())
            continue

        # A frame may hold several labelled people and this is one face, so the
        # genuine score is the best of the labels present; the rest are impostors.
        genuine_scores = [by_code[code] for code in frame.labels if code in by_code]
        if genuine_scores:
            acc.genuine.append(max(genuine_scores))
        acc.impostor.extend(
            score for code, score in by_code.items() if code not in frame.labels
        )
        acc.note_rank1(bucket, results[0].employee_code in frame.labels)


# ------------------------------------------------------------------------ reports
def render(acc: Accumulator, gallery: Gallery, detect_sizes: Sequence[int]) -> list[str]:
    """The whole report as text lines. The JSON dump carries the same numbers."""
    out: list[str] = []
    out.append("=" * 78)
    out.append("recognition evaluation")
    out.append("=" * 78)
    out.append(
        f"  frames {acc.frames}   unreadable {acc.unreadable}   "
        f"reference faces {acc.reference_faces}"
    )
    # Impostor scores rise with gallery size, so a four-employee run reads as far
    # safer than the same thresholds will be in production.
    out.append(
        f"  enrolled employees {gallery.employees}   photos {gallery.photos}   "
        f"cpus {os.cpu_count()}"
    )
    out.append("")

    out.append("A. detection recall by face width (reference pass = ground truth)")
    header = "  " + f"{'bucket':>10}" + f"{'refs':>8}"
    header += "".join(f"{'@' + str(size):>10}" for size in detect_sizes)
    out.append(header)
    for bucket in bucket_labels():
        refs = acc.reference_by_bucket.get(bucket, 0)
        if not refs:
            continue
        row = f"  {bucket:>10}{refs:>8}"
        for size in detect_sizes:
            row += f"{acc.found[size].get(bucket, 0) / refs:>9.0%} "
        out.append(row)
    out.append("")

    out.append("A2. what this camera actually delivers")
    if acc.face_widths:
        stats = describe(acc.face_widths)
        out.append(
            f"  face width px: n={stats['n']:.0f} mean={stats['mean']:.0f} "
            f"p5={stats['p5']:.0f} p50={stats['p50']:.0f} p95={stats['p95']:.0f}"
        )
        out.append(
            "  If the mass sits under 40px, no threshold or temporal rule recovers it."
        )
    else:
        out.append("  (no faces found by the reference pass)")
    out.append("")

    out.append("B. rank-1 identification")
    if acc.rank1_total:
        out.append(
            f"  overall {acc.rank1_hits}/{acc.rank1_total} = "
            f"{acc.rank1_hits / acc.rank1_total:.1%}"
        )
        for bucket in bucket_labels():
            tally = acc.rank1_by_bucket.get(bucket)
            if tally:
                out.append(f"  {bucket:>10}  {tally[0]}/{tally[1]} = {tally[0] / tally[1]:.1%}")
    else:
        out.append("  (no labelled faces reached recognition)")
    out.append("")

    out.append("C. genuine vs impostor cosine")
    for name, scores in (("genuine", acc.genuine), ("impostor", acc.impostor)):
        stats = describe(scores)
        if stats["n"]:
            out.append(
                f"  {name:>9}: n={stats['n']:.0f} mean={stats['mean']:.3f} "
                f"sd={stats['sd']:.3f} p5={stats['p5']:.3f} p50={stats['p50']:.3f} "
                f"p95={stats['p95']:.3f} max={stats['max']:.3f}"
            )
        else:
            out.append(f"  {name:>9}: none")
    out.extend(histogram(acc.genuine, acc.impostor))
    if acc.genuine and acc.impostor:
        overlap = statistics.fmean(acc.genuine) - statistics.fmean(acc.impostor)
        out.append(f"  separation of means: {overlap:+.3f}")
        if overlap < 0.05:
            out.append(
                "  STOP: genuine and impostor are not separated. The embedding carries"
            )
            out.append(
                "  no identity at these face sizes, so nothing downstream can help."
            )
    out.append("")

    sweep = far_frr_sweep(acc.genuine, acc.impostor)
    out.append("D. FAR / FRR")
    if sweep.rows and sweep.genuine and sweep.impostor:
        out.append(f"  EER {sweep.eer:.1%} at threshold {sweep.eer_threshold:.2f}")
        for target, name in ((0.0, "FAR=0"), (0.001, "FAR<=0.1%"), (0.01, "FAR<=1%")):
            point = sweep.threshold_at_far(target)
            if point is not None:
                out.append(
                    f"  {name:>10}: threshold {point.threshold:.2f}  FRR {point.frr:.1%}"
                )
        current = sweep.row_at(env_settings.recognition_threshold)
        if current is not None:
            out.append(
                f"  at the configured RECOGNITION_THRESHOLD="
                f"{env_settings.recognition_threshold:.2f}: "
                f"FAR {current.far:.1%}  FRR {current.frr:.1%}"
            )
    else:
        out.append("  (needs both genuine and impostor scores)")
    if acc.margins:
        stats = describe(acc.margins)
        out.append(
            f"  top1-top2 margin: mean={stats['mean']:.3f} p5={stats['p5']:.3f} "
            f"p50={stats['p50']:.3f}"
        )
    out.append("")

    out.append("E. latency (seconds)")
    for size in detect_sizes:
        samples = acc.detect_seconds[size]
        if samples:
            stats = describe(samples)
            out.append(
                f"  detect @{size}: p50={stats['p50']:.3f} p95={stats['p95']:.3f} "
                f"max={stats['max']:.3f}"
            )
    for name, samples in (("embed", acc.embed_seconds), ("search", acc.search_seconds)):
        if samples:
            stats = describe(samples)
            out.append(
                f"  {name:>11}: p50={stats['p50']:.4f} p95={stats['p95']:.4f} "
                f"max={stats['max']:.4f}"
            )
    out.append("")

    out.append("F. gates and temporal correlation")
    if acc.looking_total:
        out.append(
            f"  looking gate would reject {acc.not_looking}/{acc.looking_total} "
            f"= {acc.not_looking / acc.looking_total:.1%} of detected faces"
        )
    pairs: list[float] = []
    for embeddings in acc.track_embeddings.values():
        pairs.extend(consecutive_cosines(embeddings))
    if pairs:
        stats = describe(pairs)
        out.append(
            f"  consecutive-frame cosine within a track: mean={stats['mean']:.3f} "
            f"p5={stats['p5']:.3f}  (over {stats['n']:.0f} pairs)"
        )
        out.append(
            "  Near 1.0 means consecutive frames share their error: averaging cancels"
        )
        out.append(
            "  little and K-of-N confirms a wrong match about as readily as a right one."
        )
    else:
        out.append("  (no multi-frame tracks; name files t01_000.jpg, t01_001.jpg, ...)")
    return out


def as_json(acc: Accumulator, gallery: Gallery, detect_sizes: Sequence[int]) -> dict[str, Any]:
    """The same numbers, for diffing one run against another."""
    sweep = far_frr_sweep(acc.genuine, acc.impostor)
    pairs: list[float] = []
    for embeddings in acc.track_embeddings.values():
        pairs.extend(consecutive_cosines(embeddings))
    return {
        "frames": acc.frames,
        "unreadable": acc.unreadable,
        "employees": gallery.employees,
        "photos": gallery.photos,
        "reference_faces": acc.reference_faces,
        "face_width_px": describe(acc.face_widths),
        "recall": {
            str(size): {
                bucket: acc.found[size].get(bucket, 0) / count
                for bucket, count in acc.reference_by_bucket.items()
                if count
            }
            for size in detect_sizes
        },
        "rank1": {
            "hits": acc.rank1_hits,
            "total": acc.rank1_total,
            "rate": acc.rank1_hits / acc.rank1_total if acc.rank1_total else None,
            "by_bucket": {
                bucket: {"hits": tally[0], "total": tally[1]}
                for bucket, tally in acc.rank1_by_bucket.items()
            },
        },
        "genuine": describe(acc.genuine),
        "impostor": describe(acc.impostor),
        "margin": describe(acc.margins),
        "eer": None if math.isnan(sweep.eer) else sweep.eer,
        "eer_threshold": None if math.isnan(sweep.eer_threshold) else sweep.eer_threshold,
        "looking_rejected": acc.not_looking,
        "looking_total": acc.looking_total,
        "consecutive_cosine": describe(pairs),
        "latency_seconds": {
            **{f"detect_{size}": describe(acc.detect_seconds[size]) for size in detect_sizes},
            "embed": describe(acc.embed_seconds),
            "search": describe(acc.search_seconds),
        },
    }


# ---------------------------------------------------------------------------- CLI
def build_eval_gallery(
    models: Models, settings: Settings, degrade_to: int = 0
) -> Gallery:
    """Enrol from the configured photo sources with **no cache**.

    ``cache=None`` is not an optimisation choice. The cache key covers the enrolment
    path, but a harness run is free to sweep a detector input size or degrade the
    enrolment crops, and writing those vectors into ``storage/gallery/`` would hand
    the live service embeddings from a path it is not running.

    ``degrade_to`` is the symmetry experiment: low-pass each enrolment crop to the
    live faces' typical width so both sides of the comparison have lost the same
    detail. Read report A2 for the width to try.
    """
    sources = build_sources(
        settings.employee_photos_source,
        timeout=settings.employee_photos_timeout_seconds,
        auth_header=settings.employee_photos_auth_header,
        manifest_name=settings.employee_photos_manifest,
    )
    try:
        return build_gallery(models, sources, cache=None, degrade_to=degrade_to)
    finally:
        for source in sources:
            source.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.services.evaluation",
        description="Measure detection recall and identification accuracy on real frames.",
    )
    parser.add_argument(
        "--frames",
        type=Path,
        required=True,
        help="directory of per-employee-code subdirectories of frames (UNKNOWN for impostors)",
    )
    parser.add_argument(
        "--detect-sizes",
        default="640,960,1280",
        help="DETECT_INPUT_SIZE values to sweep; the FIRST is the one recognition is scored at",
    )
    parser.add_argument(
        "--reference-size",
        type=int,
        default=1600,
        help="input size for the pseudo-ground-truth pass a missed face is measured against",
    )
    parser.add_argument(
        "--degrade-enrolment-to",
        type=int,
        default=0,
        dest="degrade_to",
        help=(
            "low-pass each enrolment crop to this width before embedding, so the "
            "gallery carries the same loss of detail the live faces do; try the p50 "
            "from report A2. Judge it on rank-1 and EER, never on the mean genuine "
            "score. 0 disables it."
        ),
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the numbers here")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.frames.is_dir():
        print(f"no such directory: {args.frames}", file=sys.stderr)
        return 2

    detect_sizes = [int(part) for part in str(args.detect_sizes).split(",") if part.strip()]
    if not detect_sizes:
        print("--detect-sizes must name at least one size", file=sys.stderr)
        return 2

    frames = find_frames(args.frames)
    if not frames:
        print(f"no images under {args.frames}", file=sys.stderr)
        return 2

    models = load_models(env_settings)
    gallery = build_eval_gallery(models, env_settings, degrade_to=args.degrade_to)
    if args.degrade_to:
        print(
            f"enrolment crops low-passed to {args.degrade_to}px "
            "-- compare rank-1 and EER against a run without it\n"
        )
    acc = evaluate(
        frames,
        models,
        gallery,
        detect_sizes=detect_sizes,
        reference_size=args.reference_size,
        settings=env_settings,
    )

    for line in render(acc, gallery, detect_sizes):
        print(line)
    if args.json is not None:
        args.json.write_text(
            json.dumps(as_json(acc, gallery, detect_sizes), indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the CLI entry point
    raise SystemExit(main())
