"""The evaluation harness. Mostly model-free: this is where the arithmetic is checked.

Threshold decisions rest on `far_frr_sweep` and on the width buckets, so these tests
matter more than their size suggests -- a wrong FAR here becomes a wrong
`RECOGNITION_THRESHOLD` in production, and nothing downstream would notice.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config.settings import settings as env_settings
from app.runtime import Models
from app.services.evaluation import (
    MATCH_IOU,
    as_json,
    bucket_labels,
    bucket_of,
    build_eval_gallery,
    consecutive_cosines,
    describe,
    evaluate,
    far_frr_sweep,
    find_frames,
    histogram,
    iou,
    parse_label,
    parse_track,
    render,
)

# --- labels ------------------------------------------------------------------


def test_a_directory_name_is_the_label() -> None:
    assert parse_label("EMP1") == {"EMP1"}


def test_a_plus_joined_name_means_several_people_in_shot() -> None:
    """A face in such a frame is correct if it matches any of them."""
    assert parse_label("EMP1+EMP4") == {"EMP1", "EMP4"}


def test_unknown_is_the_empty_set_not_a_code_called_unknown() -> None:
    """That emptiness is what makes the frame an impostor sample."""
    assert parse_label("UNKNOWN") == frozenset()
    assert parse_label("unknown") == frozenset()


def test_a_track_prefix_groups_consecutive_frames() -> None:
    assert parse_track("t07_003.jpg") == 7
    assert parse_track("t01_000.jpg") == 1
    # No prefix means the frame is its own track, which is the honest default.
    assert parse_track("whatever.jpg") is None
    assert parse_track("track1_000.jpg") is None


# --- buckets -----------------------------------------------------------------


def test_the_buckets_straddle_the_min_face_pixels_gate() -> None:
    """59 and 60 must land in different buckets or the gate cannot be evaluated."""
    assert bucket_of(59) != bucket_of(60)
    assert bucket_of(60) == "60-79"
    assert bucket_of(59) == "50-59"


def test_buckets_cover_every_width() -> None:
    assert bucket_of(0) == "<30"
    assert bucket_of(29.9) == "<30"
    assert bucket_of(10_000) == "120+"
    for width in (5, 35, 45, 55, 65, 100, 200):
        assert bucket_of(width) in bucket_labels()


def test_bucket_labels_are_ordered_and_unique() -> None:
    labels = bucket_labels()
    assert len(labels) == len(set(labels))
    assert labels[0].startswith("<")
    assert labels[-1].endswith("+")


# --- geometry ----------------------------------------------------------------


def test_iou_of_identical_boxes_is_one() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero() -> None:
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # Touching edges are not overlapping.
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_of_a_half_overlap() -> None:
    # 10x10 boxes sharing a 5x10 strip: 50 / (100 + 100 - 50).
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_a_box_offset_by_two_pixels_still_matches() -> None:
    """The two detector passes letterbox differently, so boxes differ slightly."""
    assert iou((100, 100, 160, 180), (102, 101, 162, 181)) >= MATCH_IOU


# --- the sweep ---------------------------------------------------------------


def test_perfectly_separated_scores_have_no_error_between_them() -> None:
    sweep = far_frr_sweep(genuine=[0.7, 0.8], impostor=[0.2, 0.3])
    row = sweep.row_at(0.5)
    assert row is not None
    assert row.far == 0.0
    assert row.frr == 0.0
    assert sweep.eer == 0.0


def test_far_falls_and_frr_rises_as_the_threshold_climbs() -> None:
    sweep = far_frr_sweep(genuine=[0.4, 0.6, 0.8], impostor=[0.1, 0.35, 0.55])
    fars = [row.far for row in sweep.rows]
    frrs = [row.frr for row in sweep.rows]
    assert fars == sorted(fars, reverse=True)
    assert frrs == sorted(frrs)


def test_accept_is_inclusive_of_the_threshold() -> None:
    """`ArcFaceRecognition._match` uses `>=`, so the sweep must too."""
    sweep = far_frr_sweep(genuine=[0.50], impostor=[0.50])
    row = sweep.row_at(0.50)
    assert row is not None
    assert row.far == 1.0  # the impostor is accepted
    assert row.frr == 0.0  # the genuine is not rejected


def test_completely_overlapping_scores_give_a_bad_eer() -> None:
    """The stop condition: no threshold separates them, so nothing downstream can."""
    scores = [0.3, 0.31, 0.32, 0.33]
    sweep = far_frr_sweep(genuine=scores, impostor=scores)
    assert sweep.eer > 0.2


def test_the_far_zero_operating_point_is_the_kindest_one_available() -> None:
    sweep = far_frr_sweep(genuine=[0.6, 0.9], impostor=[0.1, 0.4])
    point = sweep.threshold_at_far(0.0)
    assert point is not None
    assert point.far == 0.0
    # Of the thresholds with no false accepts, it picks the one rejecting fewest
    # genuine faces -- otherwise the report would recommend a needlessly strict value.
    assert point.frr == 0.0


def test_a_sweep_with_no_impostors_reports_no_false_accepts() -> None:
    sweep = far_frr_sweep(genuine=[0.6], impostor=[])
    assert all(row.far == 0.0 for row in sweep.rows)
    assert sweep.impostor == 0


def test_an_empty_sweep_does_not_raise() -> None:
    sweep = far_frr_sweep(genuine=[], impostor=[])
    assert sweep.genuine == 0
    assert all(row.far == 0.0 and row.frr == 0.0 for row in sweep.rows)


# --- descriptive stats -------------------------------------------------------


def test_describe_of_nothing_is_a_zero_count_not_a_crash() -> None:
    assert describe([]) == {"n": 0.0}


def test_describe_reports_the_shape_of_a_distribution() -> None:
    stats = describe([0.1, 0.2, 0.3, 0.4, 0.5])
    assert stats["n"] == 5
    assert stats["mean"] == pytest.approx(0.3)
    assert stats["min"] == pytest.approx(0.1)
    assert stats["max"] == pytest.approx(0.5)
    assert stats["p50"] == pytest.approx(0.3)


def test_histogram_survives_a_single_repeated_value() -> None:
    """All-identical scores make the range zero; the divisor must not."""
    lines = histogram([0.5, 0.5], [0.5])
    assert len(lines) > 1


def test_histogram_of_nothing_says_so() -> None:
    assert histogram([], []) == ["  (no scores)"]


# --- temporal correlation ----------------------------------------------------


def test_identical_embeddings_are_perfectly_correlated() -> None:
    """Which is the finding that would kill averaging and K-of-N alike."""
    vector = np.array([1.0, 2.0, 3.0], dtype="float32")
    assert consecutive_cosines([vector, vector, vector]) == pytest.approx([1.0, 1.0])


def test_opposed_embeddings_are_anticorrelated() -> None:
    vector = np.array([1.0, 0.0], dtype="float32")
    assert consecutive_cosines([vector, -vector]) == pytest.approx([-1.0])


def test_a_single_frame_track_yields_no_pairs() -> None:
    assert consecutive_cosines([np.ones(3, dtype="float32")]) == []
    assert consecutive_cosines([]) == []


# --- finding frames ----------------------------------------------------------


def test_frames_are_found_and_labelled_from_their_directory(tmp_path: Path) -> None:
    (tmp_path / "EMP1").mkdir()
    (tmp_path / "EMP1" / "t01_000.jpg").write_bytes(b"x")
    (tmp_path / "EMP1" / "t01_001.jpg").write_bytes(b"x")
    (tmp_path / "UNKNOWN").mkdir()
    (tmp_path / "UNKNOWN" / "a.png").write_bytes(b"x")
    (tmp_path / "EMP1" / "notes.txt").write_text("ignored")

    frames = find_frames(tmp_path)
    assert len(frames) == 3
    tracks = {frame.track for frame in frames}
    # Both EMP1 frames share a track; the impostor frame gets its own.
    assert "EMP1/t01" in tracks
    assert sum(frame.is_impostor for frame in frames) == 1


def test_the_same_track_number_under_two_people_is_two_tracks(tmp_path: Path) -> None:
    """Otherwise EMP1's track 1 and EMP4's track 1 would average together."""
    for code in ("EMP1", "EMP4"):
        (tmp_path / code).mkdir()
        (tmp_path / code / "t01_000.jpg").write_bytes(b"x")

    tracks = {frame.track for frame in find_frames(tmp_path)}
    assert tracks == {"EMP1/t01", "EMP4/t01"}


def test_an_empty_directory_yields_no_frames(tmp_path: Path) -> None:
    assert find_frames(tmp_path) == []


# --- end to end, with the real models ----------------------------------------
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SNAPSHOTS = Path(__file__).resolve().parents[1] / "storage" / "snapshots"

needs_model = pytest.mark.skipif(
    not (
        (MODELS_DIR / "det_10g.onnx").is_file()
        and (MODELS_DIR / "w600k_r50.onnx").is_file()
    ),
    reason="detector/recognizer model absent; run `python -m app.ai._download`",
)


@needs_model
def test_the_harness_runs_over_real_frames_and_emits_finite_numbers(
    tmp_path: Path, models: Models
) -> None:
    """A smoke test, deliberately asserting nothing about accuracy.

    Asserting a rank-1 rate or a threshold here would turn this into a tripwire for
    the camera and the enrolment photos: it would fail when someone re-shoots a
    portrait, which is not a code regression. What must not break is the arithmetic
    running to completion over real detections.
    """
    frames = [path for path in sorted(SNAPSHOTS.glob("*.jpg"))] if SNAPSHOTS.is_dir() else []
    if not frames:
        pytest.skip("no snapshot fixtures on disk")

    for source in frames:
        # attendance_EMP1_<stamp>.jpg -> the subject's code is the second field.
        code = source.stem.split("_")[1]
        target = tmp_path / "frames" / code
        target.mkdir(parents=True, exist_ok=True)
        (target / source.name).write_bytes(source.read_bytes())

    found = find_frames(tmp_path / "frames")
    assert found

    gallery = build_eval_gallery(models, env_settings)
    acc = evaluate(
        found,
        models,
        gallery,
        detect_sizes=[640],
        reference_size=960,
        settings=env_settings,
    )
    assert acc.frames == len(found)
    assert acc.unreadable == 0

    payload = as_json(acc, gallery, [640])
    assert payload["frames"] == len(found)
    for scores in (payload["genuine"], payload["impostor"]):
        for key, value in scores.items():
            assert math.isfinite(value), f"{key} is not finite"

    lines = render(acc, gallery, [640])
    assert any("rank-1 identification" in line for line in lines)


# --- the degraded-enrolment experiment ---------------------------------------
EMPLOYEE_PHOTOS = Path(__file__).resolve().parents[1] / "uploads" / "employees"


def _an_enrolment_face(models: Models) -> tuple[np.ndarray, np.ndarray]:
    """One real photo and its landmarks, or skip. No fixture invents a face."""
    photo = next(iter(sorted(EMPLOYEE_PHOTOS.rglob("*.jpg"))), None)
    if photo is None:
        pytest.skip("no enrolment photos on disk")
    image = cv2.imread(str(photo), cv2.IMREAD_COLOR)
    faces = models.detector.detect(image)
    if not faces or faces[0].kps is None:
        pytest.skip("no detectable face with landmarks in the enrolment photo")
    return image, np.asarray(faces[0].kps, dtype="float32")
@needs_model
def test_degrading_an_enrolment_crop_changes_the_embedding(models: Models) -> None:
    """The knob has to actually do something, or the experiment measures nothing.

    Asserting only that the vector moves. Whether it moves in a *useful* direction is
    the question the harness answers on real footage, and it is not something a unit
    test can know.
    """
    image, kps = _an_enrolment_face(models)
    full = models.recognizer.embed(image, kps)
    degraded = models.recognizer.embed(image, kps, degrade_to=40)

    assert full.shape == degraded.shape
    similarity = float(
        np.dot(full, degraded) / (np.linalg.norm(full) * np.linalg.norm(degraded))
    )
    # Different, but still the same person: a value near 1.0 would mean the low-pass
    # did nothing, and one near 0.0 would mean it destroyed the identity outright.
    assert 0.2 < similarity < 0.999


@needs_model
def test_a_degrade_width_at_or_above_the_crop_size_is_a_no_op(models: Models) -> None:
    """112 is the aligned crop size, so there is nothing to throw away."""
    image, kps = _an_enrolment_face(models)
    assert models.recognizer.embed(image, kps, degrade_to=112) == pytest.approx(
        models.recognizer.embed(image, kps), abs=1e-5
    )
    assert models.recognizer.embed(image, kps, degrade_to=0) == pytest.approx(
        models.recognizer.embed(image, kps), abs=1e-5
    )
