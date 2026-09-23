"""Following one person across scans.

No models, no frames -- just boxes. The interaction being modelled is somebody walking
up to the camera, looking at it and holding a hand up for a few seconds, so the cases
that matter are "same person, slightly moved", "two people who must not swap", and
"somebody left and a different person took their place".
"""

from __future__ import annotations

from app.services.face_recognition.tracker import FaceTracker, iou

# A face roughly 100px wide at (100, 100).
FACE = (100.0, 100.0, 200.0, 220.0)


def _shift(box: tuple[float, float, float, float], dx: float, dy: float = 0.0):
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def _scale(box: tuple[float, float, float, float], factor: float):
    width, height = box[2] - box[0], box[3] - box[1]
    return (box[0], box[1], box[0] + width * factor, box[1] + height * factor)


# --- identity ----------------------------------------------------------------


def test_the_same_box_twice_is_one_person() -> None:
    tracker = FaceTracker()
    first = tracker.update([FACE])
    second = tracker.update([FACE])
    assert first == second


def test_a_person_shifting_their_weight_keeps_their_track() -> None:
    """Standing still is not being motionless; the box moves a little every scan."""
    tracker = FaceTracker()
    first = tracker.update([FACE])
    second = tracker.update([_shift(FACE, 15, 8)])
    assert first == second


def test_a_box_across_the_frame_is_a_different_person() -> None:
    tracker = FaceTracker()
    first = tracker.update([FACE])
    second = tracker.update([_shift(FACE, 900)])
    assert first != second


def test_a_face_of_a_very_different_size_is_a_different_person() -> None:
    """The camera is fixed, so apparent width is a distance proxy."""
    tracker = FaceTracker()
    first = tracker.update([FACE])
    # Half the width, nudged clear of any overlap: someone further back, not the
    # same person having shrunk.
    second = tracker.update([_shift(_scale(FACE, 0.4), 160)])
    assert first != second


def test_two_people_keep_separate_stable_ids() -> None:
    tracker = FaceTracker()
    left, right = FACE, _shift(FACE, 400)
    first = tracker.update([left, right])
    assert len(set(first)) == 2

    second = tracker.update([_shift(left, 10), _shift(right, -10)])
    assert second == first


def test_two_people_do_not_swap_ids_when_they_approach() -> None:
    """A swap here would publish attendance against the wrong person."""
    tracker = FaceTracker()
    left, right = FACE, _shift(FACE, 400)
    first = tracker.update([left, right])
    for step in (40, 80, 120):
        moved = tracker.update([_shift(left, step), _shift(right, -step)])
        assert moved == first, f"ids swapped after moving {step}px"


def test_one_box_can_only_claim_one_track() -> None:
    """Two tracks overlapping one detection must not both be assigned to it."""
    tracker = FaceTracker()
    tracker.update([FACE, _shift(FACE, 400)])
    ids = tracker.update([FACE])
    assert len(ids) == 1


# --- expiry ------------------------------------------------------------------


def test_a_track_survives_a_scan_where_the_face_was_missed() -> None:
    """A glance away must not reset the vote count."""
    tracker = FaceTracker(max_age_scans=3)
    first = tracker.update([FACE])
    tracker.update([])
    tracker.expire()
    assert tracker.update([FACE]) == first


def test_a_track_is_dropped_after_the_configured_misses() -> None:
    tracker = FaceTracker(max_age_scans=2)
    first = tracker.update([FACE])
    for _ in range(2):
        tracker.update([])
    tracker.expire()
    assert tracker.update([FACE]) != first


def test_a_published_track_is_kept_while_the_person_is_still_there() -> None:
    """Otherwise they are punched in again every confirm cycle they stand there.

    Retiring on publish would leave "one punch per visit" resting on
    `DuplicateSuppressor`, which is a rate limiter rather than a rule.
    """
    tracker = FaceTracker(max_age_scans=3, confirm_scans=1)
    (track_id,) = tracker.update([FACE])
    assert tracker.vote(track_id, "EMP1", 0.9) == "EMP1"
    tracker.expire()
    for _ in range(4):
        assert tracker.update([FACE]) == [track_id]
        assert tracker.vote(track_id, "EMP1", 0.9) is None
        tracker.expire()


def test_the_next_person_to_stand_there_is_a_new_track() -> None:
    """Once the previous person has actually left, that is."""
    tracker = FaceTracker(max_age_scans=1, confirm_scans=1)
    first = tracker.update([FACE])
    tracker.vote(first[0], "EMP1", 0.9)
    tracker.update([])  # they walked away
    tracker.expire()
    assert tracker.update([FACE]) != first


def test_the_track_count_is_bounded() -> None:
    """A busy lobby must not grow this dict without limit."""
    tracker = FaceTracker(max_tracks=4)
    for index in range(20):
        tracker.update([_shift(FACE, index * 500)])
    assert tracker.active <= 4


# --- voting ------------------------------------------------------------------


def test_a_single_scan_confirms_when_one_is_all_that_is_asked() -> None:
    """confirm_scans=1 is the behaviour this service had before tracking existed."""
    tracker = FaceTracker(confirm_scans=1)
    (track_id,) = tracker.update([FACE])
    assert tracker.vote(track_id, "EMP1", 0.9) == "EMP1"


def test_agreement_is_required_before_a_person_is_punched_in() -> None:
    tracker = FaceTracker(confirm_scans=2)
    (track_id,) = tracker.update([FACE])
    assert tracker.vote(track_id, "EMP1", 0.9) is None
    tracker.update([FACE])
    assert tracker.vote(track_id, "EMP1", 0.9) == "EMP1"


def test_a_track_publishes_at_most_once() -> None:
    """Otherwise standing there for ten scans is eight suppressed duplicate events."""
    tracker = FaceTracker(confirm_scans=1)
    (track_id,) = tracker.update([FACE])
    assert tracker.vote(track_id, "EMP1", 0.9) == "EMP1"
    for _ in range(5):
        tracker.update([FACE])
        assert tracker.vote(track_id, "EMP1", 0.9) is None


def test_a_track_that_cannot_make_up_its_mind_never_confirms() -> None:
    """The guard against a mis-associated track punching in whoever it saw last.

    Voting on the *code* rather than counting accepted scans is what makes this work:
    two people merged into one track disagree, and disagreement blocks.
    """
    tracker = FaceTracker(confirm_scans=2)
    (track_id,) = tracker.update([FACE])
    for code in ("EMP1", "EMP2", "EMP1", "EMP2"):
        tracker.update([FACE])
        assert tracker.vote(track_id, code, 0.9) is None


def test_agreement_need_not_be_on_consecutive_scans() -> None:
    """One bad scan in the middle should not restart the count."""
    tracker = FaceTracker(confirm_scans=2)
    (track_id,) = tracker.update([FACE])
    assert tracker.vote(track_id, "EMP1", 0.9) is None
    tracker.update([FACE])
    assert tracker.vote(track_id, None, 0.0) is None
    tracker.update([FACE])
    assert tracker.vote(track_id, "EMP1", 0.9) == "EMP1"


def test_voting_on_a_vanished_track_is_harmless() -> None:
    tracker = FaceTracker(confirm_scans=1)
    assert tracker.vote(9999, "EMP1", 0.9) is None


# --- why a track produced nothing --------------------------------------------


def test_a_track_that_never_reached_recognition_says_so() -> None:
    tracker = FaceTracker(confirm_scans=1, max_age_scans=1)
    tracker.update([FACE])
    tracker.update([])
    (summary,) = tracker.expire()
    assert summary.reason == "never_recognised"


def test_a_track_that_matched_nobody_says_so() -> None:
    """The gallery had no candidate, or none cleared the threshold."""
    tracker = FaceTracker(confirm_scans=1, max_age_scans=1)
    (track_id,) = tracker.update([FACE])
    tracker.vote(track_id, None, 0.0)
    tracker.update([])
    (summary,) = tracker.expire()
    assert summary.reason == "no_match"


def test_a_track_whose_scans_disagreed_says_so() -> None:
    tracker = FaceTracker(confirm_scans=3, max_age_scans=1)
    (track_id,) = tracker.update([FACE])
    tracker.vote(track_id, "EMP1", 0.9)
    tracker.update([FACE])
    tracker.vote(track_id, "EMP2", 0.8)
    tracker.update([])
    (summary,) = tracker.expire()
    assert summary.reason == "no_agreement"
    # The best guess survives, so an operator can see who it nearly was.
    assert summary.best_code == "EMP1"
    assert summary.best_score == 0.9


def test_a_track_that_ran_out_of_time_says_so() -> None:
    tracker = FaceTracker(confirm_scans=3, max_age_scans=1)
    (track_id,) = tracker.update([FACE])
    tracker.vote(track_id, "EMP1", 0.9)
    tracker.update([])
    (summary,) = tracker.expire()
    assert summary.reason == "too_few_scans"


def test_a_confirmed_track_is_not_reported_as_unresolved() -> None:
    tracker = FaceTracker(confirm_scans=1, max_age_scans=1)
    (track_id,) = tracker.update([FACE])
    tracker.vote(track_id, "EMP1", 0.9)
    assert tracker.expire() == []


def test_the_summary_carries_the_widest_the_face_ever_got() -> None:
    """So "they were only ever 30px away from the lens" is answerable from the log."""
    tracker = FaceTracker(confirm_scans=2, max_age_scans=1)
    (track_id,) = tracker.update([FACE])
    tracker.update([_scale(FACE, 1.5)])
    tracker.vote(track_id, None, 0.0)
    tracker.update([])
    (summary,) = tracker.expire()
    assert summary.max_width == 150.0


def test_scans_of_counts_appearances_and_forgets_a_dropped_track() -> None:
    tracker = FaceTracker(confirm_scans=5, max_age_scans=1)
    (track_id,) = tracker.update([FACE])
    tracker.update([FACE])
    assert tracker.scans_of(track_id) == 2
    tracker.update([])
    tracker.expire()
    assert tracker.scans_of(track_id) == 0


# --- geometry ----------------------------------------------------------------


def test_iou_basics() -> None:
    assert iou(FACE, FACE) == 1.0
    assert iou(FACE, _shift(FACE, 1000)) == 0.0
    # Touching edges do not overlap.
    assert iou((0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 20.0, 10.0)) == 0.0


def test_a_no_match_scan_is_absence_of_evidence_not_contradiction() -> None:
    """People blink. One blurred scan must not reset the run."""
    tracker = FaceTracker(confirm_scans=3)
    (track_id,) = tracker.update([FACE])
    for code in ("EMP1", None, "EMP1", None, "EMP1"):
        tracker.update([FACE])
        outcome = tracker.vote(track_id, code, 0.9 if code else 0.0)
    assert outcome == "EMP1"


def test_naming_someone_else_does_reset_the_run() -> None:
    """That is evidence the track is not one person, which is different from a blink."""
    tracker = FaceTracker(confirm_scans=3)
    (track_id,) = tracker.update([FACE])
    for code in ("EMP1", "EMP1", "EMP2", "EMP1", "EMP1"):
        tracker.update([FACE])
        assert tracker.vote(track_id, code, 0.9) is None
    # Only after a third uninterrupted EMP1 does it confirm.
    tracker.update([FACE])
    assert tracker.vote(track_id, "EMP1", 0.9) == "EMP1"
