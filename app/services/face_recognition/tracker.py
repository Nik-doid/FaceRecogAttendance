"""Following one person across scans, so a punch is one decision and not a coin flip.

Every scan used to be independent: a marginal match published an attendance event on
whichever scan happened to clear the threshold, and a stable wrong match published one
just as readily. That is the wrong shape for the interaction this camera actually sees
-- somebody walks up, looks at the lens and holds a hand up for a few seconds, which is
three to five scans of the same face. Deciding once per *person* rather than once per
frame is what turns those scans into evidence instead of repeated attempts.

Pure arithmetic. No model, no I/O, no lock -- ``CameraRunner`` only ever has one scan
in flight (``if detecting is None and now - last_scan >= scan_interval``), so this is
single-writer by construction.

What it does **not** do: track someone walking across the frame. At a scan every second
or two a moving person's boxes may not overlap at all between scans, and no amount of
matching heuristic recovers an identity from two distant boxes. It tracks a person who
has *stopped*, which is the whole interaction here. If people ever need recognising in
transit, the scan cadence is the thing to change, not the matcher.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.core.logging import get_logger

log = get_logger(__name__)

Box = tuple[float, float, float, float]

# Overlap is the cheap and unambiguous test, so it goes first.
DEFAULT_IOU_THRESHOLD = 0.3
# When boxes do not overlap, a centre within this many face-widths is still plausibly
# the same person having shifted their weight.
DEFAULT_CENTROID_FACTOR = 1.5
# ...but only if the face is still about the same size. The camera is fixed, so
# apparent width is a distance proxy: a 2x change is somebody else, not a step forward.
DEFAULT_WIDTH_RATIO = (0.7, 1.4)
# Three missed scans is a few seconds -- long enough to survive a turned head, short
# enough that the next person to stand there is not mistaken for this one.
DEFAULT_MAX_AGE = 3
# Bounded so a busy lobby cannot grow this dict without limit.
DEFAULT_MAX_TRACKS = 32


@dataclass
class Track:
    """One person, as far as consecutive scans can tell."""

    track_id: int
    box: Box
    misses: int = 0
    scans: int = 0
    # Every code this track has matched, in order. The vote reads it; the expiry log
    # reports it.
    votes: list[str | None] = field(default_factory=list)
    best_code: str | None = None
    best_score: float = 0.0
    max_width: float = 0.0
    published: bool = False

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def centre(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0


@dataclass(frozen=True)
class TrackSummary:
    """A track that ended, and why it never produced an attendance event."""

    track_id: int
    scans: int
    reason: str
    best_code: str | None
    best_score: float
    max_width: float


def iou(a: Box, b: Box) -> float:
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


def _centre(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


class FaceTracker:
    """Assigns a stable id to each face across scans, and votes on its identity.

    ``confirm_scans`` is how many scans must agree on the *same* employee code before
    the track is confirmed. Agreeing on the code rather than merely counting accepted
    frames matters: if matching ever mis-associates two people into one track, the
    codes disagree and the track correctly never confirms.

    A warning about what the confirmation buys, because it is easy to overclaim. A
    per-frame false-accept probability p does **not** become p**K over K scans. That
    would need the scans to be independent, and they are close to identical -- the same
    person, the same distance, the same lighting, the same lens, one second apart. If
    the embedding's nearest gallery neighbour is the wrong employee, it is the wrong
    employee again on the next scan for exactly the same reason. What K-of-N removes is
    the *uncorrelated* part: a single-frame landmark glitch, one blurred scan, a
    momentary occlusion. Real and worth having, but not exponential, and the size of it
    is a property of this room that has to be measured -- see the consecutive-frame
    cosine in ``app/services/evaluation.py``.
    """

    def __init__(
        self,
        *,
        confirm_scans: int = 2,
        max_age_scans: int = DEFAULT_MAX_AGE,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        centroid_factor: float = DEFAULT_CENTROID_FACTOR,
        max_tracks: int = DEFAULT_MAX_TRACKS,
    ) -> None:
        self._confirm_scans = max(1, confirm_scans)
        self._max_age = max(1, max_age_scans)
        self._iou_threshold = iou_threshold
        self._centroid_factor = centroid_factor
        self._max_tracks = max(1, max_tracks)
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    @property
    def active(self) -> int:
        return len(self._tracks)

    def scans_of(self, track_id: int) -> int:
        """How many scans this track has been seen in. 0 if it is already gone."""
        track = self._tracks.get(track_id)
        return track.scans if track is not None else 0

    def update(self, boxes: Sequence[Box]) -> list[int]:
        """Assign a track id to each box, in the order given.

        Called once per scan with every detected face -- before the gaze gate, so a
        track survives a scan in which the person glanced away.
        """
        assignments: list[int | None] = [None] * len(boxes)
        claimed: set[int] = set()

        # Greedy, best pair first, one-to-one. O(n*m) over at most a handful of faces;
        # ponytail: greedy is fine at this size, Hungarian if the frame ever holds a
        # crowd.
        candidates: list[tuple[float, int, int]] = []
        for box_index, box in enumerate(boxes):
            for track_id, track in self._tracks.items():
                score = self._affinity(box, track)
                if score > 0.0:
                    candidates.append((score, box_index, track_id))
        candidates.sort(key=lambda item: item[0], reverse=True)

        for _score, box_index, track_id in candidates:
            if assignments[box_index] is not None or track_id in claimed:
                continue
            assignments[box_index] = track_id
            claimed.add(track_id)

        matched: list[int] = []
        for box_index, box in enumerate(boxes):
            assigned = assignments[box_index]
            # Explicitly `is None`: ids happen to start at 1 today, but a falsy-id
            # check is a trap waiting for the first time they do not.
            track_id = self._open(box) if assigned is None else assigned
            track = self._tracks[track_id]
            track.box = box
            track.misses = 0
            track.scans += 1
            track.max_width = max(track.max_width, track.width)
            matched.append(track_id)

        for track_id, track in self._tracks.items():
            if track_id not in claimed and track_id not in matched:
                track.misses += 1
        return matched

    def _affinity(self, box: Box, track: Track) -> float:
        """How much this box looks like this track. 0.0 means "not the same person"."""
        overlap = iou(box, track.box)
        if overlap >= self._iou_threshold:
            return 1.0 + overlap  # Overlap always wins over a centroid guess.

        width = box[2] - box[0]
        if width <= 0 or track.width <= 0:
            return 0.0
        ratio = width / track.width
        low, high = DEFAULT_WIDTH_RATIO
        if not low <= ratio <= high:
            return 0.0

        box_centre, track_centre = _centre(box), track.centre
        distance = (
            (box_centre[0] - track_centre[0]) ** 2 + (box_centre[1] - track_centre[1]) ** 2
        ) ** 0.5
        reach = self._centroid_factor * (width + track.width) / 2.0
        if distance > reach:
            return 0.0
        # Closer is better, but always below any real overlap.
        return 1.0 - distance / reach if reach > 0 else 0.0

    def _open(self, box: Box) -> int:
        if len(self._tracks) >= self._max_tracks:
            # ponytail: evict the longest-idle track; an LRU would be tidier if a
            # crowd ever forms in front of this camera.
            oldest = max(self._tracks.values(), key=lambda track: track.misses)
            del self._tracks[oldest.track_id]
        track_id = self._next_id
        self._next_id += 1
        self._tracks[track_id] = Track(track_id=track_id, box=box)
        return track_id

    def vote(self, track_id: int, code: str | None, score: float = 0.0) -> str | None:
        """Record this scan's verdict; return the code once the track is confirmed.

        Returns a code exactly once per track. The caller publishes on that, so a
        person standing in front of the camera for ten scans produces one event rather
        than eight suppressed ones.
        """
        track = self._tracks.get(track_id)
        if track is None:
            return None
        track.votes.append(code)
        if code is not None and score > track.best_score:
            track.best_code, track.best_score = code, score
        if track.published or code is None:
            return None

        # Count back from the newest vote, and stop at the first scan that named a
        # *different* person. The two kinds of disagreement are not the same: a scan
        # that matched nobody is an absence of evidence and must not reset the run
        # (people blink, and one blurred scan is not a contradiction), whereas a scan
        # that named someone else is evidence the track is not one person at all.
        # Without that distinction a track alternating EMP1/EMP2 accumulates two votes
        # for EMP1 and punches them in -- a coin flip presented as agreement.
        agreeing = 0
        for vote in reversed(track.votes):
            if vote == code:
                agreeing += 1
            elif vote is not None:
                break
        if agreeing < self._confirm_scans:
            return None
        track.published = True
        return code

    def expire(self) -> list[TrackSummary]:
        """Drop tracks whose person has left, and say why each produced nothing.

        A track is kept until it has actually gone -- **including after it published**.
        Retiring it on publish instead looks tidier and is wrong: the person is still
        standing there, so the next scan opens a fresh track, agrees again, and punches
        them in a second time. That would leave "one punch per visit" resting on
        ``DuplicateSuppressor``, which is a publish-rate limiter and says so in its own
        docstring -- not a rule to hang correctness on. A published track lingering
        costs nothing (``vote`` returns None for it immediately) and is exactly what
        blocks the double punch until they walk away.

        The reasons are the diagnosis this system otherwise cannot give: a face too
        small to embed, a gallery with no candidate, a candidate that never cleared the
        threshold, and scans that never agreed are four different problems with four
        different fixes, and they all look identical from the outside.
        """
        gone = [track for track in self._tracks.values() if track.misses >= self._max_age]
        summaries: list[TrackSummary] = []
        for track in gone:
            del self._tracks[track.track_id]
            if track.published:
                continue
            summaries.append(
                TrackSummary(
                    track_id=track.track_id,
                    scans=track.scans,
                    reason=_reason(track),
                    best_code=track.best_code,
                    best_score=track.best_score,
                    max_width=track.max_width,
                )
            )
        return summaries


def _reason(track: Track) -> str:
    """Why a track ended without publishing."""
    if not track.votes:
        return "never_recognised"
    accepted = [vote for vote in track.votes if vote is not None]
    if not accepted:
        return "no_match"
    if len(set(accepted)) > 1:
        return "no_agreement"
    return "too_few_scans"
