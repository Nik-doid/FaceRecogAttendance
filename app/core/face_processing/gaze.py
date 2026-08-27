"""Is this face looking at the camera?

The gate that decides whether the rest of the pipeline runs at all. It is arithmetic
over the 5 landmarks SCRFD already returned -- no extra model, no extra forward pass --
which is what makes it cheap enough to sit in front of palm detection rather than
behind it.

Two angles are checked, and one deliberately is not:

- **Yaw** (head turned left/right) is the real signal. On a frontal face the nose sits
  midway between the eyes; as the head turns it slides toward the nearer eye. Measured
  as the nose's offset along the eye axis, normalised by the eye span, so it is scale
  invariant -- a face 3m away scores the same as one at 30cm.
- **Roll** (head tilted) is checked because ArcFace's alignment degrades past ~30
  degrees, so a heavily tilted face would not recognise reliably anyway. Compare
  ``app/ai/quality/quality.py``, which gates the worker's path on the same idea.
- **Pitch** (head up/down) is *not* checked. A high-mounted CCTV looks down on
  everyone, so every face it sees is pitched; gating on it would reject the entire
  intended deployment.

Landmark order is insightface's: left eye, right eye, nose, mouth left, mouth right,
where "left" is image-left (the subject's right).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# A frontal face measures near 0.0; a 45-degree turn runs past 0.3.
DEFAULT_MAX_YAW_RATIO = 0.35
DEFAULT_MAX_ROLL_DEGREES = 25.0

Landmarks = list[tuple[float, float]]


@dataclass(frozen=True)
class Gaze:
    """Where a face is pointing. ``yaw_ratio`` is signed: negative is image-left."""

    yaw_ratio: float
    roll_degrees: float
    looking: bool


def estimate_gaze(
    kps: Landmarks | None,
    max_yaw_ratio: float = DEFAULT_MAX_YAW_RATIO,
    max_roll_degrees: float = DEFAULT_MAX_ROLL_DEGREES,
) -> Gaze:
    """Estimate whether a face is turned toward the camera.

    A face with no landmarks is reported as not looking: the gate opens the expensive
    half of the pipeline, so "cannot tell" must fail closed.
    """
    if kps is None or len(kps) < 3:
        return Gaze(yaw_ratio=0.0, roll_degrees=0.0, looking=False)

    (left_x, left_y), (right_x, right_y), (nose_x, nose_y) = kps[0], kps[1], kps[2]
    axis_x, axis_y = right_x - left_x, right_y - left_y
    span = math.hypot(axis_x, axis_y)
    if span < 1e-6:
        # Both eyes at one point: a full profile, or a detection too small to trust.
        return Gaze(yaw_ratio=1.0, roll_degrees=0.0, looking=False)

    # Project the nose's offset from the eye midpoint onto the eye axis. Taking the
    # component *along* that axis rather than a plain dx keeps the measure independent
    # of roll, so a tilted-but-frontal face is not mistaken for a turned one.
    mid_x, mid_y = (left_x + right_x) / 2.0, (left_y + right_y) / 2.0
    offset_x, offset_y = nose_x - mid_x, nose_y - mid_y
    yaw_ratio = (offset_x * axis_x + offset_y * axis_y) / (span * span)
    roll_degrees = math.degrees(math.atan2(axis_y, axis_x))

    looking = abs(yaw_ratio) <= max_yaw_ratio and abs(roll_degrees) <= max_roll_degrees
    return Gaze(yaw_ratio=yaw_ratio, roll_degrees=roll_degrees, looking=looking)
