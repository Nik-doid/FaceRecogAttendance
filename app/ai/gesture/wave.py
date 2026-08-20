"""Wave gesture detection based on hand landmark tracking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class WaveTracker:
    """Track hand motion to detect wave gestures.

    A wave is detected when the wrist moves side-to-side repeatedly
    (at least 3 peaks = 1.5 oscillations) within a time window.

    Uses position-based peak detection: a peak is a local maximum or minimum
    in the wrist x-position history, with sufficient prominence.
    """

    # How many frames of history to keep (at ~30fps, 60 frames = 2 seconds)
    max_history: int = 60
    # Minimum prominence for a peak (normalized, 2.5% of frame width)
    min_prominence: float = 0.025
    # Minimum number of peaks (direction changes) to count as a wave
    min_peaks: int = 3
    # Minimum frames between peaks to avoid noise
    min_peak_distance: int = 3

    _history: dict[int, deque[float]] = field(default_factory=dict)
    _peak_counts: dict[int, int] = field(default_factory=dict)
    _last_peak_frame: dict[int, int] = field(default_factory=dict)
    _frame_count: dict[int, int] = field(default_factory=dict)

    def update(self, track_id: int, wrist_x: float) -> bool:
        """Update with wrist x-position (normalized 0-1). Returns True if wave detected."""
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.max_history)
            self._peak_counts[track_id] = 0
            self._last_peak_frame[track_id] = -self.min_peak_distance
            self._frame_count[track_id] = 0

        history = self._history[track_id]
        history.append(wrist_x)
        self._frame_count[track_id] += 1

        # Need at least 3 points to detect a peak (center + 2 neighbors)
        if len(history) < 3:
            return False

        # Check if the middle point (one frame ago) is a peak
        # A peak is where: prev < curr > next (max) or prev > curr < next (min)
        curr_idx = len(history) - 2  # middle point
        prev_x = history[curr_idx - 1]
        curr_x = history[curr_idx]
        next_x = history[curr_idx + 1]

        is_peak = False
        if (prev_x < curr_x and curr_x > next_x) or (prev_x > curr_x and curr_x < next_x):
            # Check prominence: difference from neighbors
            prominence = min(abs(curr_x - prev_x), abs(curr_x - next_x))
            if (
                prominence >= self.min_prominence
                and self._frame_count[track_id] - self._last_peak_frame[track_id]
                >= self.min_peak_distance
            ):
                is_peak = True

        if is_peak:
            self._peak_counts[track_id] += 1
            self._last_peak_frame[track_id] = self._frame_count[track_id]

        # Check if we have enough peaks for a wave
        if self._peak_counts[track_id] >= self.min_peaks:
            self.reset(track_id)
            return True

        return False

    def reset(self, track_id: int) -> None:
        """Reset state for a track."""
        self._history.pop(track_id, None)
        self._peak_counts[track_id] = 0
        self._last_peak_frame[track_id] = -self.min_peak_distance
        self._frame_count[track_id] = 0

    def get_peak_count(self, track_id: int) -> int:
        return self._peak_counts.get(track_id, 0)