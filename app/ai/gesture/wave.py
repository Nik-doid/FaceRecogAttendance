"""Wave gesture detection based on hand landmark tracking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class WaveTracker:
    """Track hand motion to detect wave gestures.

    A wave is detected when the wrist moves side-to-side repeatedly
    (at least 2 full oscillations) within a time window.
    """

    # How many frames of history to keep (at ~30fps, 60 frames = 2 seconds)
    max_history: int = 60
    # Minimum horizontal displacement to count as movement
    min_dx: float = 0.05  # normalized (5% of frame width)
    # Minimum number of direction changes (peaks) to count as a wave
    min_peaks: int = 3
    # Minimum time between peaks (frames) to avoid noise
    min_peak_distance: int = 5

    _history: dict[int, deque[float]] = field(default_factory=dict)
    _peak_counts: dict[int, int] = field(default_factory=dict)
    _last_dx: dict[int, float] = field(default_factory=dict)
    _direction: dict[int, int] = field(default_factory=dict)  # 1=right, -1=left, 0=none

    def update(self, track_id: int, wrist_x: float) -> bool:
        """Update with wrist x-position (normalized 0-1). Returns True if wave detected."""
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.max_history)
            self._peak_counts[track_id] = 0
            self._last_dx[track_id] = 0.0
            self._direction[track_id] = 0

        history = self._history[track_id]
        if not history:
            history.append(wrist_x)
            return False

        dx = wrist_x - history[-1]
        history.append(wrist_x)

        # Detect direction change (peak)
        if self._last_dx[track_id] != 0 and dx != 0:
            prev_sign = 1 if self._last_dx[track_id] > 0 else -1
            curr_sign = 1 if dx > 0 else -1
            if (
                prev_sign != curr_sign
                and abs(dx) > self.min_dx
                and self._direction[track_id] != curr_sign
            ):
                self._peak_counts[track_id] += 1
                self._direction[track_id] = curr_sign

        self._last_dx[track_id] = dx

        # Check if we have enough peaks for a wave
        if self._peak_counts[track_id] >= self.min_peaks:
            self.reset(track_id)
            return True

        return False

    def reset(self, track_id: int) -> None:
        """Reset state for a track."""
        self._history.pop(track_id, None)
        self._peak_counts[track_id] = 0
        self._last_dx[track_id] = 0.0
        self._direction[track_id] = 0

    def get_peak_count(self, track_id: int) -> int:
        return self._peak_counts.get(track_id, 0)