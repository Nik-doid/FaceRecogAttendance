"""What to write to ``ct_hr_employee_attendance_log`` for one punch.

Pure functions over the day's existing rows, with no I/O, so every branch is
exhaustively testable and a redelivered message provably takes the same branch as the
original. This is where the correctness of the whole flow lives.

The rule, two rows per employee per day:

* first punch of the day -- INSERT
* second punch -- INSERT
* third and later -- UPDATE the second row, so it always holds the latest punch

The check-in row is written once and never touched again, which is what makes it
impossible for a later punch to destroy the arrival time. The previous implementation
had one row per day and updated it, so the second punch overwrote the check-in.

Day state is re-derived from the table on every message rather than counted in memory.
The old ``InOutResolver`` kept per-process counters seeded from
``SELECT DISTINCT in_out_mode``; since every row carries the same ``in_out_mode``,
that DISTINCT collapsed N punches to at most one and the counter could never pass 1.
Counting rows fixes that, and re-reading makes the decision survive a restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.core.logging import get_logger

log = get_logger(__name__)

# Every row is written with this. It is not a state machine -- in and out are told
# apart by row order, not by this value. See the module docstring.
IN_OUT_MODE = 255


@dataclass(frozen=True)
class DayRow:
    """One existing punch for an employee on a given local date."""

    row_id: int
    log_date_time: datetime


@dataclass(frozen=True)
class Insert:
    """Append a punch."""


@dataclass(frozen=True)
class Update:
    """Move the second row's timestamp forward to this punch."""

    row_id: int


@dataclass(frozen=True)
class Skip:
    """Record nothing. ``reason`` is what the metrics and logs are keyed on."""

    reason: str


Decision = Insert | Update | Skip

SKIP_DUPLICATE = "duplicate_within_gap"


def decide(rows: list[DayRow], punch_at: datetime, min_gap_seconds: int) -> Decision:
    """Choose what to do with one punch, given the day's existing rows.

    ``rows`` must be ordered oldest first. ``punch_at`` and the row timestamps are
    both naive local values -- see :func:`to_local` for why.

    The gap check comes first and does double duty: it makes an at-least-once
    redelivery a no-op without any dedup store, and it stops someone lingering in
    front of the camera from turning one arrival into several punches.
    """
    if min_gap_seconds > 0:
        gap = timedelta(seconds=min_gap_seconds)
        for row in rows:
            if abs(punch_at - row.log_date_time) < gap:
                return Skip(SKIP_DUPLICATE)

    if len(rows) < 2:
        return Insert()
    # Two rows already exist, so the later one is the check-out and moves forward.
    # Targeted by id rather than "ORDER BY id DESC LIMIT 1", which would hit the
    # check-in row whenever only one row existed.
    return Update(rows[1].row_id)


def to_local(timestamp: datetime, timezone: str) -> datetime:
    """Convert an event timestamp into the zone the attendance table is read in.

    Events travel as aware UTC. The old code formatted that UTC value straight into
    ``log_date_time``, so an IST office's 09:15 arrival was stored as 03:45 and a
    22:00 punch landed on the previous ``log_date_only``.

    Returns a **naive** datetime, deliberately. MySQL ``DATETIME`` carries no zone, so
    the rows read back by ``day_rows`` are naive local; returning an aware value here
    would make the gap comparison in :func:`decide` raise on the first punch of a day
    that already had one.
    """
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        log.warning(
            "unknown ATTENDANCE_TIMEZONE; falling back to UTC",
            extra={"timezone": timezone},
        )
        return timestamp.astimezone(UTC).replace(tzinfo=None)
    return timestamp.astimezone(zone).replace(tzinfo=None)


def format_datetime(value: datetime) -> str:
    """MySQL DATETIME literal. Naive by construction -- the zone is already applied."""
    return value.strftime("%Y-%m-%d %H:%M:%S")


def format_date(value: datetime) -> str:
    return value.strftime("%Y-%m-%d")
