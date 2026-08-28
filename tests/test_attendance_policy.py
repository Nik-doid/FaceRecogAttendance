"""The punch rules. Pure functions, so every branch is exercised directly.

The rule under test: two rows per employee per day. First punch inserts, second
inserts, third and later update the second row. The check-in row is written once and
must never move -- ``test_the_check_in_time_is_never_moved`` is the regression test
for the old behaviour, where the second punch updated the only row and destroyed the
arrival time.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.services.attendance_consumer.policy import (
    IN_OUT_MODE,
    DayRow,
    Insert,
    Skip,
    Update,
    decide,
    format_date,
    format_datetime,
    to_local,
)

NINE_AM = datetime(2026, 3, 2, 9, 0)
GAP = 60


def _rows(*times: datetime) -> list[DayRow]:
    return [DayRow(row_id=100 + i, log_date_time=t) for i, t in enumerate(times)]


# --- the day's shape ---------------------------------------------------------


def test_first_punch_of_the_day_inserts() -> None:
    assert decide([], NINE_AM, GAP) == Insert()


def test_second_punch_inserts_a_second_row() -> None:
    """Not an update: overwriting here is what used to destroy the check-in time."""
    rows = _rows(NINE_AM)
    assert decide(rows, NINE_AM + timedelta(hours=8), GAP) == Insert()


def test_third_punch_updates_the_second_row() -> None:
    rows = _rows(NINE_AM, NINE_AM + timedelta(hours=8))
    decision = decide(rows, NINE_AM + timedelta(hours=9), GAP)
    assert decision == Update(row_id=101)


def test_fourth_and_later_punches_keep_updating_the_same_row() -> None:
    """The row count is capped at two however many times someone passes the camera."""
    rows = _rows(NINE_AM, NINE_AM + timedelta(hours=8))
    for extra_hours in (9, 10, 11):
        decision = decide(rows, NINE_AM + timedelta(hours=extra_hours), GAP)
        assert decision == Update(row_id=101)


def test_the_check_in_time_is_never_moved() -> None:
    """No sequence of punches may ever target the first row of the day."""
    rows = _rows(NINE_AM)
    for hours in range(1, 13):
        decision = decide(rows, NINE_AM + timedelta(hours=hours), GAP)
        assert decision != Update(row_id=100)
        if isinstance(decision, Insert):
            rows = _rows(NINE_AM, NINE_AM + timedelta(hours=hours))
    assert rows[0].log_date_time == NINE_AM


def test_the_update_targets_the_later_row_not_the_highest_id() -> None:
    """Rows are ordered by time; the second one is the check-out even if ids are odd."""
    rows = [
        DayRow(row_id=900, log_date_time=NINE_AM),
        DayRow(row_id=5, log_date_time=NINE_AM + timedelta(hours=8)),
    ]
    assert decide(rows, NINE_AM + timedelta(hours=9), GAP) == Update(row_id=5)


# --- redelivery and lingering ------------------------------------------------


def test_an_exact_replay_is_skipped() -> None:
    """At-least-once delivery converges with no dedup store."""
    rows = _rows(NINE_AM)
    assert decide(rows, NINE_AM, GAP) == Skip("duplicate_within_gap")


@pytest.mark.parametrize("offset", [1, 30, 59, -30])
def test_punches_inside_the_gap_are_skipped(offset: int) -> None:
    """Standing in front of the camera must not turn one arrival into many punches."""
    rows = _rows(NINE_AM)
    decision = decide(rows, NINE_AM + timedelta(seconds=offset), GAP)
    assert isinstance(decision, Skip)


def test_a_punch_just_outside_the_gap_is_recorded() -> None:
    rows = _rows(NINE_AM)
    assert decide(rows, NINE_AM + timedelta(seconds=61), GAP) == Insert()


def test_the_gap_is_checked_against_every_row() -> None:
    """A replay of the *second* punch must be caught too, not just the first."""
    rows = _rows(NINE_AM, NINE_AM + timedelta(hours=8))
    assert isinstance(decide(rows, NINE_AM + timedelta(hours=8), GAP), Skip)


def test_a_zero_gap_disables_the_check() -> None:
    rows = _rows(NINE_AM)
    assert decide(rows, NINE_AM, 0) == Insert()


# --- timezone ----------------------------------------------------------------


def test_utc_is_converted_to_the_configured_zone() -> None:
    """The bug this fixes: 03:30 UTC is a Kathmandu office's 09:15 arrival."""
    punch = to_local(datetime(2026, 3, 1, 3, 30, tzinfo=UTC), "Asia/Kathmandu")
    assert format_datetime(punch) == "2026-03-01 09:15:00"


def test_a_late_punch_lands_on_the_right_local_day() -> None:
    """22:00 in Kathmandu is still the previous day in UTC; the date must be local."""
    punch = to_local(datetime(2026, 3, 1, 16, 15, tzinfo=UTC), "Asia/Kathmandu")
    assert format_date(punch) == "2026-03-01"
    assert format_datetime(punch) == "2026-03-01 22:00:00"

    just_after_midnight = to_local(datetime(2026, 3, 1, 18, 20, tzinfo=UTC), "Asia/Kathmandu")
    assert format_date(just_after_midnight) == "2026-03-02"
    assert format_datetime(just_after_midnight) == "2026-03-02 00:05:00"


def test_the_offset_is_the_odd_forty_five_minute_one() -> None:
    """Nepal is UTC+05:45, not a whole or half hour -- a wrong zone is 15 min out."""
    punch = to_local(datetime(2026, 3, 1, 0, 0, tzinfo=UTC), "Asia/Kathmandu")
    assert format_datetime(punch) == "2026-03-01 05:45:00"


def test_an_unknown_zone_falls_back_to_utc_rather_than_raising() -> None:
    original = datetime(2026, 3, 1, 3, 45, tzinfo=UTC)
    assert format_datetime(to_local(original, "Mars/Olympus_Mons")) == "2026-03-01 03:45:00"


def test_the_local_value_is_naive() -> None:
    """MySQL DATETIME carries no zone, so comparing against an aware value raises."""
    punch = to_local(datetime(2026, 3, 1, 3, 30, tzinfo=UTC), "Asia/Kathmandu")
    assert punch.tzinfo is None
    # The comparison decide() makes must not raise against a naive row.
    assert isinstance(decide([DayRow(1, punch)], punch, 60), Skip)


def test_utc_configured_is_a_no_op() -> None:
    original = datetime(2026, 3, 1, 3, 45, tzinfo=UTC)
    assert format_datetime(to_local(original, "UTC")) == "2026-03-01 03:45:00"


# --- the constant ------------------------------------------------------------


def test_in_out_mode_is_the_agreed_constant() -> None:
    """In and out are told apart by row order, not by this value."""
    assert IN_OUT_MODE == 255
