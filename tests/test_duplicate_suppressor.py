"""Duplicate suppression window tests."""

from __future__ import annotations

from app.services.duplicate_suppressor import DuplicateSuppressor


def test_suppresses_within_window() -> None:
    s = DuplicateSuppressor(timeout_seconds=10)
    assert s.check_and_record("EMP1", now=100.0)
    assert not s.check_and_record("EMP1", now=105.0)
    assert s.check_and_record("EMP1", now=111.0)


def test_employees_are_independent() -> None:
    s = DuplicateSuppressor(timeout_seconds=10)
    assert s.check_and_record("EMP1", now=100.0)
    assert s.check_and_record("EMP2", now=100.0)


def test_timeout_is_mutable_for_runtime_tuning() -> None:
    s = DuplicateSuppressor(timeout_seconds=10)
    assert s.check_and_record("EMP1", now=100.0)
    assert not s.check_and_record("EMP1", now=105.0)
    s.timeout_seconds = 2
    assert s.check_and_record("EMP1", now=106.0)


def test_clear() -> None:
    s = DuplicateSuppressor(timeout_seconds=60)
    s.check_and_record("EMP1", now=100.0)
    s.clear()
    assert s.check_and_record("EMP1", now=100.0)
