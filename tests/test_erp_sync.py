"""ERP attendance-log sync tests: settings, mapping, service, scheduler, API."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.database.session import sync_session
from app.models.recognition_log import RecognitionLog
from app.repositories.recognition_log_repo import RecognitionLogRepository
from app.services.erp_sync import ErpSyncScheduler, ErpSyncService
from app.services.erp_sync.client import ErpDbConfig
from app.services.erp_sync.factory import build_erp_sync
from app.services.erp_sync.mapping import (
    CameraMapping,
    CameraNotMappedError,
    InOutResolver,
    format_datetime,
)
from tests.conftest import wait_until

NOW = datetime(2026, 8, 4, 9, 12, 33, tzinfo=UTC)

# The new in_out_mode is always 255 for all punches
MODE = 255


# --- settings -----------------------------------------------------------------


def test_erp_camera_mapping_json_parsed() -> None:
    s = Settings(_env_file=None, erp_camera_mapping='{"cam-01":{"device_id":5,"branch_id":2}}')
    assert s.erp_camera_mapping == {"cam-01": {"device_id": 5, "branch_id": 2}}


def test_erp_sync_defaults_disabled() -> None:
    s = Settings(_env_file=None)
    assert s.erp_sync_enabled is False
    assert s.erp_in_out_mode == "toggle"
    assert s.erp_verify_mode == "FACE"
    assert s.erp_created_by == "system"
    assert s.erp_employee_table == "ct_hr_employee_master"
    assert s.erp_employee_code_column == "emp_code"
    assert s.erp_employee_id_column == "emp_id"
    assert s.erp_employee_active_filter == "_status"


# --- mapping ------------------------------------------------------------------


def test_camera_mapping_resolve_and_missing() -> None:
    mapping = CameraMapping({"cam-01": {"device_id": 1, "branch_id": 1}})
    assert mapping.resolve("cam-01") == (1, 1)
    assert mapping.contains("cam-01")
    with pytest.raises(CameraNotMappedError):
        mapping.resolve("cam-02")


def test_in_out_resolver_literal_policy() -> None:
    r = InOutResolver("1")
    # Always returns MODE (255), first is insert (False), rest are updates (True)
    mode, is_upd = r.resolve("EMP1", NOW.date())
    assert mode == MODE
    assert is_upd is False
    mode, is_upd = r.resolve("EMP1", NOW.date())
    assert mode == MODE
    assert is_upd is True


def test_in_out_resolver_toggle_first_insert_second_update_third_skipped() -> None:
    r = InOutResolver("toggle")
    d1 = NOW.date()
    d2 = NOW.date().replace(day=5)
    # First punch -> insert (MODE, False)
    mode, is_upd = r.resolve("1001", d1)
    assert mode == MODE
    assert is_upd is False
    # Second punch -> update (MODE, True)
    mode, is_upd = r.resolve("1001", d1)
    assert mode == MODE
    assert is_upd is True
    # Third punch -> skipped (None, False)
    mode, is_upd = r.resolve("1001", d1)
    assert mode is None
    assert is_upd is False
    # New day -> insert again
    mode, is_upd = r.resolve("1001", d2)
    assert mode == MODE
    assert is_upd is False
    # Different employee -> insert
    mode, is_upd = r.resolve("1002", d1)
    assert mode == MODE
    assert is_upd is False
    r.reset()
    mode, is_upd = r.resolve("1001", d1)
    assert mode == MODE
    assert is_upd is False


def test_in_out_resolver_toggle_seeded_from_existing_rows() -> None:
    r = InOutResolver("toggle")
    d1 = NOW.date()
    # C# already recorded a punch for today, so the next punch is an update
    r.seed("1001", d1, {MODE})
    mode, is_upd = r.resolve("1001", d1)
    assert mode == MODE
    assert is_upd is True
    mode, is_upd = r.resolve("1001", d1)
    assert mode is None
    assert is_upd is False


def test_in_out_resolver_invalid_policy_falls_back_to_check_in() -> None:
    r = InOutResolver("not-a-number")
    # Falls back to MODE (255) for any non-toggle policy
    mode, is_upd = r.resolve("1001", NOW.date())
    assert mode == MODE
    assert is_upd is False


def test_format_datetime_matches_csharp_style() -> None:
    assert format_datetime(NOW) == "2026-08-04 09:12:33"


# --- client -------------------------------------------------------------------


def test_erp_db_config_defaults() -> None:
    cfg = ErpDbConfig()
    assert cfg.host == "localhost"
    assert cfg.port == 3306
    assert cfg.user == "root"
    assert cfg.employee_table == "ct_hr_employee_master"
    assert cfg.employee_code_column == "emp_code"
    assert cfg.employee_id_column == "emp_id"
    assert cfg.employee_active_filter == "_status"


# --- service ------------------------------------------------------------------


class _FakeErpClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.fail_all = False
        self.employee_ids: dict[str, str] = {}
        self.existing: dict[tuple[str, str], set[int]] = {}
        self.lookup_queries: list[tuple[str, object]] = []
        self.upsert_calls: list[tuple[list[dict[str, object]], list[bool]]] = []

    def upsert_many(self, rows: list[dict[str, object]], is_update: list[bool]) -> list[bool]:
        self.calls.append(rows)
        self.upsert_calls.append((rows, is_update))
        return [False] * len(rows) if self.fail_all else [True] * len(rows)

    # Back-compat for tests that still call insert_many
    def insert_many(self, rows: list[dict[str, object]]) -> list[bool]:
        return self.upsert_many(rows, [False] * len(rows))

    def lookup_employee_id(self, employee_code: str) -> str | None:
        self.lookup_queries.append(("lookup", employee_code))
        return self.employee_ids.get(employee_code, employee_code)

    def existing_in_out_modes(self, attendance_id_no: str, log_date_only: str) -> set[int]:
        self.lookup_queries.append(("existing", attendance_id_no, log_date_only))
        return self.existing.get((attendance_id_no, log_date_only), set())


def _add_pending(
    *,
    employee_code: str,
    camera_id: str,
    timestamp: datetime,
    reported: bool = True,
    snapshot_path: str | None = None,
) -> int:
    with sync_session() as session:
        log = RecognitionLogRepository().add_entry(
            session,
            employee_code=employee_code,
            camera_id=camera_id,
            timestamp=timestamp,
            confidence=0.9,
            reported=reported,
            attendance_response="published",
            snapshot_path=snapshot_path,
        )
        return log.id


def _make_service(
    client: _FakeErpClient, *, enabled: bool = True, in_out: str = "toggle"
) -> ErpSyncService:
    return ErpSyncService(
        repo=RecognitionLogRepository(),
        client=client,  # type: ignore[arg-type]
        camera_mapping=CameraMapping({"cam-01": {"device_id": 1, "branch_id": 2}}),
        in_out_resolver=InOutResolver(in_out),
        verify_mode="FACE",
        created_by="system",
        enabled=enabled,
    )


# --- tests --------------------------------------------------------------------


def test_sync_disabled_noops(db) -> None:  # type: ignore[no-untyped-def]
    service = _make_service(_FakeErpClient(), enabled=False)
    result = service.sync_once(sync_session)
    assert result.ok
    assert result.stats.scanned == 0
    assert result.detail == "erp sync disabled"


def test_sync_no_pending_rows(db) -> None:  # type: ignore[no-untyped-def]
    service = _make_service(_FakeErpClient())
    result = service.sync_once(sync_session)
    assert result.ok
    assert result.stats.scanned == 0
    assert result.detail == "no pending events"


def test_sync_inserts_and_marks_synced(db) -> None:  # type: ignore[no-untyped-def]
    rid = _add_pending(employee_code="EMP1", camera_id="cam-01", timestamp=NOW)
    client = _FakeErpClient()
    service = _make_service(client)
    result = service.sync_once(sync_session)

    assert result.ok
    assert result.stats.scanned == 1
    assert result.stats.inserted == 1
    assert len(client.calls) == 1
    row = client.calls[0][0]
    assert row["attendance_id_no"] == "EMP1"
    assert row["in_out_mode"] == MODE
    assert row["verify_mode"] == "FACE"
    assert row["log_date_time"] == "2026-08-04 09:12:33"
    assert row["device_id"] == 1
    assert row["branch_id"] == 2
    assert row["created_by"] == "system"
    assert row["log_date_only"] == "2026-08-04"
    # First punch -> is_update = False
    assert client.upsert_calls[0][1] == [False]

    with sync_session() as session:
        assert RecognitionLogRepository().count_pending_erp_sync(session) == 0
        assert session.get(RecognitionLog, rid) is not None


def test_sync_idempotent_second_pass(db) -> None:  # type: ignore[no-untyped-def]
    _add_pending(employee_code="EMP1", camera_id="cam-01", timestamp=NOW)
    service = _make_service(_FakeErpClient())
    first = service.sync_once(sync_session)
    second = service.sync_once(sync_session)
    assert first.stats.inserted == 1
    assert second.stats.scanned == 0
    assert second.stats.inserted == 0


def test_sync_skips_unmapped_camera(db) -> None:  # type: ignore[no-untyped-def]
    rid = _add_pending(employee_code="EMP1", camera_id="cam-zz", timestamp=NOW)
    service = _make_service(_FakeErpClient())
    result = service.sync_once(sync_session)
    assert result.ok
    assert result.stats.skipped_unmapped_camera == 1
    assert result.stats.inserted == 0
    with sync_session() as session:
        assert RecognitionLogRepository().count_pending_erp_sync(session) == 0
        skipped = session.get(RecognitionLog, rid)
        assert skipped is not None
        assert skipped.erp_skip_reason == "unmapped_camera"


def test_sync_client_failure_tracks_failed_and_keeps_pending(db) -> None:  # type: ignore[no-untyped-def]
    _add_pending(employee_code="EMP1", camera_id="cam-01", timestamp=NOW)
    client = _FakeErpClient()
    client.fail_all = True
    service = _make_service(client)
    result = service.sync_once(sync_session)
    assert result.stats.inserted == 0
    assert result.stats.failed == 1
    assert any("failed" in e for e in result.stats.errors)
    with sync_session() as session:
        assert RecognitionLogRepository().count_pending_erp_sync(session) == 1


def test_sync_first_insert_second_update(db) -> None:  # type: ignore[no-untyped-def]
    # First punch (morning) -> insert
    _add_pending(employee_code="EMP1", camera_id="cam-01", timestamp=NOW)
    # Second punch (evening) -> update
    _add_pending(
        employee_code="EMP1", camera_id="cam-01", timestamp=NOW.replace(hour=18)
    )
    client = _FakeErpClient()
    service = _make_service(client, in_out="toggle")
    result = service.sync_once(sync_session)

    assert result.stats.inserted == 2  # both count as "inserted" in stats
    assert len(client.calls) == 1
    assert len(client.calls[0]) == 2
    # Both rows should have in_out_mode = 255
    modes = [row["in_out_mode"] for row in client.calls[0]]
    assert modes == [MODE, MODE]
    # First is insert (False), second is update (True)
    is_updates = client.upsert_calls[0][1]
    assert is_updates == [False, True]


def test_sync_third_punch_skipped(db) -> None:  # type: ignore[no-untyped-def]
    _add_pending(employee_code="EMP1", camera_id="cam-01", timestamp=NOW)
    _add_pending(
        employee_code="EMP1", camera_id="cam-01", timestamp=NOW.replace(hour=18)
    )
    _add_pending(
        employee_code="EMP1", camera_id="cam-01", timestamp=NOW.replace(hour=20)
    )
    client = _FakeErpClient()
    service = _make_service(client, in_out="toggle")
    result = service.sync_once(sync_session)

    assert result.stats.inserted == 2  # only first two processed
    assert result.stats.skipped_duplicate_in_out == 1  # third skipped
    with sync_session() as session:
        assert RecognitionLogRepository().count_pending_erp_sync(session) == 0


def test_sync_does_not_duplicate_punch_already_in_erp(db) -> None:  # type: ignore[no-untyped-def]
    # The C# software already recorded a punch for EMP1 today; the next punch
    # must become an update (check-out).
    _add_pending(employee_code="EMP1", camera_id="cam-01", timestamp=NOW.replace(hour=18))
    client = _FakeErpClient()
    client.existing[("EMP1", NOW.date().isoformat())] = {MODE}
    service = _make_service(client, in_out="toggle")
    result = service.sync_once(sync_session)

    assert result.stats.inserted == 1
    row = client.calls[0][0]
    assert row["in_out_mode"] == MODE
    # Should be an update since C# already recorded one
    assert client.upsert_calls[0][1] == [True]


def test_sync_skips_no_employee_id_and_marks_reason(db) -> None:  # type: ignore[no-untyped-def]
    rid = _add_pending(employee_code="UNKNOWN-EMP", camera_id="cam-01", timestamp=NOW)
    client = _FakeErpClient()
    client.employee_ids = {"UNKNOWN-EMP": None}
    service = _make_service(client, in_out="toggle")
    result = service.sync_once(sync_session)
    assert result.stats.skipped_no_employee == 1
    assert result.stats.inserted == 0
    with sync_session() as session:
        assert RecognitionLogRepository().count_pending_erp_sync(session) == 0
        skipped = session.get(RecognitionLog, rid)
        assert skipped is not None
        assert skipped.erp_skip_reason == "no_employee_id"


def test_sync_deletes_snapshot_for_skipped_event(db, tmp_path) -> None:  # type: ignore[no-untyped-def]
    snapshot = tmp_path / "attendance_EMP1_20260804.jpg"
    snapshot.write_bytes(b"jpeg")
    _add_pending(
        employee_code="EMP1",
        camera_id="cam-zz",
        timestamp=NOW,
        snapshot_path=str(snapshot),
    )
    from app.storage.snapshot import SnapshotStorage

    client = _FakeErpClient()
    service = ErpSyncService(
        repo=RecognitionLogRepository(),
        client=client,  # type: ignore[arg-type]
        camera_mapping=CameraMapping({"cam-01": {"device_id": 1, "branch_id": 2}}),
        in_out_resolver=InOutResolver("toggle"),
        verify_mode="FACE",
        created_by="system",
        snapshot_storage=SnapshotStorage(tmp_path / "snapshots", enabled=True),
    )
    result = service.sync_once(sync_session)
    assert result.stats.skipped_unmapped_camera == 1
    assert snapshot.exists() is False


def test_sync_keeps_snapshot_for_inserted_event(db, tmp_path) -> None:  # type: ignore[no-untyped-def]
    snapshot = tmp_path / "attendance_EMP1_20260804.jpg"
    snapshot.write_bytes(b"jpeg")
    _add_pending(
        employee_code="EMP1",
        camera_id="cam-01",
        timestamp=NOW,
        snapshot_path=str(snapshot),
    )
    from app.storage.snapshot import SnapshotStorage

    client = _FakeErpClient()
    service = ErpSyncService(
        repo=RecognitionLogRepository(),
        client=client,  # type: ignore[arg-type]
        camera_mapping=CameraMapping({"cam-01": {"device_id": 1, "branch_id": 2}}),
        in_out_resolver=InOutResolver("toggle"),
        verify_mode="FACE",
        created_by="system",
        snapshot_storage=SnapshotStorage(tmp_path / "snapshots", enabled=True),
    )
    result = service.sync_once(sync_session)
    assert result.stats.inserted == 1
    assert snapshot.exists() is True


def test_build_erp_sync_from_settings_respects_enabled(db) -> None:  # type: ignore[no-untyped-def]
    s = Settings(_env_file=None, erp_sync_enabled=False)
    service = build_erp_sync(s)
    assert service.enabled is False


# --- repository ---------------------------------------------------------------


def test_list_pending_filters_reported_and_erp_synced(db) -> None:  # type: ignore[no-untyped-def]
    _add_pending(employee_code="EMP1", camera_id="cam-01", timestamp=NOW)
    _add_pending(
        employee_code="EMP2", camera_id="cam-01", timestamp=NOW, reported=False
    )
    with sync_session() as session:
        repo = RecognitionLogRepository()
        assert repo.count_pending_erp_sync(session) == 1
        pending = repo.list_pending_erp_sync(session)
        assert [p.employee_code for p in pending] == ["EMP1"]


# --- scheduler ----------------------------------------------------------------


def test_scheduler_runs_once_then_stops(db) -> None:  # type: ignore[no-untyped-def]
    calls: list[int] = []
    scheduler = ErpSyncScheduler(
        lambda: calls.append(1), interval_seconds=1
    )
    assert scheduler.interval_seconds == 1
    assert scheduler.running is False
    scheduler.start()
    assert wait_until(lambda: len(calls) >= 1)
    scheduler.stop()
    assert scheduler.running is False


# --- API ----------------------------------------------------------------------


def test_erp_sync_status_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/sync/attendance-log/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "enabled" in body
    assert "pending" in body


def test_erp_sync_run_requires_auth(client: TestClient) -> None:
    resp = client.post("/api/v1/sync/attendance-log")
    assert resp.status_code == 401


def test_erp_sync_run_audits(client: TestClient, auth_headers) -> None:  # type: ignore[no-untyped-def]
    from app.repositories.audit_repo import AuditLogRepository

    resp = client.post("/api/v1/sync/attendance-log", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "ok" in body
    assert "stats" in body

    with sync_session() as session:
        audit = AuditLogRepository().list_recent(session)
    assert any(entry.action == "sync.attendance_log" for entry in audit)