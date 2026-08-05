"""End-to-end worker loop test with scripted frames and fakes.

Drives the real RecognitionLoop thread: RTSP reader replaced by a scripted frame
source, attendance reporter a Null (records events), DB is SQLite via the shared
`db` fixture. Asserts: one report per employee (duplicate suppressed), unknown
faces persisted, snapshots written.
"""

from __future__ import annotations

import numpy as np

from app.ai.faiss.index import FaceIndex
from app.database.session import sync_session
from app.repositories.camera_repo import CameraRepository
from app.repositories.recognition_log_repo import RecognitionLogRepository
from app.repositories.setting_repo import SettingRepository
from app.repositories.unknown_face_repo import UnknownFaceRepository
from app.services.attendance_reporter.null import NullAttendanceReporter
from app.services.duplicate_suppressor import DuplicateSuppressor
from app.services.settings_service import SettingsService
from app.storage.snapshot import SnapshotStorage
from app.workers.recognition_loop.loop import RecognitionLoop
from app.workers.recognition_loop.pipeline import RecognitionPipeline
from tests.conftest import wait_until
from tests.fakes import build_ai, face_at

E1 = np.full(8, 0.5, dtype="float32")
E2 = np.full(8, -0.5, dtype="float32")


def make_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)


class FakeReader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)
        self.calls = 0
        self.connected = False

    def open(self) -> bool:
        if not self._frames:
            return False
        self.connected = True
        return True

    def read(self) -> np.ndarray | None:
        self.calls += 1
        if not self._frames:
            self.connected = False
            return None
        return self._frames.pop(0)

    def close(self) -> None:
        self.connected = False


def build_loop(db, test_settings, tmp_path, scenes):
    index = FaceIndex(dim=8)
    index.add(E1, "EMP1")
    ai = build_ai(E1, E2)
    ai.detector.scenes = scenes

    pipeline = RecognitionPipeline(ai, index, recognition_threshold=0.8)
    reporter = NullAttendanceReporter()
    reader = FakeReader([make_frame(i) for i in range(len(scenes))])

    return RecognitionLoop(
        reader=reader,
        pipeline=pipeline,
        ai=ai,
        face_index=index,
        duplicate_suppressor=DuplicateSuppressor(60),
        attendance_reporter=reporter,
        settings_service=SettingsService(SettingRepository()),
        env_settings=test_settings,
        recognition_log_repo=RecognitionLogRepository(),
        unknown_face_repo=UnknownFaceRepository(),
        camera_repo=CameraRepository(),
        snapshot_storage=SnapshotStorage(test_settings.snapshots_dir, enabled=True),
        session_factory=sync_session,
        tracker=None,
    ), reporter, reader


def test_loop_reports_once_and_records_unknown(db, test_settings, tmp_path) -> None:
    scenes = [
        [face_at(20, 20, 120, 140, 50)],   # EMP1 (left eye x=50)
        [face_at(20, 20, 120, 140, 50)],   # EMP1 again -> duplicate suppressed
        [face_at(200, 30, 300, 150, 350)],  # unknown (right eye x=350)
        [],
    ]
    loop, reporter, reader = build_loop(db, test_settings, tmp_path, scenes)
    loop.start()

    def all_consumed() -> bool:
        return reader.calls >= len(scenes)

    assert wait_until(all_consumed)
    loop.stop()

    assert len(reporter.events) == 1
    assert reporter.events[0].employee_code == "EMP1"
    assert reporter.events[0].camera_id == "cam-test"
    assert reporter.events[0].confidence > 0.8

    with sync_session() as session:
        logs = RecognitionLogRepository().list_recent(session, 10)
        unknowns = UnknownFaceRepository().list_recent(session, 10)

    assert len(logs) == 2
    reported = [log for log in logs if log.reported]
    suppressed = [log for log in logs if not log.reported]
    assert len(reported) == 1 and reported[0].employee_code == "EMP1"
    assert len(suppressed) == 1 and suppressed[0].attendance_response == "duplicate_suppressed"

    assert len(unknowns) == 1
    assert unknowns[0].camera_id == "cam-test"

    snapshots = list((test_settings.snapshots_dir).glob("*.jpg"))
    assert len(snapshots) >= 1  # at least the attendance snapshot


def test_loop_handles_stream_loss_and_stops(db, test_settings, tmp_path) -> None:
    scenes: list[list] = []
    loop, reporter, _ = build_loop(db, test_settings, tmp_path, scenes)
    loop.start()

    def errored() -> bool:
        with sync_session() as s:
            row = CameraRepository().get_by_id(s, "cam-test")
            return row is not None and row.status == "error"

    assert wait_until(errored)
    loop.stop()
    assert loop.running is False
