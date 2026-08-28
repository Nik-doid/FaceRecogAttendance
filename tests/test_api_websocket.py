"""The browser-webcam WebSocket: frame echo plus the dispatched palm/face pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints import websocket as websocket_endpoint
from app.config.settings import settings
from app.core.face_processing.dispatchers import (
    FaceDetectionDispatcher,
    FaceRecognitionDispatcher,
    MarkAttendanceDispatcher,
    PalmDetectionDispatcher,
)
from app.core.face_processing.face_detection_handlers import ScrfdFaceDetection
from app.core.face_processing.face_recognition_handlers import ArcFaceRecognition
from app.core.face_processing.gallery import Gallery, build_gallery
from app.core.face_processing.gaze import estimate_gaze
from app.core.face_processing.palm_detection_handlers import (
    BlazePalmDetection,
    palm_search_box,
    scan_regions,
)
from app.core.face_processing.photos import build_sources
from app.runtime import Models
from app.schemas.face_processing import (
    AttendanceSinkType,
    FaceDetectorType,
    FaceProcessConfig,
    FaceRecognizerType,
    FaceResult,
    PalmDetectorType,
)
from app.services.face_recognition.exceptions import StepNotImplementedError
from app.services.face_recognition.process import FaceRecognitionProcess

PALM_MODEL = settings.models_dir / "palm_detection_mediapipe_2023feb.onnx"
FACE_MODEL = settings.models_dir / "det_10g.onnx"
RECOGNIZE_MODEL = settings.models_dir / "w600k_r50.onnx"
needs_model = pytest.mark.skipif(
    not (PALM_MODEL.is_file() and FACE_MODEL.is_file() and RECOGNIZE_MODEL.is_file()),
    reason="detector/recognizer model absent; run `python -m app.ai._download`",
)

PHOTO_SOURCES = settings.employee_photos_source


@pytest.fixture(scope="module")
def enrolled(models: Models) -> Gallery:
    return build_gallery(models, build_sources(PHOTO_SOURCES))


@pytest.fixture(scope="module")
def empty_gallery(models: Models, tmp_path_factory: pytest.TempPathFactory) -> Gallery:
    return build_gallery(models, build_sources([str(tmp_path_factory.mktemp("no-photos"))]))
needs_photos = pytest.mark.skipif(
    not any(Path(root).is_dir() and any(Path(root).iterdir()) for root in PHOTO_SOURCES),
    reason="no enrolment photos under EMPLOYEE_PHOTOS_SOURCE",
)


def _jpeg(width: int = 64, height: int = 48) -> bytes:
    _, encoded = cv2.imencode(".jpg", np.zeros((height, width, 3), np.uint8))
    return bytes(encoded.tobytes())


@needs_model
def test_webcam_ws_echoes_a_jpeg_frame(client: TestClient) -> None:
    with client.websocket_connect("/api/v1/webcam/ws") as ws:
        ws.send_bytes(_jpeg())
        assert ws.receive_bytes()[:2] == b"\xff\xd8"  # JPEG start-of-image marker


@needs_model
def test_webcam_ws_ignores_undecodable_bytes(client: TestClient) -> None:
    with client.websocket_connect("/api/v1/webcam/ws") as ws:
        ws.send_bytes(b"not a jpeg")
        ws.send_bytes(_jpeg())
        assert ws.receive_bytes()[:2] == b"\xff\xd8"


@needs_model
def test_palm_dispatcher_returns_blazepalm() -> None:
    handler = PalmDetectionDispatcher.dispatch(
        PalmDetectorType.BLAZEPALM, settings.models_dir, 0.5
    )
    assert isinstance(handler, BlazePalmDetection)


@pytest.mark.parametrize("detector", [PalmDetectorType.MEDIAPIPE, "nope"])
def test_palm_dispatcher_rejects_unimplemented(detector: PalmDetectorType) -> None:
    with pytest.raises(StepNotImplementedError):
        PalmDetectionDispatcher.dispatch(detector, settings.models_dir, 0.5)


@needs_model
def test_face_dispatcher_returns_scrfd(models: Models) -> None:
    handler = FaceDetectionDispatcher.dispatch(FaceDetectorType.SCRFD, models.detector)
    assert isinstance(handler, ScrfdFaceDetection)


def test_face_dispatcher_rejects_unknown_detector(models: Models) -> None:
    with pytest.raises(StepNotImplementedError):
        FaceDetectionDispatcher.dispatch("nope", models.detector)


@needs_model
def test_recognition_dispatcher_returns_arcface(empty_gallery: Gallery) -> None:
    handler = FaceRecognitionDispatcher.dispatch(
        FaceRecognizerType.ARCFACE, empty_gallery, 0.6
    )
    assert isinstance(handler, ArcFaceRecognition)


def test_recognition_dispatcher_rejects_unknown_recognizer(empty_gallery: Gallery) -> None:
    with pytest.raises(StepNotImplementedError):
        FaceRecognitionDispatcher.dispatch("nope", empty_gallery, 0.6)


def test_step_4_is_not_implemented_yet() -> None:
    """Pins the stub contract: wiring step 4 must be a deliberate change."""
    with pytest.raises(StepNotImplementedError):
        MarkAttendanceDispatcher.dispatch(AttendanceSinkType.NULL)


@needs_model
async def test_process_frames_finds_no_palm_in_a_blank_frame(
    models: Models, empty_gallery: Gallery
) -> None:
    process = FaceRecognitionProcess(
        FaceProcessConfig(), models, empty_gallery, settings.models_dir
    )
    result = await process.process_frames(np.zeros((240, 320, 3), np.uint8))
    assert result.palm.detected is False
    assert result.faces == []


@needs_model
async def test_face_detection_finds_no_face_in_a_blank_frame(models: Models) -> None:
    """Step 2 in isolation: a blank frame has no face even though step 1 gates it."""
    handler = ScrfdFaceDetection(models.detector)
    assert await handler.detect(np.zeros((240, 320, 3), np.uint8)) == []


@needs_model
async def test_recognition_leaves_faces_unmatched_against_an_empty_gallery(
    empty_gallery: Gallery,
) -> None:
    handler = ArcFaceRecognition(empty_gallery)
    face = FaceResult(bbox=(0.0, 0.0, 10.0, 10.0), score=0.9, kps=[(1.0, 1.0)] * 5)
    (matched,) = await handler.recognize(np.zeros((240, 320, 3), np.uint8), [face])
    assert matched.employee_code is None
    assert matched.confidence == 0.0


@needs_model
@needs_photos
async def test_recognition_matches_an_enrolled_photo_to_its_own_code(
    models: Models, enrolled: Gallery
) -> None:
    """End-to-end steps 2+3: an enrolment photo must recognise as its own employee.

    This is the weakest possible identity claim (the probe *is* a gallery image), which
    is exactly what makes it a regression test for the wiring -- landmarks reaching
    ArcFace, embeddings reaching the index, codes coming back out -- rather than for
    model accuracy.
    """
    photos = sorted(
        photo
        for root in PHOTO_SOURCES
        for employee_dir in sorted(p for p in Path(root).iterdir() if p.is_dir())
        for photo in sorted(employee_dir.glob("*.jpg"))
    )
    assert photos, "expected at least one enrolment photo"
    probe = photos[0]
    expected_code = probe.parent.name

    detect = ScrfdFaceDetection(models.detector)
    recognize = ArcFaceRecognition(enrolled)
    image = cv2.imread(str(probe))
    faces = await recognize.recognize(image, await detect.detect(image))

    assert faces, f"no face detected in {probe}"
    assert faces[0].employee_code == expected_code
    assert faces[0].confidence > 0.9  # A photo against itself is a near-exact match.


# --- ROI scanning (the CCTV distance fix) -----------------------------------


def test_scan_regions_yields_only_the_frame_when_grid_is_one() -> None:
    frame = np.zeros((90, 160, 3), np.uint8)
    regions = list(scan_regions(frame, grid=1))
    assert len(regions) == 1
    assert regions[0] is frame


def test_scan_regions_yields_the_frame_then_the_grid() -> None:
    frame = np.zeros((90, 160, 3), np.uint8)
    regions = list(scan_regions(frame, grid=3, overlap=0.2))
    assert len(regions) == 1 + 3 * 3
    assert regions[0].shape == frame.shape


def test_scan_regions_tiles_overlap_and_stay_in_bounds() -> None:
    """Overlap is what stops a hand on a tile seam from being cut in half."""
    height, width = 90, 160
    frame = np.zeros((height, width, 3), np.uint8)
    tiles = list(scan_regions(frame, grid=2, overlap=0.25))[1:]

    for tile in tiles:
        assert tile.size > 0
        assert tile.shape[0] <= height and tile.shape[1] <= width
    # Each tile is wider than a bare width/grid slice, which is the overlap showing up.
    assert max(t.shape[1] for t in tiles) > width / 2


def test_scan_regions_magnifies_a_small_palm() -> None:
    """The whole point: a tile crop makes a distant palm a bigger share of the input."""
    frame = np.zeros((720, 1280, 3), np.uint8)
    tile = list(scan_regions(frame, grid=3, overlap=0.2))[1]
    # A 60px hand is 4.7% of the full frame width but ~11% of a third-width tile.
    assert 60 / tile.shape[1] > 2 * (60 / frame.shape[1])


# --- palm debounce ----------------------------------------------------------


def test_palm_debounce_holds_state_until_enough_frames_agree() -> None:
    debounce = websocket_endpoint.PalmDebounce(confirm_frames=2)
    assert debounce.update(True) is False  # one frame is not enough to flip
    assert debounce.update(True) is True
    assert debounce.update(False) is True  # nor to flip back
    assert debounce.update(False) is False


def test_palm_debounce_resets_pending_on_disagreement() -> None:
    """A lone stray detection between negatives must not accumulate toward a flip."""
    debounce = websocket_endpoint.PalmDebounce(confirm_frames=2)
    assert debounce.update(True) is False
    assert debounce.update(False) is False
    assert debounce.update(True) is False
    assert debounce.update(True) is True


# --- the camera socket ------------------------------------------------------


class _FakeReader:
    """Stands in for CameraReader: always open, endless blank frames."""

    def __init__(self, *, opens: bool = True) -> None:
        self._opens = opens
        self.closed = False

    def open(self) -> bool:
        return self._opens

    def read(self) -> np.ndarray:
        return np.zeros((120, 160, 3), np.uint8)

    def close(self) -> None:
        self.closed = True


@needs_model
def test_camera_ws_reports_a_camera_it_cannot_open(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        websocket_endpoint, "_create_camera_reader", lambda _s: _FakeReader(opens=False)
    )
    with client.websocket_connect("/api/v1/camera/ws") as ws:
        assert "error" in ws.receive_json()


@needs_model
def test_camera_ws_streams_frames_and_detection_results(
    client: TestClient, monkeypatch
) -> None:
    """The RTSP socket must emit both video and pipeline output, not just video."""
    monkeypatch.setattr(websocket_endpoint, "_create_camera_reader", lambda _s: _FakeReader())

    frames = 0
    detection = None
    with client.websocket_connect("/api/v1/camera/ws") as ws:
        for _ in range(40):
            message = ws.receive()
            if message.get("bytes"):
                frames += 1
            elif message.get("text"):
                detection = json.loads(message["text"])
                break

    assert frames > 0, "no video frames streamed"
    assert detection is not None, "detection result never arrived"
    # A blank frame has no palm, so the pipeline stops at step 1 and reports no faces.
    assert detection["palm"] is False
    assert detection["faces"] == []
    assert (detection["width"], detection["height"]) == (160, 120)


# --- the looking gate -------------------------------------------------------

# insightface landmark order: left eye, right eye, nose, mouth left, mouth right.
def _kps(nose_x: float) -> list[tuple[float, float]]:
    return [(0.0, 0.0), (100.0, 0.0), (nose_x, 40.0), (30.0, 70.0), (70.0, 70.0)]


def _rolled(kps: list[tuple[float, float]], degrees: float) -> list[tuple[float, float]]:
    """Rotate every landmark rigidly about the eye midpoint -- a real head tilt.

    Moving one eye alone would shear the face rather than roll it, and would shift the
    nose relative to the eye axis, which is exactly the yaw signal under test.
    """
    angle = math.radians(degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    cx, cy = (kps[0][0] + kps[1][0]) / 2, (kps[0][1] + kps[1][1]) / 2
    return [
        (cx + (x - cx) * cos_a - (y - cy) * sin_a, cy + (x - cx) * sin_a + (y - cy) * cos_a)
        for x, y in kps
    ]


def test_gaze_accepts_a_frontal_face() -> None:
    gaze = estimate_gaze(_kps(50.0))
    assert gaze.looking is True
    assert abs(gaze.yaw_ratio) < 0.01


def test_gaze_rejects_a_turned_head() -> None:
    """The nose slides toward the nearer eye as the head turns away."""
    assert estimate_gaze(_kps(90.0)).looking is False
    assert estimate_gaze(_kps(10.0)).looking is False


def test_gaze_is_symmetric_about_centre() -> None:
    left = estimate_gaze(_kps(30.0))
    right = estimate_gaze(_kps(70.0))
    assert left.yaw_ratio == pytest.approx(-right.yaw_ratio)


def test_gaze_rejects_a_heavily_rolled_head() -> None:
    """Past ~25 degrees of tilt ArcFace alignment degrades, so the gate closes."""
    assert estimate_gaze(_rolled(_kps(50.0), 40.0)).looking is False
    assert estimate_gaze(_rolled(_kps(50.0), -40.0)).looking is False


def test_gaze_yaw_ignores_roll() -> None:
    """A tilted-but-frontal face must not read as a turned one.

    Projecting onto the eye axis rather than taking a plain dx is what buys this.
    """
    upright = estimate_gaze(_kps(50.0))
    for degrees in (-20.0, -10.0, 10.0, 20.0):
        tilted = estimate_gaze(_rolled(_kps(50.0), degrees))
        assert tilted.yaw_ratio == pytest.approx(upright.yaw_ratio, abs=1e-6)
        assert tilted.looking is True


def test_gaze_fails_closed_without_landmarks() -> None:
    """The gate opens the expensive half of the pipeline, so 'cannot tell' means no."""
    assert estimate_gaze(None).looking is False
    assert estimate_gaze([(0.0, 0.0)]).looking is False


# --- face-anchored palm search ----------------------------------------------


def test_palm_search_box_spans_the_margin_around_the_face() -> None:
    box = palm_search_box((400.0, 300.0, 500.0, 420.0), (1080, 1920), margin=2.0)
    x1, y1, x2, y2 = box
    assert (x1, x2) == (200, 700)  # 2 face-widths (100px) either side
    assert (y1, y2) == (180, 660)  # 1 face-height above, 2 below


def test_palm_search_box_clamps_to_the_frame() -> None:
    x1, y1, x2, y2 = palm_search_box((10.0, 10.0, 60.0, 70.0), (100, 120), margin=2.0)
    assert (x1, y1) == (0, 0)
    assert (x2, y2) == (120, 100)


def test_palm_search_box_beats_the_full_frame_for_a_distant_face() -> None:
    """The point of anchoring: the crop is a small slice of a wide frame."""
    box = palm_search_box((600.0, 300.0, 660.0, 380.0), (720, 1280), margin=2.0)
    assert (box[2] - box[0]) < 1280 / 2


# --- the gate in the pipeline -----------------------------------------------


@needs_model
@needs_photos
async def test_pipeline_stops_at_face_detection_when_nobody_is_looking(
    models: Models, enrolled: Gallery
) -> None:
    """A turned head must cost one SCRFD pass and nothing else.

    ``attendance_EMP1_20260824`` is a real capture of a head turned away; it measures
    a yaw ratio of about -0.76 against the 0.35 gate.
    """
    turned = Path("storage/snapshots/attendance_EMP1_20260824_061507_000109.jpg")
    if not turned.is_file():
        pytest.skip("turned-head fixture absent")

    process = FaceRecognitionProcess(
        FaceProcessConfig(), models, enrolled, settings.models_dir
    )
    result = await process.process_frames(cv2.imread(str(turned)))

    assert result.faces, "the face itself must still be reported"
    assert all(not face.looking for face in result.faces)
    # Palm detection never ran, so its score is untouched, and nothing was embedded.
    assert result.palm.detected is False
    assert result.palm.score == 0.0
    assert all(face.employee_code is None for face in result.faces)


@needs_model
@needs_photos
async def test_pipeline_recognises_a_looking_face_that_shows_a_palm(
    models: Models, enrolled: Gallery
) -> None:
    engaged = Path("storage/snapshots/attendance_EMP1_20260820_064045_346582.jpg")
    if not engaged.is_file():
        pytest.skip("engaged-capture fixture absent")

    process = FaceRecognitionProcess(
        FaceProcessConfig(), models, enrolled, settings.models_dir
    )
    result = await process.process_frames(cv2.imread(str(engaged)))

    assert any(face.looking for face in result.faces)
    assert result.palm.detected is True
    assert {face.employee_code for face in result.faces} == {"EMP1"}


# --- palm attributed to a person, not to the frame ---------------------------


@needs_model
async def test_palm_detect_returns_one_verdict_per_region() -> None:
    """The per-region contract is what lets the pipeline say *whose* hand is up."""
    handler = BlazePalmDetection(model_path=PALM_MODEL, score_threshold=0.5)
    blank = np.zeros((240, 320, 3), np.uint8)

    boxes = [(0, 0, 100, 100), (100, 100, 200, 200), (150, 50, 300, 200)]
    assert len(await handler.detect(blank, boxes)) == len(boxes)
    # No regions means one whole-frame verdict, not zero.
    assert len(await handler.detect(blank)) == 1


@needs_model
async def test_palm_detect_skips_degenerate_regions() -> None:
    handler = BlazePalmDetection(model_path=PALM_MODEL, score_threshold=0.5)
    results = await handler.detect(np.zeros((240, 320, 3), np.uint8), [(50, 50, 50, 50)])
    assert [r.score for r in results] == [0.0]


@needs_model
@needs_photos
async def test_only_the_person_who_raised_a_hand_is_identified(
    models: Models, enrolled: Gallery
) -> None:
    """Two colleagues, one hand: the bystander is seen but never identified.

    ``attendance_EMP4_20260821`` is a real two-person capture. Both faces are looking,
    so both clear the gaze gate; only one has a hand inside its own search box.
    """
    two_people = Path("storage/snapshots/attendance_EMP4_20260821_040930_238580.jpg")
    if not two_people.is_file():
        pytest.skip("two-person fixture absent")

    process = FaceRecognitionProcess(
        FaceProcessConfig(), models, enrolled, settings.models_dir
    )
    result = await process.process_frames(cv2.imread(str(two_people)))

    assert len(result.faces) == 2
    assert all(face.looking for face in result.faces), "both faces face the camera"

    raised = [face for face in result.faces if face.palm]
    bystanders = [face for face in result.faces if not face.palm]
    assert len(raised) == 1 and len(bystanders) == 1

    # The one who waved is identified; the one who did not is never embedded.
    assert raised[0].employee_code == "EMP4"
    assert bystanders[0].employee_code is None
    assert bystanders[0].confidence == 0.0
