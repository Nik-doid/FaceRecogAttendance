"""Prometheus metrics registry for the service.

Exposed by GET /metrics in text exposition format. Kept as a module singleton so
both the API layer and the background worker report into the same registry.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

FRAMES_PROCESSED = Counter("face_frames_processed_total", "Frames fed through the pipeline")
FRAMES_SKIPPED = Counter("face_frames_skipped_total", "Frames skipped by frame-skip policy")
FACES_DETECTED = Counter("face_faces_detected_total", "Faces detected by SCRFD")
RECOGNITIONS = Counter(
    "face_recognitions_total",
    "Recognition events by employee",
    labelnames=["employee_code"],
)
UNKNOWN_FACES = Counter("face_unknown_faces_total", "Unrecognized faces seen")
REPORTS_PUBLISHED = Counter(
    "face_attendance_reports_total", "Attendance events published to MQ"
)
REPORTS_FAILED = Counter(
    "face_attendance_reports_failed_total", "Attendance events that failed to publish"
)
QUALITY_REJECTED = Counter(
    "face_quality_rejected_total",
    "Frames/faces rejected by quality gates",
    labelnames=["reason"],
)
LIVENESS_FAILED = Counter("face_liveness_failed_total", "Faces rejected by anti-spoofing")
CAMERA_RECONNECTS = Counter("face_camera_reconnects_total", "RTSP reconnection attempts")
PROCESSING_TIME = Histogram(
    "face_processing_seconds",
    "Time to process a single frame end-to-end",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
INDEX_SIZE = Gauge("face_index_size", "Number of embeddings currently in the FAISS index")
CAMERA_CONNECTED = Gauge("face_camera_connected", "1 if the camera worker is connected, else 0")
