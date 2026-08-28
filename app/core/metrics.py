"""Prometheus collectors, all of which are actually incremented.

The previous set had sixteen, thirteen of them only ever touched by the recognition
worker that no longer exists, plus a gauge that was never set at all. These follow the
one flow the service has: frames in, faces recognised, events published, rows written.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

FRAMES_PROCESSED = Counter(
    "frames_processed_total", "Frames read from the camera"
)
SCANS_COMPLETED = Counter(
    "scans_completed_total", "Frames put through the recognition pipeline"
)
SCAN_SECONDS = Histogram(
    "scan_seconds", "Wall time of one pipeline pass"
)
RECOGNITIONS = Counter(
    "recognitions_total", "Faces matched to an employee", ["employee_code"]
)
ATTENDANCE_PUBLISHED = Counter(
    "attendance_published_total", "Attendance events published to the broker"
)
ATTENDANCE_PUBLISH_FAILED = Counter(
    "attendance_publish_failed_total", "Attendance events the broker would not take"
)
ATTENDANCE_WRITTEN = Counter(
    "attendance_written_total", "Attendance rows written", ["action"]
)
ATTENDANCE_DEAD_LETTERED = Counter(
    "attendance_dead_lettered_total", "Messages parked for an operator", ["reason"]
)
CAMERA_CONNECTED = Gauge(
    "camera_connected", "1 when the camera is delivering frames"
)
GALLERY_SIZE = Gauge(
    "gallery_employees", "Employees currently enrolled"
)
