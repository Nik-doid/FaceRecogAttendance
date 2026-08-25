"""ORM models owned by this service.

Important boundary: this service owns *only* operational tables for recognition.
The attendance system's business tables (attendance records, shifts, employees)
live in the other system and are intentionally not defined here.
"""

from app.models.audit_log import AuditLog
from app.models.camera import Camera
from app.models.face_embedding import FaceEmbedding
from app.models.recognition_log import RecognitionLog
from app.models.setting import Setting
from app.models.unknown_face import UnknownFace

__all__ = [
    "AuditLog",
    "Camera",
    "FaceEmbedding",
    "RecognitionLog",
    "Setting",
    "UnknownFace",
]
