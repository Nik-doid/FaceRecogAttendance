"""Errors raised while processing a frame.

Mirrors the local convention of subclassing a builtin (``CameraAlreadyRunningError``
in ``app/services/camera_service.py``) rather than introducing an app-wide exception
base this repo does not have.
"""

from __future__ import annotations


class FaceRecognitionError(RuntimeError): ...


class StepNotImplementedError(FaceRecognitionError): ...


class PalmDetectionError(FaceRecognitionError): ...
