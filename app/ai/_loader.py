"""Lazy import helper for optional heavy AI dependencies.

insightface / onnxruntime / faiss live in the ``ai`` extra. Importing them at module
load time would make the control plane (API + DB + worker plumbing) impossible to
test or deploy on machines that only need the operational half of the service. This
helper defers the import and raises a clear, actionable error instead of a raw
ImportError.
"""

from __future__ import annotations

import importlib
from types import ModuleType

INSTALL_HINT = "Install with: uv sync --extra ai"


def import_optional(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Optional dependency '{module_name}' is not installed. {INSTALL_HINT}"
        ) from exc


def try_import(module_name: str) -> ModuleType | None:
    """Import without raising if missing; used for genuinely optional features."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None
