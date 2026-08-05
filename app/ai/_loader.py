"""Lazy import helper for optional heavy AI dependencies.

insightface / onnxruntime / faiss live in the ``ai`` extra. Importing them at module
load time would make the control plane (API + DB + worker plumbing) impossible to
test or deploy on machines that only need the operational half of the service. This
helper defers the import and raises a clear, actionable error instead of a raw
ImportError.
"""

from __future__ import annotations

import importlib
from pathlib import Path
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


def filter_providers(providers: list[str] | None) -> list[str]:
    """Keep only ONNX execution providers actually available on this machine.

    On a CPU-only box this drops ``CUDAExecutionProvider`` so the session can start
    instead of failing on an unavailable provider.
    """
    if not providers:
        return []
    onnxruntime = try_import("onnxruntime")
    if onnxruntime is None:
        return []
    available = set(onnxruntime.get_available_providers())
    return [p for p in providers if p in available]


def resolve_model_file(model_name: str, models_dir: str | None) -> str:
    """Resolve a model reference to a local ``.onnx`` file path.

    Accepts a plain filename (``det_10g.onnx``, looked up directly under
    ``models_dir``) or a zoo-style name (``scrfd_10g_bnkps``, looked up under
    ``models_dir/models/<name>/``). Raises ``FileNotFoundError`` listing every
    searched location instead of letting insightface silently return ``None``.
    """
    base = Path(models_dir) if models_dir else Path.home() / ".insightface"
    if model_name.lower().endswith(".onnx"):
        candidates = [base / model_name, base / "models" / model_name]
    else:
        candidates = [
            base / "models" / model_name / f"{model_name}.onnx",
            base / f"{model_name}.onnx",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    searched = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        f"Face model '{model_name}' was not found. Looked for:\n{searched}\n"
        "Put the .onnx file there or set MODELS_DIR / DETECT_MODEL / RECOGNIZE_MODEL."
    )
