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


def model_candidates(model_name: str, models_dir: str | None) -> list[Path]:
    """Every location a model reference may legitimately live in, in priority order.

    Accepts a plain filename (``det_10g.onnx``, looked up directly under
    ``models_dir``) or a zoo-style name (``scrfd_10g_bnkps``, looked up under
    ``models_dir/models/<name>/``).
    """
    base = Path(models_dir) if models_dir else Path.home() / ".insightface"
    if model_name.lower().endswith(".onnx"):
        return [base / model_name, base / "models" / model_name]
    return [
        base / "models" / model_name / f"{model_name}.onnx",
        base / f"{model_name}.onnx",
    ]


def find_model_file(model_name: str, models_dir: str | None) -> Path | None:
    """Locate a model file, or ``None`` when it is absent.

    Split out of ``resolve_model_file`` so the auto-downloader decides "is this
    missing?" by exactly the rule the loader resolves by -- otherwise models placed
    in the secondary candidate location would be re-downloaded on every boot.
    """
    for candidate in model_candidates(model_name, models_dir):
        if candidate.is_file():
            return candidate
    return None


def resolve_model_file(model_name: str, models_dir: str | None) -> str:
    """Resolve a model reference to a local ``.onnx`` file path.

    Raises ``FileNotFoundError`` listing every searched location instead of letting
    insightface silently return ``None``.
    """
    found = find_model_file(model_name, models_dir)
    if found is not None:
        return str(found)
    searched = "\n".join(f"  - {c}" for c in model_candidates(model_name, models_dir))
    raise FileNotFoundError(
        f"Face model '{model_name}' was not found. Looked for:\n{searched}\n"
        "Put the .onnx file there or set MODELS_DIR / DETECT_MODEL / RECOGNIZE_MODEL."
    )
