"""One-time fetch of the model files this service needs into ``MODELS_DIR``.

The README has always promised that the SCRFD/ArcFace models "auto-download into
``./models`` on first run"; nothing implemented it, so a fresh checkout died during
startup with a bare ``FileNotFoundError``. This module makes good on that promise.

Everything here is stdlib (``urllib`` + ``zipfile``) on purpose. One of the required
files feeds ``mediapipe``, which lives in the *base* dependencies, so fetching must
work in the control-plane-only install where ``insightface``/``onnxruntime`` are
absent -- and adding an HTTP client to the base deps just to download a file would be
a poor trade.

Downloads are idempotent (skipped when the file already resolves), atomic (streamed
to a ``.part`` sibling and renamed only on success) and best-effort: a failure is
logged and swallowed so the caller still raises the existing "put the file here"
error rather than replacing it with a network traceback.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from app.ai._loader import find_model_file
from app.config.settings import Settings
from app.core.logging import get_logger

_log = get_logger(__name__)

# The insightface "buffalo_l" pack. Only two of its five models are used here (the
# SCRFD detector and the ArcFace recognizer), but the release publishes no per-file
# asset, so the whole archive is fetched once and the wanted members extracted
# *flat* -- ``model_candidates`` looks under MODELS_DIR directly, never in the
# ``buffalo_l/`` subdirectory the archive would otherwise imply.
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
BUFFALO_L_MEMBERS = frozenset({"det_10g.onnx", "w600k_r50.onnx"})

# MediaPipe hand landmarker. Unlike the insightface models this is a direct file
# download; the URL matches the one HandDetector quotes in its error message.

# BlazePalm palm detector (OpenCV Zoo, Apache 2.0), run through ``cv2.dnn`` -- so it
# needs no dependency the base install lacks. Used for palm *presence* only, which is
# an order of magnitude cheaper than the hand landmarker above.
PALM_DETECTOR_NAME = "palm_detection_mediapipe_2023feb.onnx"
PALM_DETECTOR_URL = (
    "https://huggingface.co/opencv/palm_detection_mediapipe/resolve/main/"
    "palm_detection_mediapipe_2023feb.onnx"
)

_CHUNK = 1 << 20  # 1 MiB
_TIMEOUT_SECONDS = 60
_MIB = 1024 * 1024


def ensure_models(settings: Settings) -> None:
    """Download whatever model files are missing from ``settings.models_dir``.

    A no-op once everything is present, so it is safe on the hot startup path. Only
    models this module has a known source for are fetched: a custom ``DETECT_MODEL``
    is never guessed at, it just falls through to ``resolve_model_file``'s error.

    SilentFace is deliberately not handled -- there is no canonical public export,
    and ``load_ai_components`` already fails fast with instructions when liveness is
    enabled without ``SILENTFACE_MODEL_PATH``.
    """
    if not settings.models_auto_download:
        return

    models_dir = Path(settings.models_dir)
    dir_str = str(models_dir)

    wanted = sorted(
        name
        for name in {settings.detect_model, settings.recognize_model}
        if name in BUFFALO_L_MEMBERS and find_model_file(name, dir_str) is None
    )
    palm_missing = not (models_dir / PALM_DETECTOR_NAME).is_file()
    if not wanted and not palm_missing:
        return

    models_dir.mkdir(parents=True, exist_ok=True)
    if wanted:
        _fetch_buffalo_members(wanted, models_dir)
    if palm_missing:
        _fetch(PALM_DETECTOR_URL, models_dir / PALM_DETECTOR_NAME)


def _fetch_buffalo_members(members: list[str], models_dir: Path) -> None:
    """Fetch buffalo_l.zip once and extract only the members actually loaded."""
    _log.info("Fetching insightface buffalo_l pack for: %s", ", ".join(members))
    # Staged inside models_dir so extraction is a same-filesystem copy, and so the
    # ~275 MB archive is removed even if extraction raises.
    with tempfile.TemporaryDirectory(dir=models_dir) as tmp:
        archive = Path(tmp) / "buffalo_l.zip"
        if not _fetch(BUFFALO_L_URL, archive):
            return
        try:
            with zipfile.ZipFile(archive) as zf:
                # The archive nests members under a directory; match on basename.
                available = {Path(name).name: name for name in zf.namelist()}
                absent = [m for m in members if m not in available]
                if absent:
                    _log.error(
                        "buffalo_l.zip is missing %s (contains: %s)",
                        ", ".join(absent),
                        ", ".join(sorted(available)),
                    )
                    return
                for member in members:
                    dest = models_dir / member
                    with zf.open(available[member]) as src, _atomic(dest) as out:
                        shutil.copyfileobj(src, out, _CHUNK)
                    _log.info("Extracted %s (%.1f MiB)", dest, dest.stat().st_size / _MIB)
        except (zipfile.BadZipFile, OSError):
            _log.exception("Could not unpack buffalo_l.zip")


def _fetch(url: str, dest: Path) -> bool:
    """Stream ``url`` into ``dest``. Logs and returns ``False`` on any failure."""
    _log.info("Downloading %s -> %s", url, dest)
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            next_mark = 10
            with _atomic(dest) as out:
                while chunk := response.read(_CHUNK):
                    out.write(chunk)
                    done += len(chunk)
                    # A cold start pulling 275 MB in silence is indistinguishable
                    # from a hang, so report roughly every 10%.
                    percent = done * 100 // total if total else 0
                    if percent >= next_mark:
                        _log.info(
                            "  %s: %d%% of %.1f MiB", dest.name, percent, total / _MIB
                        )
                        next_mark = percent - percent % 10 + 10
    except (urllib.error.URLError, OSError, TimeoutError):
        _log.exception(
            "Failed to download %s. Fetch it manually into %s, or set "
            "MODELS_AUTO_DOWNLOAD=false if this host has no internet access.",
            url,
            dest.parent,
        )
        return False
    _log.info("Downloaded %s (%.1f MiB)", dest, dest.stat().st_size / _MIB)
    return True


@contextmanager
def _atomic(dest: Path) -> Iterator[IO[bytes]]:
    """Write through a ``.part`` sibling so a partial file never looks complete.

    Without this an interrupted download leaves a truncated ``.onnx`` in place, which
    subsequently *resolves* and fails much later inside onnxruntime.
    """
    part = dest.with_name(dest.name + ".part")
    try:
        with part.open("wb") as handle:
            yield handle
        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def main() -> int:
    """Prefetch models without booting the service: ``python -m app.ai._download``."""
    settings = Settings()
    ensure_models(settings)

    dir_str = str(settings.models_dir)
    missing = [
        name
        for name in (settings.detect_model, settings.recognize_model)
        if find_model_file(name, dir_str) is None
    ]
    if missing:
        _log.error("Still missing from %s: %s", settings.models_dir, ", ".join(missing))
        return 1
    _log.info("All required models present in %s", settings.models_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
