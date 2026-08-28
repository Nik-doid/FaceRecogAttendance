"""Enrolled-employee gallery: what is loaded, and reloading it.

Replaces the old /index endpoints. There is no database behind this now -- the
gallery is enumerated from EMPLOYEE_PHOTOS_SOURCE (local directories and HTTPS
folders) and cached on disk by photo identity, so a reload after adding a photo
re-embeds only that photo.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import AdminDep, ContainerDep
from app.core.logging import get_logger
from app.core.security import rate_limit

router = APIRouter(tags=["gallery"])
log = get_logger(__name__)

_reloading = threading.Lock()


@router.get("/gallery/status")
async def gallery_status(container: ContainerDep) -> dict[str, Any]:
    gallery = container.gallery
    return {
        "ready": container.gallery_handle.ready.is_set(),
        "employees": gallery.employees,
        "photos": gallery.photos,
        "vectors": gallery.index.size,
        "sources": list(container.settings.employee_photos_source),
    }


@router.post("/gallery/reload", dependencies=[Depends(rate_limit())])
async def gallery_reload(actor: AdminDep, container: ContainerDep) -> dict[str, Any]:
    """Re-enumerate the photo sources and swap the index in.

    Runs in the background: a cold build over hundreds of photos takes minutes, and
    recognition keeps serving the previous gallery until the new one is ready.
    """
    if not _reloading.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="a gallery reload is already running"
        )

    def run() -> None:
        try:
            container.build_gallery_now()
        except Exception:  # noqa: BLE001 - the thread must never die silently
            log.exception("gallery reload failed")
        finally:
            _reloading.release()

    threading.Thread(target=run, name="gallery-reload", daemon=True).start()
    return {"status": "reloading"}
