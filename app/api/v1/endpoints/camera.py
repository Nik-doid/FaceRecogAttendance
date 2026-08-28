"""Camera runner control and preview.

These drive the always-on :class:`CameraRunner`, not a WebSocket connection: starting
the camera here makes it capture and recognise until it is stopped, whether or not
anyone has a browser open.

``status``, ``frame`` and ``stream`` are unauthenticated because they are read-only
previews of an already-running camera. ``start`` and ``stop`` require a token: being
able to silently stop attendance capture from anywhere on the network is not a
read-only operation. Mint one with::

    uv run python -c "from app.core.security import create_access_token as t; print(t('admin'))"
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response, StreamingResponse

from app.api.deps import AdminDep, ContainerDep
from app.camera.runner import CameraRunnerAlreadyRunningError
from app.core.security import rate_limit

router = APIRouter(tags=["camera"])

STREAM_INTERVAL = 0.05


@router.get("/camera/status")
async def camera_status(container: ContainerDep) -> dict[str, Any]:
    """Runner state, gallery readiness, and the last error if there was one."""
    runner = container.camera_runner
    gallery = container.gallery
    return {
        "camera_id": container.settings.camera_id,
        "status": "running" if runner.running else "stopped",
        "camera": runner.state.as_dict(),
        "gallery": {
            "ready": container.gallery_handle.ready.is_set(),
            "employees": gallery.employees,
            "photos": gallery.photos,
        },
    }


@router.get("/camera/frame")
async def camera_frame(container: ContainerDep) -> Response:
    """Latest camera frame as JPEG, for polling previews and for quick testing."""
    jpeg = container.frame_hub.snapshot().jpeg
    if jpeg is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no camera frame available -- start the camera first",
        )
    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/camera/stream")
async def camera_stream(container: ContainerDep) -> StreamingResponse:
    """Motion JPEG stream for live previews (multipart/x-mixed-replace)."""

    async def _frames() -> AsyncIterator[bytes]:
        last_seq = -1
        while True:
            snapshot = container.frame_hub.snapshot()
            if snapshot.jpeg is not None and snapshot.frame_seq != last_seq:
                last_seq = snapshot.frame_seq
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + snapshot.jpeg + b"\r\n"
            await asyncio.sleep(STREAM_INTERVAL)

    return StreamingResponse(
        _frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/camera/start", dependencies=[Depends(rate_limit())])
async def camera_start(actor: AdminDep, container: ContainerDep) -> dict[str, Any]:
    try:
        state = container.camera_runner.start()
    except CameraRunnerAlreadyRunningError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return {
        "camera_id": container.settings.camera_id,
        "action": "start",
        "status": "running" if state.running else "error",
    }


@router.post("/camera/stop", dependencies=[Depends(rate_limit())])
async def camera_stop(actor: AdminDep, container: ContainerDep) -> dict[str, Any]:
    state = container.camera_runner.stop()
    return {
        "camera_id": container.settings.camera_id,
        "action": "stop",
        "status": "stopped" if not state.running else "error",
    }
