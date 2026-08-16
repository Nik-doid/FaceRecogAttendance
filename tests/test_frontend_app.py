"""Smoke tests for the Streamlit debug console (frontend/app.py).

Rendered with ``streamlit.testing.v1.AppTest``. API calls go to whatever server the
app is pointed at; the ``call()`` helper converts failures into ``st.error`` blocks,
so each page is verified to render without raising regardless of server availability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
APP_FILE = str(FRONTEND_DIR / "app.py")
PAGES = [
    "Dashboard",
    "Enrollment",
    "Recognition Logs",
    "Unknown Faces",
    "Metrics",
    "ERP Sync",
    "API Console",
]


def _make_app(page: str = "Dashboard") -> AppTest:
    if str(FRONTEND_DIR) not in sys.path:
        sys.path.insert(0, str(FRONTEND_DIR))
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.session_state["auto_refresh"] = False
    at.session_state["nav"] = page
    return at


def test_dashboard_renders_live_camera_and_activity() -> None:
    at = _make_app("Dashboard")
    at.run()

    assert not at.exception
    headers = [m.value for m in at.markdown]
    assert any("📷 Live camera" in h for h in headers)
    assert any("Recent activity" in h for h in headers)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page: str) -> None:
    at = _make_app(page)
    at.run()

    assert not at.exception
    assert not at.error or any(
        e.value.startswith(("Unauthorized", "Rate limited", "Conflict"))
        or "Connection to API failed" in e.value
        or "HTTP" in e.value
        for e in at.error
    )
