"""Streamlit debug console for the Face Recognition service.

Run from the project root:
    uv sync --extra frontend
    uv run streamlit run frontend/app.py

The console auto-signs a JWT with the same secret the running service uses (from
``.env``), so every admin operation (camera start/stop, index rebuild) works out of
the box against ``http://localhost:8000``.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from client import ApiClient, ApiError
from jwtgen import create_token, decode_payload, load_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV = load_env()
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VALID_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

PAGES = [
    "Dashboard",
    "Enrollment",
    "Recognition Logs",
    "Unknown Faces",
    "Metrics",
    "API Console",
]
LIVE_PAGES = {"Dashboard", "Recognition Logs", "Unknown Faces", "Metrics"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; max-width: 1280px; margin: 0 auto; }
        div[data-testid="stMetric"] {
            background: #f4f7fb; border: 1px solid #e2e8f0; border-radius: 12px;
            padding: 10px 14px; box-shadow: 0 1px 2px rgb(0 0 0 / 0.04);
        }
        div[data-testid="stMetric"] label { color: #475569; }
        h1, h2, h3 { letter-spacing: -0.01em; }
        [data-testid="stSidebar"] { border-right: 1px solid #e2e8f0; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_ts(value: Any) -> str:
    if not value:
        return "—"
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(value)


def call(client: ApiClient, fn: Callable[[], Any]) -> Any | None:
    try:
        return fn()
    except ApiError as exc:
        if exc.status_code == 401:
            st.error(
                "Unauthorized (401) — the JWT secret must match the server's "
                "JWT_SECRET_KEY, or paste a valid token in the sidebar."
            )
        elif exc.status_code == 429:
            st.error("Rate limited (429) — control endpoints allow 20 requests/min per IP.")
        elif exc.status_code == 409:
            st.info(f"Conflict (409): {exc.body}")
        else:
            st.error(f"{exc.method} {exc.path} → HTTP {exc.status_code}: {exc.body}")
    except requests.exceptions.RequestException as exc:
        st.error(f"Connection to API failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Request failed: {exc}")
    return None


def photo_sources() -> list[Path]:
    raw = (
        os.environ.get("EMPLOYEE_PHOTOS_SOURCE")
        or ENV.get("EMPLOYEE_PHOTOS_SOURCE")
        or "uploads/employees"
    )
    parts = [Path(part.strip()) for part in raw.split(",") if part.strip()]
    return [p if p.is_absolute() else PROJECT_ROOT / p for p in parts]


def _parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(\{[^}]*\})? (\d+(?:\.\d+)?(?:e[+-]?\d+)?)$")
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            name = match.group(1)
            out[name] = out.get(name, 0.0) + float(match.group(3))
    return out


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
def _generate(ss: Any, secret: str, algorithm: str, expire: int) -> None:
    try:
        ss.token = create_token(ss.subject, secret, algorithm, expire)
        ss.payload = decode_payload(ss.token)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not generate token: {exc}")
        ss.token = None
        ss.payload = None


def render_sidebar(ss: Any) -> None:
    st.title("🎥 Face Recognition")
    st.caption("Debug console · /api/v1")

    url = st.text_input("API base URL", value=ss.base_url, key="base_url_input")
    ss.base_url = url.strip().rstrip("/") or ss.base_url

    if ss.client is None or ss.client.base_url != ss.base_url:
        ss.client = ApiClient(ss.base_url, token=ss.token)
    else:
        ss.client.set_token(ss.token)

    if st.button("Test connection", width="stretch"):
        health = call(ss.client, ss.client.health)
        if health:
            st.success(
                f"OK · {health.get('service')} v{health.get('version')} · "
                f"db {health.get('database')} · index {health.get('index_size')}"
            )

    st.divider()
    st.subheader("🔑 JWT")
    subject = st.text_input("Subject", value=ss.subject, key="subject_input")
    ss.subject = subject.strip() or "debug-user"
    with st.expander("JWT settings", expanded=False):
        secret = st.text_input(
            "Secret (JWT_SECRET_KEY)",
            value=ENV.get("JWT_SECRET_KEY") or "",
            type="password",
            key="secret",
        )
        algorithm = st.selectbox("Algorithm", ["HS256"], key="algorithm")
        expire = st.number_input(
            "Expiry (minutes)",
            min_value=1,
            max_value=1440,
            value=int(ENV.get("JWT_EXPIRE_MINUTES") or "60"),
            key="expire",
        )
    secret = secret.strip() or (ENV.get("JWT_SECRET_KEY") or "")

    if ss.token is None and secret:
        _generate(ss, secret, algorithm, int(expire))

    c1, c2 = st.columns(2)
    if c1.button("Regenerate", width="stretch"):
        if secret:
            _generate(ss, secret, algorithm, int(expire))
        else:
            st.warning("No secret configured — paste a token instead.")
    if c2.button("Clear", width="stretch"):
        ss.token = None
        ss.payload = None

    if ss.token:
        st.code(ss.token, language="text")
        payload = ss.payload or {}
        exp = payload.get("exp")
        left = int(exp) - int(time.time()) if exp else None
        if left is not None:
            if left <= 0:
                st.warning("Token expired — regenerate.")
            else:
                st.caption(f"sub={payload.get('sub')} · expires in {left // 60}m {left % 60:02d}s")

    with st.expander("Or paste an external token"):
        pasted = st.text_area("Token", height=110, key="pasted")
        if st.button("Use pasted token", width="stretch"):
            value = pasted.strip()
            ss.token = value or None
            try:
                ss.payload = decode_payload(value) if value else None
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not parse pasted token: {exc}")
                ss.payload = None

    st.divider()
    ss.auto_refresh = st.toggle("Auto-refresh", value=ss.auto_refresh)
    ss["interval"] = st.slider("Refresh interval (s)", 1, 15, 3)
    st.divider()
    st.radio("Page", PAGES, key="nav", label_visibility="collapsed")


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def page_dashboard(client: ApiClient) -> None:
    st.subheader("System health")
    health = call(client, client.health)
    index = call(client, client.index_status)
    cam = call(client, client.camera_status)

    if health:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Service", health.get("status"))
        c2.metric("Database", health.get("database"))
        c3.metric("Camera", health.get("camera"))
        c4.metric("Index size", health.get("index_size"))

    st.divider()
    st.markdown("#### 📷 Live camera")
    running = bool(cam and cam.get("status") == "running")
    if running:
        stream_url = f"{client.base_url}/api/v1/camera/stream"
        st.markdown(
            f'<img src="{stream_url}" style="width:100%;border-radius:10px;'
            'border:1px solid #e2e8f0;display:block;" alt="live camera feed">',
            unsafe_allow_html=True,
        )
        st.caption("Live feed · recognized faces appear under Recent activity below.")
    else:
        st.info("Camera is stopped — start it below to see the live feed.")

    st.divider()
    st.markdown("#### Recent activity")
    recent_logs = call(client, lambda: client.recognition_logs(limit=8))
    recent_unknown = call(client, lambda: client.unknown_faces(limit=5))
    left, right = st.columns(2)
    with left:
        st.markdown("**Recognized**")
        if recent_logs and recent_logs.get("items"):
            df = pd.DataFrame(recent_logs["items"])
            df["time"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")
            df["conf"] = (df["confidence"].astype(float) * 100).round(1).astype(str) + " %"
            st.dataframe(
                df[["time", "employee_code", "conf", "reported"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No recognitions yet.")
    with right:
        st.markdown("**Unknown**")
        if recent_unknown and recent_unknown.get("items"):
            df = pd.DataFrame(recent_unknown["items"])
            df["time"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")
            st.dataframe(df[["time", "track_id"]], width="stretch", hide_index=True)
        else:
            st.caption("No unknown faces yet.")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Camera")
        if cam:
            running = cam.get("status") == "running"
            st.metric("State", cam.get("status"))
            st.caption(f"camera_id: {cam.get('camera_id')}")
            if cam.get("last_connected_at"):
                st.caption(f"last connected: {fmt_ts(cam.get('last_connected_at'))}")
            if cam.get("last_error"):
                st.warning(f"last error: {cam.get('last_error')}")
            label = "⏹ Stop camera" if running else "▶ Start camera"
            if st.button(label, key="cam_toggle", width="stretch"):
                result = call(client, client.camera_stop if running else client.camera_start)
                if result:
                    st.success(f"Camera {result.get('action')} → {result.get('status')}")
                    time.sleep(0.3)
                    st.rerun()
        else:
            st.info("Camera status unavailable.")
    with right:
        st.markdown("#### Face index")
        if index:
            building = index.get("status") == "building"
            st.metric("Status", index.get("status"))
            c1, c2 = st.columns(2)
            c1.metric("Embeddings", index.get("size"))
            c2.metric("Employees", index.get("employees"))
            st.caption(f"last built: {fmt_ts(index.get('last_built_at'))}")
            if index.get("last_error"):
                st.error(f"last error: {index.get('last_error')}")
            if st.button("🔄 Rebuild index", key="rebuild", width="stretch", disabled=building):
                result = call(client, client.index_rebuild)
                if result:
                    st.success(result.get("message", "rebuild started"))
                    time.sleep(0.3)
                    st.rerun()
        else:
            st.info("Index status unavailable.")

    st.session_state["_index_status"] = index


def _save_enrollment(code: str, upload: Any) -> Path:
    sources = photo_sources()
    target = next((p for p in sources if p.is_dir()), sources[0])
    target.mkdir(parents=True, exist_ok=True)
    employee_dir = target / code
    employee_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(upload.name).suffix.lower() or ".jpg"
    dest = employee_dir / f"{code}_{int(time.time())}{ext}"
    dest.write_bytes(upload.getvalue())
    return dest


def _enrolled_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in photo_sources():
        if not root.is_dir():
            continue
        for employee_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            images = [p for p in employee_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
            rows.append(
                {
                    "employee_code": employee_dir.name,
                    "photos": len(images),
                    "directory": str(employee_dir),
                }
            )
    return rows


def _employee_previews(rows: list[dict[str, Any]]) -> None:
    cells = []
    for row in rows:
        root = Path(row["directory"])
        images = sorted(p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS)
        if images:
            cells.append((row["employee_code"], images[0]))
    if not cells:
        return
    st.markdown("##### Photo previews")
    for start in range(0, len(cells), 5):
        cols = st.columns(min(5, len(cells) - start))
        for i, (code, image) in enumerate(cells[start : start + 5]):
            with cols[i]:
                st.image(str(image), caption=code, width=140)


def page_enrollment(client: ApiClient) -> None:
    st.subheader("Enroll an employee")
    info = call(client, client.build_info)
    if info and info.get("detail"):
        st.caption("Server reads photos from: " + str(info.get("detail")))

    with st.form("enroll_form"):
        code = st.text_input(
            "Employee code",
            help="Letters, digits, '.', '_', '-' (e.g. EMP1023)",
        )
        uploaded = st.file_uploader("Face photo", type=["jpg", "jpeg", "png", "bmp"])
        submitted = st.form_submit_button("Save enrollment photo", type="primary", width="stretch")
    if submitted:
        if not code or not VALID_CODE.match(code):
            st.error("Enter a valid employee code.")
        elif uploaded is None:
            st.error("Upload a face photo first.")
        else:
            dest = _save_enrollment(code.strip(), uploaded)
            st.success(f"Saved → {dest}")
            st.info("Now click 'Rebuild index now' below so the face can be matched.")

    st.divider()
    st.subheader("Currently enrolled")
    rows = _enrolled_rows()
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        _employee_previews(rows)
    else:
        st.info("No employees enrolled yet — upload the first photo above.")

    if st.button("🔄 Rebuild index now", type="primary", width="stretch"):
        result = call(client, client.index_rebuild)
        if result:
            st.success(result.get("message", "rebuild started"))
            time.sleep(0.3)
            st.rerun()
    st.caption("Tip: after rebuilding, watch the Dashboard — the embedding count grows.")


def _preview_picker(df: pd.DataFrame, key_prefix: str) -> None:
    cells: list[tuple[Any, str]] = []
    for row in df.to_dict("records"):
        snap = row.get("snapshot_path")
        if isinstance(snap, str) and Path(snap).is_file():
            cells.append((row.get("id"), snap))
    if not cells:
        st.caption("No snapshots readable on this machine (paths come from the server).")
        return
    labels = [f"#{record_id} — {Path(snap).name}" for record_id, snap in cells]
    choice = st.selectbox(
        "Preview snapshot",
        list(range(len(cells))),
        format_func=lambda i: labels[i],
        key=f"{key_prefix}_pick",
    )
    record_id, snap = cells[choice]
    st.image(str(snap), caption=f"Snapshot for record #{record_id}", width=320)


def page_logs(client: ApiClient) -> None:
    st.subheader("Recognition logs")
    c1, c2 = st.columns([1, 2])
    limit = int(c1.number_input("Limit", 10, 500, 50, 10))
    code = c2.text_input("Filter by employee code (empty = all)")
    data = call(client, lambda: client.recognition_logs(limit, code.strip() or None))
    if data is None:
        return
    items = data.get("items") or []
    st.caption(f"{data.get('total', len(items))} log entries · showing {len(items)}")
    if not items:
        st.info("No recognition logs yet. Start the camera and look into it.")
        return
    df = pd.DataFrame(items)
    df["confidence"] = (df["confidence"].astype(float) * 100).round(1).astype(str) + " %"
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    columns = [
        "id",
        "employee_code",
        "timestamp",
        "confidence",
        "reported",
        "attendance_response",
        "track_id",
    ]
    st.dataframe(df[columns], width="stretch", hide_index=True)
    st.divider()
    _preview_picker(df, "logs")


def page_unknown(client: ApiClient) -> None:
    st.subheader("Unknown faces")
    limit = int(st.number_input("Limit", 10, 500, 50, 10))
    data = call(client, lambda: client.unknown_faces(limit))
    if data is None:
        return
    items = data.get("items") or []
    st.caption(f"{data.get('total', len(items))} unknown faces · showing {len(items)}")
    if not items:
        st.info("No unknown faces recorded yet.")
        return
    df = pd.DataFrame(items)
    df["confidence_of_best_nonmatch"] = (
        (df["confidence_of_best_nonmatch"].astype(float) * 100).round(1).astype(str) + " %"
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    columns = ["id", "timestamp", "confidence_of_best_nonmatch", "track_id"]
    st.dataframe(df[columns], width="stretch", hide_index=True)
    st.divider()
    _preview_picker(df, "unknown")


def page_metrics(client: ApiClient) -> None:
    st.subheader("Prometheus metrics")
    text = call(client, client.metrics)
    if text is None:
        return
    parsed = _parse_metrics(text)
    key_metrics = {
        "face_frames_processed_total": "Frames processed",
        "face_faces_detected_total": "Faces detected",
        "face_recognitions_total": "Recognitions",
        "face_unknown_faces_total": "Unknown faces",
        "face_attendance_reports_total": "Reports published",
        "face_attendance_reports_failed_total": "Reports failed",
        "face_quality_rejected_total": "Quality rejects",
        "face_liveness_failed_total": "Liveness rejects",
        "face_camera_reconnects_total": "Camera reconnects",
        "face_index_size": "Index size",
        "face_camera_connected": "Camera connected",
    }
    cols = st.columns(3)
    for i, (name, label) in enumerate(key_metrics.items()):
        with cols[i % 3]:
            st.metric(label, parsed.get(name, 0))
    with st.expander("Raw exposition text", expanded=False):
        st.code(text, language="text")


def page_console(client: ApiClient) -> None:
    st.subheader("API console")
    st.caption("Requests are sent to {base}/api/v1{path}. Handy for poking endpoints without curl.")
    c1, c2 = st.columns([1, 2])
    method = c1.selectbox("Method", ["GET", "POST", "PUT", "DELETE", "PATCH"])
    path = c2.text_input("Path", value="/index/status")
    include_auth = st.checkbox("Send bearer token", value=bool(client.token))
    body_text = st.text_area("JSON body (optional)", height=130, placeholder='{"example": true}')
    if st.button("Send request", type="primary", width="stretch"):
        body = None
        if body_text.strip():
            try:
                body = json.loads(body_text)
            except json.JSONDecodeError:
                st.error("Body is not valid JSON.")
                return
        def _send() -> Any:
            return client.request(method, path, json_body=body, auth=include_auth)

        response = call(client, _send)
        if response is not None:
            st.success("HTTP 200")
            if isinstance(response, (dict, list)):
                st.json(response)
            else:
                st.code(str(response), language="text")


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Face Recognition · Debug Console",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    ss = st.session_state
    ss.setdefault("base_url", "http://localhost:8000")
    ss.setdefault("subject", "debug-user")
    ss.setdefault("token", None)
    ss.setdefault("payload", None)
    ss.setdefault("client", None)
    ss.setdefault("auto_refresh", True)

    with st.sidebar:
        render_sidebar(ss)

    client: ApiClient = ss.client
    page = ss.get("nav", "Dashboard")

    if page == "Dashboard":
        page_dashboard(client)
    elif page == "Enrollment":
        page_enrollment(client)
    elif page == "Recognition Logs":
        page_logs(client)
    elif page == "Unknown Faces":
        page_unknown(client)
    elif page == "Metrics":
        page_metrics(client)
    else:
        page_console(client)

    if page in LIVE_PAGES and ss.auto_refresh:
        interval = int(ss.get("interval", 3))
        if page == "Dashboard":
            index = ss.get("_index_status") or {}
            if index.get("status") == "building":
                interval = 1
        time.sleep(interval)
        st.rerun()


main()
