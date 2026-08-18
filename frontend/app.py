"""Streamlit dashboard for the Face Recognition service.

Run from the project root:
    uv sync --extra frontend
    uv run streamlit run frontend/app.py

The dashboard auto-signs a JWT with the same secret the running service uses (from
``.env``), so every admin operation (camera start/stop, index rebuild, ERP sync)
works out of the box against ``http://localhost:8000``. Status is fetched once per
page render — hit a button to re-check — there is no background polling loop.
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
    "ERP Sync",
    "API Console",
]


# ---------------------------------------------------------------------------
# styling
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 3rem;
            max-width: 1400px; margin: 0 auto; }
        [data-testid="stAppViewContainer"] { background: #f5f7fb; }
        [data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
        [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label { color: #e2e8f0; }
        [data-testid="stSidebar"] [data-baseweb="input"] input { color: #e2e8f0; }
        [data-testid="stSidebar"] .stCodeBlock pre { color: #cbd5e1; }
        [data-testid="stSidebar"] .stCodeBlock { background: #1e293b; }

        h1, h2, h3 { color: #0f172a; letter-spacing: -0.01em; }
        .hero {
            background: linear-gradient(135deg, #1e3a8a 0%, #4338ca 100%);
            color: #ffffff; border-radius: 16px; padding: 20px 24px; margin-bottom: 1.1rem;
            box-shadow: 0 6px 20px rgb(30 58 138 / 0.25);
        }
        .hero h1 { color: #ffffff; margin: 0 0 4px 0; font-size: 1.7rem; }
        .hero p { margin: 0; color: #dbeafe; font-size: 0.95rem; }

        div[data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
            padding: 12px 16px; box-shadow: 0 1px 3px rgb(0 0 0 / 0.06);
            height: 100%;
        }
        div[data-testid="stMetric"] label { color: #64748b; font-weight: 600; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.6rem; font-weight: 700; color: #0f172a;
        }

        .card {
            background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
            padding: 16px 18px; box-shadow: 0 1px 3px rgb(0 0 0 / 0.06);
            margin-bottom: 1rem;
        }
        .card h4 { margin: 0 0 10px 0; color: #0f172a; font-size: 1.05rem; }
        .feed-wrap {
            background: #000; border-radius: 14px; overflow: hidden;
            border: 1px solid #cbd5e1; box-shadow: 0 4px 16px rgb(0 0 0 / 0.18);
        }
        .feed-wrap img { width: 100%; display: block; }
        .section-title { margin-top: 1.4rem; margin-bottom: 0.4rem; font-weight: 700;
            color: #0f172a; }
        [data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 10px;
            overflow: hidden; }
        .stButton > button { border-radius: 10px; font-weight: 600; }
        .stButton > button[kind="primary"] { background: #2563eb; border: 1px solid #2563eb; }
        footer { visibility: hidden; }
        @media (max-width: 768px) {
            [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, tone: str = "info") -> str:
    """Inline HTML badge for status values."""
    colors = {
        "ok": "background:#dcfce7;color:#166534;",
        "warn": "background:#fef3c7;color:#92400e;",
        "err": "background:#fee2e2;color:#991b1b;",
        "info": "background:#dbeafe;color:#1e40af;",
        "muted": "background:#f1f5f9;color:#475569;",
    }
    style = colors.get(tone, colors["info"])
    return (
        f'<span style="{style}padding:3px 12px;border-radius:999px;'
        f'font-size:0.85rem;font-weight:600;display:inline-block;">{text}</span>'
    )


def status_tone(value: str | None) -> str:
    if not value:
        return "muted"
    if value in ("running", "ok", "up", "idle"):
        return "ok"
    if value in ("building", "degraded"):
        return "warn"
    return "err"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
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


def action_feedback(result: Any | None, ok_message: str) -> None:
    """Render the standard outcome of a button-triggered admin action."""
    if result is None:
        return
    if isinstance(result, dict) and result.get("ok") is False:
        st.error(result.get("detail", "action failed"))
    else:
        st.success(ok_message)


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
    st.markdown(
        '<div style="background:#1e293b;border-radius:12px;padding:14px 16px;'
        'margin-bottom:6px;">'
        '<div style="font-size:1.25rem;font-weight:700;color:#ffffff;">🎥 Face Recognition</div>'
        '<div style="color:#94a3b8;font-size:0.85rem;">attendance service dashboard</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    url = st.text_input("API base URL", value=ss.base_url, key="base_url_input")
    ss.base_url = url.strip().rstrip("/") or ss.base_url

    if ss.client is None or ss.client.base_url != ss.base_url:
        ss.client = ApiClient(ss.base_url, token=ss.token)
    else:
        ss.client.set_token(ss.token)

    if st.button("🔌 Test connection", width="stretch"):
        health = call(ss.client, ss.client.health)
        if health:
            st.success(
                f"OK · {health.get('service')} v{health.get('version')} · "
                f"db {health.get('database')} · index {health.get('index_size')}"
            )

    st.divider()
    st.markdown("##### 🔑 Admin JWT")
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
    st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def refresh_button() -> None:
    """Single manual refresh — the dashboard never polls on its own."""
    if st.button("🔄 Refresh", key="refresh_once", type="secondary"):
        st.rerun()


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def page_dashboard(client: ApiClient) -> None:
    hero("Dashboard", "Service health, live camera and quick controls.")
    refresh_button()

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
    # Determine camera running status
    cam_status = cam.get("status") if cam else None
    running = cam_status == "running" if cam_status else False
    if running:
        stream_url = f"{client.base_url}/api/v1/camera/stream"
        st.markdown(
            f'<div class="feed-wrap"><img src="{stream_url}" alt="live camera feed"></div>',
            unsafe_allow_html=True,
        )
        # Show engagement/Wave status box below the video
        st.caption(
            "Detected faces are boxed live — green = recognized employee ✓ (looking + waved), "
            "orange = engagement in progress, red = unknown, gray = low quality. "
            "Employee name appears only after 2-second gaze + wave gesture."
        )
    else:
        st.info("Camera is stopped — use Start camera below to see the live feed.")

    st.markdown('<div class="section-title">Quick controls</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card"><h4>🎥 Camera</h4>', unsafe_allow_html=True)
        if cam:
            cam_status_val = cam.get("status") or "—"
            st.markdown(
                f"{pill(cam_status_val, status_tone(cam_status_val))} "
                f"<span style='color:#64748b'>· {cam.get('camera_id')}</span>",
                unsafe_allow_html=True,
            )
            if cam.get("last_connected_at"):
                st.caption(f"last connected: {fmt_ts(cam.get('last_connected_at'))}")
            if cam.get("last_error"):
                st.warning(f"last error: {cam.get('last_error')}")
            # Use the computed running state for the button label
            label = "⏹ Stop camera" if running else "▶ Start camera"
            if st.button(label, key="cam_toggle", width="stretch"):
                result = call(client, client.camera_stop if running else client.camera_start)
                action_feedback(
                    result,
                    f"Camera {result.get('action')} → {result.get('status')}" if result else "",
                )
                st.rerun()
        else:
            st.info("Camera status unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><h4>🧠 Face index</h4>', unsafe_allow_html=True)
        if index:
            building = index.get("status") == "building"
            st.markdown(
                f"{pill(index.get('status', '—'), status_tone(index.get('status')))} "
                f"<span style='color:#64748b'>· {index.get('size')} embeddings "
                f"· {index.get('employees')} employees</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"last built: {fmt_ts(index.get('last_built_at'))}")
            if index.get("last_error"):
                st.error(f"last error: {index.get('last_error')}")
            if st.button("🔄 Rebuild index", key="rebuild", width="stretch", disabled=building):
                result = call(client, client.index_rebuild)
                action_feedback(result, result.get("message", "rebuild started") if result else "")
                st.rerun()
        else:
            st.info("Index status unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Recent activity</div>', unsafe_allow_html=True)
    recent_logs = call(client, lambda: client.recognition_logs(limit=8))
    recent_unknown = call(client, lambda: client.unknown_faces(limit=5))
    left, right = st.columns(2)
    with left:
        st.markdown("**Recognized**")
        if recent_logs and recent_logs.get("items"):
            df = pd.DataFrame(recent_logs["items"])
            df["time"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")
            df["conf"] = (df["confidence"].astype(float) * 100).round(1).astype(str) + " %"
            # Show engagement status column if available
            if "attendance_response" in df.columns:
                st.dataframe(
                    df[["time", "employee_code", "conf", "attendance_response"]],
                    width="stretch",
                    hide_index=True,
                )
            else:
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

    # Engagement summary box
    st.markdown("---")
    st.markdown("#### 👁️ Engagement overview")
    if recent_logs and recent_logs.get("items"):
        df = pd.DataFrame(recent_logs["items"])
        # Count engaged vs not-engaged
        engaged_count = 0
        not_engaged_count = 0
        if "attendance_response" in df.columns:
            for resp in df["attendance_response"]:
                if isinstance(resp, str) and "attendance captured" in resp.lower():
                    engaged_count += 1
                else:
                    not_engaged_count += 1
        else:
            # If no attendance_response, check employee_code presence
            for _, row in df.iterrows():
                if row.get("employee_code"):
                    engaged_count += 1
                else:
                    not_engaged_count += 1
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Engaged + waved ✓", engaged_count)
        with col2:
            st.metric("Looking but no wave", not_engaged_count)
    else:
        st.caption("Start the camera to see engagement stats.")


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
    hero("Enrollment", "Add employee photos so the face index can match them.")
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
        action_feedback(result, result.get("message", "rebuild started") if result else "")
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
    hero("Recognition Logs", "Every recognition decision recorded by the worker.")
    refresh_button()
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
    hero("Unknown Faces", "People the index could not match.")
    refresh_button()
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
    hero("Metrics", "Prometheus counters and gauges from /metrics.")
    refresh_button()
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
        "face_erp_sync_inserted_total": "ERP rows inserted",
        "face_erp_sync_failed_total": "ERP rows failed",
        "face_erp_sync_pending": "ERP pending",
    }
    cols = st.columns(3)
    for i, (name, label) in enumerate(key_metrics.items()):
        with cols[i % 3]:
            st.metric(label, parsed.get(name, 0))
    with st.expander("Raw exposition text", expanded=False):
        st.code(text, language="text")


def page_erp_sync(client: ApiClient) -> None:
    hero("ERP Sync", "Push recognized attendance events into the ERP attendance log.")
    refresh_button()

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card"><h4>📡 Sync status</h4>', unsafe_allow_html=True)
        status = call(client, client.erp_sync_status)
        if status:
            enabled_tone = "ok" if status.get("enabled") else "muted"
            enabled_label = "enabled" if status.get("enabled") else "disabled"
            st.markdown(f"{pill(enabled_label, enabled_tone)}", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Pending events", status.get("pending"))
            c2.metric("Interval (s)", status.get("interval_seconds") or "—")
            st.caption(status.get("message") or "")
        else:
            st.info("Sync status unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><h4>▶ Run sync now</h4>', unsafe_allow_html=True)
        st.caption("POST /api/v1/sync/attendance-log — writes pending recognitions to the ERP DB.")
        if st.button("🚀 Run ERP sync", type="primary", width="stretch"):
            result = call(client, client.erp_sync_run)
            if result:
                stats = result.get("stats") or {}
                action_feedback(
                    result,
                    "Sync finished — "
                    f"scanned={stats.get('scanned')} inserted={stats.get('inserted')} "
                    f"failed={stats.get('failed')}",
                )
        st.markdown("</div>", unsafe_allow_html=True)


def page_console(client: ApiClient) -> None:
    hero("API Console", "Hit any /api/v1 endpoint directly.")
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
        page_title="Face Recognition · Dashboard",
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
    elif page == "ERP Sync":
        page_erp_sync(client)
    else:
        page_console(client)


main()
