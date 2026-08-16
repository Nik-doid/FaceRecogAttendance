"""Thin, typed REST client for the service's API (prefix ``/api/v1``)."""

from __future__ import annotations

from typing import Any

import requests

API_PREFIX = "/api/v1"


class ApiError(Exception):
    def __init__(self, method: str, path: str, status_code: int, body: Any) -> None:
        super().__init__(f"{method} {path} -> HTTP {status_code}: {body}")
        self.method = method
        self.path = path
        self.status_code = status_code
        self.body = body


class ApiClient:
    def __init__(self, base_url: str, token: str | None = None, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def set_token(self, token: str | None) -> None:
        self.token = token

    # -- core ---------------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        auth: bool = False,
        raw: bool = False,
    ) -> Any:
        url = f"{self.base_url}{API_PREFIX}{path if path.startswith('/') else '/' + path}"
        response = requests.request(
            method,
            url,
            json=json_body,
            headers=self._headers(auth),
            timeout=self.timeout,
        )
        if not response.ok:
            try:
                body = response.json()
            except ValueError:
                body = response.text
            raise ApiError(method, path, response.status_code, body)
        if raw:
            return response.text
        try:
            return response.json()
        except ValueError:
            return response.text

    def _headers(self, include_auth: bool) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if include_auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    # -- typed helpers ------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self.request("GET", "/health")

    def index_status(self) -> dict[str, Any]:
        return self.request("GET", "/index/status")

    def build_info(self) -> dict[str, Any]:
        return self.request("GET", "/index/build-info")

    def camera_status(self) -> dict[str, Any]:
        return self.request("GET", "/camera/status")

    def camera_start(self) -> dict[str, Any]:
        return self.request("POST", "/camera/start", auth=True)

    def camera_stop(self) -> dict[str, Any]:
        return self.request("POST", "/camera/stop", auth=True)

    def index_rebuild(self) -> dict[str, Any]:
        return self.request("POST", "/index/rebuild", auth=True)

    def recognition_logs(self, limit: int = 50, employee_code: str | None = None) -> dict[str, Any]:
        path = f"/recognition/logs?limit={limit}"
        if employee_code:
            path += f"&employee_code={employee_code}"
        return self.request("GET", path)

    def unknown_faces(self, limit: int = 50) -> dict[str, Any]:
        return self.request("GET", f"/unknown-faces?limit={limit}")

    def metrics(self) -> str:
        return self.request("GET", "/metrics", raw=True)

    def erp_sync_status(self) -> dict[str, Any]:
        return self.request("GET", "/sync/attendance-log/status")

    def erp_sync_run(self) -> dict[str, Any]:
        return self.request("POST", "/sync/attendance-log", auth=True)
