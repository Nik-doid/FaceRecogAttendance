"""Security helpers: JWT tokens, bearer auth, in-memory rate limiting.

Notes on production trade-offs:
- JWT secret comes from env; in a real deployment rotate it and keep it in a secret
  manager. We only *verify* tokens here; token issuance happens in the ops tooling /
  admin system, so this service ships no user CRUD.
- The rate limiter is in-memory (per-process). That is correct for a single-instance
  service, which is the documented deployment model. For multi-replica horizontal
  scaling, swap the limiter backend for Redis with the same interface.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config.settings import settings

_bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------
def create_access_token(subject: str, extra: dict[str, object] | None = None) -> str:
    """Create a signed JWT. ``subject`` is typically the operator/CI username."""
    now = datetime.now(UTC)
    payload: dict[str, object] = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, object]:
    """Decode + validate a JWT. Raises jwt.PyJWTError on any problem."""
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    """FastAPI dependency: returns the authenticated subject or 401."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_token(credentials.credentials)
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    subject = payload.get("sub")
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject",
        )
    return str(subject)


# ---------------------------------------------------------------------------
# Rate limiting (in-memory fixed window)
# ---------------------------------------------------------------------------
class RateLimiter:
    """Fixed-window rate limiter keyed by an arbitrary string (IP, subject, ...)."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = __import__("threading").Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            window = self._hits[key]
            while window and window[0] <= now - self._window_seconds:
                window.popleft()
            if len(window) >= self._max_requests:
                return False
            window.append(now)
            return True


_rate_limiter = RateLimiter(
    settings.rate_limit_max_requests, settings.rate_limit_window_seconds
)


def rate_limit(
    max_requests: int | None = None,
    window_seconds: int | None = None,
) -> Callable[[Request], None]:
    """Dependency factory for rate limiting a specific endpoint."""

    def _check(request: Request) -> None:
        limiter = _rate_limiter
        if max_requests or window_seconds:
            limiter = RateLimiter(
                max_requests or settings.rate_limit_max_requests,
                window_seconds or settings.rate_limit_window_seconds,
            )
        key = request.client.host if request.client else "unknown"
        if not limiter.allow(key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(settings.rate_limit_window_seconds)},
            )

    return _check
