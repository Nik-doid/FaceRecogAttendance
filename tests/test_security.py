"""JWT and rate limiter tests."""

from __future__ import annotations

import jwt
import pytest

from app.core.security import RateLimiter, create_access_token, decode_token


def test_jwt_roundtrip() -> None:
    token = create_access_token("alice", extra={"role": "ops"})
    payload = decode_token(token)
    assert payload["sub"] == "alice"
    assert payload["role"] == "ops"


def test_jwt_rejects_tampered() -> None:
    token = create_access_token("alice")
    with pytest.raises(jwt.PyJWTError):
        decode_token(token + "x")


def test_rate_limiter_allows_until_max() -> None:
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    assert limiter.allow("ip-1")
    assert limiter.allow("ip-1")
    assert limiter.allow("ip-1")
    assert not limiter.allow("ip-1")
    # Different key is unaffected.
    assert limiter.allow("ip-2")


def test_rate_limiter_window_expiry() -> None:
    limiter = RateLimiter(max_requests=1, window_seconds=10)
    assert limiter.allow("ip", now=100.0)
    assert not limiter.allow("ip", now=105.0)
    assert limiter.allow("ip", now=111.0)  # outside the window
