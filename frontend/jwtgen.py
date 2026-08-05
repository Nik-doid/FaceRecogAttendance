"""Local JWT generation for the debug console.

The service itself issues no tokens (issuance belongs to the ops/admin system), so
the UI signs a token locally with the same secret / algorithm / expiry the running
service uses. PyJWT is preferred (it is already a service dependency) with a
pure-stdlib HS256 fallback so the dashboard also works in a bare venv.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> dict[str, str]:
    """Read the service ``.env`` file values (real env vars win if both set)."""
    values: dict[str, str] = {}
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        for key, value in dotenv.dotenv_values(env_file).items():
            if value is not None:
                values[key] = value
    return values


def env_value(name: str, env: dict[str, str], default: str) -> str:
    return os.environ.get(name) or env.get(name) or default


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _hs256(secret: str, subject: str, expire_minutes: int) -> str:
    """Minimal RFC 7519 HS256 signing (no external dependencies)."""
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {"sub": subject, "iat": now, "exp": now + expire_minutes * 60}

    def enc(obj: dict[str, Any]) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return _b64url(raw)

    signing_input = f"{enc(header)}.{enc(payload)}".encode()
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{signing_input.decode('ascii')}.{_b64url(signature)}"


def create_token(
    subject: str,
    secret: str,
    algorithm: str = "HS256",
    expire_minutes: int = 60,
) -> str:
    """Sign a bearer token the running service will accept."""
    if algorithm != "HS256":
        raise ValueError(f"Local generation supports HS256 only, got {algorithm!r}")
    try:
        import jwt as pyjwt
    except ImportError:
        return _hs256(secret, subject, expire_minutes)
    now = datetime.now(UTC)
    return pyjwt.encode(
        {"sub": subject, "iat": now, "exp": now + timedelta(minutes=expire_minutes)},
        secret,
        algorithm="HS256",
    )


def decode_payload(token: str) -> dict[str, Any]:
    """Decode a JWT's payload without verifying the signature (for display only)."""
    try:
        import jwt as pyjwt
    except ImportError:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("not a JWT") from None
        padding = "=" * (-len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(parts[1] + padding))
    return pyjwt.decode(
        token,
        options={"verify_signature": False, "verify_exp": False},
    )
