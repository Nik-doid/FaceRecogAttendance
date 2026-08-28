"""Where enrolment photos come from: local directories and HTTPS folders.

Both kinds use the same layout -- one subdirectory per ``employee_code``, images
inside it -- so a source is only responsible for enumerating those photos and handing
back bytes. Everything downstream (detect, embed, index) is identical either way,
which is what lets ``gallery.py`` treat a laptop's ``uploads/`` folder and an HR
server behind HTTPS as one pool.

``PhotoRef.key`` is the cache identity: a stable string that changes when the image
changes. Local files use path + mtime + size; remote ones use the URL plus an ETag or
Last-Modified when the server offers one. The embedding cache keys off this, so a key
that fails to change on edit means a stale embedding, and a key that changes when the
image did not means a needless re-embed.

Remote enumeration prefers a manifest and falls back to scraping an autoindex page --
see :meth:`HttpPhotoSource.list_photos`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin, urlparse

import httpx

from app.core.logging import get_logger

log = get_logger(__name__)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
DEFAULT_MANIFEST_NAME = "manifest.json"
# Autoindex pages list directories with a trailing slash. Anchors that are absolute,
# parent links, or nginx/Apache column-sort links are not entries.
_HREF = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
# A directory listing should never be enormous; a runaway parse means we scraped
# something that is not an index page.
MAX_REMOTE_ENTRIES = 5000


@dataclass(frozen=True)
class PhotoRef:
    """One enrolment image, not yet fetched."""

    employee_code: str
    key: str
    label: str  # human-readable origin, for logs
    _read: Callable[[], bytes | None] = field(repr=False)

    def read(self) -> bytes | None:
        """Return the image bytes, or None if it could not be fetched."""
        return self._read()


class PhotoSource(Protocol):
    """Enumerates enrolment photos from one root."""

    def list_photos(self) -> list[PhotoRef]: ...

    def close(self) -> None: ...


class LocalPhotoSource:
    """A directory whose subdirectory names are employee codes."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def list_photos(self) -> list[PhotoRef]:
        if not self._root.is_dir():
            log.warning("employee photo source missing", extra={"source": str(self._root)})
            return []

        refs: list[PhotoRef] = []
        for employee_dir in sorted(p for p in self._root.iterdir() if p.is_dir()):
            photos = sorted(
                p for p in employee_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not photos:
                log.warning(
                    "no photos for employee", extra={"employee_code": employee_dir.name}
                )
            refs.extend(self._to_ref(photo, employee_dir.name) for photo in photos)
        return refs

    def _to_ref(self, photo: Path, employee_code: str) -> PhotoRef:
        stat = photo.stat()
        return PhotoRef(
            employee_code=employee_code,
            # mtime alone is too coarse on filesystems with 1s resolution; size
            # catches same-second edits that keep the length different.
            key=f"local:{photo}:{stat.st_mtime_ns}:{stat.st_size}",
            label=str(photo),
            _read=lambda: _read_file(photo),
        )

    def close(self) -> None:
        """Nothing to release."""


class HttpPhotoSource:
    """An HTTPS folder laid out exactly like the local one.

    Enumeration is tried in two ways, in order:

    1. ``GET {base}/manifest.json`` -- one request, explicit, and able to carry an
       ETag per image so the embedding cache can skip unchanged photos without
       downloading them. This is the supported contract; publish one if you can.
    2. Scraping the server's directory listing. Works, but autoindex markup differs
       between nginx, Apache and Caddy, and object stores answer with XML instead, so
       every build using this path logs a warning recommending a manifest.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 15.0,
        auth_header: str = "",
        manifest_name: str = DEFAULT_MANIFEST_NAME,
        client: httpx.Client | None = None,
    ) -> None:
        self._base = base_url if base_url.endswith("/") else base_url + "/"
        self._manifest_name = manifest_name
        headers = {}
        if auth_header and ":" in auth_header:
            name, _, value = auth_header.partition(":")
            headers[name.strip()] = value.strip()
        # An injected client is how tests drive this against httpx.MockTransport.
        self._client = client or httpx.Client(
            timeout=timeout, follow_redirects=True, headers=headers
        )

    def list_photos(self) -> list[PhotoRef]:
        refs = self._from_manifest()
        if refs is not None:
            return refs
        log.warning(
            "no photo manifest; falling back to directory-listing scrape. "
            "Publish manifest.json for reliable enumeration.",
            extra={"source": self._base, "manifest": self._manifest_name},
        )
        return self._from_autoindex()

    # -- manifest ------------------------------------------------------------
    def _from_manifest(self) -> list[PhotoRef] | None:
        url = urljoin(self._base, self._manifest_name)
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            log.warning("photo manifest unreachable", extra={"url": url, "error": str(exc)})
            return None
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        if response.status_code >= httpx.codes.BAD_REQUEST:
            log.warning(
                "photo manifest request failed",
                extra={"url": url, "status": response.status_code},
            )
            return None
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            log.warning("photo manifest is not valid JSON", extra={"url": url})
            return None

        entries = payload.get("employees", payload) if isinstance(payload, dict) else None
        if not isinstance(entries, dict):
            log.warning("photo manifest has no employees mapping", extra={"url": url})
            return None

        refs: list[PhotoRef] = []
        for employee_code, files in entries.items():
            if not isinstance(files, list):
                continue
            for entry in files:
                ref = self._manifest_entry(str(employee_code), entry)
                if ref is not None:
                    refs.append(ref)
        log.info(
            "photo manifest read",
            extra={"url": url, "employees": len(entries), "photos": len(refs)},
        )
        return refs

    def _manifest_entry(self, employee_code: str, entry: object) -> PhotoRef | None:
        """Accept both ``"front.jpg"`` and ``{"file": "front.jpg", "etag": "..."}``."""
        if isinstance(entry, str):
            filename, version = entry, ""
        elif isinstance(entry, dict):
            filename = str(entry.get("file", ""))
            version = str(entry.get("etag") or entry.get("last_modified") or "")
        else:
            return None
        if not filename or not _is_safe_segment(filename):
            return None
        url = urljoin(self._base, f"{employee_code}/{filename}")
        return self._to_ref(employee_code, url, version)

    # -- autoindex fallback ---------------------------------------------------
    def _from_autoindex(self) -> list[PhotoRef]:
        refs: list[PhotoRef] = []
        for employee_code in self._links(self._base, want_dirs=True):
            code = employee_code.rstrip("/")
            for filename in self._links(urljoin(self._base, f"{code}/"), want_dirs=False):
                url = urljoin(self._base, f"{code}/{filename}")
                refs.append(self._to_ref(code, url, version=""))
                if len(refs) >= MAX_REMOTE_ENTRIES:
                    log.warning(
                        "stopped enumerating remote photos at the cap",
                        extra={"source": self._base, "cap": MAX_REMOTE_ENTRIES},
                    )
                    return refs
        return refs

    def _links(self, url: str, *, want_dirs: bool) -> list[str]:
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("directory listing failed", extra={"url": url, "error": str(exc)})
            return []
        if "html" not in response.headers.get("content-type", "").lower():
            log.warning(
                "directory listing is not HTML; a manifest.json is required here",
                extra={"url": url, "content_type": response.headers.get("content-type")},
            )
            return []

        found: list[str] = []
        for href in _HREF.findall(response.text):
            if not _is_safe_segment(href):
                continue
            is_dir = href.endswith("/")
            if is_dir != want_dirs:
                continue
            if not want_dirs and Path(href).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            found.append(href)
        return sorted(set(found))

    # -- shared ---------------------------------------------------------------
    def _to_ref(self, employee_code: str, url: str, version: str) -> PhotoRef:
        if not version:
            version = self._probe_version(url)
        return PhotoRef(
            employee_code=employee_code,
            key=f"http:{url}:{version}",
            label=url,
            _read=lambda: _fetch(self._client, url),
        )

    def _probe_version(self, url: str) -> str:
        """A HEAD for ETag/Last-Modified, so an unchanged photo can skip embedding.

        Without it every remote photo re-embeds on every build, which at ~0.7s each
        is the whole cold-start cost paid again.
        """
        try:
            response = self._client.head(url)
        except httpx.HTTPError:
            return ""
        etag = response.headers.get("etag") or response.headers.get("last-modified") or ""
        return etag.strip()

    def close(self) -> None:
        self._client.close()


def _read_file(photo: Path) -> bytes | None:
    try:
        return photo.read_bytes()
    except OSError as exc:
        log.warning("unreadable enrolment photo", extra={"photo": str(photo), "error": str(exc)})
        return None


def _fetch(client: httpx.Client, url: str) -> bytes | None:
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("enrolment photo download failed", extra={"url": url, "error": str(exc)})
        return None
    return response.content


def _is_safe_segment(href: str) -> bool:
    """Reject traversal, absolute URLs and listing-sort links.

    An autoindex page carries links back to the parent, to itself, and (on nginx and
    Apache) to re-sorted views of the same directory. Following those walks the tree
    sideways or off-host.
    """
    if not href or href.startswith(("/", "#", "?", "..")):
        return False
    if "://" in href or "\\" in href:
        return False
    return ".." not in Path(href).parts


def is_url(uri: str) -> bool:
    return urlparse(uri).scheme in {"http", "https"}


def build_sources(
    uris: Sequence[str],
    *,
    timeout: float = 15.0,
    auth_header: str = "",
    manifest_name: str = DEFAULT_MANIFEST_NAME,
) -> list[PhotoSource]:
    """Turn the configured URIs into sources. ``http(s)://`` is remote, else local."""
    sources: list[PhotoSource] = []
    for uri in uris:
        text = str(uri).strip()
        if not text:
            continue
        if is_url(text):
            sources.append(
                HttpPhotoSource(
                    text,
                    timeout=timeout,
                    auth_header=auth_header,
                    manifest_name=manifest_name,
                )
            )
        else:
            sources.append(LocalPhotoSource(text))
    return sources


def collect(sources: Iterable[PhotoSource]) -> list[PhotoRef]:
    """List every source in order. A code in two sources simply gets more vectors."""
    refs: list[PhotoRef] = []
    for source in sources:
        refs.extend(source.list_photos())
    return refs
