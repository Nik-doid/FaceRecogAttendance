"""Enrolment photo enumeration: local directories and HTTPS folders."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.core.face_processing.photos import (
    HttpPhotoSource,
    LocalPhotoSource,
    build_sources,
    collect,
    is_url,
)

BASE = "https://hr.example.com/photos/"


def _photo_tree(root: Path) -> Path:
    for code, names in (("EMP1", ["a.jpg", "b.png"]), ("EMP2", ["c.jpeg"])):
        directory = root / code
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).write_bytes(b"not-really-an-image")
    (root / "EMP3").mkdir()  # no photos: warned about, not fatal
    (root / "loose.jpg").write_bytes(b"ignored")  # not inside an employee dir
    return root


# --- local -------------------------------------------------------------------


def test_local_source_uses_directory_names_as_employee_codes(tmp_path: Path) -> None:
    refs = LocalPhotoSource(_photo_tree(tmp_path)).list_photos()
    assert sorted({r.employee_code for r in refs}) == ["EMP1", "EMP2"]
    assert len(refs) == 3
    assert refs[0].read() == b"not-really-an-image"


def test_local_source_missing_directory_is_not_fatal(tmp_path: Path) -> None:
    assert LocalPhotoSource(tmp_path / "nope").list_photos() == []


def test_local_key_changes_when_the_photo_does(tmp_path: Path) -> None:
    """The embedding cache keys off this, so an edited photo must miss and re-embed."""
    photo = tmp_path / "EMP1" / "a.jpg"
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b"one")
    before = LocalPhotoSource(tmp_path).list_photos()[0].key

    photo.write_bytes(b"a different length entirely")
    after = LocalPhotoSource(tmp_path).list_photos()[0].key
    assert before != after


# --- remote ------------------------------------------------------------------


def _source(handler: object, **kwargs: object) -> HttpPhotoSource:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE)
    return HttpPhotoSource(BASE, client=client, **kwargs)


def test_manifest_is_preferred() -> None:
    manifest = {"employees": {"EMP1": ["front.jpg", "side.jpg"], "EMP2": ["only.jpg"]}}
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("manifest.json"):
            return httpx.Response(200, json=manifest)
        return httpx.Response(200, content=b"jpeg-bytes")

    refs = _source(handler).list_photos()
    assert sorted(r.employee_code for r in refs) == ["EMP1", "EMP1", "EMP2"]
    assert refs[0].read() == b"jpeg-bytes"
    # Nothing scraped a directory listing.
    assert not any(path.endswith("/") for path in seen)


def test_manifest_accepts_the_degenerate_shape() -> None:
    """A hand-written mapping works without the outer "employees" key."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("manifest.json"):
            return httpx.Response(200, json={"EMP1": ["a.jpg"]})
        return httpx.Response(200, content=b"x")

    refs = _source(handler).list_photos()
    assert [r.employee_code for r in refs] == ["EMP1"]


def test_manifest_etag_lands_in_the_key() -> None:
    """An ETag in the manifest is what lets a warm start skip download and embed."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("manifest.json"):
            return httpx.Response(
                200, json={"employees": {"EMP1": [{"file": "a.jpg", "etag": "W/abc"}]}}
            )
        return httpx.Response(200, content=b"x")

    (ref,) = _source(handler).list_photos()
    assert "W/abc" in ref.key


def test_autoindex_fallback_when_manifest_is_absent() -> None:
    listing = '<a href="../">up</a><a href="EMP1/">EMP1/</a><a href="?C=N;O=D">sort</a>'
    files = '<a href="../">up</a><a href="a.jpg">a.jpg</a><a href="notes.txt">notes</a>'

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("manifest.json"):
            return httpx.Response(404)
        body = files if path.rstrip("/").endswith("EMP1") else listing
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})

    refs = _source(handler).list_photos()
    assert [(r.employee_code, r.label) for r in refs] == [("EMP1", f"{BASE}EMP1/a.jpg")]


def test_autoindex_rejects_traversal_and_offsite_links() -> None:
    listing = (
        '<a href="../../etc/">up</a>'
        '<a href="https://evil.example.com/x/">offsite</a>'
        '<a href="/absolute/">absolute</a>'
        '<a href="EMP1/">EMP1/</a>'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("manifest.json"):
            return httpx.Response(404)
        body = '<a href="a.jpg">a</a>' if "EMP1" in request.url.path else listing
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})

    refs = _source(handler).list_photos()
    assert {r.employee_code for r in refs} == {"EMP1"}


def test_non_html_listing_is_refused() -> None:
    """Object stores answer with XML; scraping that would invent employee codes."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("manifest.json"):
            return httpx.Response(404)
        return httpx.Response(
            200, text="<ListBucketResult/>", headers={"content-type": "application/xml"}
        )

    assert _source(handler).list_photos() == []


def test_unreachable_server_yields_nothing_rather_than_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert _source(handler).list_photos() == []


def test_download_failure_reads_as_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("manifest.json"):
            return httpx.Response(200, json={"EMP1": ["a.jpg"]})
        return httpx.Response(500)

    (ref,) = _source(handler).list_photos()
    assert ref.read() is None


def test_auth_header_is_sent() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"EMP1": ["a.jpg"]})

    client = httpx.Client(
        transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer tok"}
    )
    HttpPhotoSource(BASE, client=client).list_photos()
    assert seen and seen[0] == "Bearer tok"


# --- wiring ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("https://h/p/", True),
        ("http://h/p/", True),
        ("uploads/employees", False),
        ("/var/photos", False),
    ],
)
def test_is_url(uri: str, expected: bool) -> None:
    assert is_url(uri) is expected


def test_build_sources_picks_a_source_per_uri(tmp_path: Path) -> None:
    sources = build_sources([str(tmp_path), BASE, "  "])
    assert [type(s).__name__ for s in sources] == ["LocalPhotoSource", "HttpPhotoSource"]
    for source in sources:
        source.close()


def test_collect_merges_sources_so_a_code_can_gain_vectors(tmp_path: Path) -> None:
    """A code in two sources is not a conflict -- FaceIndex keeps both vectors."""
    first, second = tmp_path / "one", tmp_path / "two"
    for root in (first, second):
        (root / "EMP1").mkdir(parents=True)
        (root / "EMP1" / "a.jpg").write_bytes(b"x")

    refs = collect([LocalPhotoSource(first), LocalPhotoSource(second)])
    assert [r.employee_code for r in refs] == ["EMP1", "EMP1"]
    assert len({r.key for r in refs}) == 2
