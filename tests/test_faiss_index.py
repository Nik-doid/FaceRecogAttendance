"""FAISS index behaviour tests (cosine similarity, atomic rebuild)."""

from __future__ import annotations

import threading

import numpy as np

from app.ai.faiss.index import FaceIndex, IndexItem, l2_normalize


def test_add_and_search_exact() -> None:
    rng = np.random.default_rng(42)
    idx = FaceIndex(dim=8)
    emb = rng.normal(size=8).astype("float32")
    idx.add(emb, "EMP1")
    results = idx.search(emb, k=1)
    assert len(results) == 1
    assert results[0].employee_code == "EMP1"
    assert results[0].score > 0.999


def test_cosine_similarity_order() -> None:
    rng = np.random.default_rng(1)
    idx = FaceIndex(dim=8)
    emb_a = rng.normal(size=8).astype("float32")
    emb_b = -emb_a.copy()
    idx.add(emb_a, "EMP_A")
    idx.add(emb_b, "EMP_B")
    results = idx.search(emb_a, k=2)
    assert results[0].employee_code == "EMP_A"
    assert results[0].score > results[1].score


def test_unnormalized_input_still_cosine() -> None:
    idx = FaceIndex(dim=8)
    vec = np.full(8, 5.0, dtype="float32")  # not unit length
    idx.add(vec, "EMP1")
    results = idx.search(vec, k=1)
    assert results[0].score > 0.99  # normalization made it near-cosine


def test_rebuild_is_atomic_and_replaces() -> None:
    idx = FaceIndex(dim=8)
    rng = np.random.default_rng(7)
    old = rng.normal(size=8).astype("float32")
    idx.add(old, "OLD")
    assert idx.size == 1

    new_emb = rng.normal(size=8).astype("float32")
    idx.rebuild([IndexItem("NEW", new_emb)])
    assert idx.size == 1
    assert idx.employee_codes == {"NEW"}
    assert idx.search(new_emb, k=1)[0].employee_code == "NEW"


def test_empty_index_search() -> None:
    idx = FaceIndex(dim=8)
    assert idx.search(np.zeros(8, dtype="float32")) == []


def test_l2_normalize() -> None:
    v = np.array([3.0, 4.0], dtype="float32")
    n = l2_normalize(v)
    assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-6


# --- the parallel _codes list ------------------------------------------------
# The index returns *row numbers*, and a row's employee code is only correct while the
# list it was numbered against is still in place. A mistake here does not degrade
# accuracy -- it records a punch against the wrong person.


def test_a_rebuild_during_a_search_never_yields_a_crossed_code() -> None:
    """``rebuild`` swaps index and codes together; a search must see one or the other.

    Reading ``_codes`` after releasing the lock maps this search's row numbers onto the
    *next* gallery's codes. At GALLERY_REFRESH_SECONDS=3600 that is a wrong employee
    roughly once an hour, silently.
    """
    rng = np.random.default_rng(11)
    idx = FaceIndex(dim=8)
    old = [
        IndexItem(f"OLD{i}", rng.normal(size=8).astype("float32")) for i in range(50)
    ]
    new = [IndexItem(f"NEW{i}", rng.normal(size=8).astype("float32")) for i in range(3)]
    idx.rebuild(old)

    probe = rng.normal(size=8).astype("float32")
    seen: list[str] = []
    errors: list[BaseException] = []
    stop = threading.Event()

    def search_forever() -> None:
        try:
            while not stop.is_set():
                results = idx.search(probe, k=1)
                if results:
                    seen.append(results[0].employee_code)
        except BaseException as exc:  # noqa: BLE001 - the assertion is that none escape
            errors.append(exc)

    worker = threading.Thread(target=search_forever, daemon=True)
    worker.start()
    for _ in range(200):
        idx.rebuild(new)
        idx.rebuild(old)
    stop.set()
    worker.join(timeout=5)

    assert not errors
    assert seen, "the search thread never completed a search"
    assert set(seen) <= {item.employee_code for item in old + new}


def test_top_2_survives_an_employee_with_many_photos() -> None:
    """Top-2 must be the second-best *employee*, not the same person's 11th photo.

    The candidate pool used to be a flat ``k * 10`` rows, so one employee with ten
    enrolment photos could fill it and hide everybody else. A rank margin computed from
    that is comparing someone against themselves.
    """
    rng = np.random.default_rng(23)
    idx = FaceIndex(dim=8)
    probe = rng.normal(size=8).astype("float32")
    items = [IndexItem("EMP1", probe + 0.01 * i) for i in range(15)]
    items.append(IndexItem("EMP2", -probe))
    idx.rebuild(items)

    results = idx.search(probe, k=2)
    assert [r.employee_code for r in results] == ["EMP1", "EMP2"]
