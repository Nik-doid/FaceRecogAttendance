"""FAISS index behaviour tests (cosine similarity, atomic rebuild)."""

from __future__ import annotations

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
