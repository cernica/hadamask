"""Behaviour every ProtectedStore adapter must satisfy, checked against a real database."""

from __future__ import annotations

import uuid
from typing import Any, Callable, Optional

import numpy as np

from hadamask import MasterKey, MetadataPolicy, Protector
from hadamask.adapters.base import ProtectedStore
from hadamask.metadata import is_ciphertext

DIM = 384
N_DOCS = 600
N_QUERIES = 25
K = 10

POLICY = MetadataPolicy(plaintext=["year"], blind=["client"])


def make_protector() -> Protector:
    return Protector(MasterKey.generate("test1"), POLICY)


def make_corpus(seed: int = 7):
    """Clustered unit vectors (like real embeddings) with sensitive payloads."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((12, DIM))
    labels = rng.integers(0, 12, N_DOCS)
    docs = centers[labels] + 0.6 * rng.standard_normal((N_DOCS, DIM))
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    queries = centers[rng.integers(0, 12, N_QUERIES)] + 0.6 * rng.standard_normal((N_QUERIES, DIM))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    ids = [str(uuid.UUID(int=i + 1)) for i in range(N_DOCS)]
    payloads = [
        {
            "text": f"Wire transfer #{i}: {rng.integers(1_000, 9_999_999)} EUR to IBAN RO49AAAA{i:016d}",
            "client": f"client-{i % 5}",
            "year": 2020 + i % 3,
            "amount": float(rng.integers(100, 10_000_000)) / 100,
        }
        for i in range(N_DOCS)
    ]
    return ids, docs.astype(np.float32), queries.astype(np.float32), payloads


def exact_top_k(docs: np.ndarray, query: np.ndarray, k: int, mask: Optional[np.ndarray] = None) -> list[int]:
    scores = docs.astype(np.float64) @ query.astype(np.float64)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    order = np.argsort(-scores, kind="stable")[:k]
    return [int(i) for i in order if np.isfinite(scores[i])]


# -- reference filter semantics, evaluated on CLEAR payloads ----------------------------------


def matches(payload: dict, where: dict) -> bool:
    """Ground truth for the hadamask filter syntax (MongoDB-like; $ne/$nin match missing fields)."""
    for key, cond in where.items():
        if key == "$and":
            ok = all(matches(payload, w) for w in cond)
        elif key == "$or":
            ok = any(matches(payload, w) for w in cond)
        else:
            ops = cond if isinstance(cond, dict) else {"$eq": cond}
            ok = all(_check(payload.get(key), op, operand) for op, operand in ops.items())
        if not ok:
            return False
    return True


def _check(value, op, operand) -> bool:
    return {
        "$eq": lambda: value == operand,
        "$ne": lambda: value != operand,
        "$in": lambda: value in operand,
        "$nin": lambda: value not in operand,
        "$gt": lambda: value is not None and value > operand,
        "$gte": lambda: value is not None and value >= operand,
        "$lt": lambda: value is not None and value < operand,
        "$lte": lambda: value is not None and value <= operand,
    }[op]()


FILTER_CASES = [
    {"client": "client-3"},
    {"year": 2021, "client": "client-1"},
    {"year": {"$gte": 2021}},
    {"year": {"$gt": 2020, "$lte": 2021}},
    {"year": {"$lt": 2021}},
    {"client": {"$in": ["client-0", "client-4"]}},
    {"client": {"$ne": "client-0"}},
    {"client": {"$nin": ["client-0", "client-1"]}},
    {"year": {"$in": [2020, 2022]}},
    {"year": {"$nin": [2020]}},
    {"$or": [{"client": "client-2"}, {"year": 2022}]},
    {"$and": [{"year": {"$ne": 2020}}, {"$or": [{"client": "client-1"}, {"client": "client-2"}]}]},
]


def run_contract(
    store: ProtectedStore,
    name: str,
    raw_payload: Optional[Callable[[str, Any], dict]] = None,
    min_recall: float = 0.97,
    settle: Callable[[], None] = lambda: None,
    clone: Optional[Callable[[Protector], ProtectedStore]] = None,
) -> dict:
    """Run the full contract against a live database.

    ``raw_payload(name, id)`` reads a record bypassing hadamask (at-rest check). ``settle()``
    waits until preceding writes are searchable. ``clone(protector)`` builds a store of the
    same kind with another key (used to test ``copy_to`` / key rotation).
    """
    ids, docs, queries, payloads = make_corpus()
    index_of = {rid: i for i, rid in enumerate(ids)}
    for stale in (name, name + "_copy"):
        try:
            store.drop_collection(stale)
        except Exception:
            pass
    store.create_collection(name, DIM, metric="cosine")
    try:
        # 0. One upsert call, split into batches by the base class.
        store.upsert(name, ids, docs, payloads, batch_size=128)
        settle()
        assert store.count(name) == N_DOCS

        # 1. Retrieval quality: protected search vs exact search on the clear vectors.
        recalls = []
        for q in queries:
            truth = set(exact_top_k(docs, q, K))
            hits = store.search(name, q, k=K)
            assert len(hits) == K, f"expected {K} hits, got {len(hits)}"
            recalls.append(len(truth & {index_of[str(h.id)] for h in hits}) / K)
            for h in hits:
                assert h.payload == payloads[index_of[str(h.id)]], "payload did not decrypt to the original"
            assert all(a.score >= b.score for a, b in zip(hits, hits[1:])), "hits are not sorted best first"
        recall = float(np.mean(recalls))
        assert recall >= min_recall, f"recall@{K} {recall:.4f} < {min_recall}"

        # 2. Every filter operator, checked against the reference semantics on clear payloads.
        filter_recalls = []
        for case_no, where in enumerate(FILTER_CASES):
            mask = np.array([matches(p, where) for p in payloads])
            q = queries[case_no % N_QUERIES]
            hits = store.search(name, q, k=K, where=where)
            for h in hits:
                assert matches(h.payload, where), f"hit violates filter {where}: {h.payload}"
            truth = exact_top_k(docs, q, K, mask)
            assert len(hits) == min(K, len(truth)), f"filter {where}: {len(hits)} hits, expected {len(truth)}"
            filter_recalls.append(len(set(truth) & {index_of[str(h.id)] for h in hits}) / len(truth))
            assert store.count(name, where) == int(mask.sum()), f"count mismatch for {where}"
        filter_recall = float(np.mean(filter_recalls))
        assert filter_recall >= min_recall, f"filtered recall {filter_recall:.4f} < {min_recall}"

        # 3. Refused filters: encrypted-only fields, ranges on blind fields, unknown operators.
        for bad in ({"amount": 1.0}, {"client": {"$gt": "client-1"}}, {"year": {"$regex": "20"}}, {"$not": {}}):
            try:
                store.search(name, queries[0], k=K, where=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"filter {bad} must be refused")

        # 4. Paging, thresholds, projections, vectors, batches of queries.
        top = store.search(name, queries[2], k=2 * 5)
        page1 = store.search(name, queries[2], k=5)
        page2 = store.search(name, queries[2], k=5, offset=5)
        assert [h.id for h in page1 + page2] == [h.id for h in top], "offset paging is inconsistent"
        threshold = top[4].score
        above = store.search(name, queries[2], k=K, score_threshold=threshold)
        assert 5 <= len(above) <= K and all(h.score >= threshold for h in above)
        slim = store.search(name, queries[2], k=3, fields=["client"])
        assert all(set(h.payload) == {"client"} for h in slim)
        with_vec = store.search(name, queries[2], k=3, with_vectors=True)
        for h in with_vec:
            assert np.allclose(h.vector, docs[index_of[str(h.id)]], atol=1e-5)
        batched = store.search_many(name, queries[:3], k=5)
        assert [[h.id for h in hs] for hs in batched] == [
            [h.id for h in store.search(name, q, k=5)] for q in queries[:3]
        ]

        # 5. Get: vectors decode to the native embedding; projections.
        records = store.get(name, ids[:3] + ["00000000-0000-0000-0000-00000000ffff"], with_vectors=True)
        assert sorted(str(r.id) for r in records) == sorted(ids[:3])
        for r in records:
            i = index_of[str(r.id)]
            assert r.payload == payloads[i]
            assert r.vector is not None and r.vector.shape == (DIM,)
            assert np.allclose(r.vector, docs[i], atol=1e-5), "decoded vector differs from original"
        assert store.get(name, ids[:1], fields=["year"])[0].payload == {"year": payloads[0]["year"]}

        # 6. Scroll: every record exactly once; with a filter; with vectors.
        seen = [str(r.id) for r in store.scroll(name, batch_size=97)]
        assert sorted(seen) == sorted(ids), "scroll must return every record exactly once"
        where = {"client": "client-2", "year": {"$gte": 2021}}
        expected = {ids[i] for i, p in enumerate(payloads) if matches(p, where)}
        scrolled = list(store.scroll(name, where, batch_size=13, with_vectors=True))
        assert {str(r.id) for r in scrolled} == expected and len(scrolled) == len(expected)
        for r in scrolled[:5]:
            assert np.allclose(r.vector, docs[index_of[str(r.id)]], atol=1e-5)

        # 7. What the database actually stores: ciphertext only (except declared plaintext fields).
        if raw_payload is not None:
            raw = raw_payload(name, ids[0])
            assert raw["year"] == payloads[0]["year"]
            for field in ("text", "client", "amount"):
                assert is_ciphertext(raw[field]), f"{field} is not encrypted at rest: {raw[field]!r}"
            assert "_hmb_client" in raw
            assert "IBAN RO49" not in repr(raw) and "client-0" not in repr(raw)

        # 8. Partial payload update keeps the vector and re-encrypts.
        store.update_payload(name, ids[0], {"amount": 1.5, "note": "reviewed"})
        settle()
        updated = store.get(name, ids[:1], with_vectors=True)[0]
        assert updated.payload == {**payloads[0], "amount": 1.5, "note": "reviewed"}
        assert np.allclose(updated.vector, docs[0], atol=1e-5)
        assert any(
            str(h.id) == ids[0] for h in store.search(name, docs[0], k=3, where={"client": payloads[0]["client"]})
        )

        # 9. Copy into a collection protected with ANOTHER key (key rotation / migration).
        copied_ok = None
        if clone is not None:
            target = clone(Protector(MasterKey.generate("test2"), POLICY))
            target.create_collection(name + "_copy", DIM, metric="cosine")
            try:
                assert store.copy_to(name, target, name + "_copy", batch_size=150) == N_DOCS
                settle()
                assert target.count(name + "_copy") == N_DOCS
                for q in queries[:3]:
                    a = [str(h.id) for h in store.search(name, q, k=K)]
                    b = [str(h.id) for h in target.search(name + "_copy", q, k=K)]
                    assert len(set(a) & set(b)) >= K - 1, "rotated copy returns different neighbours"
                assert target.get(name + "_copy", ids[:1])[0].payload == updated.payload
                assert target.count(name + "_copy", {"client": "client-3"}) == store.count(name, {"client": "client-3"})
                copied_ok = True
            finally:
                target.drop_collection(name + "_copy")

        # 10. Delete by ids and by filter.
        store.delete(name, ids[:2])
        settle()
        assert store.get(name, ids[:2]) == []
        remaining = [p for i, p in enumerate(payloads) if i >= 2]
        store.delete(name, where={"client": "client-4", "year": {"$lte": 2021}})
        settle()
        gone = sum(matches(p, {"client": "client-4", "year": {"$lte": 2021}}) for p in remaining)
        assert store.count(name) == len(remaining) - gone
        assert store.count(name, {"client": "client-4", "year": {"$lte": 2021}}) == 0
        for bad in ({}, None):
            try:
                store.delete(name, where=bad) if bad is not None else store.delete(name)
            except ValueError:
                pass
            else:
                raise AssertionError("delete without ids or filter must raise")
        return {"recall": recall, "filtered_recall": filter_recall, "copy": copied_ok}
    finally:
        store.drop_collection(name)
