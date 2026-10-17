"""MongoDB Atlas Vector Search adapter (Atlas or ``mongodb-atlas-local``).

Document layout: ``{_id: id, embedding: [protected vector], payload: {encrypted fields}}``.
Keeping the payload in a sub-document avoids collisions with ``_id`` and ``embedding``.

A ``vectorSearch`` index named ``hadamask_vector`` is created with the protected dimension
and ``filter`` entries for every plaintext field and every blind token field of the policy
(``payload.<field>`` / ``payload._hmb_<field>``), since Atlas only filters indexed paths.

The search index is maintained asynchronously by ``mongot``: :meth:`MongoStore.wait_until_synced`
blocks until all documents are searchable. Search uses exact (ENN) mode by default, which
returns the true top-k; pass ``exact=False`` to use ANN with ``num_candidates``.

Filters compile to MQL on ``payload.<field>``; the same document is used as the
``$vectorSearch`` pre-filter (fields declared in the index) and for ``count_documents``,
``find`` (scroll) and ``delete_many``.

Search options: ``exact`` (bool) and ``num_candidates`` (int) override the store defaults
for one query.

Scores: Atlas normalizes ``vectorSearchScore`` to [0, 1]; the adapter converts it back to
the native measure: cosine similarity / inner product (``2*s - 1``) for ``cosine``/``dot``,
Euclidean distance (``sqrt(1/s - 1)``, Atlas reports ``1 / (1 + d^2)``) for ``l2``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from typing import Any, Optional

import numpy as np

from .._deps import missing_extra

try:
    from pymongo import ReplaceOne
    from pymongo.database import Database
    from pymongo.operations import SearchIndexModel
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise missing_extra("mongodb", exc) from exc

from ..core import Protector
from .base import And, Condition, Filter, ProtectedStore, RawHit, RawRecord

_SIMILARITY = {"cosine": "cosine", "dot": "dotProduct", "l2": "euclidean"}
INDEX_NAME = "hadamask_vector"
_VEC = "embedding"


def compile_filter(flt: Optional[Filter]) -> dict:
    """Translate a hadamask filter into MQL over the ``payload`` sub-document."""
    if flt is None:
        return {}
    if isinstance(flt, Condition):
        return {f"payload.{flt.field}": {f"${flt.op}": flt.value}}
    return {"$and" if isinstance(flt, And) else "$or": [compile_filter(i) for i in flt.items]}


class MongoStore(ProtectedStore):
    def __init__(
        self,
        database: Database,
        protector: Protector,
        exact: bool = True,
        num_candidates: int = 200,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(protector)
        self.db = database
        self.exact = exact
        self.num_candidates = num_candidates
        self.timeout = timeout

    def create_collection(self, name: str, dim: int, metric: str = "cosine") -> None:
        self._check_metric(metric)
        policy = self.protector.policy
        filter_paths = [f"payload.{f}" for f in sorted(policy.plaintext)]
        filter_paths += [f"payload.{self.protector.metadata.blind_field(f)}" for f in sorted(policy.blind)]
        fields = [
            {
                "type": "vector",
                "path": _VEC,
                "numDimensions": self.protector.stored_dim(dim),
                "similarity": _SIMILARITY[metric],
            }
        ] + [{"type": "filter", "path": p} for p in filter_paths]
        self.db.create_collection(name)
        self.db[name].create_search_index(
            SearchIndexModel(definition={"fields": fields}, name=INDEX_NAME, type="vectorSearch")
        )
        self.register_collection(name, dim, metric)
        self._wait(lambda: any(i.get("queryable") for i in self.db[name].list_search_indexes(INDEX_NAME)))

    def drop_collection(self, name: str) -> None:
        self.db.drop_collection(name)

    def _wait(self, ready) -> None:
        deadline = time.monotonic() + self.timeout
        while not ready():
            if time.monotonic() > deadline:
                raise TimeoutError("MongoDB search index did not become ready in time")
            time.sleep(0.5)

    def _stored_dim(self, name: str) -> int:
        if name in self._dims:
            return self.protector.stored_dim(self._dims[name])
        for index in self.db[name].list_search_indexes(INDEX_NAME):
            definition = index.get("latestDefinition") or index.get("definition") or {}
            for f in definition.get("fields", []):
                if f.get("type") == "vector":
                    return int(f["numDimensions"])
        raise KeyError(f"collection {name!r} has no {INDEX_NAME} vector index")

    def wait_until_synced(self, name: str) -> None:
        """Block until the vector index covers exactly the documents in the collection."""
        coll = self.db[name]
        probe = np.ones(self._stored_dim(name)).tolist()

        def synced() -> bool:
            count = coll.count_documents({})
            if count == 0:
                return True
            found = list(
                coll.aggregate(
                    [
                        {
                            "$vectorSearch": {
                                "index": INDEX_NAME,
                                "path": _VEC,
                                "queryVector": probe,
                                "exact": True,
                                "limit": min(count + 1, 10_000),
                            }
                        },
                        {"$count": "n"},
                    ]
                )
            )
            return bool(found) and found[0]["n"] == count

        self._wait(synced)

    # -- hooks ----------------------------------------------------------------------------------

    def _upsert(self, name, ids, vectors, payloads) -> None:
        ops = [
            ReplaceOne({"_id": rid}, {"_id": rid, _VEC: vec, "payload": p}, upsert=True)
            for rid, vec, p in zip(ids, vectors, payloads, strict=True)
        ]
        self.db[name].bulk_write(ops, ordered=False)

    def _score(self, name: str, s: float) -> float:
        if self._metric(name) == "l2":
            # Atlas euclidean score is 1 / (1 + d^2).
            return math.sqrt(max(1.0 / s - 1.0, 0.0)) if s > 0 else float("inf")
        return 2.0 * s - 1.0

    def _search(self, name, vector, k, flt, with_vectors, params) -> list[RawHit]:
        unknown = set(params) - {"exact", "num_candidates"}
        if unknown:
            raise TypeError(f"unknown MongoStore search options: {sorted(unknown)}")
        stage: dict[str, Any] = {"index": INDEX_NAME, "path": _VEC, "queryVector": vector, "limit": int(k)}
        if params.get("exact", self.exact):
            stage["exact"] = True
        else:
            stage["numCandidates"] = max(int(params.get("num_candidates", self.num_candidates)), 20 * int(k))
        if flt is not None:
            stage["filter"] = compile_filter(flt)
        project: dict[str, Any] = {"payload": 1, "score": {"$meta": "vectorSearchScore"}}
        if with_vectors:
            project[_VEC] = 1
        return [
            RawHit(d["_id"], self._score(name, d["score"]), d.get("payload"), d.get(_VEC))
            for d in self.db[name].aggregate([{"$vectorSearch": stage}, {"$project": project}])
        ]

    def _get(self, name, ids, with_vectors) -> list[RawRecord]:
        projection = None if with_vectors else {_VEC: 0}
        return [
            RawRecord(d["_id"], d.get("payload"), d.get(_VEC))
            for d in self.db[name].find({"_id": {"$in": ids}}, projection)
        ]

    def _delete(self, name, ids) -> None:
        self.db[name].delete_many({"_id": {"$in": ids}})

    def _delete_where(self, name, flt) -> None:
        self.db[name].delete_many(compile_filter(flt))

    def _count(self, name, flt) -> int:
        return self.db[name].count_documents(compile_filter(flt))

    def _scroll(self, name, flt, batch_size, with_vectors) -> Iterator[RawRecord]:
        projection = None if with_vectors else {_VEC: 0}
        cursor = self.db[name].find(compile_filter(flt), projection).sort("_id", 1).batch_size(batch_size)
        for d in cursor:
            yield RawRecord(d["_id"], d.get("payload"), d.get(_VEC))
