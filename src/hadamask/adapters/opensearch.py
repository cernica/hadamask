"""OpenSearch adapter (2.x, k-NN plugin).

Database-specific choices:

* The protected vector lives in ``_hm_vector``, a ``knn_vector`` of the stored dimension,
  indexed with HNSW on the ``lucene`` engine, which supports efficient filtered k-NN search
  through the ``filter`` clause of the ``knn`` query. No quantization is configured, so the
  full float32 protected vectors are searched.
* Payload fields are top-level ``_source`` fields. Dynamic templates index blind tokens and
  plaintext strings as ``keyword`` (for ``term`` filters); ciphertexts are only stored.
* The lucene engine uses the query's ``k`` as its candidate list size, so the adapter asks
  for ``max(num_candidates, k)`` neighbours and returns the top ``k``; asking for ``k=5``
  directly loses true neighbours.
* OpenSearch reports transformed scores; :class:`Hit.score` is converted back to the native
  similarity (cosine / inner product) or to the Euclidean distance for ``l2``.
* Filters run inside the ``knn`` clause (efficient filtering), so filtered searches still
  return ``k`` hits. Search options: ``num_candidates`` and ``ef_search`` (sent as
  ``method_parameters``, OpenSearch 2.16+).
* The record id is also stored as the keyword ``_hm_id``; :meth:`_scroll` pages on it with
  ``search_after``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Optional

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from ..core import Protector
from ._mappings import ID_FIELD, compile_filter, dynamic_templates
from .base import ProtectedStore, RawHit, RawRecord

VECTOR_FIELD = "_hm_vector"
_SPACE = {"cosine": "cosinesimil", "dot": "innerproduct", "l2": "l2"}
_SEARCH_PARAMS = {"num_candidates", "ef_search"}


def _native_score(metric: str, score: float) -> float:
    if metric == "cosine":
        return 2.0 * score - 1.0  # (1 + cos) / 2
    if metric == "dot":
        return score - 1.0 if score >= 1.0 else 1.0 - 1.0 / score  # ip + 1, or 1 / (1 - ip)
    return math.sqrt(max(0.0, 1.0 / score - 1.0))  # 1 / (1 + d^2)


def _split(source: dict) -> tuple[dict, Optional[list]]:
    payload = dict(source)
    vector = payload.pop(VECTOR_FIELD, None)
    payload.pop(ID_FIELD, None)
    return payload, vector


class OpenSearchStore(ProtectedStore):
    def __init__(
        self, client: OpenSearch, protector: Protector, engine: str = "lucene", num_candidates: int = 100
    ) -> None:
        super().__init__(protector)
        self.client = client
        self.engine = engine
        self.num_candidates = num_candidates

    def create_collection(self, name: str, dim: int, metric: str = "cosine") -> None:
        self._check_metric(metric)
        self.client.indices.create(
            index=name,
            body={
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "dynamic_templates": dynamic_templates(self.protector),
                    "properties": {
                        ID_FIELD: {"type": "keyword"},
                        VECTOR_FIELD: {
                            "type": "knn_vector",
                            "dimension": self.protector.stored_dim(dim),
                            "method": {"name": "hnsw", "engine": self.engine, "space_type": _SPACE[metric]},
                        },
                    },
                },
            },
        )
        self.register_collection(name, dim, metric)

    def drop_collection(self, name: str) -> None:
        self.client.indices.delete(index=name, ignore_unavailable=True)

    def _upsert(self, name, ids, vectors, payloads) -> None:
        actions = (
            {"_index": name, "_id": str(i), "_source": {**p, ID_FIELD: str(i), VECTOR_FIELD: v}}
            for i, v, p in zip(ids, vectors, payloads, strict=True)
        )
        bulk(self.client, actions, refresh="wait_for")

    def _search(self, name, vector, k, flt, with_vectors, params) -> list[RawHit]:
        unknown = set(params) - _SEARCH_PARAMS
        if unknown:
            raise TypeError(f"unknown OpenSearch search options: {sorted(unknown)}")
        knn: dict = {"vector": vector, "k": max(params.get("num_candidates", self.num_candidates), k)}
        if flt is not None:
            knn["filter"] = compile_filter(flt)
        if "ef_search" in params:
            knn["method_parameters"] = {"ef_search": params["ef_search"]}
        body: dict = {"size": k, "query": {"knn": {VECTOR_FIELD: knn}}}
        if not with_vectors:
            body["_source"] = {"excludes": [VECTOR_FIELD]}
        response = self.client.search(index=name, body=body)
        metric = self._metric(name)
        hits = []
        for h in response["hits"]["hits"]:
            payload, vec = _split(h["_source"])
            hits.append(RawHit(h["_id"], _native_score(metric, h["_score"]), payload, vec))
        return hits

    def _get(self, name, ids, with_vectors) -> list[RawRecord]:
        params = {} if with_vectors else {"_source_excludes": VECTOR_FIELD}
        docs = self.client.mget(index=name, body={"ids": [str(i) for i in ids]}, params=params)["docs"]
        return [RawRecord(d["_id"], *_split(d["_source"])) for d in docs if d.get("found")]

    def _delete(self, name, ids) -> None:
        actions = ({"_op_type": "delete", "_index": name, "_id": str(i)} for i in ids)
        bulk(self.client, actions, refresh="wait_for", raise_on_error=False)

    def _delete_where(self, name, flt) -> None:
        self.client.delete_by_query(
            index=name,
            body={"query": compile_filter(flt)},
            params={"refresh": "true", "conflicts": "proceed", "wait_for_completion": "true"},
        )

    def _count(self, name, flt) -> int:
        return self.client.count(index=name, body={"query": compile_filter(flt)})["count"]

    def _scroll(self, name, flt, batch_size, with_vectors) -> Iterator[RawRecord]:
        after = None
        while True:
            body: dict = {"query": compile_filter(flt), "size": batch_size, "sort": [{ID_FIELD: "asc"}]}
            if not with_vectors:
                body["_source"] = {"excludes": [VECTOR_FIELD]}
            if after is not None:
                body["search_after"] = after
            hits = self.client.search(index=name, body=body)["hits"]["hits"]
            for h in hits:
                yield RawRecord(h["_id"], *_split(h["_source"]))
            if len(hits) < batch_size:
                return
            after = hits[-1]["sort"]

    def close(self) -> None:
        self.client.close()
