"""Elasticsearch adapter (8.x).

Database-specific choices:

* The protected vector lives in ``_hm_vector`` (a ``dense_vector`` of the stored dimension).
  ``index_options`` is set explicitly to ``hnsw``: since 8.14 Elasticsearch defaults to
  ``int8_hnsw`` (scalar quantization) for float vectors. Quantization is applied to the
  *stored* coordinates, so whether search is exact or approximate is decided by the engine
  configuration, not by the transform (paper, Section 5.7). We keep full float32 vectors.
* Payload fields are top-level ``_source`` fields. Dynamic templates index blind tokens and
  plaintext strings as ``keyword`` (for ``term`` filters); ciphertexts are only stored.
* Elasticsearch reports transformed scores; :class:`Hit.score` is converted back to the
  native similarity (cosine / dot product) or to the Euclidean distance for ``l2``.
* Filters run inside the ``knn`` clause (pre-filtering), so filtered searches still return
  ``k`` hits. ``num_candidates`` (constructor default 100, or per search) is raised to ``k``
  when needed. ``dot`` requires unit-length vectors (an Elasticsearch rule).
* The record id is also stored as the keyword ``_hm_id``; :meth:`_scroll` pages on it with
  ``search_after`` because sorting on ``_id`` is disabled by default.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from ..core import Protector
from ._mappings import ID_FIELD, compile_filter, dynamic_templates
from .base import ProtectedStore, RawHit, RawRecord

VECTOR_FIELD = "_hm_vector"
_SIMILARITY = {"cosine": "cosine", "dot": "dot_product", "l2": "l2_norm"}
_SEARCH_PARAMS = {"num_candidates"}


def _native_score(metric: str, score: float) -> float:
    if metric in ("cosine", "dot"):
        # cosine: (1 + cos) / 2; dot_product on unit vectors: (1 + dot) / 2.
        return 2.0 * score - 1.0
    # l2_norm: 1 / (1 + d^2) -> d
    return math.sqrt(max(0.0, 1.0 / score - 1.0))


def _split(source: dict) -> tuple[dict, Optional[list]]:
    payload = dict(source)
    vector = payload.pop(VECTOR_FIELD, None)
    payload.pop(ID_FIELD, None)
    return payload, vector


class ElasticsearchStore(ProtectedStore):
    def __init__(self, client: Elasticsearch, protector: Protector, num_candidates: int = 100) -> None:
        super().__init__(protector)
        self.client = client
        self.num_candidates = num_candidates

    def create_collection(self, name: str, dim: int, metric: str = "cosine") -> None:
        self._check_metric(metric)
        self.client.indices.create(
            index=name,
            mappings={
                "dynamic_templates": dynamic_templates(self.protector),
                "properties": {
                    ID_FIELD: {"type": "keyword"},
                    VECTOR_FIELD: {
                        "type": "dense_vector",
                        "dims": self.protector.stored_dim(dim),
                        "index": True,
                        "similarity": _SIMILARITY[metric],
                        "index_options": {"type": "hnsw"},
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
            raise TypeError(f"unknown Elasticsearch search options: {sorted(unknown)}")
        knn = {
            "field": VECTOR_FIELD,
            "query_vector": vector,
            "k": k,
            "num_candidates": max(params.get("num_candidates", self.num_candidates), k),
        }
        if flt is not None:
            knn["filter"] = compile_filter(flt)
        source = {} if with_vectors else {"source_excludes": [VECTOR_FIELD]}
        response = self.client.search(index=name, knn=knn, size=k, **source)
        metric = self._metric(name)
        hits = []
        for h in response["hits"]["hits"]:
            payload, vec = _split(h["_source"])
            hits.append(RawHit(h["_id"], _native_score(metric, h["_score"]), payload, vec))
        return hits

    def _get(self, name, ids, with_vectors) -> list[RawRecord]:
        kwargs = {} if with_vectors else {"source_excludes": [VECTOR_FIELD]}
        docs = self.client.mget(index=name, ids=[str(i) for i in ids], **kwargs)["docs"]
        return [RawRecord(d["_id"], *_split(d["_source"])) for d in docs if d.get("found")]

    def _delete(self, name, ids) -> None:
        actions = ({"_op_type": "delete", "_index": name, "_id": str(i)} for i in ids)
        bulk(self.client, actions, refresh="wait_for", raise_on_error=False)

    def _delete_where(self, name, flt) -> None:
        self.client.delete_by_query(
            index=name, query=compile_filter(flt), refresh=True, conflicts="proceed", wait_for_completion=True
        )

    def _count(self, name, flt) -> int:
        return self.client.count(index=name, query=compile_filter(flt))["count"]

    def _scroll(self, name, flt, batch_size, with_vectors) -> Iterator[RawRecord]:
        source = {} if with_vectors else {"source_excludes": [VECTOR_FIELD]}
        after = None
        while True:
            extra = {"search_after": after} if after is not None else {}
            response = self.client.search(
                index=name, query=compile_filter(flt), size=batch_size, sort=[{ID_FIELD: "asc"}], **source, **extra
            )
            hits = response["hits"]["hits"]
            for h in hits:
                yield RawRecord(h["_id"], *_split(h["_source"]))
            if len(hits) < batch_size:
                return
            after = hits[-1]["sort"]

    def close(self) -> None:
        self.client.close()
