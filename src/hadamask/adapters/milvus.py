"""Milvus adapter (``pymilvus.MilvusClient``).

Schema per collection: ``id`` VARCHAR primary key, ``vector`` FLOAT_VECTOR of the protected
dimension, and ``payload`` JSON holding the encrypted metadata. A JSON field (rather than
dynamic fields) keeps payload keys from colliding with ``id``/``vector`` and gives one filter
syntax, ``payload["field"] == value``.

Collections use ``consistency_level="Strong"`` so reads see preceding writes. Filter values
are rendered as JSON literals; only str, int, float and bool are accepted, which rules out
expression injection through user-supplied values. Field names are JSON-quoted as well.

Filter semantics: ``$eq`` also matches a list-valued field that contains the value
(``json_contains``); ``$ne``/``$nin`` are the negation of ``$eq``/``$in``, so records without
the field match, as in MongoDB.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from typing import Any, Optional

from .._deps import missing_extra

try:
    from pymilvus import DataType, MilvusClient
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise missing_extra("milvus", exc) from exc

from ..core import Protector
from .base import And, Condition, Filter, ProtectedStore, RawHit, RawRecord

_METRIC = {"cosine": "COSINE", "dot": "IP", "l2": "L2"}
_SEARCH_PARAMS = frozenset({"ef", "nprobe", "search_list", "radius", "range_filter", "level"})
_RANGE = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("filter values must be finite numbers")
    if isinstance(value, (int, float, str)):
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(f"unsupported filter value type for Milvus: {type(value).__name__}")


def compile_filter(flt: Optional[Filter]) -> str:
    """Translate a hadamask filter into a Milvus boolean expression ("" for no filter)."""
    if flt is None:
        return ""
    if isinstance(flt, Condition):
        return _condition(flt)
    joiner = " and " if isinstance(flt, And) else " or "
    return "(" + joiner.join(compile_filter(i) for i in flt.items) + ")"


def _condition(c: Condition) -> str:
    path = f"payload[{json.dumps(c.field)}]"
    if c.op in ("eq", "ne"):
        lit = _literal(c.value)
        expr = f"({path} == {lit} or json_contains({path}, {lit}))"
    elif c.op in ("in", "nin"):
        lits = "[" + ", ".join(_literal(v) for v in c.value) + "]"
        expr = f"({path} in {lits} or json_contains_any({path}, {lits}))"
    else:
        return f"({path} {_RANGE[c.op]} {_literal(c.value)})"
    return f"(not {expr})" if c.op in ("ne", "nin") else expr


class MilvusStore(ProtectedStore):
    """Protected store on Milvus.

    ``index_type`` defaults to HNSW (``M=16``, ``efConstruction=200``); pass ``"FLAT"`` for
    exhaustive search. Scores follow :class:`~hadamask.adapters.base.Hit` (Milvus reports
    squared L2; the adapter returns the Euclidean distance). Search options
    (``search(..., ef=200)``; also ``nprobe``, ``search_list``, ``radius``, ``range_filter``,
    ``level``) go to Milvus' search params.
    """

    def __init__(self, client: MilvusClient, protector: Protector, index_type: str = "HNSW") -> None:
        super().__init__(protector)
        self.client = client
        self.index_type = index_type

    def create_collection(self, name: str, dim: int, metric: str = "cosine") -> None:
        self._check_metric(metric)
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.protector.stored_dim(dim))
        schema.add_field("payload", DataType.JSON)
        index = self.client.prepare_index_params()
        params = {"M": 16, "efConstruction": 200} if self.index_type == "HNSW" else {}
        index.add_index("vector", index_type=self.index_type, metric_type=_METRIC[metric], params=params)
        self.client.create_collection(name, schema=schema, index_params=index, consistency_level="Strong")
        self.client.load_collection(name)
        self.register_collection(name, dim, metric)

    def drop_collection(self, name: str) -> None:
        self.client.drop_collection(name)

    def _score(self, name: str, distance: float) -> float:
        # Milvus reports squared L2; COSINE and IP are already similarities.
        if self._metric(name) == "l2":
            return math.sqrt(max(float(distance), 0.0))
        return float(distance)

    def _upsert(self, name, ids, vectors, payloads) -> None:
        rows = [{"id": str(i), "vector": v, "payload": p} for i, v, p in zip(ids, vectors, payloads, strict=True)]
        self.client.upsert(name, data=rows)

    def _search(self, name, vector, k, flt, with_vectors, params) -> list[RawHit]:
        unknown = set(params) - _SEARCH_PARAMS
        if unknown:
            raise TypeError(f"unsupported Milvus search options: {sorted(unknown)}")
        native = {"ef": max(100, k)} if self.index_type == "HNSW" else {}
        native.update(params)
        results = self.client.search(
            name,
            data=[vector],
            filter=compile_filter(flt),
            limit=k,
            output_fields=["payload", "vector"] if with_vectors else ["payload"],
            search_params={"params": native},
        )
        return [
            RawHit(
                r["id"],
                self._score(name, r["distance"]),
                r["entity"]["payload"],
                list(r["entity"]["vector"]) if with_vectors else None,
            )
            for r in results[0]
        ]

    def _get(self, name, ids, with_vectors) -> list[RawRecord]:
        fields = ["payload", "vector"] if with_vectors else ["payload"]
        rows = self.client.get(name, ids=[str(i) for i in ids], output_fields=fields)
        return [RawRecord(r["id"], r["payload"], list(r["vector"]) if with_vectors else None) for r in rows]

    def _delete(self, name, ids) -> None:
        self.client.delete(name, ids=[str(i) for i in ids])

    def _delete_where(self, name, flt) -> None:
        self.client.delete(name, filter=compile_filter(flt))

    def _count(self, name, flt) -> int:
        rows = self.client.query(name, filter=compile_filter(flt), output_fields=["count(*)"])
        return int(rows[0]["count(*)"])

    def _scroll(self, name, flt, batch_size, with_vectors) -> Iterator[RawRecord]:
        fields = ["payload", "vector"] if with_vectors else ["payload"]
        iterator = self.client.query_iterator(
            name, batch_size=batch_size, filter=compile_filter(flt), output_fields=fields
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    return
                for r in batch:
                    yield RawRecord(r["id"], r["payload"], list(r["vector"]) if with_vectors else None)
        finally:
            iterator.close()

    def close(self) -> None:
        self.client.close()
