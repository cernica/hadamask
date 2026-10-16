"""Redis (RediSearch / Redis Stack) adapter.

Storage layout: one HASH per record at key ``"{collection}:{id}"`` and one RediSearch index
named after the collection over that prefix.

* ``vec`` - protected vector, FLOAT32 bytes, dimension ``protector.stored_dim(dim)``;
* each payload field is its own hash field. Encrypted fields hold the ``hm1:`` ciphertext;
  plaintext fields are JSON-encoded so their types (int, float, bool, str...) round-trip;
* blind tokens (``_hmb_<field>``) are case-sensitive TAG fields. List-valued blind fields are
  stored as comma-separated tokens (base64url never contains commas) and ``$eq`` matches
  any element;
* plaintext fields listed in ``numeric_fields`` are NUMERIC fields (their JSON encoding of a
  number is a plain number RediSearch can parse) and support every operator, ranges
  included. They must hold numbers: other values are rejected on write, because RediSearch
  silently drops documents whose NUMERIC field does not parse;
* other plaintext fields are case-sensitive TAG fields over their JSON encoding (separator
  ``\\x1f`` so commas inside values do not split them) and support ``$eq $ne $in $nin`` only.

Filters compile to RediSearch query syntax (AND = space, OR = ``|``, NOT = ``-``). Negated
conditions (``$ne``/``$nin``) match records that lack the field, as in MongoDB.

Scores follow :class:`~hadamask.adapters.base.Hit`: RediSearch reports ``1 - similarity``
for COSINE/IP and squared L2; the adapter converts both.

Iteration (``scroll``, ``delete(where=...)``) uses an ``FT.AGGREGATE`` cursor, so it is not
bound by ``MAXSEARCHRESULTS``. ``count`` uses ``FT.SEARCH ... LIMIT 0 0``.

Search options: ``ef_runtime`` (HNSW only).

The client must be created with ``decode_responses=False`` because vectors are binary.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from typing import Any, Iterable, Optional

import numpy as np
import redis
from redis.commands.search.aggregation import AggregateRequest
from redis.commands.search.field import NumericField, TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from ..core import Protector
from .base import And, Condition, Filter, ProtectedStore, RawHit, RawRecord

_METRIC = {"cosine": "COSINE", "dot": "IP", "l2": "L2"}
_VEC = "vec"
_TAG_ESCAPE = re.compile(r"([^A-Za-z0-9])")
_PLAIN_SEPARATOR = "\x1f"


def _escape_tag(value: str) -> str:
    return _TAG_ESCAPE.sub(r"\\\1", value)


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError(f"numeric filter values must be finite numbers, got {value!r}")
    return repr(float(value)) if isinstance(value, float) else str(value)


class RedisStore(ProtectedStore):
    """Protected store on Redis Stack.

    ``numeric_fields`` are plaintext fields holding numbers; they get NUMERIC indexes and
    support ranges. ``algorithm`` is ``"HNSW"`` or ``"FLAT"``.
    """

    def __init__(
        self,
        client: redis.Redis,
        protector: Protector,
        algorithm: str = "HNSW",
        ef_runtime: int = 200,
        numeric_fields: Iterable[str] = (),
    ) -> None:
        super().__init__(protector)
        if client.get_connection_kwargs().get("decode_responses"):
            raise ValueError("RedisStore needs a client created with decode_responses=False")
        self.client = client
        self.algorithm = algorithm
        self.ef_runtime = ef_runtime
        self.numeric_fields = frozenset(numeric_fields)
        not_plain = self.numeric_fields - protector.policy.plaintext
        if not_plain:
            raise ValueError(f"numeric_fields must be plaintext fields: {sorted(not_plain)}")

    def _key(self, name: str, rid: Any) -> str:
        return f"{name}:{rid}"

    # -- collections -------------------------------------------------------------------------

    def create_collection(self, name: str, dim: int, metric: str = "cosine") -> None:
        self._check_metric(metric)
        attrs = {"TYPE": "FLOAT32", "DIM": self.protector.stored_dim(dim), "DISTANCE_METRIC": _METRIC[metric]}
        if self.algorithm == "HNSW":
            attrs["EF_RUNTIME"] = self.ef_runtime
        policy = self.protector.policy
        schema: list = [VectorField(_VEC, self.algorithm, attrs)]
        schema += [TagField(self.protector.metadata.blind_field(f), case_sensitive=True) for f in sorted(policy.blind)]
        for f in sorted(policy.plaintext):
            if f in self.numeric_fields:
                schema.append(NumericField(f))
            else:
                schema.append(TagField(f, separator=_PLAIN_SEPARATOR, case_sensitive=True))
        self.client.ft(name).create_index(
            schema, definition=IndexDefinition(prefix=[f"{name}:"], index_type=IndexType.HASH)
        )
        self.register_collection(name, dim, metric)

    def drop_collection(self, name: str) -> None:
        self.client.ft(name).dropindex(delete_documents=True)

    # -- stored representation ------------------------------------------------------------------

    def _to_hash(self, vector: list[float], payload: dict) -> dict[str, Any]:
        mapping: dict[str, Any] = {_VEC: np.asarray(vector, dtype=np.float32).tobytes()}
        for field, value in payload.items():
            if field in self.protector.policy.plaintext:
                if field in self.numeric_fields and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    raise ValueError(f"numeric field {field!r} must hold a number, got {value!r}")
                mapping[field] = json.dumps(value)
            elif isinstance(value, list):  # blind tokens of a list-valued field
                mapping[field] = ",".join(value)
            else:
                mapping[field] = value
        return mapping

    def _from_hash(self, rid: Any, raw: dict, with_vectors: bool) -> RawRecord:
        stored: dict[str, Any] = {}
        vector = None
        for k, v in raw.items():
            field = _text(k)
            if field == _VEC:
                if with_vectors:
                    vector = np.frombuffer(v, dtype=np.float32).tolist()
            elif field in self.protector.policy.plaintext:
                stored[field] = json.loads(_text(v))
            else:
                stored[field] = _text(v)
        return RawRecord(rid, stored, vector)

    # -- filters --------------------------------------------------------------------------------

    def _compile(self, flt: Optional[Filter]) -> str:
        if flt is None:
            return "*"
        if isinstance(flt, Condition):
            return self._condition(flt)
        joiner = " " if isinstance(flt, And) else " | "
        return "(" + joiner.join(self._compile(i) for i in flt.items) + ")"

    def _condition(self, c: Condition) -> str:
        values = c.value if c.op in ("in", "nin") else [c.value]
        negate = "-" if c.op in ("ne", "nin") else ""
        if c.field in self.numeric_fields:
            if c.op in ("eq", "ne", "in", "nin"):
                body = " | ".join(f"@{c.field}:[{_number(v)} {_number(v)}]" for v in values)
                return f"{negate}({body})"
            bound = _number(c.value)
            ranges = {
                "gt": f"[({bound} +inf]",
                "gte": f"[{bound} +inf]",
                "lt": f"[-inf ({bound}]",
                "lte": f"[-inf {bound}]",
            }
            return f"@{c.field}:{ranges[c.op]}"
        if c.op not in ("eq", "ne", "in", "nin"):
            raise ValueError(
                f"range filter on {c.field!r} needs a NUMERIC index: pass numeric_fields=[{c.field!r}] to RedisStore"
            )
        encoded = [v if c.blind else json.dumps(v) for v in values]
        return f"{negate}@{c.field}:{{{' | '.join(_escape_tag(v) for v in encoded)}}}"

    # -- hooks ----------------------------------------------------------------------------------

    def _upsert(self, name, ids, vectors, payloads) -> None:
        pipe = self.client.pipeline(transaction=False)
        for rid, vec, payload in zip(ids, vectors, payloads, strict=True):
            key = self._key(name, rid)
            pipe.delete(key)  # replace semantics: drop fields of the previous version
            pipe.hset(key, mapping=self._to_hash(vec, payload))
        pipe.execute()

    def _search(self, name, vector, k, flt, with_vectors, params) -> list[RawHit]:
        unknown = set(params) - {"ef_runtime"}
        if unknown:
            raise TypeError(f"unknown RedisStore search options: {sorted(unknown)}")
        query_params: dict[str, Any] = {"q": np.asarray(vector, dtype=np.float32).tobytes()}
        knn = f"KNN {int(k)} @{_VEC} $q"
        if "ef_runtime" in params:
            knn += " EF_RUNTIME $ef"
            query_params["ef"] = int(params["ef_runtime"])
        prefilter = "*" if flt is None else f"({self._compile(flt)})"
        query = Query(f"{prefilter}=>[{knn} AS score]").sort_by("score").return_fields("score").paging(0, k).dialect(2)
        docs = self.client.ft(name).search(query, query_params=query_params).docs
        pipe = self.client.pipeline(transaction=False)
        for d in docs:
            pipe.hgetall(_text(d.id))
        raws = pipe.execute()
        metric = self._metric(name)
        prefix = f"{name}:"
        hits = []
        for d, raw in zip(docs, raws, strict=True):
            distance = float(d.score)
            score = math.sqrt(max(distance, 0.0)) if metric == "l2" else 1.0 - distance
            record = self._from_hash(_text(d.id)[len(prefix) :], raw, with_vectors)
            hits.append(RawHit(record.id, score, record.payload, record.vector))
        return hits

    def _get(self, name, ids, with_vectors) -> list[RawRecord]:
        pipe = self.client.pipeline(transaction=False)
        for rid in ids:
            pipe.hgetall(self._key(name, rid))
        return [self._from_hash(rid, raw, with_vectors) for rid, raw in zip(ids, pipe.execute(), strict=True) if raw]

    def _delete(self, name, ids) -> None:
        self.client.delete(*[self._key(name, rid) for rid in ids])

    def _keys(self, name: str, flt: Optional[Filter], batch_size: int) -> Iterator[list[str]]:
        """Keys of matching records, one batch per FT.AGGREGATE cursor read."""
        request = AggregateRequest(self._compile(flt)).load("@__key").cursor(count=batch_size).dialect(2)
        index = self.client.ft(name)
        result = index.aggregate(request)
        while True:
            keys = []
            for row in result.rows:
                fields = {_text(k): v for k, v in zip(row[::2], row[1::2], strict=True)}
                keys.append(_text(fields["__key"]))
            if keys:
                yield keys
            if result.cursor is None or not result.cursor.cid:
                return
            result = index.aggregate(result.cursor)

    def _delete_where(self, name, flt) -> None:
        keys = [k for batch in self._keys(name, flt, 1000) for k in batch]
        for start in range(0, len(keys), 1000):
            self.client.delete(*keys[start : start + 1000])

    def _count(self, name, flt) -> int:
        return self.client.ft(name).search(Query(self._compile(flt)).no_content().paging(0, 0).dialect(2)).total

    def _scroll(self, name, flt, batch_size, with_vectors) -> Iterator[RawRecord]:
        prefix = f"{name}:"
        for keys in self._keys(name, flt, batch_size):
            pipe = self.client.pipeline(transaction=False)
            for key in keys:
                pipe.hgetall(key)
            for key, raw in zip(keys, pipe.execute(), strict=True):
                if raw:
                    yield self._from_hash(key[len(prefix) :], raw, with_vectors)

    def close(self) -> None:
        self.client.close()
