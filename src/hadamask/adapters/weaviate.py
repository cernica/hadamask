"""Weaviate adapter (weaviate-client v4).

Two entry points:

* :class:`WeaviateStore` - the uniform :class:`~hadamask.adapters.base.ProtectedStore` API;
* :class:`ProtectedWeaviateCollection` - wraps a v4 ``Collection`` and mirrors the SDK names
  (``data.insert``, ``data.insert_many``, ``query.near_vector``, ``query.fetch_object_by_id``,
  ``data.delete_by_id``), for applications that already use the official SDK.

Weaviate-specific choices:

* Collections must use self-provided vectors. A server-side vectorizer (``text2vec-*``)
  would receive the plaintext, so the wrapper refuses collections that have one.
* Weaviate class names start with an upper-case letter; :class:`WeaviateStore` capitalizes
  the first letter of the collection name it is given.
* Object ids must be UUIDs.
* Properties are declared before first use: ciphertext fields as non-indexed ``text``,
  blind-token fields as ``text`` with ``field`` tokenization (exact-match filtering), and
  plaintext fields with the type of the first value seen. Without this, auto-schema would
  build a BM25 index over ciphertexts and word-tokenize the blind tokens.
* Filters: ``$ne``/``$nin`` match objects that lack the property (as in MongoDB).
  ``scroll`` without a filter uses the cursor API; with a filter (which the cursor API does
  not allow) it pages with ``offset`` sorted by id, so it is bounded by the server's
  ``QUERY_MAXIMUM_RESULTS`` (10,000 by default).
* ``search`` takes no adapter-specific options.
* Scores follow :class:`~hadamask.adapters.base.Hit`: cosine similarity (``1 - distance``),
  inner product (Weaviate reports its negative) and Euclidean distance (Weaviate reports it
  squared).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, Iterable, Mapping, Optional, Sequence

import weaviate.classes.config as wc
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter, MetadataQuery, Sort

from ..core import Protector
from ..metadata import BLIND_PREFIX
from .base import And, Condition, ProtectedStore, RawHit, RawRecord

_DISTANCE = {"cosine": wc.VectorDistances.COSINE, "dot": wc.VectorDistances.DOT, "l2": wc.VectorDistances.L2_SQUARED}


def class_name(name: str) -> str:
    return name[:1].upper() + name[1:]


def _score(distance: Optional[float], metric: Optional[str]) -> Optional[float]:
    if distance is None:
        return None
    if metric == "cosine":
        return 1.0 - distance
    if metric == "l2":
        return math.sqrt(max(distance, 0.0))  # Weaviate reports squared L2
    return -distance  # Weaviate reports the negative dot product


def _plain_type(value: Any) -> Optional[wc.DataType]:
    if isinstance(value, bool):
        return wc.DataType.BOOL
    if isinstance(value, int):
        return wc.DataType.INT
    if isinstance(value, float):
        return wc.DataType.NUMBER
    if isinstance(value, str):
        return wc.DataType.TEXT
    return None  # let auto-schema decide


def _main_vector(vector):
    """Weaviate returns ``{"default": [...]}`` for collections created with one unnamed vector."""
    if isinstance(vector, Mapping):
        return vector.get("default", next(iter(vector.values()), None))
    return vector


def compile_filter(flt) -> Optional[Filter]:
    """Translate a hadamask :data:`~hadamask.adapters.base.Filter` into a Weaviate filter."""
    if flt is None:
        return None
    if isinstance(flt, Condition):
        return _condition(flt)
    parts = [compile_filter(i) for i in flt.items]
    return Filter.all_of(parts) if isinstance(flt, And) else Filter.any_of(parts)


def _condition(c: Condition) -> Filter:
    prop = Filter.by_property(c.field)
    if c.op == "eq":
        return prop.equal(c.value)
    if c.op == "ne":
        return prop.not_equal(c.value)
    if c.op == "in":
        return prop.contains_any(c.value)
    if c.op == "nin":  # not_equal keeps objects without the property, like $nin in MongoDB
        parts = [Filter.by_property(c.field).not_equal(v) for v in c.value]
        return parts[0] if len(parts) == 1 else Filter.all_of(parts)
    return {"gt": prop.greater_than, "gte": prop.greater_or_equal, "lt": prop.less_than, "lte": prop.less_or_equal}[
        c.op
    ](c.value)


def _check_no_vectorizer(collection) -> str:
    """Raise if any vector of ``collection`` is produced server-side; return its distance metric."""
    config = collection.config.get()
    metric = None
    for vec_name, vec in (config.vector_config or {}).items():
        vectorizer = getattr(vec.vectorizer, "vectorizer", None)
        if vectorizer is not None and str(getattr(vectorizer, "value", vectorizer)) != "none":
            raise ValueError(
                f"collection {config.name!r} vector {vec_name!r} uses vectorizer {vectorizer}; "
                "server-side vectorizers see the plaintext - use self-provided vectors"
            )
        metric = metric or getattr(vec.vector_index_config, "distance_metric", None)
    legacy = getattr(config, "vectorizer", None)
    if legacy is not None and str(getattr(legacy, "value", legacy)) != "none":
        raise ValueError(f"collection {config.name!r} uses vectorizer {legacy}; use self-provided vectors")
    by_metric = {v: k for k, v in _DISTANCE.items()}
    return by_metric.get(metric) if metric is not None else None


class _PropertyRegistry:
    """Declares Weaviate properties for protected payloads before they are first written.

    Works on *stored* (already encrypted) payloads: blind-token fields become exact-match
    ``text``/``text[]``, plaintext fields get the type of their value, and every other field
    holds a ciphertext string and is stored without any index.
    """

    def __init__(self, collection, protector: Protector) -> None:
        self._collection = collection
        self._protector = protector
        self._known: Optional[set[str]] = None

    def ensure(self, stored_payloads: Iterable[Optional[Mapping[str, Any]]]) -> None:
        if self._known is None:
            self._known = {p.name for p in self._collection.config.get().properties}
        policy = self._protector.policy
        new: dict[str, wc.Property] = {}
        for payload in stored_payloads:
            for field, value in (payload or {}).items():
                if field in self._known or field in new:
                    continue
                if field.startswith(BLIND_PREFIX):
                    many = isinstance(value, (list, tuple))
                    new[field] = wc.Property(
                        name=field,
                        data_type=wc.DataType.TEXT_ARRAY if many else wc.DataType.TEXT,
                        tokenization=wc.Tokenization.FIELD,
                        index_searchable=False,
                    )
                elif field in policy.plaintext:
                    if _plain_type(value) is not None:
                        new[field] = wc.Property(name=field, data_type=_plain_type(value))
                else:
                    new[field] = wc.Property(
                        name=field, data_type=wc.DataType.TEXT, index_filterable=False, index_searchable=False
                    )
        for prop in new.values():
            self._collection.config.add_property(prop)
            self._known.add(prop.name)


class WeaviateStore(ProtectedStore):
    """Uniform protected store on Weaviate.

    ``index_type`` selects the vector index (``"hnsw"``, ``"flat"``, ``"dynamic"`` or
    ``"hfresh"``). Some Weaviate Cloud plans only allow ``"hfresh"``.
    """

    def __init__(self, client, protector: Protector, index_type: str = "hnsw") -> None:
        super().__init__(protector)
        if index_type not in ("hnsw", "flat", "dynamic", "hfresh"):
            raise ValueError(f"unsupported index_type {index_type!r}")
        self.client = client
        self.index_type = index_type
        self._checked: dict[str, Optional[str]] = {}
        self._registries: dict[str, _PropertyRegistry] = {}

    def _collection(self, name: str):
        return self.client.collections.get(class_name(name))

    def _metric(self, name: str) -> Optional[str]:
        if name not in self._checked:
            self._checked[name] = _check_no_vectorizer(self._collection(name))
        return self._checked[name] or self._metrics.get(name, "cosine")

    def create_collection(self, name: str, dim: int, metric: str = "cosine") -> None:
        self._check_metric(metric)
        self.client.collections.create(
            class_name(name),
            vector_config=wc.Configure.Vectors.self_provided(
                vector_index_config=getattr(wc.Configure.VectorIndex, self.index_type)(
                    distance_metric=_DISTANCE[metric]
                )
            ),
        )
        self._checked[name] = metric
        self.register_collection(name, dim, metric)

    def drop_collection(self, name: str) -> None:
        self.client.collections.delete(class_name(name))
        self._registries.pop(name, None)
        self._checked.pop(name, None)

    def _upsert(self, name, ids, vectors, payloads) -> None:
        self._metric(name)  # refuses collections with a server-side vectorizer
        collection = self._collection(name)
        self._registries.setdefault(name, _PropertyRegistry(collection, self.protector)).ensure(payloads)
        objects = [DataObject(properties=p, vector=v, uuid=i) for i, v, p in zip(ids, vectors, payloads, strict=True)]
        result = collection.data.insert_many(objects)
        if result.has_errors:
            raise RuntimeError(f"Weaviate insert failed: {list(result.errors.values())[:3]}")

    def _search(self, name, vector, k, flt, with_vectors, params) -> list[RawHit]:
        if params:
            raise TypeError(f"WeaviateStore.search takes no adapter options, got {sorted(params)}")
        metric = self._metric(name)
        response = self._collection(name).query.near_vector(
            near_vector=vector,
            limit=k,
            filters=compile_filter(flt),
            include_vector=with_vectors,
            return_metadata=MetadataQuery(distance=True),
        )
        return [
            RawHit(
                o.uuid,
                _score(o.metadata.distance, metric),
                o.properties,
                _main_vector(o.vector) if with_vectors else None,
            )
            for o in response.objects
        ]

    def _get(self, name, ids, with_vectors) -> list[RawRecord]:
        response = self._collection(name).query.fetch_objects(
            filters=Filter.by_id().contains_any(ids), include_vector=with_vectors, limit=len(ids)
        )
        return [
            RawRecord(o.uuid, o.properties, _main_vector(o.vector) if with_vectors else None) for o in response.objects
        ]

    def _delete(self, name, ids) -> None:
        self._collection(name).data.delete_many(where=Filter.by_id().contains_any(ids))

    def _delete_where(self, name, flt) -> None:
        self._collection(name).data.delete_many(where=compile_filter(flt))

    def _count(self, name, flt) -> int:
        return self._collection(name).aggregate.over_all(total_count=True, filters=compile_filter(flt)).total_count

    def _scroll(self, name, flt, batch_size, with_vectors) -> Iterator[RawRecord]:
        collection = self._collection(name)
        after = None
        offset = 0
        while True:
            if flt is None:  # cursor API: no upper bound on collection size
                objects = collection.query.fetch_objects(
                    limit=batch_size, after=after, include_vector=with_vectors
                ).objects
            else:  # the cursor API does not accept filters: offset paging in id order
                objects = collection.query.fetch_objects(
                    filters=compile_filter(flt),
                    limit=batch_size,
                    offset=offset,
                    sort=Sort.by_property("_id"),
                    include_vector=with_vectors,
                ).objects
            for o in objects:
                yield RawRecord(o.uuid, o.properties, _main_vector(o.vector) if with_vectors else None)
            if len(objects) < batch_size:
                return
            after = objects[-1].uuid
            offset += len(objects)

    def close(self) -> None:
        self.client.close()


class _ProtectedData:
    def __init__(self, owner: ProtectedWeaviateCollection) -> None:
        self._owner = owner

    def _object(self, properties, vector, uuid):
        p = self._owner.protector
        return p.encrypt_payload(properties), (p.encode_list(vector) if vector is not None else None), uuid

    def insert(self, properties: Mapping[str, Any], vector=None, uuid=None, **kwargs):
        props, vec, uuid = self._object(properties, vector, uuid)
        self._owner._registry.ensure([props])
        return self._owner._collection.data.insert(properties=props, vector=vec, uuid=uuid, **kwargs)

    def insert_many(self, objects: Sequence[DataObject]):
        protected = []
        for o in objects:
            props, vec, uuid = self._object(o.properties, o.vector, o.uuid)
            protected.append(DataObject(properties=props, vector=vec, uuid=uuid, references=o.references))
        self._owner._registry.ensure(o.properties for o in protected)
        return self._owner._collection.data.insert_many(protected)

    def update(self, uuid, properties: Optional[Mapping[str, Any]] = None, vector=None, **kwargs):
        p = self._owner.protector
        props = p.encrypt_payload(properties) if properties is not None else None
        if props is not None:
            self._owner._registry.ensure([props])
        return self._owner._collection.data.update(
            uuid,
            properties=props,
            vector=p.encode_list(vector) if vector is not None else None,
            **kwargs,
        )

    def replace(self, uuid, properties: Mapping[str, Any], vector=None, **kwargs):
        props, vec, _ = self._object(properties, vector, uuid)
        self._owner._registry.ensure([props])
        return self._owner._collection.data.replace(uuid, properties=props, vector=vec, **kwargs)

    def delete_by_id(self, uuid) -> bool:
        return self._owner._collection.data.delete_by_id(uuid)

    def delete_many(self, where, **kwargs):
        return self._owner._collection.data.delete_many(where=where, **kwargs)

    def exists(self, uuid) -> bool:
        return self._owner._collection.data.exists(uuid)


class _ProtectedQuery:
    def __init__(self, owner: ProtectedWeaviateCollection) -> None:
        self._owner = owner

    def near_vector(self, near_vector, **kwargs):
        if isinstance(near_vector, Mapping):
            raise TypeError("ProtectedWeaviateCollection.query.near_vector supports a single raw vector")
        response = self._owner._collection.query.near_vector(
            near_vector=self._owner.protector.encode_list(near_vector), **kwargs
        )
        return self._owner._reveal_response(response)

    def fetch_object_by_id(self, uuid, **kwargs):
        obj = self._owner._collection.query.fetch_object_by_id(uuid, **kwargs)
        return None if obj is None else self._owner._reveal(obj)

    def fetch_objects(self, **kwargs):
        return self._owner._reveal_response(self._owner._collection.query.fetch_objects(**kwargs))


class ProtectedWeaviateCollection:
    """Wrap a weaviate v4 ``Collection`` with transparent protection.

    ``dim`` is the native embedding dimension, needed to decode vectors returned with
    ``include_vector=True``. Filters are native Weaviate filters; use :meth:`match` for
    conditions on blind or plaintext fields.

    Only attributes that cannot leak data are forwarded to the underlying collection (see
    ``PASSTHROUGH``). ``batch``, ``generate``, ``iterator``, ``query.bm25``/``hybrid`` and the
    like raise ``AttributeError`` instead of bypassing protection; use :attr:`unprotected`
    if you really need the raw collection.
    """

    PASSTHROUGH = frozenset({"name", "config", "aggregate", "tenants", "exists"})

    def __init__(self, collection, protector: Protector, dim: Optional[int] = None) -> None:
        _check_no_vectorizer(collection)
        self._collection = collection
        self.protector = protector
        self.dim = dim
        self._registry = _PropertyRegistry(collection, protector)
        self.data = _ProtectedData(self)
        self.query = _ProtectedQuery(self)

    def __getattr__(self, item):
        if item in self.PASSTHROUGH:
            return getattr(self._collection, item)
        raise AttributeError(
            f"ProtectedWeaviateCollection does not support {item!r}: it would read or write data "
            "without protection. Use a wrapped method, or .unprotected for raw access."
        )

    @property
    def unprotected(self):
        """The raw collection. Data read or written through it is NOT protected."""
        return self._collection

    def with_tenant(self, tenant) -> ProtectedWeaviateCollection:
        return ProtectedWeaviateCollection(self._collection.with_tenant(tenant), self.protector, self.dim)

    def with_consistency_level(self, level) -> ProtectedWeaviateCollection:
        return ProtectedWeaviateCollection(self._collection.with_consistency_level(level), self.protector, self.dim)

    def _reveal(self, obj):
        obj.properties = self.protector.decrypt_payload(obj.properties)
        if obj.vector:
            if self.dim is None:
                raise ValueError("pass dim= to ProtectedWeaviateCollection to read vectors")
            obj.vector = {k: self.protector.decode(v, self.dim).tolist() for k, v in obj.vector.items()}
        return obj

    def _reveal_response(self, response):
        for obj in response.objects:
            self._reveal(obj)
        return response

    def match(self, field: str, value: Any) -> Filter:
        """Native equality filter on a blind or plaintext field."""
        if field in self.protector.policy.plaintext:
            return Filter.by_property(field).equal(value)
        stored, token = self.protector.blind(field, value)
        return Filter.by_property(stored).equal(token)
