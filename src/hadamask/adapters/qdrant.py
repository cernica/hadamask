"""Qdrant adapter.

Two entry points:

* :class:`QdrantStore` - the uniform :class:`~hadamask.adapters.base.ProtectedStore` API;
* :class:`ProtectedQdrantClient` - a drop-in wrapper with the ``qdrant_client`` method
  names (``create_collection``, ``upsert``, ``query_points``, ``retrieve``, ``scroll``),
  for applications that already use the official SDK.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .._deps import missing_extra

try:
    from qdrant_client import QdrantClient, models
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise missing_extra("qdrant", exc) from exc

from ..core import Protector
from .base import And, Condition, Filter, ProtectedStore, RawHit, RawRecord

_DISTANCE = {"cosine": models.Distance.COSINE, "dot": models.Distance.DOT, "l2": models.Distance.EUCLID}


def compile_filter(flt: Optional[Filter]) -> Optional[models.Filter]:
    """Translate a hadamask :data:`~hadamask.adapters.base.Filter` into a Qdrant filter.

    ``$ne``/``$nin`` use ``must_not``, so (as in MongoDB) records without the field match.
    """
    if flt is None:
        return None
    if isinstance(flt, Condition):
        return _condition(flt)
    parts = [compile_filter(i) for i in flt.items]
    return models.Filter(must=parts) if isinstance(flt, And) else models.Filter(should=parts)


def _condition(c: Condition) -> models.Filter:
    if c.op in ("eq", "ne"):
        fc = models.FieldCondition(key=c.field, match=models.MatchValue(value=c.value))
    elif c.op in ("in", "nin"):
        fc = models.FieldCondition(key=c.field, match=models.MatchAny(any=c.value))
    else:
        fc = models.FieldCondition(key=c.field, range=models.Range(**{c.op: c.value}))
    return models.Filter(must_not=[fc]) if c.op in ("ne", "nin") else models.Filter(must=[fc])


def create_filter_indexes(
    client: QdrantClient,
    name: str,
    protector: Protector,
    payload_schema: Optional[Mapping[str, models.PayloadSchemaType]] = None,
) -> None:
    """Create keyword indexes on blind-token fields and the given plaintext field indexes."""
    for field in sorted(protector.policy.blind):
        client.create_payload_index(
            name, protector.metadata.blind_field(field), field_schema=models.PayloadSchemaType.KEYWORD, wait=True
        )
    for field, schema in (payload_schema or {}).items():
        if field not in protector.policy.plaintext:
            raise ValueError(f"payload_schema field {field!r} is not a plaintext field")
        client.create_payload_index(name, field, field_schema=schema, wait=True)


class QdrantStore(ProtectedStore):
    """Uniform protected store on Qdrant.

    ``payload_schema`` maps plaintext fields to Qdrant payload index types (e.g.
    ``{"year": models.PayloadSchemaType.INTEGER}``). Blind fields always get a keyword index.
    Indexes are required for filtering on Qdrant Cloud (strict mode) and speed up filters
    everywhere.

    Search options (``search(..., hnsw_ef=128, exact=True)``) are passed to
    ``models.SearchParams``. Ids must be unsigned integers or UUID strings.
    """

    def __init__(
        self,
        client: QdrantClient,
        protector: Protector,
        payload_schema: Optional[Mapping[str, models.PayloadSchemaType]] = None,
    ) -> None:
        super().__init__(protector)
        self.client = client
        self.payload_schema = dict(payload_schema or {})

    def create_collection(self, name: str, dim: int, metric: str = "cosine") -> None:
        self._check_metric(metric)
        self.client.create_collection(
            name,
            vectors_config=models.VectorParams(size=self.protector.stored_dim(dim), distance=_DISTANCE[metric]),
        )
        create_filter_indexes(self.client, name, self.protector, self.payload_schema)
        self.register_collection(name, dim, metric)

    def drop_collection(self, name: str) -> None:
        self.client.delete_collection(name)

    def _upsert(self, name, ids, vectors, payloads) -> None:
        points = [models.PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads, strict=True)]
        self.client.upsert(name, points=points, wait=True)

    def _search(self, name, vector, k, flt, with_vectors, params) -> list[RawHit]:
        response = self.client.query_points(
            name,
            query=vector,
            query_filter=compile_filter(flt),
            limit=k,
            with_payload=True,
            with_vectors=with_vectors,
            search_params=models.SearchParams(**params) if params else None,
        )
        return [RawHit(p.id, p.score, p.payload, p.vector if with_vectors else None) for p in response.points]

    def _get(self, name, ids, with_vectors) -> list[RawRecord]:
        points = self.client.retrieve(name, ids=ids, with_payload=True, with_vectors=with_vectors)
        return [RawRecord(p.id, p.payload, p.vector if with_vectors else None) for p in points]

    def _delete(self, name, ids) -> None:
        self.client.delete(name, points_selector=models.PointIdsList(points=ids), wait=True)

    def _delete_where(self, name, flt) -> None:
        self.client.delete(name, points_selector=models.FilterSelector(filter=compile_filter(flt)), wait=True)

    def _count(self, name, flt) -> int:
        return self.client.count(name, count_filter=compile_filter(flt), exact=True).count

    def _scroll(self, name, flt, batch_size, with_vectors) -> Iterator[RawRecord]:
        offset = None
        while True:
            points, offset = self.client.scroll(
                name,
                scroll_filter=compile_filter(flt),
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            for p in points:
                yield RawRecord(p.id, p.payload, p.vector if with_vectors else None)
            if offset is None:
                return

    def close(self) -> None:
        self.client.close()


class ProtectedQdrantClient:
    """Wrap a :class:`qdrant_client.QdrantClient`, protecting vectors and payloads transparently.

    Supports single unnamed vectors and named dense vectors. Sizes in ``vectors_config`` are
    native dimensions; the wrapper creates the collection with the protected (padded) size.
    Filters are native Qdrant filters; use :meth:`match` to build conditions on blind fields.

    Only administrative methods that cannot leak data are forwarded to the underlying client
    (see ``PASSTHROUGH``). Anything else that reads or writes points (``set_payload``,
    ``upload_points``, ``query_batch_points``, ...) raises ``AttributeError`` instead of
    silently bypassing protection; use :attr:`unprotected` if you really need the raw client.
    """

    PASSTHROUGH = frozenset(
        {
            "get_collection",
            "get_collections",
            "collection_exists",
            "update_collection",
            "delete_collection",
            "count",
            "delete",
            "create_payload_index",
            "delete_payload_index",
            "get_aliases",
            "get_collection_aliases",
            "update_collection_aliases",
            "create_snapshot",
            "list_snapshots",
            "delete_snapshot",
            "info",
            "close",
        }
    )

    def __init__(self, client: QdrantClient, protector: Protector) -> None:
        self._client = client
        self.protector = protector
        self._dims: dict[str, Any] = {}  # collection -> native dim, or {vector_name: dim}

    def __getattr__(self, item):
        if item in self.PASSTHROUGH:
            return getattr(self._client, item)
        raise AttributeError(
            f"ProtectedQdrantClient does not support {item!r}: it would read or write data without "
            "protection. Use a wrapped method, or .unprotected for raw access."
        )

    @property
    def unprotected(self) -> QdrantClient:
        """The raw client. Data read or written through it is NOT protected."""
        return self._client

    # -- collections ---------------------------------------------------------------------

    def _protected_params(self, params: models.VectorParams) -> models.VectorParams:
        return params.model_copy(update={"size": self.protector.stored_dim(params.size)})

    def create_collection(self, collection_name: str, vectors_config, payload_schema=None, **kwargs):
        """Like ``QdrantClient.create_collection``; also indexes blind fields (see :func:`create_filter_indexes`)."""
        if isinstance(vectors_config, Mapping):
            self._dims[collection_name] = {k: v.size for k, v in vectors_config.items()}
            vectors_config = {k: self._protected_params(v) for k, v in vectors_config.items()}
        else:
            self._dims[collection_name] = vectors_config.size
            vectors_config = self._protected_params(vectors_config)
        result = self._client.create_collection(collection_name, vectors_config=vectors_config, **kwargs)
        create_filter_indexes(self._client, collection_name, self.protector, payload_schema)
        return result

    def register_collection(self, collection_name: str, dims) -> None:
        """Declare native dims (int, or ``{vector_name: dim}``) of an existing collection."""
        self._dims[collection_name] = dims

    # -- vectors and payloads -------------------------------------------------------------

    def _encode_vector(self, vector):
        if vector is None:
            return None
        if isinstance(vector, Mapping):
            return {name: self.protector.encode_list(v) for name, v in vector.items()}
        return self.protector.encode_list(vector)

    def _decode_vector(self, collection_name: str, vector):
        if vector is None:
            return None
        dims = self._dims.get(collection_name)
        if dims is None:
            raise KeyError(f"call register_collection({collection_name!r}, dims) before reading vectors")
        if isinstance(vector, Mapping):
            return {n: self.protector.decode(v, dims[n]).tolist() for n, v in vector.items()}
        return self.protector.decode(vector, dims).tolist()

    def _protect_point(self, point: models.PointStruct) -> models.PointStruct:
        return point.model_copy(
            update={
                "vector": self._encode_vector(point.vector),
                "payload": self.protector.encrypt_payload(point.payload),
            }
        )

    def _reveal(self, collection_name: str, record):
        update = {"payload": self.protector.decrypt_payload(record.payload) if record.payload else record.payload}
        if getattr(record, "vector", None) is not None:
            update["vector"] = self._decode_vector(collection_name, record.vector)
        return record.model_copy(update=update)

    def upsert(self, collection_name: str, points: Sequence[models.PointStruct], **kwargs):
        if not isinstance(points, (list, tuple)):
            raise TypeError("ProtectedQdrantClient.upsert supports a list of PointStruct")
        return self._client.upsert(collection_name, points=[self._protect_point(p) for p in points], **kwargs)

    def query_points(self, collection_name: str, query=None, **kwargs):
        if query is not None and not isinstance(query, (list, np.ndarray)):
            raise TypeError("ProtectedQdrantClient.query_points supports a raw dense vector as query")
        if kwargs.get("prefetch") is not None:
            raise TypeError("prefetch is not supported: its queries would reach Qdrant untransformed")
        if query is not None:
            query = self.protector.encode_list(query)
        response = self._client.query_points(collection_name, query=query, **kwargs)
        return response.model_copy(update={"points": [self._reveal(collection_name, p) for p in response.points]})

    def retrieve(self, collection_name: str, ids, **kwargs):
        return [self._reveal(collection_name, r) for r in self._client.retrieve(collection_name, ids=ids, **kwargs)]

    def scroll(self, collection_name: str, **kwargs):
        records, offset = self._client.scroll(collection_name, **kwargs)
        return [self._reveal(collection_name, r) for r in records], offset

    def match(self, field: str, value: Any) -> models.FieldCondition:
        """Equality condition usable in a native ``models.Filter`` (blind or plaintext field)."""
        if field in self.protector.policy.plaintext:
            return models.FieldCondition(key=field, match=models.MatchValue(value=value))
        stored, token = self.protector.blind(field, value)
        return models.FieldCondition(key=stored, match=models.MatchValue(value=token))
