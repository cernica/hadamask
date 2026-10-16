"""Database-independent interface implemented by every adapter.

The public methods of :class:`ProtectedStore` (``upsert``, ``search``, ``get``, ``scroll``,
``count``, ``delete``, ...) do all protection work: they transform vectors, encrypt and
decrypt payloads, translate filters to stored fields and tokens, and apply paging and score
thresholds. Adapters only implement the native ``_``-prefixed hooks, which receive data that
is already protected and return what the database stored. Adapters never see keys.

Filters (``where``) use a MongoDB-like syntax::

    {"year": 2026}                                  # equality
    {"year": {"$gte": 2020, "$lt": 2027}}           # ranges: $gt $gte $lt $lte (plaintext fields)
    {"client": {"$in": ["ACME", "Globex"]}}         # $in $nin $eq $ne
    {"$or": [{"client": "ACME"}, {"year": 2026}]}   # $or / $and, nested freely

Several fields in one dict are combined with AND. Plaintext fields accept every operator.
Blind fields accept ``$eq $ne $in $nin`` (their tokens do not preserve order). Encrypted-only
fields cannot be filtered. For a list-valued blind field, ``$eq`` matches records whose list
contains the value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from ..core import Protector

METRICS = ("cosine", "dot", "l2")
OPERATORS = ("eq", "ne", "in", "nin", "gt", "gte", "lt", "lte")
_BLIND_OPERATORS = ("eq", "ne", "in", "nin")


# -- results ------------------------------------------------------------------------------


@dataclass
class Hit:
    """One search result, with metadata already decrypted.

    ``score`` has the same meaning on every database:

    * ``cosine`` - cosine similarity (higher is closer);
    * ``dot`` - inner product (higher is closer);
    * ``l2`` - Euclidean distance, not squared (lower is closer).

    Because the transform is orthogonal, these equal the values computed on the clear
    vectors (up to float32 rounding). ``vector`` is the native vector when requested.
    """

    id: Any
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    vector: Optional[np.ndarray] = None


@dataclass
class Record:
    """A stored record, with metadata decrypted and (optionally) the native vector restored."""

    id: Any
    payload: dict[str, Any] = field(default_factory=dict)
    vector: Optional[np.ndarray] = None


@dataclass
class RawHit:
    """What an adapter returns from ``_search``: stored payload, protected vector."""

    id: Any
    score: float
    payload: Optional[Mapping[str, Any]]
    vector: Optional[Sequence[float]] = None


@dataclass
class RawRecord:
    """What an adapter returns from ``_get`` / ``_scroll``: stored payload, protected vector."""

    id: Any
    payload: Optional[Mapping[str, Any]]
    vector: Optional[Sequence[float]] = None


# -- filters --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """A condition on the stored representation.

    ``field`` is the stored field name and ``value`` the stored value: the plaintext value for
    plaintext fields, the blind token (or list of tokens for ``in``/``nin``) for blind fields.
    ``op`` is one of ``OPERATORS``; ``blind`` tells adapters the field holds tokens.
    """

    field: str
    value: Any
    op: str = "eq"
    blind: bool = False


@dataclass(frozen=True)
class And:
    items: tuple[Filter, ...]


@dataclass(frozen=True)
class Or:
    items: tuple[Filter, ...]


Filter = Union[Condition, And, Or]


def compile_where(protector: Protector, where: Optional[Mapping[str, Any]]) -> Optional[Filter]:
    """Translate a ``where`` dict into a :data:`Filter` on stored fields (``None`` if empty)."""
    if not where:
        return None
    return _simplify(_compile(protector, where))


def _compile(protector: Protector, where: Mapping[str, Any]) -> Filter:
    if not isinstance(where, Mapping):
        raise TypeError(f"a filter must be a dict, got {type(where).__name__}")
    items: list[Filter] = []
    for key, value in where.items():
        if key in ("$and", "$or"):
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"{key} needs a non-empty list of filters")
            parts = tuple(_compile(protector, w) for w in value)
            items.append(And(parts) if key == "$and" else Or(parts))
        elif key.startswith("$"):
            raise ValueError(f"unknown filter operator {key!r}")
        elif isinstance(value, Mapping):
            if not value:
                raise ValueError(f"empty condition for field {key!r}")
            for op_key, operand in value.items():
                items.append(_condition(protector, key, op_key, operand))
        else:
            items.append(_condition(protector, key, "$eq", value))
    return And(tuple(items))


def _condition(protector: Protector, name: str, op_key: str, value: Any) -> Condition:
    op = op_key[1:] if op_key.startswith("$") else ""
    if op not in OPERATORS:
        raise ValueError(f"unknown operator {op_key!r} for field {name!r}; use one of {['$' + o for o in OPERATORS]}")
    if op in ("in", "nin"):
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"${op} on {name!r} needs a non-empty list")
        value = list(value)
    policy = protector.policy
    if name in policy.plaintext:
        return Condition(name, value, op)
    if name in policy.blind:
        if op not in _BLIND_OPERATORS:
            raise ValueError(
                f"blind field {name!r} supports only $eq, $ne, $in, $nin: its tokens do not preserve order. "
                "Declare it plaintext if you need ranges."
            )
        stored = protector.metadata.blind_field(name)
        if op in ("in", "nin"):
            return Condition(stored, [protector.metadata.blind_token(name, v) for v in value], op, blind=True)
        return Condition(stored, protector.metadata.blind_token(name, value), op, blind=True)
    raise ValueError(
        f"cannot filter on encrypted field {name!r}; declare it as 'blind' (equality filters) "
        "or 'plaintext' in the MetadataPolicy"
    )


def _simplify(flt: Filter) -> Filter:
    if isinstance(flt, Condition):
        return flt
    items = tuple(_simplify(i) for i in flt.items)
    if len(items) == 1:
        return items[0]
    flat: list[Filter] = []
    for item in items:  # flatten And(And(...)) and Or(Or(...))
        flat.extend(item.items if type(item) is type(flt) else (item,))
    return type(flt)(tuple(flat))


def conditions_of(flt: Optional[Filter]) -> list[Condition]:
    """The conditions of a filter that is a plain AND of conditions (else ``ValueError``)."""
    if flt is None:
        return []
    if isinstance(flt, Condition):
        return [flt]
    if isinstance(flt, And) and all(isinstance(i, Condition) for i in flt.items):
        return list(flt.items)
    raise ValueError("this operation supports only AND-combined conditions, not $or")


def translate_where(protector: Protector, where: Optional[Mapping[str, Any]]) -> list[Condition]:
    """``where`` as a flat list of AND-combined conditions (used by the framework integrations)."""
    return conditions_of(compile_where(protector, where))


# -- the store ------------------------------------------------------------------------------


class ProtectedStore(ABC):
    """Uniform protected CRUD, kNN search and iteration over one database.

    See the module docstring for the filter syntax and :class:`Hit` for score semantics.
    """

    #: Records per write request when ``upsert`` is called without ``batch_size``.
    default_batch_size = 256

    def __init__(self, protector: Protector) -> None:
        self.protector = protector
        self._dims: dict[str, int] = {}
        self._metrics: dict[str, str] = {}

    # -- collection bookkeeping ------------------------------------------------------------

    def register_collection(self, name: str, dim: int, metric: str = "cosine") -> None:
        """Declare an existing collection's native dimension and metric.

        Needed for collections this object did not create: the dimension to decode vectors,
        the metric to report scores consistently (see :class:`Hit`).
        """
        self._check_metric(metric)
        self._dims[name] = dim
        self._metrics[name] = metric

    def _native_dim(self, name: str) -> int:
        try:
            return self._dims[name]
        except KeyError:
            raise KeyError(
                f"native dimension of collection {name!r} is unknown; call "
                "create_collection() or register_collection() first"
            ) from None

    def _metric(self, name: str) -> str:
        return self._metrics.get(name, "cosine")

    @staticmethod
    def _check_metric(metric: str) -> None:
        if metric not in METRICS:
            # Only rotation-invariant metrics are preserved by the transform (not L1, Hamming...).
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")

    def _filter(self, where: Optional[Mapping[str, Any]]) -> Optional[Filter]:
        return compile_where(self.protector, where)

    # -- native hooks implemented by adapters -------------------------------------------------

    @abstractmethod
    def create_collection(self, name: str, dim: int, metric: str = "cosine") -> None:
        """Create a collection for native dimension ``dim`` (stored dimension may be larger)."""

    @abstractmethod
    def drop_collection(self, name: str) -> None: ...

    @abstractmethod
    def _upsert(self, name: str, ids: list, vectors: list[list[float]], payloads: list[dict]) -> None:
        """Write one batch: protected vectors, encrypted payloads. Replaces existing ids."""

    @abstractmethod
    def _search(
        self, name: str, vector: list[float], k: int, flt: Optional[Filter], with_vectors: bool, params: dict
    ) -> list[RawHit]:
        """Top-``k`` by the collection metric, best first, scores converted per :class:`Hit`.

        ``params`` are adapter-specific search options (e.g. ``ef``). Returning protected
        vectors is optional even when ``with_vectors`` is true; the base class fetches the
        missing ones.
        """

    @abstractmethod
    def _get(self, name: str, ids: list, with_vectors: bool) -> list[RawRecord]:
        """Records for ``ids`` that exist (any order)."""

    @abstractmethod
    def _delete(self, name: str, ids: list) -> None: ...

    @abstractmethod
    def _delete_where(self, name: str, flt: Filter) -> None: ...

    @abstractmethod
    def _count(self, name: str, flt: Optional[Filter]) -> int: ...

    @abstractmethod
    def _scroll(self, name: str, flt: Optional[Filter], batch_size: int, with_vectors: bool) -> Iterator[RawRecord]:
        """Every matching record exactly once, fetched ``batch_size`` at a time."""

    def close(self) -> None:
        """Release the underlying client, if the adapter owns one."""

    # -- public API -------------------------------------------------------------------------

    def upsert(
        self,
        name: str,
        ids: Sequence[Any],
        vectors,
        payloads: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Insert or replace records. Vectors are native embeddings, payloads clear dicts."""
        ids = list(ids)
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or len(vectors) != len(ids):
            raise ValueError(f"expected {len(ids)} vectors (one per id), got array of shape {vectors.shape}")
        payloads = list(payloads) if payloads is not None else [None] * len(ids)
        if len(payloads) != len(ids):
            raise ValueError(f"expected {len(ids)} payloads (one per id), got {len(payloads)}")
        size = batch_size or self.default_batch_size
        if size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(ids), size):
            stop = start + size
            self._upsert(
                name,
                ids[start:stop],
                self.protector.encode_many(vectors[start:stop]),
                [self.protector.encrypt_payload(p) for p in payloads[start:stop]],
            )

    def search(
        self,
        name: str,
        vector,
        k: int = 10,
        where: Optional[Mapping[str, Any]] = None,
        *,
        offset: int = 0,
        score_threshold: Optional[float] = None,
        with_vectors: bool = False,
        fields: Optional[Sequence[str]] = None,
        **params: Any,
    ) -> list[Hit]:
        """k-nearest-neighbour search.

        ``offset`` skips the first results (paging); ``score_threshold`` keeps hits with a
        score of at least the threshold (``cosine``/``dot``) or at most it (``l2``);
        ``fields`` limits which payload fields are decrypted and returned; extra keyword
        arguments are adapter-specific search options.
        """
        if k < 1 or offset < 0:
            raise ValueError("k must be positive and offset non-negative")
        raw = self._search(
            name, self.protector.encode_list(vector), k + offset, self._filter(where), with_vectors, params
        )[offset : offset + k]
        if score_threshold is not None:
            if self._metric(name) == "l2":
                raw = [h for h in raw if h.score <= score_threshold]
            else:
                raw = [h for h in raw if h.score >= score_threshold]
        vectors = self._vectors_for(name, raw) if with_vectors else {}
        return [
            Hit(h.id, float(h.score), self.protector.decrypt_payload(h.payload, fields), vectors.get(_key(h.id)))
            for h in raw
        ]

    def search_many(self, name: str, vectors, k: int = 10, where=None, **options: Any) -> list[list[Hit]]:
        """Run :meth:`search` for each query vector (same options for all)."""
        return [self.search(name, v, k, where, **options) for v in np.asarray(vectors, dtype=np.float64)]

    def get(
        self, name: str, ids: Sequence[Any], with_vectors: bool = False, fields: Optional[Sequence[str]] = None
    ) -> list[Record]:
        """Fetch records by id (missing ids are skipped)."""
        ids = list(ids)
        if not ids:
            return []
        return [self._reveal(name, r, with_vectors, fields) for r in self._get(name, ids, with_vectors)]

    def scroll(
        self,
        name: str,
        where: Optional[Mapping[str, Any]] = None,
        *,
        batch_size: int = 256,
        with_vectors: bool = False,
        fields: Optional[Sequence[str]] = None,
    ) -> Iterator[Record]:
        """Iterate over every record (optionally filtered), ``batch_size`` per request."""
        for raw in self._scroll(name, self._filter(where), batch_size, with_vectors):
            yield self._reveal(name, raw, with_vectors, fields)

    def count(self, name: str, where: Optional[Mapping[str, Any]] = None) -> int:
        return self._count(name, self._filter(where))

    def delete(
        self, name: str, ids: Optional[Sequence[Any]] = None, *, where: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Delete by ids, or every record matching ``where`` (exactly one of the two)."""
        if (ids is None) == (where is None):
            raise ValueError("pass either ids or where")
        if ids is not None:
            if ids:
                self._delete(name, list(ids))
        else:
            flt = self._filter(where)
            if flt is None:
                raise ValueError("where must not be empty; use drop_collection to delete everything")
            self._delete_where(name, flt)

    def update_payload(self, name: str, id: Any, payload: Mapping[str, Any], *, merge: bool = True) -> None:
        """Change a record's metadata, keeping its vector.

        With ``merge=True`` the given fields are added or replaced and the others kept;
        with ``merge=False`` the payload is replaced. Values are re-encrypted with fresh nonces.
        """
        records = self.get(name, [id], with_vectors=True)
        if not records:
            raise KeyError(f"record {id!r} not found in {name!r}")
        record = records[0]
        new_payload = {**record.payload, **payload} if merge else dict(payload)
        self.upsert(name, [id], record.vector[None, :], [new_payload])

    def copy_to(
        self,
        name: str,
        target: ProtectedStore,
        target_name: Optional[str] = None,
        *,
        where: Optional[Mapping[str, Any]] = None,
        batch_size: int = 256,
    ) -> int:
        """Copy records into ``target`` (another store, database, or key), returning the count.

        Records are decrypted with this store's key and re-protected with the target's, which
        makes this the tool for key rotation and for moving between databases. The target
        collection must exist.
        """
        target_name = target_name or name
        copied = 0
        ids: list = []
        vectors: list = []
        payloads: list = []
        for record in self.scroll(name, where, batch_size=batch_size, with_vectors=True):
            ids.append(record.id)
            vectors.append(record.vector)
            payloads.append(record.payload)
            if len(ids) == batch_size:
                target.upsert(target_name, ids, np.stack(vectors), payloads, batch_size=batch_size)
                copied += len(ids)
                ids, vectors, payloads = [], [], []
        if ids:
            target.upsert(target_name, ids, np.stack(vectors), payloads, batch_size=batch_size)
            copied += len(ids)
        return copied

    # -- helpers --------------------------------------------------------------------------------

    def _reveal(self, name: str, raw: RawRecord, with_vectors: bool, fields) -> Record:
        vector = None
        if with_vectors:
            if raw.vector is None:
                raise RuntimeError(f"{type(self).__name__} did not return the vector of {raw.id!r}")
            vector = self.protector.decode(np.asarray(raw.vector, dtype=np.float64), self._native_dim(name))
        return Record(raw.id, self.protector.decrypt_payload(raw.payload, fields), vector)

    def _vectors_for(self, name: str, hits: list[RawHit]) -> dict:
        dim = self._native_dim(name)
        out = {
            _key(h.id): self.protector.decode(np.asarray(h.vector, dtype=np.float64), dim)
            for h in hits
            if h.vector is not None
        }
        missing = [h.id for h in hits if _key(h.id) not in out]
        if missing:
            for r in self._get(name, missing, True):
                out[_key(r.id)] = self.protector.decode(np.asarray(r.vector, dtype=np.float64), dim)
        return out


def _key(rid: Any) -> str:
    return str(rid)
