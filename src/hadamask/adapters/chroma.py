"""Chroma adapter (chromadb 1.x HTTP client).

Chroma-specific choices:

* Collections are created with ``embedding_function=None`` so Chroma never embeds text
  itself; vectors are always supplied, already protected.
* The whole payload goes to ``metadatas`` (Chroma's ``documents`` field is not used, so
  no plaintext reaches its full-text index). Metadata values must be scalars or arrays of
  scalars; encrypted values are strings, so any payload works, but plaintext fields must
  respect Chroma's type rules.
* List values are refused for plaintext and blind fields (encrypted lists are fine, they
  become one string). Chroma 1.5.9 attaches array metadata from deleted collections to
  records of unrelated collections (observed against chromadb/chroma:1.5.9), and ``$eq``
  filters would not match array elements anyway.
* Filters map one-to-one to Chroma's ``where`` operators; ``$ne``/``$nin`` match records
  without the field, as in MongoDB. ``scroll`` pages with ``limit``/``offset`` in insertion
  order.
* Scores follow :class:`~hadamask.adapters.base.Hit`: ``1 - distance`` for cosine and dot
  (Chroma reports ``1 - <q, v>`` for ``ip``) and the Euclidean distance for l2 (Chroma
  reports it squared).
* ``search`` takes no adapter-specific options.
* Chroma's HNSW construction is nondeterministic, so rankings may vary slightly between
  re-indexings of the same clear corpus (paper, Section 5.7), with or without protection.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Optional

from ..core import Protector
from .base import And, Condition, Filter, ProtectedStore, RawHit, RawRecord

_SPACE = {"cosine": "cosine", "dot": "ip", "l2": "l2"}


def _score(distance: float, space: Optional[str]) -> float:
    # Chroma reports squared L2, and 1 - similarity for cosine and ip.
    return math.sqrt(max(distance, 0.0)) if space == "l2" else 1.0 - distance


def compile_filter(flt: Optional[Filter]) -> Optional[dict]:
    """Translate a hadamask :data:`~hadamask.adapters.base.Filter` into a Chroma ``where``."""
    if flt is None:
        return None
    if isinstance(flt, Condition):
        return {flt.field: {f"${flt.op}": flt.value}}
    return {"$and" if isinstance(flt, And) else "$or": [compile_filter(i) for i in flt.items]}


def _vector(embeddings, i: int, with_vectors: bool):
    if not with_vectors or embeddings is None:
        return None
    return [float(x) for x in embeddings[i]]


class ChromaStore(ProtectedStore):
    def __init__(self, client, protector: Protector) -> None:
        super().__init__(protector)
        self.client = client
        self._spaces: dict[str, Optional[str]] = {}

    def _collection(self, name: str):
        return self.client.get_collection(name, embedding_function=None)

    def _space(self, name: str, collection) -> Optional[str]:
        if name not in self._spaces:
            self._spaces[name] = ((collection.configuration or {}).get("hnsw") or {}).get("space")
        return self._spaces[name]

    def create_collection(self, name: str, dim: int, metric: str = "cosine") -> None:
        self._check_metric(metric)
        self.client.create_collection(name, configuration={"hnsw": {"space": _SPACE[metric]}}, embedding_function=None)
        self._spaces[name] = _SPACE[metric]
        self.register_collection(name, dim, metric)

    def drop_collection(self, name: str) -> None:
        self.client.delete_collection(name)
        self._spaces.pop(name, None)

    def _upsert(self, name, ids, vectors, payloads) -> None:
        for m in payloads:
            for field, value in m.items():
                if isinstance(value, (list, tuple)):
                    raise ValueError(f"Chroma adapter does not store list values in clear (field {field!r})")
        # Chroma rejects empty metadata dicts; send None for records without payload.
        self._collection(name).upsert(
            ids=[str(i) for i in ids], embeddings=vectors, metadatas=[m or None for m in payloads]
        )

    def _search(self, name, vector, k, flt, with_vectors, params) -> list[RawHit]:
        if params:
            raise TypeError(f"ChromaStore.search takes no adapter options, got {sorted(params)}")
        collection = self._collection(name)
        include = ["metadatas", "distances"] + (["embeddings"] if with_vectors else [])
        result = collection.query(query_embeddings=[vector], n_results=k, where=compile_filter(flt), include=include)
        space = self._space(name, collection)
        embeddings = result["embeddings"][0] if with_vectors else None
        return [
            RawHit(i, _score(d, space), m, _vector(embeddings, n, with_vectors))
            for n, (i, d, m) in enumerate(
                zip(result["ids"][0], result["distances"][0], result["metadatas"][0], strict=True)
            )
        ]

    def _get(self, name, ids, with_vectors) -> list[RawRecord]:
        include = ["metadatas", "embeddings"] if with_vectors else ["metadatas"]
        result = self._collection(name).get(ids=[str(i) for i in ids], include=include)
        return self._records(result, with_vectors)

    def _records(self, result, with_vectors: bool) -> list[RawRecord]:
        embeddings = result["embeddings"] if with_vectors else None
        return [
            RawRecord(i, m, _vector(embeddings, n, with_vectors))
            for n, (i, m) in enumerate(zip(result["ids"], result["metadatas"], strict=True))
        ]

    def _delete(self, name, ids) -> None:
        self._collection(name).delete(ids=[str(i) for i in ids])

    def _delete_where(self, name, flt) -> None:
        self._collection(name).delete(where=compile_filter(flt))

    def _count(self, name, flt) -> int:
        collection = self._collection(name)
        if flt is None:
            return collection.count()
        return len(collection.get(where=compile_filter(flt), include=[])["ids"])

    def _scroll(self, name, flt, batch_size, with_vectors) -> Iterator[RawRecord]:
        collection = self._collection(name)
        include = ["metadatas", "embeddings"] if with_vectors else ["metadatas"]
        offset = 0
        while True:
            result = collection.get(where=compile_filter(flt), limit=batch_size, offset=offset, include=include)
            yield from self._records(result, with_vectors)
            if len(result["ids"]) < batch_size:
                return
            offset += len(result["ids"])
