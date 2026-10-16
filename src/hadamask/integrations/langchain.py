"""LangChain integration: works with any LangChain ``VectorStore``.

Two levels of protection:

* :class:`ProtectedEmbeddings` - wrap your ``Embeddings``; every vector (documents and
  queries) is transformed. Texts and metadata are stored **in clear** by the vector store.
  Use it only when the vectors are the sole concern.
* :class:`ProtectedVectorStore` - wrap a vector store; vectors are transformed AND
  ``page_content`` and metadata are encrypted per field with AES-256-GCM. The inner store
  only ever sees ciphertexts and transformed vectors.

Never combine the two (vectors would be transformed twice): ``ProtectedVectorStore`` takes
the plain ``Embeddings``.

Example::

    store = ProtectedVectorStore(
        build_inner=lambda emb: QdrantVectorStore(client, "docs", embedding=emb),
        embedding=OpenAIEmbeddings(),
        protector=protector,
    )
    store.add_texts(["..."], metadatas=[{"client": "ACME"}])
    store.similarity_search("query", k=4, filter={"client": "ACME"})

Filters use the hadamask syntax (``{"client": "ACME", "year": {"$gte": 2021}}``; operators
``$eq $ne $in $nin`` on blind fields, plus ``$gt $gte $lt $lte`` on plaintext fields;
``$or`` is not supported here). They are translated to stored names/tokens and handed to
the inner store through ``filter_builder`` (a callable receiving a list of
:class:`~hadamask.adapters.base.Condition`). The default builds a callable for
``InMemoryVectorStore`` and a MongoDB-style dict for any other store; pass a builder for
stores with their own filter syntax (e.g. Qdrant).
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Iterable, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from ..adapters.base import Condition
from ..core import Protector
from . import translate_where

PAGE_CONTENT_FIELD = "page_content"

FilterBuilder = Callable[[list[Condition]], Any]


class ProtectedEmbeddings(Embeddings):
    """Vector-only protection: transforms the output of an inner ``Embeddings``.

    Texts and metadata given to the vector store are NOT encrypted with this class.
    """

    def __init__(self, inner: Embeddings, protector: Protector) -> None:
        self.inner = inner
        self.protector = protector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.protector.encode_many(self.inner.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.protector.encode_list(self.inner.embed_query(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.protector.encode_many(await self.inner.aembed_documents(texts))

    async def aembed_query(self, text: str) -> list[float]:
        return self.protector.encode_list(await self.inner.aembed_query(text))


class _PrecomputedEmbeddings(Embeddings):
    """Embeddings handed to the inner store: returns vectors the wrapper registered.

    The wrapper registers ``ciphertext -> protected vector`` right before calling the inner
    store's ``add_texts``. Outside of a write, unknown texts (dimension probes such as
    ``"dummy_text"`` that some stores send at construction) get a unit placeholder vector of
    the right size. During a write, an unknown text means the inner store is embedding
    something the wrapper did not prepare; that raises instead of storing a wrong vector.
    """

    def __init__(self, stored_dim: Callable[[], int]) -> None:
        self._stored_dim = stored_dim
        self._pending: dict[str, list[float]] = {}

    def _placeholder(self) -> list[float]:
        vec = [0.0] * self._stored_dim()
        vec[0] = 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._pending:
            return [self._placeholder() for _ in texts]
        missing = [t for t in texts if t not in self._pending]
        if missing:
            raise RuntimeError(
                f"the inner vector store asked to embed {len(missing)} text(s) that hadamask did not "
                "prepare; refusing to store placeholder vectors"
            )
        return [self._pending[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        # The wrapper always searches by vector; only probes reach this method.
        return self._placeholder()


_PY_OPS = {
    "eq": lambda stored, v: stored == v or (isinstance(stored, list) and v in stored),
    "ne": lambda stored, v: not (stored == v or (isinstance(stored, list) and v in stored)),
    "in": lambda stored, v: stored in v or (isinstance(stored, list) and any(x in v for x in stored)),
    "nin": lambda stored, v: not (stored in v or (isinstance(stored, list) and any(x in v for x in stored))),
    "gt": lambda stored, v: stored is not None and stored > v,
    "gte": lambda stored, v: stored is not None and stored >= v,
    "lt": lambda stored, v: stored is not None and stored < v,
    "lte": lambda stored, v: stored is not None and stored <= v,
}


def _default_filter_builder(inner: VectorStore) -> FilterBuilder:
    """Callable filter for ``InMemoryVectorStore``; MongoDB-style dict for any other store.

    The dict form (``{"year": {"$gte": 2021}, "client_token_field": "..."}``) is what
    Chroma, MongoDB, PGVector and several other LangChain stores accept; pass your own
    ``filter_builder`` for stores with another syntax.
    """
    if isinstance(inner, InMemoryVectorStore):

        def build(conditions: list[Condition]):
            def match(doc: Document) -> bool:
                return all(_PY_OPS[c.op](doc.metadata.get(c.field), c.value) for c in conditions)

            return match

        return build

    def build_dict(conditions: list[Condition]) -> dict:
        out: dict = {}
        for c in conditions:
            if c.op == "eq" and c.field not in out:
                out[c.field] = c.value
            else:
                existing = out.get(c.field)
                ops = existing if isinstance(existing, dict) else ({} if existing is None else {"$eq": existing})
                ops["$" + c.op] = c.value
                out[c.field] = ops
        return out

    return build_dict


class ProtectedVectorStore(VectorStore):
    """Full protection around any LangChain vector store.

    Parameters
    ----------
    build_inner:
        ``Callable[[Embeddings], VectorStore]`` that constructs the inner store with the
        embeddings object it is given (that object serves the precomputed, protected
        vectors - the inner store must not be given your real embeddings).
    embedding:
        Your real (plain) ``Embeddings``.
    protector:
        The tenant :class:`~hadamask.Protector`.
    dim:
        Native embedding dimension; probed once from ``embedding`` if omitted.
    filter_builder:
        Translates stored-field equality conditions into the inner store's filter format.
    """

    def __init__(
        self,
        build_inner: Callable[[Embeddings], VectorStore],
        embedding: Embeddings,
        protector: Protector,
        dim: Optional[int] = None,
        filter_builder: Optional[FilterBuilder] = None,
    ) -> None:
        if isinstance(embedding, ProtectedEmbeddings):
            raise ValueError("pass the plain Embeddings; ProtectedVectorStore transforms vectors itself")
        self.embedding = embedding
        self.protector = protector
        self._dim = dim
        self._precomputed = _PrecomputedEmbeddings(lambda: self.protector.stored_dim(self.dim))
        self.inner = build_inner(self._precomputed)
        self.filter_builder = filter_builder or _default_filter_builder(self.inner)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embedding.embed_query("dimension probe"))
        return self._dim

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    # -- protection helpers ---------------------------------------------------------------

    def _reveal(self, doc: Document) -> Document:
        text = doc.page_content
        if text:
            text = self.protector.metadata.decrypt_value(PAGE_CONTENT_FIELD, text)
        metadata = self.protector.decrypt_payload(doc.metadata)
        # Some stores (e.g. langchain_qdrant) report the id only as metadata["_id"].
        doc_id = doc.id if doc.id is not None else metadata.get("_id")
        return Document(id=None if doc_id is None else str(doc_id), page_content=text, metadata=metadata)

    def _native_filter(self, filter: Any) -> Any:
        if filter is None:
            return None
        if callable(filter) and isinstance(self.inner, InMemoryVectorStore):
            # A callable is evaluated client-side on the decrypted document.
            return lambda doc: filter(self._reveal(doc))
        if isinstance(filter, dict):
            return self.filter_builder(translate_where(self.protector, filter))
        raise TypeError("filter must be a {field: value} dict (or a callable for InMemoryVectorStore)")

    # -- writes ----------------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        if not texts:
            return []
        metadatas = metadatas or [{} for _ in texts]
        ids = list(ids) if ids else [str(uuid.uuid4()) for _ in texts]
        if not (len(metadatas) == len(ids) == len(texts)):
            raise ValueError("texts, metadatas and ids must have the same length")

        plain = self.embedding.embed_documents(texts)
        self._dim = self._dim or len(plain[0])
        vectors = self.protector.encode_many(plain)
        ciphertexts = [self.protector.metadata.encrypt_value(PAGE_CONTENT_FIELD, t) for t in texts]
        encrypted = [self.protector.encrypt_payload(m) for m in metadatas]
        for ct, vec in zip(ciphertexts, vectors, strict=True):
            self._precomputed._pending[ct] = vec
        try:
            return self.inner.add_texts(ciphertexts, metadatas=encrypted, ids=ids, **kwargs)
        finally:
            for ct in ciphertexts:
                self._precomputed._pending.pop(ct, None)

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        if "ids" not in kwargs and any(d.id for d in documents):
            kwargs["ids"] = [d.id or str(uuid.uuid4()) for d in documents]
        return self.add_texts(
            [d.page_content for d in documents], metadatas=[dict(d.metadata) for d in documents], **kwargs
        )

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        return self.inner.delete(ids=ids, **kwargs)

    # -- reads -----------------------------------------------------------------------------

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        return [self._reveal(d) for d in self.inner.get_by_ids(ids)]

    def _encode_query(self, query: str) -> list[float]:
        vector = self.embedding.embed_query(query)
        self._dim = self._dim or len(vector)
        return self.protector.encode_list(vector)

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, filter: Any = None, **kwargs: Any
    ) -> list[Document]:
        """``embedding`` is a plain (untransformed) query vector."""
        docs = self.inner.similarity_search_by_vector(
            self.protector.encode_list(embedding), k=k, filter=self._native_filter(filter), **kwargs
        )
        return [self._reveal(d) for d in docs]

    def similarity_search(self, query: str, k: int = 4, filter: Any = None, **kwargs: Any) -> list[Document]:
        docs = self.inner.similarity_search_by_vector(
            self._encode_query(query), k=k, filter=self._native_filter(filter), **kwargs
        )
        return [self._reveal(d) for d in docs]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Any = None, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        search = getattr(self.inner, "similarity_search_with_score_by_vector", None)
        if search is None:
            raise NotImplementedError(f"{type(self.inner).__name__} has no similarity_search_with_score_by_vector")
        pairs = search(self._encode_query(query), k=k, filter=self._native_filter(filter), **kwargs)
        return [(self._reveal(d), s) for d, s in pairs]

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, filter: Any = None, **kwargs: Any
    ) -> list[Document]:
        # MMR only uses similarities, which the transform preserves exactly.
        docs = self.inner.max_marginal_relevance_search_by_vector(
            self._encode_query(query),
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=self._native_filter(filter),
            **kwargs,
        )
        return [self._reveal(d) for d in docs]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self.inner._select_relevance_score_fn()

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        protector: Optional[Protector] = None,
        build_inner: Optional[Callable[[Embeddings], VectorStore]] = None,
        **kwargs: Any,
    ) -> ProtectedVectorStore:
        if protector is None or build_inner is None:
            raise TypeError("ProtectedVectorStore.from_texts requires protector= and build_inner=")
        store = cls(build_inner, embedding, protector, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store
