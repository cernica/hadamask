"""LlamaIndex integration: works with any LlamaIndex vector store.

Two levels of protection (never combine them - vectors would be transformed twice):

* :class:`ProtectedEmbedding` - wrap your embedding model; vectors are transformed, while
  node text and metadata are stored in clear. Vector-only protection.
* :class:`ProtectedVectorStore` - wrap a vector store and keep your plain embedding model in
  ``Settings.embed_model``. Node embeddings are transformed, node text and metadata are
  encrypted per field with AES-256-GCM before the inner store serializes anything, and
  query results are decrypted on the way back.

Example::

    store = ProtectedVectorStore(QdrantVectorStore(client=client, collection_name="docs"), protector, embed_dim=1536)
    index = VectorStoreIndex.from_documents(docs, storage_context=StorageContext.from_defaults(vector_store=store))
    index.as_query_engine(filters=MetadataFilters(filters=[MetadataFilter(key="client", value="ACME")]))

Text location: if the inner store keeps node text (``stores_text=True``, e.g. Qdrant,
Weaviate, Milvus, pgvector), everything sent to the database is protected. If it does not
(e.g. ``SimpleVectorStore``), LlamaIndex keeps the original nodes in its docstore, which must
then live in your trusted environment; only vectors and metadata reach the vector store.

Filters: ``EQ``/``NE``/``IN``/``NIN`` work on blind fields (values are mapped to tokens);
plaintext fields accept every operator; encrypted-only fields cannot be filtered.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, RelatedNodeInfo
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

from ..core import Protector
from ..metadata import RESERVED_PREFIX

TEXT_FIELD = "text"
_BLIND_OPERATORS = {FilterOperator.EQ, FilterOperator.NE, FilterOperator.IN, FilterOperator.NIN}
_UNSUPPORTED_MODES = {
    VectorStoreQueryMode.SPARSE,
    VectorStoreQueryMode.HYBRID,
    VectorStoreQueryMode.TEXT_SEARCH,
    VectorStoreQueryMode.SEMANTIC_HYBRID,
}


class ProtectedEmbedding(BaseEmbedding):
    """Vector-only protection: transforms the output of an inner embedding model."""

    _inner: BaseEmbedding = PrivateAttr()
    _protector: Protector = PrivateAttr()

    def __init__(self, inner: BaseEmbedding, protector: Protector, **kwargs: Any) -> None:
        super().__init__(model_name=f"hadamask:{inner.model_name}", embed_batch_size=inner.embed_batch_size, **kwargs)
        self._inner = inner
        self._protector = protector

    @classmethod
    def class_name(cls) -> str:
        return "HadamaskProtectedEmbedding"

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._protector.encode_list(self._inner.get_query_embedding(query))

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._protector.encode_list(await self._inner.aget_query_embedding(query))

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._protector.encode_list(self._inner.get_text_embedding(text))

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return self._protector.encode_many(self._inner.get_text_embedding_batch(texts))

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return self._protector.encode_list(await self._inner.aget_text_embedding(text))


class ProtectedVectorStore(BasePydanticVectorStore):
    """Full protection around any LlamaIndex vector store.

    Nodes must carry PLAIN embeddings (use your normal embedding model, not
    :class:`ProtectedEmbedding`). Pass ``embed_dim`` to reject vectors of the wrong size,
    which also catches accidental double transformation when ``embed_dim`` is not a power
    of two.
    """

    stores_text: bool = True
    flat_metadata: bool = True

    _inner: BasePydanticVectorStore = PrivateAttr()
    _protector: Protector = PrivateAttr()
    _embed_dim: Optional[int] = PrivateAttr(default=None)

    def __init__(self, inner: BasePydanticVectorStore, protector: Protector, embed_dim: Optional[int] = None) -> None:
        super().__init__(stores_text=inner.stores_text, flat_metadata=getattr(inner, "flat_metadata", True))
        self._inner = inner
        self._protector = protector
        self._embed_dim = embed_dim

    @classmethod
    def class_name(cls) -> str:
        return "HadamaskProtectedVectorStore"

    @property
    def client(self) -> Any:
        return self._inner.client

    @property
    def inner(self) -> BasePydanticVectorStore:
        return self._inner

    # -- node protection -------------------------------------------------------------------

    def _check_dim(self, vector: Sequence[float]) -> None:
        if self._embed_dim is not None and len(vector) != self._embed_dim:
            raise ValueError(
                f"expected a plain embedding of size {self._embed_dim}, got {len(vector)}; "
                "do not combine ProtectedEmbedding with ProtectedVectorStore"
            )

    def _protect(self, node: BaseNode) -> BaseNode:
        embedding = node.get_embedding()
        self._check_dim(embedding)
        metadata = self._protector.encrypt_payload(node.metadata)
        hidden = [k for k in metadata if k.startswith(RESERVED_PREFIX)]
        update: dict[str, Any] = {
            "embedding": self._protector.encode_list(embedding),
            "metadata": metadata,
            "excluded_embed_metadata_keys": list(node.excluded_embed_metadata_keys) + hidden,
            "excluded_llm_metadata_keys": list(node.excluded_llm_metadata_keys) + hidden,
        }
        if hasattr(node, "text"):
            update["text"] = self._protector.metadata.encrypt_value(TEXT_FIELD, node.text)
        update["relationships"] = {
            k: self._map_related(v, self._protect_related) for k, v in node.relationships.items()
        }
        return node.model_copy(update=update, deep=True)

    @staticmethod
    def _map_related(value, fn):
        return [fn(r) for r in value] if isinstance(value, list) else fn(value)

    def _protect_related(self, info: RelatedNodeInfo) -> RelatedNodeInfo:
        # Relationships embed the parent document's metadata and a hash of its plaintext;
        # encrypt the former and drop the latter (a plaintext hash allows guess confirmation).
        return info.model_copy(update={"metadata": self._protector.encrypt_payload(info.metadata), "hash": None})

    def _reveal_related(self, info: RelatedNodeInfo) -> RelatedNodeInfo:
        return info.model_copy(update={"metadata": self._protector.decrypt_payload(info.metadata)})

    def _reveal(self, node: BaseNode) -> BaseNode:
        update: dict[str, Any] = {
            "metadata": self._protector.decrypt_payload(node.metadata),
            "excluded_embed_metadata_keys": [
                k for k in node.excluded_embed_metadata_keys if not k.startswith(RESERVED_PREFIX)
            ],
            "excluded_llm_metadata_keys": [
                k for k in node.excluded_llm_metadata_keys if not k.startswith(RESERVED_PREFIX)
            ],
            "embedding": None,
            "relationships": {k: self._map_related(v, self._reveal_related) for k, v in node.relationships.items()},
        }
        text = getattr(node, "text", None)
        if text:
            update["text"] = self._protector.metadata.decrypt_value(TEXT_FIELD, text)
        return node.model_copy(update=update)

    def _translate(self, filters: Optional[MetadataFilters]) -> Optional[MetadataFilters]:
        if filters is None:
            return None
        policy = self._protector.policy
        out: list[Any] = []
        for f in filters.filters:
            if isinstance(f, MetadataFilters):
                out.append(self._translate(f))
            elif f.key in policy.plaintext:
                out.append(f)
            elif f.key in policy.blind:
                if f.operator not in _BLIND_OPERATORS:
                    raise ValueError(f"blind field {f.key!r} only supports ==, !=, in, nin (got {f.operator})")
                field = self._protector.metadata.blind_field(f.key)
                if isinstance(f.value, list):
                    value: Any = [self._protector.blind(f.key, v)[1] for v in f.value]
                else:
                    value = self._protector.blind(f.key, f.value)[1]
                out.append(MetadataFilter(key=field, value=value, operator=f.operator))
            else:
                raise ValueError(f"cannot filter on encrypted field {f.key!r}; declare it as 'blind' or 'plaintext'")
        return MetadataFilters(filters=out, condition=filters.condition)

    # -- vector store API ------------------------------------------------------------------

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        return self._inner.add([self._protect(n) for n in nodes], **add_kwargs)

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        return await self._inner.async_add([self._protect(n) for n in nodes], **kwargs)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._inner.delete(ref_doc_id, **delete_kwargs)

    def delete_nodes(
        self, node_ids: Optional[List[str]] = None, filters: Optional[MetadataFilters] = None, **kwargs: Any
    ) -> None:
        self._inner.delete_nodes(node_ids=node_ids, filters=self._translate(filters), **kwargs)

    def clear(self) -> None:
        self._inner.clear()

    def get_nodes(
        self, node_ids: Optional[List[str]] = None, filters: Optional[MetadataFilters] = None
    ) -> List[BaseNode]:
        return [self._reveal(n) for n in self._inner.get_nodes(node_ids=node_ids, filters=self._translate(filters))]

    def _protected_query(self, query: VectorStoreQuery) -> VectorStoreQuery:
        if query.mode in _UNSUPPORTED_MODES:
            raise NotImplementedError(f"query mode {query.mode} needs plaintext text; only dense modes are supported")
        if query.query_embedding is None:
            raise ValueError("ProtectedVectorStore needs query.query_embedding")
        self._check_dim(query.query_embedding)
        return VectorStoreQuery(
            **{
                **query.__dict__,
                "query_embedding": self._protector.encode_list(query.query_embedding),
                "query_str": None,  # never send the clear query text to the database
                "filters": self._translate(query.filters),
            }
        )

    def _revealed(self, result: VectorStoreQueryResult) -> VectorStoreQueryResult:
        nodes = [self._reveal(n) for n in result.nodes] if result.nodes is not None else None
        return VectorStoreQueryResult(nodes=nodes, similarities=result.similarities, ids=result.ids)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        return self._revealed(self._inner.query(self._protected_query(query), **kwargs))

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        return self._revealed(await self._inner.aquery(self._protected_query(query), **kwargs))
