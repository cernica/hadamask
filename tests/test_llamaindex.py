import hashlib
import json
from typing import List

import numpy as np
import pytest

pytest.importorskip("llama_index.core")

from conftest import service_up
from contract import make_protector
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import Document, TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from hadamask.integrations.llamaindex import ProtectedEmbedding, ProtectedVectorStore
from hadamask.metadata import is_ciphertext

DIM = 384


class HashEmbedding(BaseEmbedding):
    """Deterministic pseudo-embeddings: same text -> same unit vector."""

    def _vec(self, text: str) -> List[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        v = np.random.default_rng(seed).standard_normal(DIM)
        return (v / np.linalg.norm(v)).tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._vec(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._vec(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._vec(text)


EMB = HashEmbedding()
TEXTS = [f"Loan agreement {i}: principal {50_000 + 113 * i} EUR, IBAN RO49BBBB{i:016d}" for i in range(50)]
METAS = [{"client": f"client-{i % 4}", "year": 2020 + i % 3, "amount": 50_000 + 113 * i} for i in range(50)]


def make_nodes():
    return [
        TextNode(id_=f"node-{i}", text=t, metadata=dict(m), embedding=EMB.get_text_embedding(t))
        for i, (t, m) in enumerate(zip(TEXTS, METAS))
    ]


def query(text, k=6, filters=None):
    return VectorStoreQuery(query_embedding=EMB.get_query_embedding(text), similarity_top_k=k, filters=filters)


def test_simple_store_same_results_and_ciphertext():
    plain = SimpleVectorStore()
    plain.add(make_nodes())
    store = ProtectedVectorStore(SimpleVectorStore(), make_protector(), embed_dim=DIM)
    store.add(make_nodes())
    for t in (TEXTS[2], TEXTS[31], "unrelated"):
        a, b = plain.query(query(t)), store.query(query(t))
        assert a.ids == b.ids
        assert np.allclose(a.similarities, b.similarities, atol=1e-5)
    dump = json.dumps(store.inner.to_dict())
    assert "IBAN RO49" not in dump and "client-1" not in dump
    meta = store.inner.data.metadata_dict["node-0"]
    assert is_ciphertext(meta["client"]) and is_ciphertext(meta["amount"]) and meta["year"] == 2020
    assert len(store.inner.data.embedding_dict["node-0"]) == 512


def test_simple_store_filters():
    store = ProtectedVectorStore(SimpleVectorStore(), make_protector())
    store.add(make_nodes())
    by_id = {f"node-{i}": m for i, m in enumerate(METAS)}
    f = MetadataFilters(filters=[MetadataFilter(key="client", value="client-2")])
    res = store.query(query(TEXTS[0], filters=f))
    assert len(res.ids) == 6 and all(by_id[i]["client"] == "client-2" for i in res.ids)
    f = MetadataFilters(
        filters=[
            MetadataFilter(key="client", value=["client-1", "client-3"], operator=FilterOperator.IN),
            MetadataFilter(key="year", value=2021, operator=FilterOperator.GTE),
        ]
    )
    res = store.query(query(TEXTS[0], k=50, filters=f))
    assert res.ids and all(by_id[i]["client"] in ("client-1", "client-3") and by_id[i]["year"] >= 2021 for i in res.ids)
    with pytest.raises(ValueError):
        store.query(query(TEXTS[0], filters=MetadataFilters(filters=[MetadataFilter(key="amount", value=1)])))


def test_double_transform_guard_and_vector_only():
    protector = make_protector()
    store = ProtectedVectorStore(SimpleVectorStore(), protector, embed_dim=DIM)
    wrapped = ProtectedEmbedding(EMB, protector)
    node = TextNode(text="x", embedding=wrapped.get_text_embedding("x"))
    with pytest.raises(ValueError):
        store.add([node])
    # ProtectedEmbedding alone: rankings unchanged.
    plain, vec_only = SimpleVectorStore(), SimpleVectorStore()
    plain.add(make_nodes())
    vec_only.add(
        [TextNode(id_=f"node-{i}", text=t, embedding=wrapped.get_text_embedding(t)) for i, t in enumerate(TEXTS)]
    )
    q = VectorStoreQuery(query_embedding=wrapped.get_query_embedding(TEXTS[7]), similarity_top_k=5)
    assert vec_only.query(q).ids == plain.query(query(TEXTS[7], k=5)).ids


def plain_top(text, k=3):
    index = VectorStoreIndex.from_documents(
        [Document(text=t, metadata=m) for t, m in zip(TEXTS, METAS)], embed_model=EMB
    )
    return [h.node.get_content() for h in index.as_retriever(similarity_top_k=k).retrieve(text)]


def test_index_end_to_end_simple():
    Settings.embed_model = EMB
    store = ProtectedVectorStore(SimpleVectorStore(), make_protector(), embed_dim=DIM)
    index = VectorStoreIndex.from_documents(
        [Document(text=t, metadata=m) for t, m in zip(TEXTS, METAS)],
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=EMB,
    )
    hits = index.as_retriever(similarity_top_k=3).retrieve(TEXTS[12])
    assert [h.node.get_content() for h in hits] == plain_top(TEXTS[12])


@pytest.mark.integration
def test_qdrant_backend_end_to_end():
    pytest.importorskip("llama_index.vector_stores.qdrant")
    if not service_up(6333):
        pytest.skip("qdrant is not running")
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    client = QdrantClient(url="http://127.0.0.1:6333")
    name = "hadamask_llamaindex"
    if client.collection_exists(name):
        client.delete_collection(name)
    try:
        store = ProtectedVectorStore(
            QdrantVectorStore(client=client, collection_name=name), make_protector(), embed_dim=DIM
        )
        index = VectorStoreIndex.from_documents(
            [Document(text=t, metadata=m) for t, m in zip(TEXTS, METAS)],
            storage_context=StorageContext.from_defaults(vector_store=store),
            embed_model=EMB,
        )
        hits = index.as_retriever(similarity_top_k=3).retrieve(TEXTS[12])
        assert [h.node.get_content() for h in hits] == plain_top(TEXTS[12])
        assert hits[0].node.metadata == METAS[TEXTS.index(hits[0].node.get_content())]
        f = MetadataFilters(filters=[MetadataFilter(key="client", value="client-3")])
        hits = index.as_retriever(similarity_top_k=4, filters=f).retrieve(TEXTS[0])
        assert len(hits) == 4 and all(h.node.metadata["client"] == "client-3" for h in hits)
        points, _ = client.scroll(name, limit=100, with_payload=True)
        dump = json.dumps([p.payload for p in points])
        assert points and "IBAN RO49" not in dump and "client-1" not in dump and "Loan agreement" not in dump
    finally:
        client.delete_collection(name)
        client.close()
