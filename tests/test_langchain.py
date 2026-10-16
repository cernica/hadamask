import json

import numpy as np
import pytest

pytest.importorskip("langchain_core")

from conftest import service_up
from contract import make_protector
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from hadamask.integrations.langchain import ProtectedEmbeddings, ProtectedVectorStore
from hadamask.metadata import is_ciphertext

DIM = 384
TEXTS = [f"Contract {i}: client pays {1000 + 37 * i} EUR to IBAN RO49AAAA{i:016d}" for i in range(60)]
METAS = [{"client": f"client-{i % 4}", "year": 2020 + i % 3, "amount": 1000 + 37 * i} for i in range(60)]
IDS = [f"00000000-0000-0000-0000-{i:012d}" for i in range(60)]


@pytest.fixture
def embedding():
    return DeterministicFakeEmbedding(size=DIM)


def make_store(embedding, protector=None):
    protector = protector or make_protector()
    return ProtectedVectorStore(InMemoryVectorStore, embedding, protector)


def test_same_results_as_unprotected(embedding):
    plain = InMemoryVectorStore(embedding)
    plain.add_texts(TEXTS, metadatas=METAS, ids=IDS)
    store = make_store(embedding)
    store.add_texts(TEXTS, metadatas=METAS, ids=IDS)
    for q in (TEXTS[3], TEXTS[17], "something else entirely"):
        expected = plain.similarity_search_with_score(q, k=8)
        got = store.similarity_search_with_score(q, k=8)
        assert [d.id for d, _ in got] == [d.id for d, _ in expected]
        assert np.allclose([s for _, s in got], [s for _, s in expected], atol=1e-5)
        for d, _ in got:
            i = IDS.index(d.id)
            assert d.page_content == TEXTS[i] and d.metadata == METAS[i]


def test_inner_store_holds_only_ciphertext(embedding):
    store = make_store(embedding)
    store.add_texts(TEXTS, metadatas=METAS, ids=IDS)
    raw = store.inner.store
    for rec in raw.values():
        assert is_ciphertext(rec["text"])
        assert len(rec["vector"]) == 512
        assert rec["metadata"]["year"] in (2020, 2021, 2022)  # declared plaintext
        assert is_ciphertext(rec["metadata"]["client"]) and is_ciphertext(rec["metadata"]["amount"])
    dump = json.dumps(list(raw.values()))
    assert "IBAN RO49" not in dump and "client-1" not in dump and " EUR to " not in dump


def test_filters(embedding):
    store = make_store(embedding)
    store.add_texts(TEXTS, metadatas=METAS, ids=IDS)
    docs = store.similarity_search(TEXTS[0], k=5, filter={"client": "client-2"})
    assert len(docs) == 5 and all(d.metadata["client"] == "client-2" for d in docs)
    docs = store.similarity_search(TEXTS[0], k=50, filter={"client": "client-1", "year": 2021})
    assert docs and all(d.metadata["client"] == "client-1" and d.metadata["year"] == 2021 for d in docs)
    # Callable filters see decrypted documents (in-memory store only).
    docs = store.similarity_search(TEXTS[0], k=50, filter=lambda d: d.metadata["amount"] > 2500)
    assert docs and all(d.metadata["amount"] > 2500 for d in docs)
    with pytest.raises(ValueError):
        store.similarity_search(TEXTS[0], k=5, filter={"amount": 1000})


def test_get_delete_retriever_mmr(embedding):
    store = make_store(embedding)
    store.add_documents(
        [
            __import__("langchain_core.documents").documents.Document(id=i, page_content=t, metadata=m)
            for i, t, m in zip(IDS, TEXTS, METAS)
        ]
    )
    got = store.get_by_ids(IDS[:2])
    assert sorted(d.page_content for d in got) == sorted(TEXTS[:2])
    docs = store.as_retriever(search_kwargs={"k": 3}).invoke(TEXTS[5])
    assert docs[0].page_content == TEXTS[5]
    assert store.max_marginal_relevance_search(TEXTS[5], k=3)[0].page_content == TEXTS[5]
    store.delete(IDS[:2])
    assert store.get_by_ids(IDS[:2]) == []


def test_protected_embeddings_only_vectors(embedding):
    protector = make_protector()
    plain = InMemoryVectorStore(embedding)
    plain.add_texts(TEXTS, ids=IDS)
    vec_only = InMemoryVectorStore(ProtectedEmbeddings(embedding, protector))
    vec_only.add_texts(TEXTS, ids=IDS)
    assert [d.id for d in vec_only.similarity_search(TEXTS[9], k=6)] == [
        d.id for d in plain.similarity_search(TEXTS[9], k=6)
    ]
    rec = vec_only.store[IDS[0]]
    assert rec["text"] == TEXTS[0]  # text is NOT protected at this level
    assert not np.allclose(rec["vector"][:DIM], embedding.embed_query(TEXTS[0]))
    with pytest.raises(ValueError):
        ProtectedVectorStore(InMemoryVectorStore, ProtectedEmbeddings(embedding, protector), protector)


@pytest.mark.integration
def test_qdrant_backend(embedding):
    pytest.importorskip("langchain_qdrant")
    if not service_up(6333):
        pytest.skip("qdrant is not running")
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url="http://127.0.0.1:6333")
    name = "hadamask_langchain"
    protector = make_protector()
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name, vectors_config=models.VectorParams(size=protector.stored_dim(DIM), distance=models.Distance.COSINE)
    )

    def qdrant_filter(conditions):
        return models.Filter(
            must=[
                models.FieldCondition(key=f"metadata.{c.field}", match=models.MatchValue(value=c.value))
                for c in conditions
            ]
        )

    try:
        store = ProtectedVectorStore(
            lambda emb: QdrantVectorStore(client, name, embedding=emb),
            embedding,
            protector,
            filter_builder=qdrant_filter,
        )
        store.add_texts(TEXTS, metadatas=METAS, ids=IDS)
        plain = InMemoryVectorStore(embedding)
        plain.add_texts(TEXTS, metadatas=METAS, ids=IDS)
        assert [d.id for d in store.similarity_search(TEXTS[4], k=5)] == [
            d.id for d in plain.similarity_search(TEXTS[4], k=5)
        ]
        docs = store.similarity_search(TEXTS[4], k=5, filter={"client": "client-3"})
        assert len(docs) == 5 and all(d.metadata["client"] == "client-3" for d in docs)
        assert docs[0].page_content.startswith("Contract")
        raw = client.retrieve(name, ids=[IDS[0]], with_payload=True)[0].payload
        assert is_ciphertext(raw["page_content"]) and is_ciphertext(raw["metadata"]["client"])
        assert "IBAN RO49" not in json.dumps(raw)
    finally:
        client.delete_collection(name)
        client.close()


def test_operator_filters(embedding):
    store = make_store(embedding)
    store.add_texts(TEXTS, metadatas=METAS, ids=IDS)
    cases = [
        ({"year": {"$gte": 2021}}, lambda m: m["year"] >= 2021),
        ({"client": {"$in": ["client-0", "client-3"]}}, lambda m: m["client"] in ("client-0", "client-3")),
        (
            {"client": {"$ne": "client-1"}, "year": {"$lt": 2022}},
            lambda m: m["client"] != "client-1" and m["year"] < 2022,
        ),
        ({"client": {"$nin": ["client-0"]}}, lambda m: m["client"] != "client-0"),
    ]
    for where, predicate in cases:
        docs = store.similarity_search(TEXTS[0], k=60, filter=where)
        assert docs and all(predicate(d.metadata) for d in docs), where
        assert len(docs) == sum(predicate(m) for m in METAS), where


def test_dict_builder_keeps_operators():
    from hadamask.adapters.base import Condition
    from hadamask.integrations.langchain import _default_filter_builder

    build = _default_filter_builder(object())
    assert build([Condition("year", 2021, "gte"), Condition("year", 2023, "lt"), Condition("_hmb_c", "tok")]) == {
        "year": {"$gte": 2021, "$lt": 2023},
        "_hmb_c": "tok",
    }
