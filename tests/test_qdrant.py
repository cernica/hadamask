import numpy as np
import pytest
from conftest import require_service
from contract import DIM, make_corpus, make_protector, run_contract
from qdrant_client import QdrantClient, models

from hadamask.adapters.qdrant import ProtectedQdrantClient, QdrantStore

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    require_service(6333, "qdrant")
    c = QdrantClient(url="http://127.0.0.1:6333")
    yield c
    c.close()


def test_contract(client):
    store = QdrantStore(client, make_protector())

    def raw(name, rid):
        return client.retrieve(name, ids=[rid], with_payload=True)[0].payload

    result = run_contract(store, "hadamask_contract", raw_payload=raw, clone=lambda p: QdrantStore(client, p))
    print(result)


def test_sdk_wrapper(client):
    protector = make_protector()
    pc = ProtectedQdrantClient(client, protector)
    ids, docs, queries, payloads = make_corpus()
    name = "hadamask_sdk"
    if client.collection_exists(name):
        client.delete_collection(name)
    pc.create_collection(name, vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
    try:
        assert client.get_collection(name).config.params.vectors.size == protector.stored_dim(DIM)
        pc.upsert(
            name,
            points=[models.PointStruct(id=i, vector=v.tolist(), payload=p) for i, v, p in zip(ids, docs, payloads)],
            wait=True,
        )
        res = pc.query_points(
            name,
            query=queries[0].tolist(),
            limit=5,
            with_payload=True,
            query_filter=models.Filter(must=[pc.match("client", "client-2")]),
        )
        assert len(res.points) == 5
        assert all(p.payload["client"] == "client-2" for p in res.points)
        # Scores are the exact cosine similarities of the clear vectors.
        idx = {rid: i for i, rid in enumerate(ids)}
        for p in res.points:
            assert abs(p.score - float(docs[idx[p.id]] @ queries[0])) < 1e-5
        rec = pc.retrieve(name, ids=[ids[0]], with_vectors=True)[0]
        assert np.allclose(rec.vector, docs[0], atol=1e-5)
        recs, _ = pc.scroll(name, limit=3, with_payload=True)
        assert all("text" in r.payload and not r.payload["text"].startswith("hm1:") for r in recs)
    finally:
        client.delete_collection(name)
