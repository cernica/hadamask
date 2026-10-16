import numpy as np
import pytest
import weaviate
import weaviate.classes.config as wc
from conftest import require_service
from contract import DIM, make_corpus, make_protector, run_contract
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery

from hadamask.adapters.weaviate import (
    ProtectedWeaviateCollection,
    WeaviateStore,
    class_name,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    require_service(8080, "weaviate")
    c = weaviate.connect_to_local()
    yield c
    c.close()


def test_contract(client):
    store = WeaviateStore(client, make_protector())

    def raw(name, rid):
        return client.collections.get(class_name(name)).query.fetch_object_by_id(rid).properties

    result = run_contract(store, "hadamask_contract", raw_payload=raw, clone=lambda p: WeaviateStore(client, p))
    print(result)


def test_sdk_wrapper(client):
    protector = make_protector()
    ids, docs, queries, payloads = make_corpus()
    name = "HadamaskSdk"
    client.collections.delete(name)
    collection = client.collections.create(
        name,
        vector_config=wc.Configure.Vectors.self_provided(
            vector_index_config=wc.Configure.VectorIndex.hnsw(distance_metric=wc.VectorDistances.COSINE)
        ),
    )
    try:
        pc = ProtectedWeaviateCollection(collection, protector, dim=DIM)
        pc.data.insert(payloads[0], vector=docs[0].tolist(), uuid=ids[0])
        result = pc.data.insert_many(
            [DataObject(properties=p, vector=v.tolist(), uuid=i) for i, v, p in zip(ids[1:], docs[1:], payloads[1:])]
        )
        assert not result.has_errors
        res = pc.query.near_vector(
            queries[0].tolist(),
            limit=5,
            filters=pc.match("client", "client-2"),
            return_metadata=MetadataQuery(distance=True),
        )
        assert len(res.objects) == 5
        assert all(o.properties["client"] == "client-2" for o in res.objects)
        idx = {i: n for n, i in enumerate(ids)}
        for o in res.objects:
            assert abs((1 - o.metadata.distance) - float(docs[idx[str(o.uuid)]] @ queries[0])) < 1e-5
        obj = pc.query.fetch_object_by_id(ids[0], include_vector=True)
        assert obj.properties == payloads[0]
        assert np.allclose(obj.vector["default"], docs[0], atol=1e-5)
        assert pc.data.delete_by_id(ids[0])
        assert pc.query.fetch_object_by_id(ids[0]) is None
    finally:
        client.collections.delete(name)


def test_refuses_server_side_vectorizer(client):
    name = "HadamaskVectorizer"
    client.collections.delete(name)
    try:
        collection = client.collections.create(
            name,
            vector_config=wc.Configure.Vectors.text2vec_openai(),
        )
    except Exception:
        pytest.skip("server cannot create a text2vec collection (module not enabled)")
    try:
        with pytest.raises(ValueError, match="vectorizer"):
            ProtectedWeaviateCollection(collection, make_protector(), dim=DIM)
    finally:
        client.collections.delete(name)
