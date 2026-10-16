import pytest
from conftest import require_service
from contract import make_protector, run_contract
from elasticsearch import Elasticsearch

from hadamask.adapters.elasticsearch import ElasticsearchStore

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    require_service(9200, "elasticsearch")
    c = Elasticsearch("http://127.0.0.1:9200")
    yield c
    c.close()


@pytest.mark.parametrize("metric", ["cosine", "dot", "l2"])
def test_scores_are_native(client, metric):
    import numpy as np
    from contract import DIM, make_corpus

    store = ElasticsearchStore(client, make_protector())
    ids, docs, queries, payloads = make_corpus()
    name = f"hadamask_scores_{metric}"
    store.drop_collection(name)
    store.create_collection(name, DIM, metric=metric)
    try:
        store.upsert(name, ids[:100], docs[:100], payloads[:100])
        idx = {rid: i for i, rid in enumerate(ids)}
        for h in store.search(name, queries[0], k=5):
            d = docs[idx[h.id]].astype(np.float64)
            q = queries[0].astype(np.float64)
            expected = float(np.linalg.norm(d - q)) if metric == "l2" else float(d @ q)
            assert abs(h.score - expected) < 1e-3
    finally:
        store.drop_collection(name)


def test_contract(client):
    store = ElasticsearchStore(client, make_protector())

    def raw(name, rid):
        return client.get(index=name, id=rid)["_source"]

    print(run_contract(store, "hadamask_contract", raw_payload=raw, clone=lambda p: ElasticsearchStore(client, p)))
