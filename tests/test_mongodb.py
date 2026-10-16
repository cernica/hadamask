import pytest
from conftest import require_service
from contract import make_protector, run_contract
from pymongo import MongoClient

from hadamask.adapters.mongodb import MongoStore

pytestmark = pytest.mark.integration

NAME = "hadamask_contract"


def settle_all(store, db, name):
    """Wait for the contract collection and, when copy_to created it, its copy."""

    def settle():
        existing = set(db.list_collection_names())
        for n in (name, name + "_copy"):
            if n in existing:
                store.wait_until_synced(n)

    return settle


@pytest.fixture
def client():
    require_service(27017, "mongodb")
    c = MongoClient("mongodb://127.0.0.1:27017/?directConnection=true")
    yield c
    c.close()


def test_contract(client):
    db = client["hadamask_test"]
    store = MongoStore(db, make_protector())

    def raw(name, rid):
        return db[name].find_one({"_id": rid})["payload"]

    result = run_contract(
        store, NAME, raw_payload=raw, settle=settle_all(store, db, NAME), clone=lambda p: MongoStore(db, p)
    )
    print(result)


def test_ann_mode(client):
    """ANN (numCandidates) mode also reaches the contract recall on this corpus."""
    db = client["hadamask_test"]
    store = MongoStore(db, make_protector(), exact=False, num_candidates=200)
    name = NAME + "_ann"
    result = run_contract(
        store,
        name,
        settle=settle_all(store, db, name),
        clone=lambda p: MongoStore(db, p, exact=False, num_candidates=200),
    )
    print(result)


def test_negation_and_search_options(client):
    """$ne/$nin semantics on missing fields in count/find vs. $vectorSearch, and search options."""
    import numpy as np

    from hadamask import MasterKey, MetadataPolicy, Protector

    db = client["hadamask_test"]
    protector = Protector(MasterKey.generate(), MetadataPolicy(plaintext=["year"], blind=["client"]))
    store = MongoStore(db, protector)
    name = NAME + "_missing"
    store.drop_collection(name)
    store.create_collection(name, 4)
    try:
        vecs = np.eye(4)[:3]
        store.upsert(name, ["a", "b", "c"], vecs, [{"client": "x", "year": 1}, {"year": 2}, {}])
        store.wait_until_synced(name)
        assert {r.id for r in store.scroll(name, {"client": {"$ne": "x"}})} == {"b", "c"}
        assert store.count(name, {"year": {"$nin": [1]}}) == 2
        hits = store.search(name, vecs[1], k=3, where={"client": {"$ne": "x"}})
        print("vectorSearch $ne ids:", sorted(h.id for h in hits))
        assert {h.id for h in hits} == {"b", "c"}
        assert [h.id for h in store.search(name, vecs[0], k=1, exact=False, num_candidates=50)] == ["a"]
        with pytest.raises(TypeError, match="unknown"):
            store.search(name, vecs[0], k=1, ef=10)
    finally:
        store.drop_collection(name)
