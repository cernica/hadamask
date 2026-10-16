import json

import pytest
import redis
from conftest import require_service
from contract import make_protector, run_contract

from hadamask.adapters.redis import RedisStore

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    require_service(6379, "redis")
    c = redis.Redis(host="127.0.0.1", port=6379)
    yield c
    c.close()


def test_contract(client):
    store = RedisStore(client, make_protector(), numeric_fields=["year"])

    def raw(name, rid):
        out = {}
        for k, v in client.hgetall(f"{name}:{rid}").items():
            k = k.decode()
            if k == "vec":
                continue
            v = v.decode()
            out[k] = json.loads(v) if k == "year" else v
        return out

    result = run_contract(
        store,
        "hadamask_contract",
        raw_payload=raw,
        clone=lambda p: RedisStore(client, p, numeric_fields=["year"]),
    )
    print(result)


def test_negation_matches_missing_fields_and_range_needs_numeric(client):
    """$ne/$nin match records that lack the field (MongoDB semantics); ranges need NUMERIC."""
    import numpy as np

    from hadamask import MasterKey, MetadataPolicy, Protector

    protector = Protector(MasterKey.generate(), MetadataPolicy(plaintext=["year", "kind"], blind=["client"]))
    store = RedisStore(client, protector, numeric_fields=["year"])
    name = "hadamask_redis_missing"
    try:
        store.drop_collection(name)
    except Exception:
        pass
    store.create_collection(name, 4)
    try:
        vecs = np.eye(4)[:3]
        store.upsert(name, ["a", "b", "c"], vecs, [{"client": "x", "year": 1, "kind": "a, b"}, {"year": 2}, {}])
        assert {r.id for r in store.scroll(name, {"client": {"$ne": "x"}})} == {"b", "c"}
        assert {r.id for r in store.scroll(name, {"year": {"$nin": [1]}})} == {"b", "c"}
        assert {r.id for r in store.scroll(name, {"kind": "a, b"})} == {"a"}  # commas do not split values
        assert store.count(name, {"year": {"$gte": 2}}) == 1
        with pytest.raises(ValueError, match="numeric_fields"):
            store.count(name, {"kind": {"$gt": "a"}})
        with pytest.raises(ValueError, match="must hold a number"):
            store.upsert(name, ["d"], vecs[:1], [{"year": "2026"}])
        with pytest.raises(TypeError, match="unknown"):
            store.search(name, vecs[0], k=1, nprobe=3)
        assert [h.id for h in store.search(name, vecs[0], k=1, ef_runtime=50)] == ["a"]
    finally:
        store.drop_collection(name)
