"""Weaviate Cloud (managed service) round trip."""

import time
import uuid

import cloud_credentials
import pytest
import weaviate
import weaviate.classes.config as wc
from contract import make_protector, run_contract
from weaviate.classes.init import AdditionalConfig, Auth, Timeout

from hadamask.adapters.weaviate import WeaviateStore, class_name

pytestmark = pytest.mark.cloud


@pytest.fixture
def client():
    url, key = cloud_credentials.get("WEAVIATE_CLOUD_URL"), cloud_credentials.get("WEAVIATE_CLOUD_API_KEY")
    if not (url and key):
        pytest.skip("Weaviate Cloud credentials not configured")
    c = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(key),
        additional_config=AdditionalConfig(timeout=Timeout(init=60, query=120, insert=120)),
    )
    yield c
    c.close()


def _allowed_index_type(client) -> str:
    """Weaviate Cloud plans may restrict index types; use hnsw when allowed, else hfresh."""
    probe = f"Hadamask_probe_{uuid.uuid4().hex[:8]}"
    try:
        client.collections.create(
            probe, vector_config=wc.Configure.Vectors.self_provided(vector_index_config=wc.Configure.VectorIndex.hnsw())
        )
        return "hnsw"
    except Exception as exc:
        return "hfresh" if "hfresh" in str(exc) else "hnsw"
    finally:
        if client.collections.exists(probe):
            client.collections.delete(probe)


def test_contract_weaviate_cloud(client):
    index_type = _allowed_index_type(client)
    store = WeaviateStore(client, make_protector(), index_type=index_type)
    print("index:", index_type)

    def raw(name, rid):
        return client.collections.get(class_name(name)).query.fetch_object_by_id(rid).properties

    name = f"hadamask_ci_{uuid.uuid4().hex[:8]}"

    def settle(timeout: float = 180.0) -> None:
        """hfresh indexes asynchronously: wait until stored vectors find themselves.

        Works on the raw collections (the contract's ``_copy`` collection uses another key).
        """
        deadline = time.monotonic() + timeout
        for coll_name in (name, name + "_copy"):
            if not client.collections.exists(class_name(coll_name)):
                continue
            coll = client.collections.get(class_name(coll_name))
            while True:
                sample = coll.query.fetch_objects(limit=20, include_vector=True).objects
                found = 0
                for o in sample:
                    top = coll.query.near_vector(near_vector=o.vector["default"], limit=1).objects
                    found += bool(top) and top[0].uuid == o.uuid
                if found >= 0.9 * len(sample):  # hfresh is approximate; the contract checks recall
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Weaviate Cloud did not finish indexing {coll_name}")
                time.sleep(3)

    print(
        run_contract(
            store,
            name,
            raw_payload=raw,
            settle=settle if index_type == "hfresh" else lambda: None,
            # No clone: this Weaviate Cloud plan allows a single collection, so copy_to (which
            # needs a second one) is covered by the local Weaviate test instead.
        )
    )
