"""Qdrant Cloud (managed service) round trip."""

import uuid

import cloud_credentials
import pytest
from contract import make_protector, run_contract
from qdrant_client import QdrantClient, models

from hadamask.adapters.qdrant import QdrantStore

pytestmark = pytest.mark.cloud


@pytest.fixture
def client():
    url, key = cloud_credentials.get("QDRANT_CLOUD_URL"), cloud_credentials.get("QDRANT_CLOUD_API_KEY")
    if not (url and key):
        pytest.skip("Qdrant Cloud credentials not configured")
    c = QdrantClient(url=url, api_key=key, timeout=60)
    yield c
    c.close()


def test_contract_qdrant_cloud(client):
    store = QdrantStore(client, make_protector(), payload_schema={"year": models.PayloadSchemaType.INTEGER})

    def raw(name, rid):
        return client.retrieve(name, ids=[rid], with_payload=True)[0].payload

    schema = {"year": models.PayloadSchemaType.INTEGER}
    print(
        run_contract(
            store,
            f"hadamask_ci_{uuid.uuid4().hex[:8]}",
            raw_payload=raw,
            clone=lambda p: QdrantStore(client, p, payload_schema=schema),
        )
    )
