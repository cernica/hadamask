import chromadb
import pytest
from conftest import require_service
from contract import make_protector, run_contract

from hadamask.adapters.chroma import ChromaStore

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    require_service(8000, "chroma")
    return chromadb.HttpClient(host="127.0.0.1", port=8000)


def test_contract(client):
    store = ChromaStore(client, make_protector())

    def raw(name, rid):
        return client.get_collection(name, embedding_function=None).get(ids=[rid], include=["metadatas"])["metadatas"][
            0
        ]

    result = run_contract(store, "hadamask_contract", raw_payload=raw, clone=lambda p: ChromaStore(client, p))
    print(result)


def test_refuses_list_values_in_clear(client):
    from hadamask import MasterKey, MetadataPolicy, Protector

    store = ChromaStore(client, Protector(MasterKey.generate(), MetadataPolicy(blind=["tags"])))
    store.create_collection("hadamask_lists", 4)
    try:
        with pytest.raises(ValueError, match="list values"):
            store.upsert("hadamask_lists", ["a"], [[1.0, 0, 0, 0]], [{"tags": ["x", "y"]}])
    finally:
        store.drop_collection("hadamask_lists")
