import pytest
from conftest import require_service
from contract import make_protector, run_contract
from pymilvus import MilvusClient

from hadamask.adapters.base import And, Condition, Or
from hadamask.adapters.milvus import MilvusStore, compile_filter

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    require_service(19530, "milvus")
    c = MilvusClient(uri="http://127.0.0.1:19530")
    yield c
    c.close()


@pytest.mark.parametrize("index_type", ["HNSW", "FLAT"])
def test_contract(client, index_type):
    store = MilvusStore(client, make_protector(), index_type=index_type)

    def raw(name, rid):
        return client.get(name, ids=[rid], output_fields=["payload"])[0]["payload"]

    result = run_contract(
        store,
        f"hadamask_contract_{index_type.lower()}",
        raw_payload=raw,
        clone=lambda p: MilvusStore(client, p, index_type=index_type),
    )
    print(index_type, result)


def test_filter_injection_rejected():
    with pytest.raises(TypeError):
        compile_filter(Condition("year", ["2020) or (1 == 1"]))
    # String values and field names are JSON-quoted, so quotes inside them cannot close the literal.
    expr = compile_filter(Condition('y"] or 1==1 or payload["x', '2020" or payload["x"] != "'))
    assert expr.count('"') == expr.count('\\"') + 8
    expr = compile_filter(Or((Condition("a", 1, "gte"), And((Condition("b", ["x", "y"], "nin"),)))))
    assert (
        expr
        == '((payload["a"] >= 1) or ((not (payload["b"] in ["x", "y"] or json_contains_any(payload["b"], ["x", "y"])))))'
    )


def test_unknown_search_option_rejected(client):
    store = MilvusStore(client, make_protector())
    with pytest.raises(TypeError, match="unsupported"):
        store._search("whatever", [0.0] * 8, 3, None, False, {"bogus": 1})
