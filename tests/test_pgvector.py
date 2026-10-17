import os

import pytest
from conftest import require_service
from contract import make_protector, run_contract

from hadamask.adapters.base import Condition
from hadamask.adapters.pgvector import PgVectorStore, compile_filter

pytestmark = pytest.mark.integration

DSN = os.environ.get("PGVECTOR_DSN", "postgresql://hadamask:hadamask@127.0.0.1:5433/hadamask")


def _require():
    require_service(int(DSN.rsplit(":", 1)[1].split("/")[0]), "pgvector")


@pytest.mark.parametrize("hnsw", [True, False])
def test_contract(hnsw):
    _require()
    store = PgVectorStore(DSN, make_protector(), hnsw=hnsw)
    clones = []

    def clone(p):
        clones.append(PgVectorStore(DSN, p, hnsw=hnsw))
        return clones[-1]

    def raw(name, rid):
        return store.conn.execute(f"SELECT payload FROM {name} WHERE id = %s", [rid]).fetchone()[0]

    try:
        result = run_contract(store, f"hadamask_contract_{'hnsw' if hnsw else 'flat'}", raw_payload=raw, clone=clone)
        print("hnsw" if hnsw else "exhaustive", result)
    finally:
        store.close()
        for c in clones:
            c.close()


def test_rejects_unsafe_collection_names():
    _require()
    store = PgVectorStore(DSN, make_protector())
    try:
        for bad in ["x; DROP TABLE y", 'a"b', "", "1abc", "a" * 64]:
            with pytest.raises(ValueError):
                store.create_collection(bad, 8)
    finally:
        store.close()


def test_filter_values_are_bound_parameters():
    evil = "x'); DROP TABLE t; --"
    for op in ("eq", "ne", "gt"):
        expr, params = compile_filter(Condition(evil, evil, op))
        assert evil not in expr.as_string(None) and any(evil in str(p) for p in params)
    with pytest.raises(TypeError):
        compile_filter(Condition("f", [1], "gt"))


def test_unknown_search_option_rejected():
    _require()
    store = PgVectorStore(DSN, make_protector())
    try:
        with pytest.raises(TypeError, match="unsupported"):
            store._search("whatever", [0.0] * 8, 3, None, False, {"bogus": 1})
    finally:
        store.close()


def test_hnsw_dimension_limit_is_explained():
    from hadamask import MasterKey, Protector

    require_service(5433, "pgvector")
    with pytest.raises(ValueError, match="transform='haar'"):
        PgVectorStore(DSN, Protector(MasterKey.generate()), hnsw=True).create_collection("too_wide", 1536)
    store = PgVectorStore(DSN, Protector(MasterKey.generate(), transform="haar"), hnsw=True)
    store.create_collection("haar_1536", 1536)
    store.drop_collection("haar_1536")
