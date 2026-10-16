"""Every adapter reports Hit.score with the same meaning, for every metric.

cosine -> cosine similarity, dot -> inner product, l2 -> Euclidean distance (not squared),
all equal to the values computed on the CLEAR vectors (the transform is orthogonal).
"""

from __future__ import annotations

import os
import uuid
from typing import Callable

import numpy as np
import pytest
from conftest import require_service
from contract import make_protector

pytestmark = pytest.mark.integration

N, DIM, K = 200, 96, 5


def _qdrant():
    require_service(6333, "qdrant")
    from qdrant_client import QdrantClient

    from hadamask.adapters.qdrant import QdrantStore

    return QdrantStore(QdrantClient(url="http://127.0.0.1:6333"), make_protector()), None


def _weaviate():
    require_service(8080, "weaviate")
    import weaviate

    from hadamask.adapters.weaviate import WeaviateStore

    return WeaviateStore(weaviate.connect_to_local(), make_protector()), None


def _milvus():
    require_service(19530, "milvus")
    from pymilvus import MilvusClient

    from hadamask.adapters.milvus import MilvusStore

    return MilvusStore(MilvusClient(uri="http://127.0.0.1:19530"), make_protector(), index_type="FLAT"), None


def _chroma():
    require_service(8000, "chroma")
    import chromadb

    from hadamask.adapters.chroma import ChromaStore

    return ChromaStore(chromadb.HttpClient(host="127.0.0.1", port=8000), make_protector()), None


def _pgvector():
    require_service(5433, "pgvector")
    from hadamask.adapters.pgvector import PgVectorStore

    dsn = os.environ.get("PGVECTOR_DSN", "postgresql://hadamask:hadamask@127.0.0.1:5433/hadamask")
    return PgVectorStore(dsn, make_protector(), hnsw=False), None


def _elasticsearch():
    require_service(9200, "elasticsearch")
    from elasticsearch import Elasticsearch

    from hadamask.adapters.elasticsearch import ElasticsearchStore

    return ElasticsearchStore(Elasticsearch("http://127.0.0.1:9200"), make_protector()), None


def _opensearch():
    require_service(9201, "opensearch")
    from opensearchpy import OpenSearch

    from hadamask.adapters.opensearch import OpenSearchStore

    return OpenSearchStore(OpenSearch(hosts=[{"host": "127.0.0.1", "port": 9201}]), make_protector()), None


def _redis():
    require_service(6379, "redis")
    import redis

    from hadamask.adapters.redis import RedisStore

    return RedisStore(redis.Redis(host="127.0.0.1", port=6379), make_protector(), algorithm="FLAT"), None


def _mongodb():
    require_service(27017, "mongodb")
    from pymongo import MongoClient

    from hadamask.adapters.mongodb import MongoStore

    store = MongoStore(
        MongoClient("mongodb://127.0.0.1:27017/?directConnection=true")["hadamask_test"], make_protector()
    )
    return store, store.wait_until_synced


FACTORIES: dict[str, Callable] = {
    "qdrant": _qdrant,
    "weaviate": _weaviate,
    "milvus": _milvus,
    "chroma": _chroma,
    "pgvector": _pgvector,
    "elasticsearch": _elasticsearch,
    "opensearch": _opensearch,
    "redis": _redis,
    "mongodb": _mongodb,
}


def _exact(metric: str, docs: np.ndarray, q: np.ndarray) -> np.ndarray:
    if metric == "l2":
        return np.linalg.norm(docs - q, axis=1)
    return docs @ q  # unit vectors: cosine == dot


@pytest.mark.parametrize("metric", ["cosine", "dot", "l2"])
@pytest.mark.parametrize("db", list(FACTORIES))
def test_scores_match_clear_values(db, metric):
    store, settle = FACTORIES[db]()
    rng = np.random.default_rng(3)
    docs = rng.standard_normal((N, DIM))
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)  # some engines require unit vectors for dot
    q = rng.standard_normal(DIM)
    q /= np.linalg.norm(q)
    ids = [str(uuid.UUID(int=i + 1)) for i in range(N)]
    name = f"hadamask_scores_{metric}_{uuid.uuid4().hex[:6]}"
    store.create_collection(name, DIM, metric=metric)
    try:
        store.upsert(name, ids, docs, [{"i": i} for i in range(N)])
        if settle:
            settle(name)
        hits = store.search(name, q, k=K)
        exact = _exact(metric, docs.astype(np.float32).astype(np.float64), q)
        order = np.argsort(exact) if metric == "l2" else np.argsort(-exact)
        assert [h.payload["i"] for h in hits] == [int(i) for i in order[:K]]
        for h in hits:
            assert abs(h.score - exact[h.payload["i"]]) < 2e-3, (db, metric, h.score, exact[h.payload["i"]])
    finally:
        store.drop_collection(name)
