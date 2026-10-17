"""Reproducible benchmark: hadamask-protected vs unprotected storage on the same database.

For each database, two collections are built with identical index settings from the same
real embeddings: a CLEAR one written with the native client, and a PROTECTED one written
through hadamask (transformed vectors, encrypted payloads). Then it measures:

* quality      recall@k of each collection against exact search on the clear vectors,
               and the overlap between clear and protected results;
* speed        end-to-end query latency (p50/p95/p99) and throughput, ingest time;
* resources    on-disk size, server CPU-seconds (from the container's cgroup), client CPU
               for transform/encryption, server memory;
* per million  storage and CPU extrapolated to 1M vectors / 1M queries.

Data: DBpedia entities with OpenAI text-embedding-ada-002 vectors (1536-d), the parquet
shards of https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M.
Queries are held-out vectors from the same dataset (standard practice for ANN benchmarks).

    docker compose up -d qdrant pgvector
    python benchmarks/run.py --data /path/to/parquet/dir --n 100000 --queries 1000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from hadamask import MasterKey, MetadataPolicy, Protector

K_DEFAULT = 10
COMPOSE_DIR = Path(__file__).resolve().parent.parent


# -- data -----------------------------------------------------------------------------------


def load_dbpedia(data_dir: Path, n_docs: int, n_queries: int):
    """Return (doc_vectors, doc_payloads, query_vectors) from the parquet shards."""
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files in {data_dir}")
    need = n_docs + n_queries
    chunks, titles, texts, have = [], [], [], 0
    for path in files:
        table = pq.read_table(path, columns=["title", "text", "openai"])
        column = table.column("openai").combine_chunks()  # list<double>, flattened without Python objects
        dim = len(column[0])
        chunks.append(np.asarray(column.values, dtype=np.float32).reshape(-1, dim))
        titles.extend(table.column("title").to_pylist())
        texts.extend(table.column("text").to_pylist())
        have += len(table)
        if have >= need:
            break
    if have < need:
        raise SystemExit(f"only {have} vectors available, need {need}")
    x = np.concatenate(chunks)[:need]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    payloads = [{"title": t, "text": s} for t, s in zip(titles[:n_docs], texts[:n_docs], strict=False)]
    return x[:n_docs], payloads, x[n_docs:need]


def exact_top_k(docs: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), 256):
        scores = queries[start : start + 256] @ docs.T
        part = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.take_along_axis(scores, part, axis=1).argsort(axis=1)[:, ::-1]
        out[start : start + 256] = np.take_along_axis(part, order, axis=1)
    return out


# -- measurement helpers -----------------------------------------------------------------------


def percentiles(samples_s: list[float]) -> dict:
    ms = np.asarray(samples_s) * 1000
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "mean_ms": float(ms.mean()),
    }


def container_id(service: str) -> str:
    out = subprocess.run(
        ["docker", "compose", "ps", "-q", service], cwd=COMPOSE_DIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    if not out:
        raise SystemExit(f"service {service} is not running (docker compose up -d {service})")
    return out


def container_cpu_seconds(cid: str) -> float:
    """CPU time consumed by the container so far (cgroup v2 cpu.stat)."""
    stat = subprocess.run(
        ["docker", "exec", cid, "cat", "/sys/fs/cgroup/cpu.stat"], capture_output=True, text=True, check=True
    ).stdout
    usage = next(line for line in stat.splitlines() if line.startswith("usage_usec"))
    return int(usage.split()[1]) / 1e6


def container_memory_mb(cid: str) -> float:
    raw = subprocess.run(
        ["docker", "exec", cid, "cat", "/sys/fs/cgroup/memory.current"], capture_output=True, text=True, check=True
    ).stdout
    return int(raw.strip()) / 1e6


@dataclass
class Result:
    db: str
    variant: str  # "clear" or "protected"
    stored_dim: int
    ingest_s: float = 0.0
    ready_s: float = 0.0
    storage_mb: float = 0.0
    server_memory_mb: float = 0.0
    recall: float = 0.0
    latency: dict = field(default_factory=dict)
    qps: float = 0.0
    server_cpu_s_per_1k_queries: float = 0.0
    client_cpu_s_per_1k_queries: float = 0.0
    note: str = ""
    ids: list = field(default_factory=list, repr=False)


def run_queries(search, queries: np.ndarray, warmup: int, cid: str):
    # Warm caches with one full untimed pass, so a variant is not penalised by running right
    # after another one evicted the page cache. ``warmup`` extra queries settle the client.
    for q in queries:
        search(q)
    for q in queries[:warmup]:
        search(q)
    latencies, ids = [], []
    cpu0, srv0, t0 = time.process_time(), container_cpu_seconds(cid), time.perf_counter()
    for q in queries:
        s = time.perf_counter()
        ids.append(search(q))
        latencies.append(time.perf_counter() - s)
    wall = time.perf_counter() - t0
    client_cpu, server_cpu = time.process_time() - cpu0, container_cpu_seconds(cid) - srv0
    n = len(queries)
    return ids, percentiles(latencies), n / wall, server_cpu / n * 1000, client_cpu / n * 1000


def recall_at_k(found: list[list[int]], truth: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(f[:k]) & set(t[:k])) / k for f, t in zip(found, truth, strict=False)]))


# -- client-side costs (no database) -------------------------------------------------------------


def client_costs(protector: Protector, docs: np.ndarray, payloads: list[dict]) -> dict:
    dim = docs.shape[1]
    protector.encode(docs[:10])  # build the transform once
    t0, c0 = time.perf_counter(), time.process_time()
    for start in range(0, len(docs), 1000):
        protector.encode(docs[start : start + 1000])
    encode_wall, encode_cpu = time.perf_counter() - t0, time.process_time() - c0

    single = [docs[i] for i in range(1000)]
    for v in single[:100]:
        protector.encode(v)
    lat = []
    for v in single:
        s = time.perf_counter()
        protector.encode(v)
        lat.append(time.perf_counter() - s)

    sample = payloads[:5000]
    c0 = time.process_time()
    encrypted = [protector.encrypt_payload(p) for p in sample]
    enc_cpu = time.process_time() - c0
    c0 = time.process_time()
    for e in encrypted:
        protector.decrypt_payload(e)
    dec_cpu = time.process_time() - c0
    clear_bytes = sum(len(json.dumps(p, ensure_ascii=False).encode()) for p in sample) / len(sample)
    enc_bytes = sum(len(json.dumps(e).encode()) for e in encrypted) / len(sample)
    return {
        "native_dim": dim,
        "stored_dim": protector.stored_dim(dim),
        "encode_vectors_per_s": len(docs) / encode_wall,
        "encode_cpu_s_per_1M_vectors": encode_cpu / len(docs) * 1e6,
        "encode_single_query": percentiles(lat),
        "encrypt_us_per_record": enc_cpu / len(sample) * 1e6,
        "decrypt_us_per_record": dec_cpu / len(sample) * 1e6,
        "payload_bytes_clear": clear_bytes,
        "payload_bytes_encrypted": enc_bytes,
    }


# -- databases ------------------------------------------------------------------------------------
#
# Each benchmark builds one collection per variant, with identical index settings:
#   clear     native client, raw vectors, plaintext payloads (the baseline)
#   hadamard  hadamask default transform (padded to a power of two), encrypted payloads
#   haar      hadamask dense Haar transform (native dimension), encrypted payloads


def variants(native_dim: int) -> list[tuple[str, Protector | None]]:
    key = MasterKey.generate("bench")
    return [
        ("clear", None),
        ("hadamard", Protector(key, MetadataPolicy())),
        ("haar", Protector(key, MetadataPolicy(), transform="haar")),
    ]


def bench_qdrant(docs, payloads, queries, truth, k, ef):
    from qdrant_client import QdrantClient, models

    from hadamask.adapters.qdrant import QdrantStore

    client = QdrantClient(url="http://127.0.0.1:6333", timeout=600)
    cid = container_id("qdrant")
    ids = list(range(len(docs)))
    hnsw = models.HnswConfigDiff(m=16, ef_construct=100)
    # Bulk-load pattern recommended by Qdrant: upload without indexing, then build HNSW once
    # over every segment (the default threshold would leave small segments unindexed).
    no_index = models.OptimizersConfigDiff(indexing_threshold=0)
    index_all = models.OptimizersConfigDiff(indexing_threshold=1)
    results = []

    def wait_indexed(name):
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            info = client.get_collection(name)
            if info.status == models.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= len(docs):
                return
            time.sleep(1)
        raise TimeoutError(f"{name}: Qdrant did not finish indexing")

    def storage_mb(name):
        out = subprocess.run(
            ["docker", "exec", cid, "du", "-sm", f"/qdrant/storage/collections/{name}"], capture_output=True, text=True
        ).stdout
        return float(out.split()[0]) if out else float("nan")

    for variant, protector in variants(docs.shape[1]):
        name = f"bench_{variant}"
        if client.collection_exists(name):
            client.delete_collection(name)
        stored = protector.stored_dim(docs.shape[1]) if protector else docs.shape[1]
        r = Result("qdrant", variant, stored)
        t0 = time.perf_counter()
        if protector is None:
            client.create_collection(
                name,
                vectors_config=models.VectorParams(size=stored, distance=models.Distance.COSINE),
                hnsw_config=hnsw,
                optimizers_config=no_index,
            )
            for start in range(0, len(docs), 500):
                client.upsert(
                    name,
                    wait=True,
                    points=[
                        models.PointStruct(id=i, vector=v.tolist(), payload=p)
                        for i, v, p in zip(
                            ids[start : start + 500],
                            docs[start : start + 500],
                            payloads[start : start + 500],
                            strict=False,
                        )
                    ],
                )

            def search(q, name=name):
                res = client.query_points(
                    name, query=q.tolist(), limit=k, with_payload=True, search_params=models.SearchParams(hnsw_ef=ef)
                )
                return [p.id for p in res.points]
        else:
            store = QdrantStore(client, protector)
            store.create_collection(name, docs.shape[1], metric="cosine")
            client.update_collection(name, hnsw_config=hnsw, optimizers_config=no_index)
            store.upsert(name, ids, docs, payloads, batch_size=500)

            def search(q, name=name, store=store):
                return [h.id for h in store.search(name, q, k=k, hnsw_ef=ef)]

        r.ingest_s = time.perf_counter() - t0
        client.update_collection(name, optimizers_config=index_all)
        wait_indexed(name)
        r.ready_s = time.perf_counter() - t0
        r.storage_mb = storage_mb(name)
        r.server_memory_mb = container_memory_mb(cid)
        r.ids, r.latency, r.qps, r.server_cpu_s_per_1k_queries, r.client_cpu_s_per_1k_queries = run_queries(
            search, queries, 50, cid
        )
        r.recall = recall_at_k(r.ids, truth, k)
        results.append(r)
        print(
            f"  qdrant/{variant}: recall {r.recall:.4f}, p50 {r.latency['p50_ms']:.2f} ms, {r.storage_mb:.0f} MB",
            flush=True,
        )
        client.delete_collection(name)
    return results


def bench_pgvector(docs, payloads, queries, truth, k, ef):
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg.types.json import Jsonb

    from hadamask.adapters.pgvector import HNSW_MAX_DIM, PgVectorStore

    dsn = os.environ.get("PGVECTOR_DSN", "postgresql://hadamask:hadamask@127.0.0.1:5433/hadamask")
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    cid = container_id("pgvector")
    ids = [str(i) for i in range(len(docs))]
    results = []

    for variant, protector in variants(docs.shape[1]):
        name = f"bench_{variant}"
        conn.execute(f"DROP TABLE IF EXISTS {name}")
        stored = protector.stored_dim(docs.shape[1]) if protector else docs.shape[1]
        hnsw = stored <= HNSW_MAX_DIM
        r = Result("pgvector", variant, stored, note="" if hnsw else f"exact scan: HNSW limited to {HNSW_MAX_DIM} dims")
        t0 = time.perf_counter()
        if protector is None:
            # Same layout and index as PgVectorStore, without protection.
            conn.execute(
                f"CREATE TABLE {name} (id TEXT PRIMARY KEY, embedding vector({stored}) NOT NULL, payload JSONB)"
            )
            conn.execute(f"CREATE INDEX ON {name} USING GIN (payload jsonb_path_ops)")
            conn.execute(
                f"CREATE INDEX ON {name} USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)"
            )
            with conn.cursor() as cur:
                for start in range(0, len(docs), 500):
                    cur.executemany(
                        f"INSERT INTO {name} (id, embedding, payload) VALUES (%s, %s, %s)",
                        [
                            (i, v, Jsonb(p))
                            for i, v, p in zip(
                                ids[start : start + 500],
                                docs[start : start + 500],
                                payloads[start : start + 500],
                                strict=False,
                            )
                        ],
                    )

            def search(q, name=name):
                with conn.transaction():
                    conn.execute(f"SET LOCAL hnsw.ef_search = {ef}")
                    rows = conn.execute(
                        f"SELECT id, 1 - (embedding <=> %s) AS score, payload FROM {name} "
                        "ORDER BY embedding <=> %s LIMIT %s",
                        (q, q, k),
                    ).fetchall()
                return [int(row[0]) for row in rows]
        else:
            store = PgVectorStore(conn, protector, hnsw=hnsw)
            store.create_collection(name, docs.shape[1], metric="cosine")
            store.upsert(name, ids, docs, payloads, batch_size=500)

            def search(q, name=name, store=store, hnsw=hnsw):
                options = {"ef_search": ef} if hnsw else {}
                return [int(h.id) for h in store.search(name, q, k=k, **options)]

        r.ingest_s = r.ready_s = time.perf_counter() - t0  # the HNSW index is maintained during inserts
        r.storage_mb = conn.execute("SELECT pg_total_relation_size(%s)", (name,)).fetchone()[0] / 1e6
        r.server_memory_mb = container_memory_mb(cid)
        r.ids, r.latency, r.qps, r.server_cpu_s_per_1k_queries, r.client_cpu_s_per_1k_queries = run_queries(
            search, queries, 20, cid
        )
        r.recall = recall_at_k(r.ids, truth, k)
        results.append(r)
        print(
            f"  pgvector/{variant}: recall {r.recall:.4f}, p50 {r.latency['p50_ms']:.2f} ms, {r.storage_mb:.0f} MB"
            f"{' (' + r.note + ')' if r.note else ''}",
            flush=True,
        )
        conn.execute(f"DROP TABLE IF EXISTS {name}")
    return results


BENCHES = {"qdrant": bench_qdrant, "pgvector": bench_pgvector}


# -- report -------------------------------------------------------------------------------------

VARIANTS = ("clear", "hadamard", "haar")
LABELS = {"clear": "Unprotected", "hadamard": "hadamask (hadamard)", "haar": "hadamask (haar)"}


def summarize(results: list[Result], n_docs: int, k: int) -> dict:
    out = {}
    for db in sorted({r.db for r in results}):
        by = {r.variant: r for r in results if r.db == db}
        clear = by["clear"]
        rows = {}
        for variant, r in by.items():
            rows[variant] = {
                "stored_dim": r.stored_dim,
                "note": r.note,
                "recall_at_k": r.recall,
                "overlap_with_clear": float(
                    np.mean([len(set(a) & set(b)) / k for a, b in zip(clear.ids, r.ids, strict=False)])
                ),
                "latency": r.latency,
                "qps_single_client": r.qps,
                "ingest_s": r.ingest_s,
                "ready_s": r.ready_s,
                "storage_mb": r.storage_mb,
                "storage_gb_per_1M_vectors": r.storage_mb / n_docs * 1e6 / 1e3,
                "server_memory_mb": r.server_memory_mb,
                "server_cpu_s_per_1M_queries": r.server_cpu_s_per_1k_queries * 1e3,
                "client_cpu_s_per_1M_queries": r.client_cpu_s_per_1k_queries * 1e3,
            }
        out[db] = rows
    return out


def markdown(summary: dict, client: dict, meta: dict) -> str:
    lines = [
        "# hadamask benchmark",
        "",
        f"{meta['n_docs']:,} DBpedia documents with OpenAI ada-002 embeddings ({client['hadamard']['native_dim']}-d), "
        f"{meta['n_queries']:,} held-out queries, k = {meta['k']}, HNSW search breadth ef = {meta['ef']}. "
        f"{meta['platform']}. Only the benchmarked database runs during its measurement; each variant gets one full untimed warm-up pass. One single-threaded client; latency is end to end (for hadamask it includes the "
        "transform, payload decryption and result handling). Ratios are relative to the unprotected collection "
        "on the same database with the same index settings.",
        "",
        "## Client-side cost (no database)",
        "",
        "| | hadamard | haar |",
        "|---|---:|---:|",
        f"| Stored dimension | {client['hadamard']['stored_dim']} | {client['haar']['stored_dim']} |",
        f"| Transform throughput, batches of 1000 | {client['hadamard']['encode_vectors_per_s']:,.0f} vectors/s | "
        f"{client['haar']['encode_vectors_per_s']:,.0f} vectors/s |",
        f"| Transform CPU per 1M vectors | {client['hadamard']['encode_cpu_s_per_1M_vectors']:.0f} CPU-s | "
        f"{client['haar']['encode_cpu_s_per_1M_vectors']:.0f} CPU-s |",
        f"| Transform latency, one query (p50 / p95) | {client['hadamard']['encode_single_query']['p50_ms']:.3f} / "
        f"{client['hadamard']['encode_single_query']['p95_ms']:.3f} ms | {client['haar']['encode_single_query']['p50_ms']:.3f} / "
        f"{client['haar']['encode_single_query']['p95_ms']:.3f} ms |",
        f"| Encrypt / decrypt one payload (title + text) | {client['hadamard']['encrypt_us_per_record']:.0f} / "
        f"{client['hadamard']['decrypt_us_per_record']:.0f} µs | same |",
        f"| Mean payload size, clear → encrypted | {client['hadamard']['payload_bytes_clear']:.0f} → "
        f"{client['hadamard']['payload_bytes_encrypted']:.0f} bytes | same |",
        "",
    ]
    for db, rows in summary.items():
        present = [v for v in VARIANTS if v in rows]
        c = rows["clear"]
        header = "| Metric | " + " | ".join(LABELS[v] for v in present) + " |"
        lines += [f"## {db}", "", header, "|---|" + "---:|" * len(present)]

        def row(label, fmt, key, sub=None, ratio=True):
            cells = []
            for v in present:
                val = rows[v][key][sub] if sub else rows[v][key]
                base = c[key][sub] if sub else c[key]
                cell = fmt.format(val)
                if ratio and v != "clear" and base:
                    cell += f" ({val / base:.2f}×)"
                cells.append(cell)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

        row("Stored dimension", "{}", "stored_dim")
        row(f"Recall@{meta['k']} vs exact search", "{:.4f}", "recall_at_k", ratio=False)
        row(f"Top-{meta['k']} overlap with unprotected", "{:.4f}", "overlap_with_clear", ratio=False)
        row("Query latency p50", "{:.2f} ms", "latency", "p50_ms")
        row("Query latency p95", "{:.2f} ms", "latency", "p95_ms")
        row("Query latency p99", "{:.2f} ms", "latency", "p99_ms")
        row("Throughput, one client", "{:.0f} q/s", "qps_single_client")
        row("Ingest until index ready", "{:.0f} s", "ready_s")
        row("Storage on disk", "{:,.0f} MB", "storage_mb")
        row("Storage per 1M vectors", "{:.1f} GB", "storage_gb_per_1M_vectors")
        row("Server CPU per 1M queries", "{:,.0f} CPU-s", "server_cpu_s_per_1M_queries")
        row("Client CPU per 1M queries", "{:,.0f} CPU-s", "client_cpu_s_per_1M_queries")
        row("Server memory after load (indicative)", "{:,.0f} MB", "server_memory_mb", ratio=False)
        notes = [f"{LABELS[v]}: {rows[v]['note']}" for v in present if rows[v]["note"]]
        lines += [""] + [f"> {n}" for n in notes] + ([""] if notes else [])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="directory with the DBpedia OpenAI parquet shards")
    ap.add_argument("--n", type=int, default=100_000, help="documents to index")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--ef", type=int, default=128, help="HNSW search breadth, same for both variants")
    ap.add_argument("--db", nargs="+", default=list(BENCHES), choices=list(BENCHES))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = ap.parse_args()

    print(f"loading {args.n + args.queries:,} vectors ...", flush=True)
    docs, payloads, queries = load_dbpedia(args.data, args.n, args.queries)
    print("exact ground truth ...", flush=True)
    truth = exact_top_k(docs, queries, args.k)
    print("client-side costs ...", flush=True)
    key = MasterKey.generate("bench")
    client = {
        "hadamard": client_costs(Protector(key, MetadataPolicy()), docs, payloads),
        "haar": client_costs(Protector(key, MetadataPolicy(), transform="haar"), docs, payloads),
    }

    results = []
    for db in args.db:
        print(f"{db} ...", flush=True)
        results += BENCHES[db](docs, payloads, queries, truth, args.k, args.ef)

    meta = {
        "n_docs": args.n,
        "n_queries": args.queries,
        "k": args.k,
        "ef": args.ef,
        "platform": f"{platform.machine()}, {platform.system()} {platform.release()}, Python {sys.version.split()[0]}, "
        f"databases in Docker (colima)",
    }
    summary = summarize(results, args.n, args.k)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps({"meta": meta, "client": client, "databases": summary}, indent=2))
    (args.out / "RESULTS.md").write_text(markdown(summary, client, meta))
    print(markdown(summary, client, meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
