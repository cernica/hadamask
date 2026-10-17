# Benchmarks

`run.py` compares, on the same database and with identical index settings, an unprotected
collection (written with the native client) against hadamask-protected collections with
the `hadamard` and `dense` transforms. It reports recall against exact search, latency,
throughput, ingest time, on-disk size, server CPU (read from the container's cgroup),
client CPU and memory, and extrapolates storage and CPU to one million vectors or queries.

## Data

DBpedia entities with OpenAI `text-embedding-ada-002` vectors (1536-d), from
[KShivendu/dbpedia-entities-openai-1M](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M).
Three parquet shards (~1.1 GB, 115k vectors) are enough for 100k documents plus 1,000
held-out queries:

```bash
mkdir -p data/dbpedia && cd data/dbpedia
for f in train-00000-of-00026-3c7b99d1c7eda36e train-00001-of-00026-2b24035a6390fdcb \
         train-00002-of-00026-b05ce48965853dad; do
  curl -fLO "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/resolve/main/data/$f.parquet"
done
```

## Run

```bash
pip install -e ".[qdrant,pgvector]" pyarrow
docker compose up -d qdrant pgvector     # stop other services for cleaner numbers
python benchmarks/run.py --data data/dbpedia --n 100000 --queries 1000
```

Results go to `benchmarks/results/RESULTS.md` and `results.json`. Allow about 30 minutes
and 10 GB of free Docker disk space: pgvector maintains its HNSW index during inserts, and
Qdrant is loaded first and indexed afterwards, as Qdrant recommends for bulk loads.

## Method notes

- Queries are held-out vectors from the same dataset, a standard choice for ANN
  benchmarks.
- Every variant gets one full, untimed warm-up pass before measurement, so a variant does
  not pay for a cache evicted by the previous one.
- Qdrant is configured to index every segment (`indexing_threshold=1` after loading). By
  default it leaves small segments unindexed, which would mix exact and approximate search.
- With the `hadamard` transform, pgvector cannot build an HNSW index on 2048-d vectors, so
  that variant is measured with an exact scan.
- Server CPU is read from cgroup v2 `cpu.stat`, so it counts only the database container.
