# hadamask benchmark

100,000 DBpedia documents with OpenAI ada-002 embeddings (1536-d), 1,000 held-out queries, k = 10, HNSW search breadth ef = 128. arm64, Darwin 25.6.0, Python 3.12.13, databases in Docker (colima). Only the benchmarked database runs during its measurement; each variant gets one full untimed warm-up pass. One single-threaded client; latency is end to end (for hadamask it includes the transform, payload decryption and result handling). Ratios are relative to the unprotected collection on the same database with the same index settings.

## Client-side cost (no database)

| | hadamard | dense |
|---|---:|---:|
| Stored dimension | 2048 | 1536 |
| Transform throughput, batches of 1000 | 11,949 vectors/s | 76,970 vectors/s |
| Transform CPU per 1M vectors | 83 CPU-s | 13 CPU-s |
| Transform latency, one query (p50 / p95) | 0.100 / 0.105 ms | 0.167 / 0.410 ms |
| Encrypt / decrypt one payload (title + text) | 7 / 5 µs | same |
| Mean payload size, clear → encrypted | 404 → 838 bytes | same |

## pgvector

| Metric | Unprotected | hadamask (hadamard) | hadamask (dense) |
|---|---:|---:|---:|
| Stored dimension | 1536 | 2048 (1.33×) | 1536 (1.00×) |
| Recall@10 vs exact search | 0.9951 | 1.0000 | 0.9952 |
| Top-10 overlap with unprotected | 1.0000 | 0.9951 | 0.9994 |
| Query latency p50 | 5.69 ms | 212.45 ms (37.33×) | 6.34 ms (1.11×) |
| Query latency p95 | 7.98 ms | 233.74 ms (29.30×) | 10.23 ms (1.28×) |
| Query latency p99 | 10.08 ms | 318.26 ms (31.57×) | 11.68 ms (1.16×) |
| Throughput, one client | 178 q/s | 5 q/s (0.03×) | 154 q/s (0.86×) |
| Ingest until index ready | 667 s | 16 s (0.02×) | 659 s (0.99×) |
| Storage on disk | 1,703 MB | 950 MB (0.56×) | 1,745 MB (1.02×) |
| Storage per 1M vectors | 17.0 GB | 9.5 GB (0.56×) | 17.4 GB (1.02×) |
| Server CPU per 1M queries | 4,218 CPU-s | 637,855 CPU-s (151.23×) | 4,466 CPU-s (1.06×) |
| Client CPU per 1M queries | 244 CPU-s | 718 CPU-s (2.94×) | 693 CPU-s (2.83×) |
| Server memory after load (indicative) | 3,405 MB | 2,630 MB | 3,446 MB |

> hadamask (hadamard): exact scan: HNSW limited to 2000 dims

## qdrant

| Metric | Unprotected | hadamask (hadamard) | hadamask (dense) |
|---|---:|---:|---:|
| Stored dimension | 1536 | 2048 (1.33×) | 1536 (1.00×) |
| Recall@10 vs exact search | 0.9911 | 0.9916 | 0.9908 |
| Top-10 overlap with unprotected | 1.0000 | 0.9902 | 0.9898 |
| Query latency p50 | 3.63 ms | 4.13 ms (1.14×) | 4.01 ms (1.11×) |
| Query latency p95 | 5.46 ms | 6.03 ms (1.10×) | 5.99 ms (1.10×) |
| Query latency p99 | 6.34 ms | 6.55 ms (1.03×) | 6.32 ms (1.00×) |
| Throughput, one client | 260 q/s | 229 q/s (0.88×) | 235 q/s (0.91×) |
| Ingest until index ready | 115 s | 143 s (1.25×) | 102 s (0.89×) |
| Storage on disk | 1,319 MB | 1,769 MB (1.34×) | 1,381 MB (1.05×) |
| Storage per 1M vectors | 13.2 GB | 17.7 GB (1.34×) | 13.8 GB (1.05×) |
| Server CPU per 1M queries | 2,338 CPU-s | 2,473 CPU-s (1.06×) | 2,245 CPU-s (0.96×) |
| Client CPU per 1M queries | 1,069 CPU-s | 1,454 CPU-s (1.36×) | 1,521 CPU-s (1.42×) |
| Server memory after load (indicative) | 1,573 MB | 2,014 MB | 1,599 MB |
