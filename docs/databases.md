# Database notes

Every adapter passes `tests/contract.py` against a real server: 600 clustered 384-d vectors
(stored at 512-d), recall@10 against exact clear search, decrypted payloads, blind and
plaintext equality filters, vector decoding, deletion, and a check that the database stores
only ciphertext.

| Database | Tested server | Class | Index used | recall@10 |
|---|---|---|---|---|
| Qdrant | 1.19.0, Qdrant Cloud | `QdrantStore`, `ProtectedQdrantClient` | default HNSW | 1.0 |
| Weaviate | 1.33.0, Weaviate Cloud 1.40 | `WeaviateStore`, `ProtectedWeaviateCollection` | HNSW (local), hfresh (cloud) | 1.0 |
| Milvus | 2.6.0 standalone | `MilvusStore` | HNSW or FLAT | 1.0 |
| Chroma | 1.5.9 | `ChromaStore` | default HNSW | 1.0 |
| pgvector | pg17, extension 0.8.7 | `PgVectorStore` | HNSW or exhaustive | 1.0 |
| Elasticsearch | 8.17.4 | `ElasticsearchStore` | `hnsw` (unquantized) | 1.0 |
| OpenSearch | 2.19.1 | `OpenSearchStore` | HNSW, lucene engine | 1.0 |
| Redis Stack | 7.4.0 | `RedisStore` | HNSW, EF_RUNTIME=200 | 1.0 |
| MongoDB Atlas (local) | 8.0 | `MongoStore` | exact (ENN) or ANN | 1.0 |

**Scores.** Every adapter converts the engine's native score so that `Hit.score` is the
cosine similarity, the inner product, or the Euclidean distance (`l2`, not squared), equal
to the value on the clear vectors. `tests/test_scores.py` checks this on all 9 databases for
all 3 metrics. Engines report very different things (Weaviate and Chroma report squared L2,
Milvus and Redis too, Elasticsearch and OpenSearch rescale to (1 + cos)/2 or 1/(1 + d²),
Atlas reports 1/(1 + d²) for euclidean).

## Filters, search options and limits

Every adapter passes the same extended contract: all filter operators (`$eq $ne $in $nin
$gt $gte $lt $lte $and $or`) checked against a reference implementation on the clear data,
`count`, `offset` paging, `score_threshold`, `fields`, `with_vectors`, `search_many`,
`scroll`, `update_payload`, `copy_to` into a collection with another key, and delete by
filter. `$ne`/`$nin` match records without the field on every database.

| Database | Search options (`search(..., **options)`) | Limits |
|---|---|---|
| Qdrant | `models.SearchParams` fields: `hnsw_ef`, `exact`, `quantization`, ... | Qdrant Cloud needs payload indexes (`payload_schema`) for filtered fields |
| Weaviate | none | Filtered `scroll` pages by offset and stops at `QUERY_MAXIMUM_RESULTS` (10,000 by default); unfiltered scroll has no cap. Weaviate Cloud plans may limit the number of collections |
| Milvus | `ef`, `nprobe`, `search_list`, `radius`, `range_filter`, `level` | |
| Chroma | none | List values in clear are refused (server bug in 1.5.9) |
| pgvector | `ef_search`, `iterative_scan` | |
| Elasticsearch | `num_candidates` | `dot` requires unit-length vectors |
| OpenSearch | `num_candidates`, `ef_search` | |
| Redis Stack | `ef_runtime` | Ranges only on `numeric_fields` (declared at construction, must hold numbers); other plaintext fields support `$eq $ne $in $nin` |
| MongoDB Atlas | `num_candidates`, `exact` | |

Unknown options raise `TypeError` rather than being silently ignored.

## Common rules

- **No server-side vectorizers.** The database must only ever see protected vectors.
  `WeaviateStore` and `ProtectedWeaviateCollection` refuse collections with a vectorizer.
- **Create collections with the stored dimension** (`protector.stored_dim(dim)`, the next
  power of two). Adapters do this automatically.
- **Filter fields are fixed when the index is created** on Qdrant (payload indexes),
  Redis (TAG fields) and MongoDB (`filter` fields in the vector index). Changing the
  metadata policy requires recreating those indexes.
- **Quantization:** prefer unquantized indexes for exact parity. Elasticsearch 8.14+
  defaults to `int8_hnsw`; the adapter sets `hnsw` explicitly.

## Per database

**Qdrant.** Blind fields get keyword payload indexes automatically. Pass
`payload_schema={"year": PayloadSchemaType.INTEGER}` for plaintext filter fields; Qdrant
Cloud (strict mode) rejects filters on unindexed fields.

**Weaviate.** Collection names are capitalized (`class_name()`); ids must be UUIDs.
Ciphertext properties are declared as `text` with no inverted index, so no BM25 index is
built over ciphertexts. Some Weaviate Cloud plans only allow the `hfresh` index
(`WeaviateStore(..., index_type="hfresh")`). It indexes asynchronously: new objects
become searchable after roughly 10 seconds.

**Milvus.** The payload is stored in a JSON field `payload`; filters are generated as
`payload["f"] == <literal>` with escaped literals. `consistency_level="Strong"`.
The Docker image needs `DEPLOY_MODE=STANDALONE` with embedded etcd (see `docker-compose.yml`).

**Chroma.** The payload goes into `metadatas` (not `documents`, which is full-text indexed).
List values in clear are refused: chromadb 1.5.9 leaked array metadata across collections
in our tests (a server bug), and `$eq` does not match array elements anyway.

**pgvector.** One table per collection (`id TEXT`, `embedding vector(D)`, `payload JSONB`);
filters use `payload @> %s::jsonb` with bound parameters plus a GIN index. Collection names
are validated against a strict identifier regex. The compose file maps Postgres to port 5433.

**Elasticsearch / OpenSearch.** Payload fields are top-level `_source` fields. Dynamic
templates index blind tokens and plaintext strings as `keyword`; ciphertexts are stored but
neither indexed nor kept in doc values. Writes use `refresh="wait_for"`. OpenSearch's lucene
engine treats the query `k` as its candidate list size, so the adapter asks for
`max(num_candidates, k)` neighbours (default 100): asking for `k=5` directly loses true
neighbours.

**Redis.** One HASH per record and a RediSearch index per collection. Blind and plaintext
fields are case-sensitive TAG fields (base64url tokens are case-sensitive). Create the
client with `decode_responses=False`.

**MongoDB.** The payload lives in a `payload` subdocument. Search is exact by default
(`exact: True`); `wait_until_synced()` waits for asynchronous mongot indexing.

## Framework integrations

**LangChain** (`hadamask.integrations.langchain`). `ProtectedEmbeddings` protects vectors
only. `ProtectedVectorStore` wraps any LangChain vector store and also encrypts
`page_content` and metadata; give it a `filter_builder` for stores whose filter syntax is
not a plain dict.

**LlamaIndex** (`hadamask.integrations.llamaindex`). `ProtectedEmbedding` protects vectors
only. `ProtectedVectorStore` takes plaintext embeddings on nodes and protects vectors, text,
metadata and relationship metadata. LlamaIndex copies parent-document metadata and a
SHA-256 hash of the text into `relationships`; the wrapper encrypts that metadata and drops
the hash. Never combine the `Protected*Embedding(s)` wrappers with `ProtectedVectorStore`:
that would transform the vectors twice.
