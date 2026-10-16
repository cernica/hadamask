# Changelog

## 0.1.0 (unreleased)

First public version.

- Keyed orthogonal transform `Q = H·D2·P·H·D1` with HMAC-SHA256 key derivation, padding to
  the next power of two, decoding, key re-mapping (`rekey_to`) and optional noise padding.
- Per-field AES-256-GCM metadata encryption with a fresh nonce per value, field-bound
  associated data, length padding (minimum bucket, power-of-two classes, fixed lengths),
  blind equality tokens and decryption with previous keys.
- Adapters with one uniform API: Qdrant, Weaviate, Milvus, Chroma, pgvector,
  Elasticsearch, OpenSearch, Redis Stack, MongoDB Atlas Vector Search. `Hit.score` has the
  same meaning on all of them.
- SDK wrappers for Qdrant and Weaviate that refuse unprotected data paths.
- LangChain and LlamaIndex integrations that wrap any vector store.
- Store API on every adapter: MongoDB-style filters (`$eq $ne $in $nin $gt $gte $lt $lte
  $and $or`), `offset`, `score_threshold`, `fields`, `with_vectors`, engine-specific search
  options, `search_many`, batched `upsert`, `get`, `scroll`, `count`, `update_payload`,
  delete by ids or filter, and `copy_to` for migration and key rotation.
- Key management: envelope encryption of the master key with HashiCorp Vault (Transit,
  KV v2), AWS KMS, Google Cloud KMS and Azure Key Vault; `Keyring` for rotation.
- `hadamask keygen` (optionally `--wrap <provider>`) / `hadamask info` command line helpers.
