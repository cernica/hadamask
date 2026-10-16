# Security

## Threat model

hadamask protects data **stored in a vector database** against an attacker who obtains a
copy of that database (a snapshot, a backup, an over-privileged database account) but not
the application, its memory, or the master key.

| Protected | Mechanism | What remains visible |
|---|---|---|
| Encrypted metadata fields and document text | AES-256-GCM, fresh 96-bit nonce per value, field name and key id as associated data, length padding | Which fields exist; a power-of-two size class for long values (nothing with `fixed_lengths`) |
| Blind fields | Encrypted as above, plus an HMAC-SHA256 equality token | Which records share a value, and value frequencies |
| Embeddings | Keyed orthogonal transform `Q = H·D2·P·H·D1` | All pairwise similarities (the Gram matrix): clusters, duplicates, neighbourhoods, spectrum |

The vector transform is **obfuscation with measured leakage, not encryption**. Known
plaintext/vector pairs recover the map, and public data from the same distribution supports
topic inference. See Cernica & Popescu, *Applied Sciences* 2026, 16, 10259, Sections 3.4, 5
and Table 8.

Out of scope: compromise of the application or key, attackers who can insert chosen
documents and later read a snapshot, query and access-pattern leakage, and server-side
vectorizers (the database must only ever receive protected vectors).

## Operational guidance

- Keep the master key in a secret manager (Vault, a cloud KMS, an HSM); never in the
  database, the repository or container images.
- Use one key per tenant.
- Declare as few fields as possible `blind` or `plaintext`. Low-cardinality blind fields
  (status, country, product type) are exposed to frequency analysis.
- Do not log raw embeddings, queries or decrypted payloads.
- Prefer unquantized indexes when exact parity with clear search matters.

## Reporting a vulnerability

Please do not open a public issue. Email ionut.cernica@gmail.com with a description and,
if possible, a minimal reproduction. You will get an answer within 5 working days.
