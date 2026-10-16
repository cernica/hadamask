"""hadamask quickstart: runs as is, with an in-memory Qdrant and toy embeddings.

    pip install "hadamask[qdrant]"
    python examples/quickstart.py

Replace ``embed`` with your real embedding model and ``QdrantClient(":memory:")`` with your
server. In production, load the key from a secret manager (``MasterKey.from_env()``).
"""

import hashlib

import numpy as np
from qdrant_client import QdrantClient, models

from hadamask import MasterKey, MetadataPolicy, Protector
from hadamask.adapters.qdrant import QdrantStore

DIM = 384


def embed(text: str) -> np.ndarray:
    """Toy embedding: bag of hashed words. Use a real model in practice."""
    vec = np.zeros(DIM)
    for word in text.lower().split():
        vec[int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % DIM] += 1.0
    return vec / (np.linalg.norm(vec) or 1.0)


documents = [
    ("Wire transfer of 1,250,000 EUR to IBAN RO49AAAA1B31007593840000", {"client": "ACME", "year": 2026}),
    ("Loan agreement: principal 480,000 EUR at 4.1% fixed", {"client": "ACME", "year": 2025}),
    ("Quarterly audit report for the treasury department", {"client": "Globex", "year": 2026}),
    ("Wire transfer of 9,900 EUR to a supplier in Cluj", {"client": "Globex", "year": 2026}),
]

# 1. One key per tenant. Fields: "client" can be filtered (blind token), "year" stays in clear,
#    everything else (here: "text") is encrypted with AES-256-GCM and padded.
protector = Protector(MasterKey.generate("demo1"), MetadataPolicy(plaintext=["year"], blind=["client"]))

# 2. Wrap the database client.
client = QdrantClient(":memory:")
store = QdrantStore(client, protector, payload_schema={"year": models.PayloadSchemaType.INTEGER})
store.create_collection("docs", dim=DIM, metric="cosine")

# 3. Write: vectors are transformed, payloads encrypted, before leaving the process.
ids = [i + 1 for i in range(len(documents))]
store.upsert(
    "docs",
    ids,
    np.stack([embed(text) for text, _ in documents]),
    [{"text": text, **meta} for text, meta in documents],
)

# 4. Search: the query gets the same transform; results come back decrypted.
for hit in store.search("docs", embed("wire transfer EUR"), k=2, where={"client": "ACME"}):
    print(f"{hit.score:.3f}  {hit.payload['client']:6}  {hit.payload['text']}")

# 5. What the database actually stores:
raw = client.retrieve("docs", ids=[1], with_payload=True, with_vectors=True)[0]
print("\nstored payload:", {k: (v[:40] + "...") if isinstance(v, str) else v for k, v in raw.payload.items()})
print("stored vector :", np.round(raw.vector[:6], 3), f"... ({len(raw.vector)} dims)")
