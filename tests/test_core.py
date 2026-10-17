import numpy as np
import pytest

from hadamask import (
    DecryptionError,
    MasterKey,
    MetadataCipher,
    MetadataPolicy,
    OrthogonalTransform,
    Protector,
)
from hadamask.keys import HmacStream
from hadamask.transform import fwht


@pytest.fixture
def key():
    return MasterKey(bytes(range(32)), "k1")


def unit_rows(n, d, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# -- transform ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 3, 64, 384, 768, 1024, 1536, 3072])
def test_transform_is_orthogonal_and_invertible(key, d):
    t = OrthogonalTransform(key, d)
    x = unit_rows(50, d)
    y = t.encode(x)
    assert y.shape == (50, t.processing_dim) and y.dtype == np.float32
    assert np.abs(x @ x.T - y.astype(np.float64) @ y.T.astype(np.float64)).max() < 1e-5
    assert np.allclose(t.decode(y), x, atol=1e-6)


def test_matrix_is_exactly_orthogonal(key):
    t = OrthogonalTransform(key, 256)
    m = t.encode(np.eye(256)).astype(np.float64)
    assert np.allclose(m @ m.T, np.eye(256), atol=1e-6)


def test_preserves_cosine_dot_and_l2(key):
    t = OrthogonalTransform(key, 384)
    a, b = np.random.default_rng(1).standard_normal((2, 384)) * 3
    ea, eb = t.encode(a).astype(np.float64), t.encode(b).astype(np.float64)
    assert np.isclose(a @ b, ea @ eb, rtol=1e-5)
    assert np.isclose(np.linalg.norm(a - b), np.linalg.norm(ea - eb), rtol=1e-5)
    cos = lambda u, v: u @ v / np.linalg.norm(u) / np.linalg.norm(v)  # noqa: E731
    assert np.isclose(cos(a, b), cos(ea, eb), atol=1e-6)


def test_same_key_same_map_different_key_different_map(key):
    x = unit_rows(5, 384)
    assert np.array_equal(
        OrthogonalTransform(key, 384).encode(x), OrthogonalTransform(MasterKey(key.secret), 384).encode(x)
    )
    other = OrthogonalTransform(MasterKey(bytes(32)), 384).encode(x)
    assert not np.allclose(OrthogonalTransform(key, 384).encode(x), other)


def test_output_is_not_the_input(key):
    t = OrthogonalTransform(key, 512)
    x = unit_rows(100, 512)
    y = t.encode(x).astype(np.float64)
    # No coordinate of the output is (anti)correlated with the same coordinate of the input.
    corr = [abs(np.corrcoef(x[:, j], y[:, j])[0, 1]) for j in range(0, 512, 16)]
    assert max(corr) < 0.5


def test_ranking_is_preserved(key):
    t = OrthogonalTransform(key, 384)
    docs, q = unit_rows(1000, 384), unit_rows(1, 384, seed=2)[0]
    clear = np.argsort(-(docs @ q))[:10]
    prot = np.argsort(-(t.encode(docs).astype(np.float64) @ t.encode(q).astype(np.float64)))[:10]
    assert set(clear) == set(prot)


def test_noise_padding_keeps_dot_products_with_queries(key):
    t = OrthogonalTransform(key, 384)  # D = 512, 128 padding coordinates
    docs, q = unit_rows(20, 384), unit_rows(1, 384, seed=3)[0]
    noisy = t.encode(docs, noise_norm=0.5).astype(np.float64)
    assert np.allclose(noisy @ t.encode(q).astype(np.float64), docs @ q, atol=1e-5)
    assert np.allclose(np.linalg.norm(noisy, axis=1), np.sqrt(1.25), atol=1e-5)
    assert np.allclose(t.decode(noisy), docs, atol=1e-5)


def test_rekey_without_reembedding(key):
    old, new = OrthogonalTransform(key, 384), OrthogonalTransform(MasterKey.generate("k2"), 384)
    x = unit_rows(10, 384)
    assert np.allclose(old.rekey_to(new, old.encode(x)), new.encode(x), atol=1e-6)


def test_fwht_is_self_inverse():
    x = np.random.default_rng(0).standard_normal((4, 64))
    assert np.allclose(fwht(fwht(x)), x)


def test_permutation_sampling_is_uniform_enough(key):
    counts = np.zeros(4)
    for i in range(4000):
        counts[HmacStream(key.secret, f"t{i}").uniform_below(4)] += 1
    assert np.all(np.abs(counts - 1000) < 150)


def test_processing_dim_validation(key):
    with pytest.raises(ValueError):
        OrthogonalTransform(key, 384, processing_dim=300)
    with pytest.raises(ValueError):
        OrthogonalTransform(key, 384, processing_dim=256)
    assert OrthogonalTransform(key, 384, processing_dim=1024).processing_dim == 1024


# -- metadata ----------------------------------------------------------------------------


def test_roundtrip_preserves_types(key):
    c = MetadataCipher(key)
    payload = {"s": "Ünïcødé 1,250,000.00 EUR", "i": 42, "f": 3.5, "b": True, "n": None, "l": [1, "x"], "d": {"a": 1}}
    enc = c.encrypt(payload)
    assert all(isinstance(v, str) and v.startswith("hm1:k1:") for v in enc.values())
    assert c.decrypt(enc) == payload


def test_fresh_nonce_for_every_value(key):
    c = MetadataCipher(key)
    a = c.encrypt({"x": "same", "y": "same"})
    b = c.encrypt({"x": "same", "y": "same"})
    envelopes = [a["x"], a["y"], b["x"], b["y"]]
    assert len(set(envelopes)) == 4
    nonces = {e.split(":")[2][:16] for e in envelopes}
    assert len(nonces) == 4


def test_tampering_is_detected(key):
    c = MetadataCipher(key)
    env = c.encrypt({"x": "secret"})["x"]
    body = env.split(":")[2]
    flipped = body[:-2] + ("A" if body[-2] != "A" else "B") + body[-1]
    with pytest.raises(DecryptionError):
        c.decrypt({"x": f"hm1:k1:{flipped}"})


def test_ciphertext_bound_to_field_name(key):
    c = MetadataCipher(key)
    enc = c.encrypt({"amount": 1000, "fee": 1})
    with pytest.raises(DecryptionError):
        c.decrypt({"fee": enc["amount"]})


def test_wrong_key_fails(key):
    enc = MetadataCipher(key).encrypt({"x": 1})
    with pytest.raises(DecryptionError):
        MetadataCipher(MasterKey(bytes(32), "k1")).decrypt(enc)


def test_key_rotation_reads_old_records(key):
    enc_old = MetadataCipher(key).encrypt({"x": "old"})
    new = MetadataCipher(MasterKey.generate("k2"), previous_keys=[key])
    enc_new = new.encrypt({"x": "new"})
    assert enc_new["x"].startswith("hm1:k2:")
    assert new.decrypt(enc_old) == {"x": "old"} and new.decrypt(enc_new) == {"x": "new"}


def test_policy_plaintext_and_blind(key):
    c = MetadataCipher(key, MetadataPolicy(plaintext=["year"], blind=["client", "tags"]))
    enc = c.encrypt({"year": 2026, "client": "ACME", "tags": ["a", "b"], "text": "t"})
    assert enc["year"] == 2026
    assert enc["_hmb_client"] == c.blind_token("client", "ACME")
    assert enc["_hmb_tags"] == [c.blind_token("tags", "a"), c.blind_token("tags", "b")]
    assert c.blind_token("client", "ACME") != c.blind_token("other", "ACME")
    assert c.decrypt(enc) == {"year": 2026, "client": "ACME", "tags": ["a", "b"], "text": "t"}


def test_reserved_names_rejected(key):
    with pytest.raises(ValueError):
        MetadataCipher(key).encrypt({"_hmb_x": 1})
    with pytest.raises(ValueError):
        MetadataPolicy(blind=["_hm"])
    with pytest.raises(ValueError):
        MetadataPolicy(plaintext=["a"], blind=["a"])


def test_protector_blind_requires_declaration(key):
    p = Protector(key, MetadataPolicy(blind=["client"]))
    assert p.blind("client", "x")[0] == "_hmb_client"
    with pytest.raises(ValueError):
        p.blind("text", "x")


def test_master_key_validation():
    with pytest.raises(ValueError):
        MasterKey(b"short")
    with pytest.raises(ValueError):
        MasterKey(bytes(32), "bad:id")
    k = MasterKey.generate()
    assert MasterKey.from_base64(k.to_base64()).secret == k.secret
    assert "secret" not in repr(k) or k.secret.hex() not in repr(k)


# -- length hiding -------------------------------------------------------------------------


def envelope_len(c, name, value):
    return len(c.encrypt_value(name, value))


def test_amounts_of_any_magnitude_have_the_same_ciphertext_length(key):
    c = MetadataCipher(key)
    values = [0, 7, 100.0, 9_999_999.99, -1_250_000.5, 123456789012345, "1,250,000.00 EUR", None, True]
    assert len({envelope_len(c, "amount", v) for v in values}) == 1


def test_identifiers_and_short_strings_share_one_bucket(key):
    c = MetadataCipher(key)
    values = ["", "RO", "RO49AAAA1B31007593840000", "GB33BUKB20201555555555", "2026-10-16", "Ion Popescu SRL"]
    assert len({envelope_len(c, "x", v) for v in values}) == 1


def test_long_values_reveal_only_their_power_of_two_class(key):
    c = MetadataCipher(key)
    assert envelope_len(c, "text", "a" * 70) == envelope_len(c, "text", "b" * 120)  # both pad to 128
    assert envelope_len(c, "text", "a" * 130) == envelope_len(c, "text", "b" * 250)  # both pad to 256
    assert envelope_len(c, "text", "a" * 120) < envelope_len(c, "text", "a" * 130)


def test_fixed_length_fields_hide_length_completely(key):
    c = MetadataCipher(key, MetadataPolicy(fixed_lengths={"text": 1024}))
    lengths = {envelope_len(c, "text", "x" * n) for n in (0, 10, 500, 1000)}
    assert len(lengths) == 1
    assert c.decrypt(c.encrypt({"text": "x" * 1000})) == {"text": "x" * 1000}
    with pytest.raises(ValueError, match="fixed_lengths"):
        c.encrypt({"text": "x" * 1100})


def test_padding_roundtrip_edge_cases(key):
    c = MetadataCipher(key, MetadataPolicy(min_padded_length=16))
    for value in ["", "Ѐ", "ț" * 7, "a" * 13, "a" * 14, "a" * 15, {"€": [1, None]}, 0.0, False]:
        assert c.decrypt(c.encrypt({"v": value})) == {"v": value}


def test_padding_validation():
    from hadamask.metadata import pad, padded_length, unpad

    assert padded_length(0) == 64 and padded_length(63) == 64 and padded_length(64) == 128
    assert unpad(pad(b'"x"', 8)) == b'"x"'
    with pytest.raises(ValueError):
        unpad(b'"x"\x00\x00')
    with pytest.raises(ValueError):
        MetadataPolicy(plaintext=["year"], fixed_lengths={"year": 8})
    with pytest.raises(ValueError):
        MetadataPolicy(min_padded_length=0)


def test_rejects_non_finite_vectors(key):
    t = OrthogonalTransform(key, 8)
    for bad in (np.nan, np.inf):
        v = np.ones(8)
        v[3] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            t.encode(v)


# -- SDK wrappers do not leak through passthrough --------------------------------------------


def test_qdrant_wrapper_blocks_unprotected_data_methods(key):
    from qdrant_client import QdrantClient

    from hadamask.adapters.qdrant import ProtectedQdrantClient

    raw = QdrantClient(":memory:")
    pc = ProtectedQdrantClient(raw, Protector(key))
    for method in (
        "set_payload",
        "overwrite_payload",
        "upload_points",
        "upload_collection",
        "batch_update_points",
        "update_vectors",
        "query_batch_points",
        "query_points_groups",
    ):
        with pytest.raises(AttributeError, match="without protection"):
            getattr(pc, method)
    assert pc.collection_exists("nope") is False  # admin methods still pass through
    assert pc.unprotected is raw
    with pytest.raises(TypeError, match="prefetch"):
        pc.query_points("c", query=[0.0] * 8, prefetch=[object()])


def test_langchain_refuses_to_store_unprepared_vectors(key):
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.vectorstores import InMemoryVectorStore

    from hadamask.integrations.langchain import ProtectedVectorStore

    class ReEmbeddingStore(InMemoryVectorStore):
        def add_texts(self, texts, metadatas=None, **kwargs):
            # A store that embeds something else than the texts it was given.
            self.embedding.embed_documents([t + " (modified)" for t in texts])
            return super().add_texts(texts, metadatas=metadatas, **kwargs)

    store = ProtectedVectorStore(ReEmbeddingStore, DeterministicFakeEmbedding(size=16), Protector(key))
    with pytest.raises(RuntimeError, match="did not prepare"):
        store.add_texts(["hello"])


# -- dense Haar transform ---------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 3, 7, 384, 1536])
def test_dense_transform_is_orthogonal_without_padding(key, d):
    from hadamask import DenseTransform

    t = DenseTransform(key, d)
    assert t.processing_dim == d
    x = unit_rows(40, d)
    y = t.encode(x)
    assert y.shape == (40, d)
    assert np.abs(x @ x.T - y.astype(np.float64) @ y.T.astype(np.float64)).max() < 1e-5
    assert np.allclose(t.decode(y), x, atol=1e-6)
    assert np.array_equal(t.encode(x), DenseTransform(MasterKey(key.secret), d).encode(x))


def test_dense_and_hadamard_are_independent_maps(key):
    x = unit_rows(5, 512)
    assert not np.allclose(Protector(key, transform="dense").encode(x), Protector(key).encode(x))


def test_dense_protector_keeps_dimension_and_rekeys(key):
    from hadamask import DenseTransform

    p = Protector(key, transform="dense")
    assert p.stored_dim(1536) == 1536
    old, new = DenseTransform(key, 64), DenseTransform(MasterKey.generate("k2"), 64)
    x = unit_rows(4, 64)
    assert np.allclose(old.rekey_to(new, old.encode(x)), new.encode(x), atol=1e-6)
    with pytest.raises(ValueError):
        Protector(key, transform="dense", processing_dims={384: 1024})
    with pytest.raises(ValueError):
        Protector(key, transform="rot13")
    with pytest.raises(ValueError, match="noise padding"):
        p.transform(16).encode(np.ones(16), noise_norm=0.1)


def test_dense_first_column_is_uniform_on_the_sphere():
    from hadamask import DenseTransform

    cols = np.array([DenseTransform(MasterKey.generate(), 4)._q[:, 0] for _ in range(3000)])
    assert np.allclose(cols.mean(axis=0), 0, atol=0.05)
    assert np.allclose((cols**2).mean(axis=0), 0.25, atol=0.02)


# -- optional dependencies -------------------------------------------------------------------


@pytest.mark.parametrize(
    "module, missing, extra",
    [
        ("hadamask.adapters.qdrant", "qdrant_client", "qdrant"),
        ("hadamask.adapters.pgvector", "psycopg", "pgvector"),
        ("hadamask.adapters.milvus", "pymilvus", "milvus"),
        ("hadamask.integrations.langchain", "langchain_core", "langchain"),
    ],
)
def test_missing_extra_gives_install_hint(monkeypatch, module, missing, extra):
    import importlib
    import sys

    # None in sys.modules makes "import <missing>" raise ImportError; block already-imported
    # submodules too, or "from <missing>.sub import x" would succeed from the cache.
    for name in [m for m in sys.modules if m == missing or m.startswith(missing + ".")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, missing, None)
    monkeypatch.delitem(sys.modules, module, raising=False)
    with pytest.raises(ImportError, match=rf'pip install "hadamask\[{extra}\]"'):
        importlib.import_module(module)
