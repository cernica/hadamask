"""Key management: wrapping master keys with Vault, AWS KMS, GCP KMS, Azure Key Vault.

Real emulators (docker compose up -d vault localstack lowkey-vault):
* Vault 1.18 dev server (Transit and KV v2);
* LocalStack (AWS KMS API);
* Lowkey Vault (Azure Key Vault API).
GCP Cloud KMS has no emulator; it is tested against a fake client with AAD semantics.
"""

from __future__ import annotations

import os
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import require_service
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from hadamask import MasterKey, MetadataPolicy
from hadamask.kms import (
    KeyManagementError,
    Keyring,
    LocalKeyWrapper,
    generate_wrapped_key,
    parse_wrapped_key,
    unwrap_master_key,
    wrap_master_key,
)

VAULT_URL = "http://127.0.0.1:8200"
VAULT_TOKEN = "hadamask-dev-token"  # noqa: S105 - Vault dev-mode root token from docker-compose.yml


def _tamper_key_id(wrapped: str, new_id: str) -> str:
    prefix, provider, _, blob = wrapped.split(":")
    return ":".join([prefix, provider, new_id, blob])


def _assert_roundtrip(make_wrapper):
    """Generate, unwrap with a fresh wrapper instance ("new process"), and check tampering fails."""
    key, wrapped = generate_wrapped_key(make_wrapper(), "prod-2026")
    assert key.to_base64() not in wrapped
    restored = unwrap_master_key(make_wrapper(), wrapped)
    assert restored.secret == key.secret and restored.key_id == "prod-2026"
    with pytest.raises(KeyManagementError):
        unwrap_master_key(make_wrapper(), _tamper_key_id(wrapped, "other"))
    return key, wrapped


# -- format and local wrapper ---------------------------------------------------------------


def test_local_wrapper_roundtrip_and_tamper():
    kek = os.urandom(32)
    key, wrapped = _assert_roundtrip(lambda: LocalKeyWrapper(kek))
    provider, key_id, _ = parse_wrapped_key(wrapped)
    assert (provider, key_id) == ("local", "prod-2026")
    bad = f"hmk1:local:prod-2026:{wrapped.split(':')[3][:-2]}AA"
    with pytest.raises(KeyManagementError):
        unwrap_master_key(LocalKeyWrapper(kek), bad)
    with pytest.raises(KeyManagementError):
        unwrap_master_key(LocalKeyWrapper(os.urandom(32)), wrapped)


def test_wrong_provider_and_malformed_strings():
    _, wrapped = generate_wrapped_key(LocalKeyWrapper(os.urandom(32)))
    other = SimpleNamespace(provider="aws-kms")
    with pytest.raises(KeyManagementError, match="wrapped by 'local'"):
        unwrap_master_key(other, wrapped)
    for bad in ("", "hmk1:local:k1", "hmk2:local:k1:AAAA", "hmk1:local:k1:@@@"):
        with pytest.raises(KeyManagementError):
            unwrap_master_key(LocalKeyWrapper(os.urandom(32)), bad)


def test_keyring_protector_reads_old_and_writes_new():
    wrapper = LocalKeyWrapper(os.urandom(32))
    old, old_wrapped = generate_wrapped_key(wrapper, "k1")
    _, new_wrapped = generate_wrapped_key(wrapper, "k2")
    policy = MetadataPolicy(blind=["client"])
    old_record = Keyring(old).protector(policy).encrypt_payload({"text": "old", "client": "ACME"})
    protector = Keyring.from_wrapped(wrapper, new_wrapped, [old_wrapped]).protector(policy)
    assert protector.decrypt_payload(old_record) == {"text": "old", "client": "ACME"}
    assert protector.encrypt_payload({"text": "new"})["text"].startswith("hm1:k2:")


def test_end_to_end_same_transform_after_unwrap():
    kek = os.urandom(32)
    key, wrapped = generate_wrapped_key(LocalKeyWrapper(kek), "k1")
    a = Keyring(key).protector()
    b = Keyring.from_wrapped(LocalKeyWrapper(kek), wrapped).protector()
    v = np.random.default_rng(0).standard_normal((3, 384))
    assert np.array_equal(a.encode(v), b.encode(v))
    assert b.decrypt_payload(a.encrypt_payload({"amount": 1250000})) == {"amount": 1250000}


# -- Vault ----------------------------------------------------------------------------------


@pytest.fixture
def vault():
    require_service(8200, "vault")
    hvac = pytest.importorskip("hvac")
    client = hvac.Client(url=VAULT_URL, token=VAULT_TOKEN)
    if "transit/" not in client.sys.list_mounted_secrets_engines()["data"]:
        client.sys.enable_secrets_engine("transit")
    return client


@pytest.mark.integration
def test_vault_transit(vault):
    from hadamask.kms.vault import VaultTransitKeyWrapper

    name = f"hadamask-{uuid.uuid4().hex[:8]}"
    VaultTransitKeyWrapper.ensure_key(vault, name)
    _assert_roundtrip(lambda: VaultTransitKeyWrapper(vault, name))


@pytest.mark.integration
def test_vault_kv_keyring(vault):
    from hadamask.kms.vault import VaultKVKeyStore

    store = VaultKVKeyStore(vault, f"hadamask/test-{uuid.uuid4().hex[:8]}")
    k1, k2 = MasterKey.generate("k1"), MasterKey.generate("k2")
    assert store.store(k1) == 1 and store.store(k2) == 2
    ring = store.keyring()
    assert ring.current.key_id == "k2" and ring.current.secret == k2.secret
    assert [k.key_id for k in ring.previous] == ["k1"]
    assert store.load(1).secret == k1.secret
    old_record = Keyring(k1).protector().encrypt_payload({"text": "written with k1"})
    protector = ring.protector()
    assert protector.decrypt_payload(old_record) == {"text": "written with k1"}
    assert protector.encrypt_payload({"text": "x"})["text"].startswith("hm1:k2:")


# -- AWS KMS (LocalStack) -------------------------------------------------------------------


@pytest.mark.integration
def test_aws_kms_localstack():
    require_service(4566, "localstack")
    boto3 = pytest.importorskip("boto3")
    from hadamask.kms.aws import AwsKmsKeyWrapper

    client = boto3.client(
        "kms",
        region_name="us-east-1",
        endpoint_url="http://127.0.0.1:4566",
        aws_access_key_id="test",
        aws_secret_access_key="test",  # noqa: S106 - LocalStack dummy credentials
    )
    key_id = client.create_key(Description="hadamask test")["KeyMetadata"]["KeyId"]
    _, wrapped = _assert_roundtrip(lambda: AwsKmsKeyWrapper(key_id, client=client))
    # The encryption context is enforced by KMS itself, not only by our payload binding.
    blob = parse_wrapped_key(wrapped)[2]
    with pytest.raises(Exception):  # noqa: B017 - botocore InvalidCiphertextException
        client.decrypt(KeyId=key_id, CiphertextBlob=blob, EncryptionContext={"hadamask-context": "wrong"})


# -- Azure Key Vault (Lowkey Vault) ---------------------------------------------------------


class _DummyCredential:
    """Lowkey Vault accepts any bearer token."""

    def get_token(self, *scopes, **kwargs):
        from azure.core.credentials import AccessToken

        return AccessToken("dummy", int(time.time()) + 3600)

    def get_token_info(self, *scopes, **kwargs):
        from azure.core.credentials import AccessTokenInfo

        return AccessTokenInfo("dummy", int(time.time()) + 3600)


@pytest.mark.integration
def test_azure_key_vault_lowkey():
    require_service(8443, "lowkey-vault")
    pytest.importorskip("azure.keyvault.keys")
    import urllib3
    from azure.keyvault.keys import KeyClient, KeyOperation

    from hadamask.kms.azure import AzureKeyVaultKeyWrapper

    urllib3.disable_warnings()
    # Lowkey Vault: self-signed TLS, no challenge resource, API 7.5, and it checks the
    # encrypt/decrypt operations for wrapKey/unwrapKey (real Key Vault needs only wrap/unwrap).
    kwargs = {"connection_verify": False, "verify_challenge_resource": False, "api_version": "7.5"}
    ops = [KeyOperation.wrap_key, KeyOperation.unwrap_key, KeyOperation.encrypt, KeyOperation.decrypt]
    key = KeyClient("https://localhost:8443", _DummyCredential(), **kwargs).create_rsa_key(
        f"hadamask-{uuid.uuid4().hex[:8]}", size=2048, key_operations=ops
    )
    _assert_roundtrip(lambda: AzureKeyVaultKeyWrapper(key.id, credential=_DummyCredential(), **kwargs))


# -- GCP Cloud KMS (fake) -------------------------------------------------------------------


class _FakeGcpKms:
    """Mimics KeyManagementServiceClient.encrypt/decrypt with AAD (no emulator exists)."""

    def __init__(self):
        self._keys: dict[str, AESGCM] = {}

    def _aead(self, name):
        return self._keys.setdefault(name, AESGCM(AESGCM.generate_key(256)))

    def encrypt(self, request):
        nonce = os.urandom(12)
        ct = self._aead(request["name"]).encrypt(nonce, request["plaintext"], request["additional_authenticated_data"])
        return SimpleNamespace(ciphertext=nonce + ct)

    def decrypt(self, request):
        blob = request["ciphertext"]
        pt = self._aead(request["name"]).decrypt(blob[:12], blob[12:], request["additional_authenticated_data"])
        return SimpleNamespace(plaintext=pt)


def test_gcp_kms_fake_client():
    from hadamask.kms.gcp import GcpKmsKeyWrapper

    fake = _FakeGcpKms()
    name = "projects/p/locations/europe-west3/keyRings/r/cryptoKeys/hadamask"
    _assert_roundtrip(lambda: GcpKmsKeyWrapper(name, client=fake))


def test_wrapped_key_never_contains_raw_key():
    key = MasterKey.generate("k1")
    wrapped = wrap_master_key(LocalKeyWrapper(os.urandom(32)), key)
    assert key.to_base64() not in wrapped and key.secret.hex() not in wrapped


def test_cli_keygen_wrap_local(monkeypatch, capsys):
    import base64

    from hadamask.cli import main

    kek = os.urandom(32)
    monkeypatch.setenv("HADAMASK_LOCAL_KEK", base64.b64encode(kek).decode())
    assert main(["keygen", "--wrap", "local", "--key-id", "prod1", "--env"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("HADAMASK_WRAPPED_KEY=hmk1:local:prod1:") and "MASTER_KEY" not in out
    key = unwrap_master_key(LocalKeyWrapper(kek), out.split("=", 1)[1])
    assert key.key_id == "prod1"
