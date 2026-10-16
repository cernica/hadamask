"""Azure Key Vault / Managed HSM key wrapper."""

from __future__ import annotations

from .base import KeyWrapper


class AzureKeyVaultKeyWrapper(KeyWrapper):
    """Wrap with an Azure Key Vault key (provider ``"azure-kv"``).

    ``key_id`` is the key URL (``https://<vault>.vault.azure.net/keys/<name>[/<version>]``).
    Pin the version in production so that unwrapping keeps working after the Key Vault key
    is rotated. ``wrapKey`` takes no associated data; the binding to the hadamask key id is
    carried in the wrapped payload and verified on unwrap. Minimum permissions: the
    ``wrapKey`` and ``unwrapKey`` key operations (role "Key Vault Crypto User").
    """

    provider = "azure-kv"

    def __init__(self, key_id: str, credential=None, crypto_client=None, algorithm=None, **client_kwargs) -> None:
        from azure.keyvault.keys.crypto import CryptographyClient, KeyWrapAlgorithm

        if crypto_client is None:
            if credential is None:
                from azure.identity import DefaultAzureCredential

                credential = DefaultAzureCredential()
            crypto_client = CryptographyClient(key_id, credential, **client_kwargs)
        self.crypto_client = crypto_client
        self.key_id = key_id
        self.algorithm = algorithm or KeyWrapAlgorithm.rsa_oaep_256

    def wrap(self, plaintext: bytes, context: bytes) -> bytes:
        return self.crypto_client.wrap_key(self.algorithm, plaintext).encrypted_key

    def unwrap(self, blob: bytes, context: bytes) -> bytes:
        return self.crypto_client.unwrap_key(self.algorithm, blob).key
