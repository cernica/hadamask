"""Google Cloud KMS key wrapper."""

from __future__ import annotations

from .base import KeyWrapper


class GcpKmsKeyWrapper(KeyWrapper):
    """Wrap with a Cloud KMS symmetric key (provider ``"gcp-kms"``).

    ``key_name`` is the full resource name
    ``projects/<p>/locations/<l>/keyRings/<r>/cryptoKeys/<k>``. The binding context is passed
    as additional authenticated data. Minimum permissions on this key:
    ``cloudkms.cryptoKeyVersions.useToEncrypt`` and ``cloudkms.cryptoKeyVersions.useToDecrypt``
    (role ``roles/cloudkms.cryptoKeyEncrypterDecrypter``).
    """

    provider = "gcp-kms"

    def __init__(self, key_name: str, client=None) -> None:
        if client is None:
            from google.cloud import kms

            client = kms.KeyManagementServiceClient()
        self.client = client
        self.key_name = key_name

    def wrap(self, plaintext: bytes, context: bytes) -> bytes:
        response = self.client.encrypt(
            request={"name": self.key_name, "plaintext": plaintext, "additional_authenticated_data": context}
        )
        return response.ciphertext

    def unwrap(self, blob: bytes, context: bytes) -> bytes:
        response = self.client.decrypt(
            request={"name": self.key_name, "ciphertext": blob, "additional_authenticated_data": context}
        )
        return response.plaintext
