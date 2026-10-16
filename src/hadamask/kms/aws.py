"""AWS KMS key wrapper (also works with LocalStack via ``endpoint_url``)."""

from __future__ import annotations

from .base import KeyWrapper

_CONTEXT_KEY = "hadamask-context"


class AwsKmsKeyWrapper(KeyWrapper):
    """Wrap with an AWS KMS symmetric key (provider ``"aws-kms"``).

    The binding context is passed as the KMS encryption context, so it also appears in
    CloudTrail and can be required by key policy conditions (``kms:EncryptionContext:*``).
    Minimum permissions: ``kms:Encrypt`` and ``kms:Decrypt`` on this key only.
    """

    provider = "aws-kms"

    def __init__(self, key_id: str, client=None, region_name: str | None = None, endpoint_url: str | None = None):
        if client is None:
            import boto3

            client = boto3.client("kms", region_name=region_name, endpoint_url=endpoint_url)
        self.client = client
        self.key_id = key_id

    def wrap(self, plaintext: bytes, context: bytes) -> bytes:
        response = self.client.encrypt(
            KeyId=self.key_id, Plaintext=plaintext, EncryptionContext={_CONTEXT_KEY: context.decode()}
        )
        return response["CiphertextBlob"]

    def unwrap(self, blob: bytes, context: bytes) -> bytes:
        response = self.client.decrypt(
            KeyId=self.key_id, CiphertextBlob=blob, EncryptionContext={_CONTEXT_KEY: context.decode()}
        )
        return response["Plaintext"]
