"""HashiCorp Vault: Transit (wrap keys) and KV v2 (store keys).

* :class:`VaultTransitKeyWrapper` - envelope encryption with the Transit secrets engine. The
  key-encryption key never leaves Vault; the application needs only
  ``update`` on ``transit/encrypt/<name>`` and ``transit/decrypt/<name>``.
* :class:`VaultKVKeyStore` - for teams that use Vault as a secret store: the master key is
  kept in a KV v2 secret, one KV version per hadamask key.
"""

from __future__ import annotations

import base64

from ..keys import MasterKey
from .base import KeyManagementError, Keyring, KeyWrapper


class VaultTransitKeyWrapper(KeyWrapper):
    """Wrap with Vault Transit (provider ``"vault-transit"``).

    Transit keys without key derivation take no associated data, so binding to the key id
    relies on the payload binding verified by :func:`~hadamask.kms.unwrap_master_key`.
    """

    provider = "vault-transit"

    def __init__(self, client, key_name: str, mount_point: str = "transit") -> None:
        self.client = client
        self.key_name = key_name
        self.mount_point = mount_point

    @classmethod
    def ensure_key(cls, client, key_name: str, mount_point: str = "transit") -> VaultTransitKeyWrapper:
        """Create the transit key (``aes256-gcm96``) if it does not exist. Needs admin rights."""
        try:
            client.secrets.transit.read_key(name=key_name, mount_point=mount_point)
        except Exception:
            client.secrets.transit.create_key(name=key_name, key_type="aes256-gcm96", mount_point=mount_point)
        return cls(client, key_name, mount_point)

    def wrap(self, plaintext: bytes, context: bytes) -> bytes:
        response = self.client.secrets.transit.encrypt_data(
            name=self.key_name, plaintext=base64.b64encode(plaintext).decode(), mount_point=self.mount_point
        )
        return response["data"]["ciphertext"].encode()

    def unwrap(self, blob: bytes, context: bytes) -> bytes:
        response = self.client.secrets.transit.decrypt_data(
            name=self.key_name, ciphertext=blob.decode(), mount_point=self.mount_point
        )
        return base64.b64decode(response["data"]["plaintext"])


class VaultKVKeyStore:
    """Master keys stored in a Vault KV v2 secret; each :meth:`store` creates a new version."""

    def __init__(self, client, path: str, mount_point: str = "secret") -> None:
        self.client = client
        self.path = path
        self.mount_point = mount_point

    def store(self, key: MasterKey) -> int:
        """Write ``key`` as a new version and return the version number."""
        response = self.client.secrets.kv.v2.create_or_update_secret(
            path=self.path,
            secret={"key": key.to_base64(), "key_id": key.key_id},
            mount_point=self.mount_point,
        )
        return int(response["data"]["version"])

    def load(self, version: int | None = None) -> MasterKey:
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=self.path, version=version, mount_point=self.mount_point, raise_on_deleted_version=True
            )
        except Exception as exc:
            raise KeyManagementError(f"vault-kv: cannot read {self.path!r} version {version}: {exc}") from exc
        data = response["data"]["data"]
        return MasterKey.from_base64(data["key"], data["key_id"])

    def latest_version(self) -> int:
        meta = self.client.secrets.kv.v2.read_secret_metadata(path=self.path, mount_point=self.mount_point)
        return int(meta["data"]["current_version"])

    def keyring(self, previous_versions: int = 2) -> Keyring:
        """Current key = latest version; previous = up to ``previous_versions`` earlier ones that exist."""
        latest = self.latest_version()
        previous = []
        for version in range(latest - 1, max(0, latest - 1 - previous_versions), -1):
            try:
                previous.append(self.load(version))
            except KeyManagementError:
                continue  # deleted or destroyed versions are skipped
        return Keyring(self.load(latest), tuple(previous))
