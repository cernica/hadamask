"""Envelope encryption of hadamask master keys.

The 256-bit master key never needs to sit in clear on disk or in an environment variable.
It is *wrapped* (encrypted) by a key-encryption key held in a KMS, an HSM or Vault, and the
wrapped form is what you store in configuration::

    hmk1:<provider>:<key_id>:<base64url(blob)>

At startup the application asks the KMS to unwrap it, and the clear key only lives in memory.

Binding: the wrapped payload is ``b"hadamask-mk1\\x00" + context + b"\\x00" + key`` with
``context = b"hadamask/master-key/v1/" + key_id``. It is verified on unwrap, so a blob cannot
be relabelled with another key id or reused for another purpose. Providers that support
associated data (AWS encryption context, GCP AAD, local AES-GCM) also receive the context as
AAD.
"""

from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ..core import Protector
from ..keys import KEY_BYTES, MasterKey
from ..metadata import MetadataPolicy

WRAPPED_PREFIX = "hmk1"
_MAGIC = b"hadamask-mk1\x00"
_CONTEXT_PREFIX = b"hadamask/master-key/v1/"


class KeyManagementError(Exception):
    """A wrapped key could not be produced, parsed, unwrapped or verified."""


def master_key_context(key_id: str) -> bytes:
    return _CONTEXT_PREFIX + key_id.encode()


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class KeyWrapper(ABC):
    """Wraps and unwraps small secrets with a key held by an external key manager."""

    provider: str

    @abstractmethod
    def wrap(self, plaintext: bytes, context: bytes) -> bytes:
        """Encrypt ``plaintext``; ``context`` is associated data where the provider supports it."""

    @abstractmethod
    def unwrap(self, blob: bytes, context: bytes) -> bytes:
        """Decrypt ``blob`` produced by :meth:`wrap` with the same ``context``."""


def _bind(context: bytes, key: bytes) -> bytes:
    if b"\x00" in context:
        raise KeyManagementError("context must not contain NUL bytes")
    return _MAGIC + context + b"\x00" + key


def _unbind(payload: bytes, context: bytes) -> bytes:
    expected = _MAGIC + context + b"\x00"
    if not payload.startswith(expected) or len(payload) != len(expected) + KEY_BYTES:
        raise KeyManagementError("unwrapped key is not bound to this key id")
    return payload[len(expected) :]


def wrap_master_key(wrapper: KeyWrapper, key: MasterKey) -> str:
    """Wrap ``key`` and return the ``hmk1:...`` string to store in configuration."""
    context = master_key_context(key.key_id)
    try:
        blob = wrapper.wrap(_bind(context, key.secret), context)
    except KeyManagementError:
        raise
    except Exception as exc:  # provider SDK errors
        raise KeyManagementError(f"{wrapper.provider}: wrapping failed: {exc}") from exc
    return f"{WRAPPED_PREFIX}:{wrapper.provider}:{key.key_id}:{_b64e(blob)}"


def generate_wrapped_key(wrapper: KeyWrapper, key_id: str = "k1") -> tuple[MasterKey, str]:
    """Generate a new master key and return it together with its wrapped form."""
    key = MasterKey.generate(key_id)
    return key, wrap_master_key(wrapper, key)


def parse_wrapped_key(wrapped: str) -> tuple[str, str, bytes]:
    """Split ``hmk1:<provider>:<key_id>:<blob>`` into ``(provider, key_id, blob)``."""
    parts = wrapped.strip().split(":")
    if len(parts) != 4 or parts[0] != WRAPPED_PREFIX:
        raise KeyManagementError("not a hadamask wrapped key (expected hmk1:<provider>:<key_id>:<blob>)")
    _, provider, key_id, body = parts
    try:
        return provider, key_id, _b64d(body)
    except ValueError as exc:
        raise KeyManagementError("wrapped key blob is not valid base64url") from exc


def unwrap_master_key(wrapper: KeyWrapper, wrapped: str) -> MasterKey:
    """Ask the key manager to unwrap ``wrapped`` and return the master key (in memory only)."""
    provider, key_id, blob = parse_wrapped_key(wrapped)
    if provider != wrapper.provider:
        raise KeyManagementError(f"key was wrapped by {provider!r}, not {wrapper.provider!r}")
    context = master_key_context(key_id)
    try:
        payload = wrapper.unwrap(blob, context)
    except KeyManagementError:
        raise
    except Exception as exc:
        raise KeyManagementError(f"{provider}: unwrapping failed: {exc}") from exc
    return MasterKey(_unbind(payload, context), key_id)


@dataclass(frozen=True)
class Keyring:
    """The current master key plus older keys still needed to read existing records."""

    current: MasterKey
    previous: tuple[MasterKey, ...] = field(default_factory=tuple)

    @classmethod
    def from_wrapped(cls, wrapper: KeyWrapper, current: str, previous: Iterable[str] = ()) -> Keyring:
        return cls(unwrap_master_key(wrapper, current), tuple(unwrap_master_key(wrapper, w) for w in previous))

    def protector(
        self, policy: Optional[MetadataPolicy] = None, processing_dims: Optional[Mapping[int, int]] = None
    ) -> Protector:
        return Protector(self.current, policy, processing_dims, previous_keys=self.previous)


class LocalKeyWrapper(KeyWrapper):
    """AES-256-GCM with a local key-encryption key (provider ``"local"``).

    For tests, development and air-gapped setups. The KEK is just another secret on the same
    machine, so this is NOT a substitute for a KMS or an HSM.
    """

    provider = "local"

    def __init__(self, kek: bytes) -> None:
        if len(kek) != 32:
            raise ValueError("the local key-encryption key must be 32 bytes")
        self._aead = AESGCM(kek)

    def wrap(self, plaintext: bytes, context: bytes) -> bytes:
        nonce = os.urandom(12)
        return nonce + self._aead.encrypt(nonce, plaintext, context)

    def unwrap(self, blob: bytes, context: bytes) -> bytes:
        try:
            return self._aead.decrypt(blob[:12], blob[12:], context)
        except InvalidTag as exc:
            raise KeyManagementError("local: authentication failed") from exc
