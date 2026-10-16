"""Key material and domain-separated derivation.

A tenant owns one 256-bit master key. Every secret used by the library (sign vectors,
permutation, metadata encryption key, blind-index key) is derived from it with HMAC-SHA256
under a distinct label, so compromising one derived value does not reveal the others.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass, field

KEY_BYTES = 32
_LABEL_PREFIX = b"hadamask/v1/"


@dataclass(frozen=True)
class MasterKey:
    """A 256-bit tenant master key with a public identifier.

    ``key_id`` is stored next to protected data so that readers can tell which key (and
    which key version) produced a record. It must not contain secret information.
    """

    secret: bytes = field(repr=False)
    key_id: str = "k1"

    def __post_init__(self) -> None:
        if len(self.secret) != KEY_BYTES:
            raise ValueError(f"master key must be {KEY_BYTES} bytes, got {len(self.secret)}")
        if not self.key_id or any(c in self.key_id for c in ":. "):
            raise ValueError("key_id must be non-empty and must not contain ':', '.' or spaces")

    @classmethod
    def generate(cls, key_id: str = "k1") -> MasterKey:
        return cls(secrets.token_bytes(KEY_BYTES), key_id)

    @classmethod
    def from_base64(cls, value: str, key_id: str = "k1") -> MasterKey:
        return cls(base64.b64decode(value, validate=True), key_id)

    @classmethod
    def from_env(cls, var: str = "HADAMASK_MASTER_KEY", key_id_var: str = "HADAMASK_KEY_ID") -> MasterKey:
        value = os.environ.get(var)
        if not value:
            raise KeyError(f"environment variable {var} is not set")
        return cls.from_base64(value, os.environ.get(key_id_var, "k1"))

    def to_base64(self) -> str:
        return base64.b64encode(self.secret).decode()

    def derive(self, label: str, length: int = KEY_BYTES) -> bytes:
        """Derive ``length`` bytes bound to ``label`` (HMAC-SHA256 in counter mode)."""
        return b"".join(hmac_stream(self.secret, label, (length + 31) // 32))[:length]


def hmac_stream(key: bytes, label: str, blocks: int):
    """Yield ``blocks`` 32-byte blocks of HMAC-SHA256(key, prefix || label || counter)."""
    tag = _LABEL_PREFIX + label.encode()
    for counter in range(blocks):
        yield hmac.new(key, tag + b"\x00" + counter.to_bytes(8, "big"), hashlib.sha256).digest()


class HmacStream:
    """Deterministic byte stream from a key and a domain label (HMAC-SHA256, counter mode)."""

    def __init__(self, key: bytes, label: str) -> None:
        self._key = key
        self._tag = _LABEL_PREFIX + label.encode()
        self._counter = 0
        self._buffer = b""

    def read(self, n: int) -> bytes:
        while len(self._buffer) < n:
            block = hmac.new(self._key, self._tag + b"\x00" + self._counter.to_bytes(8, "big"), hashlib.sha256).digest()
            self._counter += 1
            self._buffer += block
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out

    def uniform_below(self, bound: int) -> int:
        """Uniform integer in [0, bound) by rejection sampling (no modulo bias)."""
        if bound <= 0:
            raise ValueError("bound must be positive")
        if bound == 1:
            return 0
        nbytes = ((bound - 1).bit_length() + 7) // 8
        limit = (256**nbytes // bound) * bound
        while True:
            value = int.from_bytes(self.read(nbytes), "big")
            if value < limit:
                return value % bound
