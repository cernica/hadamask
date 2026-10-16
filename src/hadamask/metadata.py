"""Field-level metadata protection with AES-256-GCM and keyed blind indexes.

Every protected field value is serialized to JSON and encrypted separately with AES-256-GCM
under a fresh random 96-bit nonce, so two records holding the same value produce different
ciphertexts. The field name and key id are bound as associated data: a ciphertext moved to
another field fails authentication instead of decrypting under the wrong name.

Ciphertexts are stored as ASCII strings so that every database accepts them::

    hm1:<key_id>:<base64url(nonce || ciphertext || tag)>

Before encryption every serialized value is padded (ISO/IEC 7816-4: 0x80 then zero bytes), so
ciphertext length does not reveal the value's exact length. Short values (amounts, IBANs,
dates, names) all pad to the same ``min_padded_length`` bucket, which hides e.g. whether an
amount has 3 or 9 digits. Longer values pad to the next power of two, revealing only their
size class. Fields in ``fixed_lengths`` always pad to exactly that many bytes, and longer
values are rejected instead of leaking.

Fields listed as ``blind`` are encrypted the same way and additionally get a deterministic
HMAC token in ``_hmb_<field>``, which the database can match for equality filters. Blind
tokens reveal which records share a value (and, over time, value frequencies), so only mark
fields as blind when you need to filter on them.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .keys import MasterKey

ENVELOPE_PREFIX = "hm1:"
BLIND_PREFIX = "_hmb_"
RESERVED_PREFIX = "_hm"
NONCE_BYTES = 12
TOKEN_BYTES = 16
DEFAULT_MIN_PADDED_LENGTH = 64
_PAD_MARKER = b"\x80"


class DecryptionError(Exception):
    """A ciphertext failed authentication or was produced with an unknown key."""


@dataclass(frozen=True)
class MetadataPolicy:
    """Which metadata fields are encrypted, left in clear, or made filterable.

    By default every field is encrypted. ``plaintext`` fields are stored unchanged and stay
    fully usable for filtering, sorting and ranges. ``blind`` fields are encrypted and also
    get an equality token (see module docstring).

    Length hiding: every encrypted value is padded to at least ``min_padded_length`` bytes,
    then to the next power of two. ``fixed_lengths`` pads chosen fields to an exact size
    (e.g. ``{"text": 8192}``) so that even long values reveal nothing about their length;
    values that do not fit raise ``ValueError``.
    """

    plaintext: frozenset[str] = field(default_factory=frozenset)
    blind: frozenset[str] = field(default_factory=frozenset)
    min_padded_length: int = DEFAULT_MIN_PADDED_LENGTH
    fixed_lengths: Mapping[str, int] = field(default_factory=dict)

    def __init__(
        self,
        plaintext: Iterable[str] = (),
        blind: Iterable[str] = (),
        min_padded_length: int = DEFAULT_MIN_PADDED_LENGTH,
        fixed_lengths: Optional[Mapping[str, int]] = None,
    ) -> None:
        object.__setattr__(self, "plaintext", frozenset(plaintext))
        object.__setattr__(self, "blind", frozenset(blind))
        object.__setattr__(self, "min_padded_length", min_padded_length)
        object.__setattr__(self, "fixed_lengths", dict(fixed_lengths or {}))
        if min_padded_length < 1:
            raise ValueError("min_padded_length must be at least 1")
        bad = {f: n for f, n in self.fixed_lengths.items() if n < 1}
        if bad:
            raise ValueError(f"fixed lengths must be positive: {bad}")
        clear_fixed = self.plaintext & set(self.fixed_lengths)
        if clear_fixed:
            raise ValueError(f"plaintext fields are not encrypted, so they cannot be padded: {sorted(clear_fixed)}")
        both = self.plaintext & self.blind
        if both:
            raise ValueError(f"fields cannot be both plaintext and blind: {sorted(both)}")
        reserved = [f for f in self.plaintext | self.blind if f.startswith(RESERVED_PREFIX)]
        if reserved:
            raise ValueError(f"field names starting with {RESERVED_PREFIX!r} are reserved: {reserved}")


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def padded_length(n: int, min_length: int = DEFAULT_MIN_PADDED_LENGTH) -> int:
    """Bucket for an ``n``-byte value: ``min_length``, else the next power of two above ``n``."""
    needed = n + len(_PAD_MARKER)
    if needed <= min_length:
        return min_length
    return 1 << (needed - 1).bit_length()


def pad(data: bytes, length: int) -> bytes:
    if len(data) + len(_PAD_MARKER) > length:
        raise ValueError(f"value of {len(data)} bytes does not fit in {length} padded bytes")
    return data + _PAD_MARKER + b"\x00" * (length - len(data) - len(_PAD_MARKER))


def unpad(data: bytes) -> bytes:
    # Canonical JSON never ends in 0x00 or 0x80 (it ends in a quote, bracket, digit or letter),
    # so stripping trailing zeros and then one marker byte is unambiguous.
    body = data.rstrip(b"\x00")
    if not body.endswith(_PAD_MARKER):
        raise ValueError("invalid padding")
    return body[: -len(_PAD_MARKER)]


def is_ciphertext(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(ENVELOPE_PREFIX)


class MetadataCipher:
    """Encrypts and decrypts metadata dictionaries according to a :class:`MetadataPolicy`.

    ``key`` encrypts new data. ``previous_keys`` are only used for decryption, which lets a
    deployment read records written before a key rotation.
    """

    def __init__(
        self,
        key: MasterKey,
        policy: Optional[MetadataPolicy] = None,
        previous_keys: Iterable[MasterKey] = (),
    ) -> None:
        self.policy = policy or MetadataPolicy()
        self.key_id = key.key_id
        self._aead = {k.key_id: AESGCM(k.derive("metadata/aes-256-gcm")) for k in (*previous_keys, key)}
        self._blind_key = key.derive("metadata/blind-index")

    # -- single values ---------------------------------------------------------------

    def _aad(self, key_id: str, name: str) -> bytes:
        return b"hadamask/v1/metadata\x00" + key_id.encode() + b"\x00" + name.encode()

    def _padded(self, name: str, data: bytes) -> bytes:
        fixed = self.policy.fixed_lengths.get(name)
        if fixed is not None:
            try:
                return pad(data, fixed)
            except ValueError as exc:
                raise ValueError(f"field {name!r}: {exc} (fixed_lengths[{name!r}] = {fixed})") from None
        return pad(data, padded_length(len(data), self.policy.min_padded_length))

    def encrypt_value(self, name: str, value: Any) -> str:
        nonce = os.urandom(NONCE_BYTES)
        plain = self._padded(name, _canonical(value))
        ct = self._aead[self.key_id].encrypt(nonce, plain, self._aad(self.key_id, name))
        return f"{ENVELOPE_PREFIX}{self.key_id}:{_b64e(nonce + ct)}"

    def decrypt_value(self, name: str, envelope: str) -> Any:
        try:
            _, key_id, body = envelope.split(":", 2)
            aead = self._aead[key_id]
            raw = _b64d(body)
            plain = aead.decrypt(raw[:NONCE_BYTES], raw[NONCE_BYTES:], self._aad(key_id, name))
        except KeyError as exc:
            raise DecryptionError(f"field {name!r}: unknown key id") from exc
        except (InvalidTag, ValueError) as exc:
            raise DecryptionError(f"field {name!r}: authentication failed") from exc
        try:
            return json.loads(unpad(plain))
        except ValueError as exc:
            raise DecryptionError(f"field {name!r}: malformed plaintext") from exc

    def blind_field(self, name: str) -> str:
        """Name of the stored field that holds equality tokens for ``name``."""
        return BLIND_PREFIX + name

    def blind_token(self, name: str, value: Any) -> str:
        """Deterministic equality token for ``value`` in field ``name``."""
        mac = hmac.new(self._blind_key, name.encode() + b"\x00" + _canonical(value), hashlib.sha256)
        return _b64e(mac.digest()[:TOKEN_BYTES])

    # -- whole payloads --------------------------------------------------------------

    def encrypt(self, payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, value in (payload or {}).items():
            if name.startswith(RESERVED_PREFIX):
                raise ValueError(f"field names starting with {RESERVED_PREFIX!r} are reserved: {name}")
            if name in self.policy.plaintext:
                out[name] = value
                continue
            out[name] = self.encrypt_value(name, value)
            if name in self.policy.blind:
                if isinstance(value, (list, tuple)):
                    out[self.blind_field(name)] = [self.blind_token(name, v) for v in value]
                else:
                    out[self.blind_field(name)] = self.blind_token(name, value)
        return out

    def decrypt(self, stored: Optional[Mapping[str, Any]], fields: Optional[Iterable[str]] = None) -> dict[str, Any]:
        """Decrypt a stored payload; with ``fields``, only those fields are decrypted and returned."""
        wanted = None if fields is None else set(fields)
        out: dict[str, Any] = {}
        for name, value in (stored or {}).items():
            if name.startswith(RESERVED_PREFIX) or (wanted is not None and name not in wanted):
                continue
            if name not in self.policy.plaintext and is_ciphertext(value):
                out[name] = self.decrypt_value(name, value)
            else:
                out[name] = value
        return out
