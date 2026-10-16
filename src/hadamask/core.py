"""The :class:`Protector`: one object holding everything a client needs to protect data."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import numpy as np

from .keys import MasterKey
from .metadata import MetadataCipher, MetadataPolicy
from .transform import OrthogonalTransform, next_power_of_two


class Protector:
    """Client-side protection for one tenant key.

    * vectors: keyed orthogonal transform (exact similarity preservation, measured leakage);
    * metadata: AES-256-GCM per field, optional blind equality tokens.

    Parameters
    ----------
    key:
        The tenant master key used for new writes.
    policy:
        Metadata policy; by default every field is encrypted.
    processing_dims:
        Optional ``{native_dim: processing_dim}`` overrides. By default each native
        dimension is padded to the next power of two.
    previous_keys:
        Older keys, used only to decrypt metadata written before a key rotation.
    """

    def __init__(
        self,
        key: MasterKey,
        policy: Optional[MetadataPolicy] = None,
        processing_dims: Optional[Mapping[int, int]] = None,
        previous_keys: Iterable[MasterKey] = (),
    ) -> None:
        self.key = key
        self.metadata = MetadataCipher(key, policy, previous_keys)
        self._processing_dims = dict(processing_dims or {})
        self._transforms: dict[int, OrthogonalTransform] = {}

    @property
    def key_id(self) -> str:
        return self.key.key_id

    @property
    def policy(self) -> MetadataPolicy:
        return self.metadata.policy

    # -- vectors ---------------------------------------------------------------------

    def transform(self, dim: int) -> OrthogonalTransform:
        if dim not in self._transforms:
            self._transforms[dim] = OrthogonalTransform(self.key, dim, self._processing_dims.get(dim))
        return self._transforms[dim]

    def stored_dim(self, dim: int) -> int:
        """Dimension of protected vectors for native dimension ``dim`` (create collections with it)."""
        return self._processing_dims.get(dim) or next_power_of_two(dim)

    def encode(self, vectors) -> np.ndarray:
        """Protect document or query vectors: ``(d,)`` or ``(n, d)`` -> float32 ``(D,)``/``(n, D)``."""
        dim = np.shape(vectors)[-1]
        return self.transform(dim).encode(vectors)

    def encode_list(self, vector) -> list[float]:
        """Protect a single vector and return a plain list (what most clients expect)."""
        return self.encode(vector).tolist()

    def encode_many(self, vectors) -> list[list[float]]:
        if len(vectors) == 0:
            return []
        return self.encode(np.asarray(vectors)).tolist()

    def decode(self, vectors, dim: int) -> np.ndarray:
        """Recover native vectors (native dimension ``dim``) from protected ones."""
        return self.transform(dim).decode(vectors)

    # -- metadata --------------------------------------------------------------------

    def encrypt_payload(self, payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        return self.metadata.encrypt(payload)

    def decrypt_payload(
        self, stored: Optional[Mapping[str, Any]], fields: Optional[Iterable[str]] = None
    ) -> dict[str, Any]:
        return self.metadata.decrypt(stored, fields)

    def blind(self, field: str, value: Any) -> tuple[str, str]:
        """``(stored_field, token)`` to use in a native equality filter on a blind field."""
        if field not in self.policy.blind:
            raise ValueError(f"field {field!r} is not declared as blind in the metadata policy")
        return self.metadata.blind_field(field), self.metadata.blind_token(field, value)
