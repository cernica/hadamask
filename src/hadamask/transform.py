"""Keyed orthogonal transformation of embeddings.

Implements the structured map from Cernica & Popescu (Appl. Sci. 2026, 16, 10259), Eq. (3):

    Q = H D2 P H D1

where H is the normalized Walsh-Hadamard matrix (applied with an FWHT), D1 and D2 are
diagonal sign matrices and P is a permutation, all derived from the tenant key through
domain-separated HMAC-SHA256 streams (Fisher-Yates with rejection sampling for P).
Native dimensions that are not a power of two are zero-padded to the processing dimension D.

Because Q is orthogonal, dot products, cosine similarity and Euclidean distances between
transformed vectors equal those between the original vectors. This is obfuscation with
measured leakage, not encryption: the stored corpus still reveals its Gram matrix
(all pairwise similarities). See the paper, Section 3.4 and Table 8.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .keys import HmacStream, MasterKey


def next_power_of_two(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def fwht(x: np.ndarray) -> np.ndarray:
    """Normalized fast Walsh-Hadamard transform along the last axis (length a power of two)."""
    n, dim = x.shape
    y = x.copy()
    h = 1
    while h < dim:
        y = y.reshape(n, dim // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = np.stack((a + b, a - b), axis=2)
        h *= 2
    return y.reshape(n, dim) / np.sqrt(dim)


class OrthogonalTransform:
    """The keyed map v -> Q (v, 0) for one native dimension.

    Parameters
    ----------
    key:
        Tenant master key.
    dim:
        Native embedding dimension ``d``.
    processing_dim:
        Processing dimension ``D >= d``; must be a power of two. Defaults to the next power
        of two. Stored vectors have this dimension, so the database collection must use it.
    """

    def __init__(self, key: MasterKey, dim: int, processing_dim: Optional[int] = None) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        big_d = processing_dim or next_power_of_two(dim)
        if big_d < dim or big_d & (big_d - 1):
            raise ValueError("processing_dim must be a power of two and >= dim")
        self.key_id = key.key_id
        self.dim = dim
        self.processing_dim = big_d

        prefix = f"transform/D={big_d}/"
        self._d1 = self._signs(HmacStream(key.secret, prefix + "D1"), big_d)
        self._d2 = self._signs(HmacStream(key.secret, prefix + "D2"), big_d)
        self._perm = self._permutation(HmacStream(key.secret, prefix + "P"), big_d)
        self._inv_perm = np.argsort(self._perm)

    @staticmethod
    def _signs(stream: HmacStream, n: int) -> np.ndarray:
        bits = np.unpackbits(np.frombuffer(stream.read((n + 7) // 8), dtype=np.uint8))[:n]
        return np.where(bits == 1, -1.0, 1.0)

    @staticmethod
    def _permutation(stream: HmacStream, n: int) -> np.ndarray:
        perm = np.arange(n)
        for i in range(n - 1, 0, -1):
            j = stream.uniform_below(i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    @property
    def output_dim(self) -> int:
        return self.processing_dim

    def _as_matrix(self, vectors, width: int) -> tuple[np.ndarray, bool]:
        arr = np.asarray(vectors, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != width:
            raise ValueError(f"expected vectors of dimension {width}, got shape {np.shape(vectors)}")
        if not np.isfinite(arr).all():
            # The FWHT mixes every coordinate into every output: one NaN would poison them all.
            raise ValueError("vectors contain NaN or infinite values")
        return arr, single

    def _forward(self, padded: np.ndarray) -> np.ndarray:
        # Apply D1, FWHT, P, D2, FWHT in that order (paper, Algorithm 1, step 4).
        y = fwht(padded * self._d1)
        y = y[:, self._perm] * self._d2
        return fwht(y)

    def _backward(self, y: np.ndarray) -> np.ndarray:
        # Q^T = D1 H P^T D2 H, since H and the D's are symmetric and P^T = P^-1.
        x = fwht(y) * self._d2
        x = x[:, self._inv_perm]
        return fwht(x) * self._d1

    def encode(self, vectors, noise_norm: float = 0.0) -> np.ndarray:
        """Transform document or query vectors (shape ``(d,)`` or ``(n, d)``) to float32.

        ``noise_norm > 0`` fills the padding coordinates of *documents* with random noise of
        that norm (paper, Section 3.5.5). It keeps dot-product rankings exact but changes
        cosine and L2 scores, so only use it with dot-product collections, and never for
        queries.
        """
        arr, single = self._as_matrix(vectors, self.dim)
        padded = np.zeros((arr.shape[0], self.processing_dim))
        padded[:, : self.dim] = arr
        if noise_norm > 0:
            pad = self.processing_dim - self.dim
            if pad == 0:
                raise ValueError("noise padding needs processing_dim > dim")
            rng = np.random.default_rng(np.random.SeedSequence())
            noise = rng.standard_normal((arr.shape[0], pad))
            noise *= noise_norm / np.linalg.norm(noise, axis=1, keepdims=True)
            padded[:, self.dim :] = noise
        out = self._forward(padded).astype(np.float32)
        return out[0] if single else out

    def decode(self, vectors) -> np.ndarray:
        """Invert :meth:`encode`, dropping the padding coordinates."""
        arr, single = self._as_matrix(vectors, self.processing_dim)
        out = self._backward(arr)[:, : self.dim].astype(np.float32)
        return out[0] if single else out

    def rekey_to(self, new: OrthogonalTransform, vectors) -> np.ndarray:
        """Re-map stored vectors from this key to ``new`` without re-embedding (Q_new Q_old^T).

        Note (paper, Section 6.2): rotating the key does not erase geometry already exposed
        in an earlier snapshot, and persistent record ids link old and new snapshots.
        """
        if new.processing_dim != self.processing_dim or new.dim != self.dim:
            raise ValueError("both transforms must have the same dimensions")
        arr, single = self._as_matrix(vectors, self.processing_dim)
        out = new._forward(self._backward(arr)).astype(np.float32)
        return out[0] if single else out
