"""hadamask: client-side protection for vector databases.

Embeddings are transformed with a keyed orthogonal map (exact similarity preservation,
measured leakage - not encryption); metadata is encrypted per field with AES-256-GCM.
"""

from .core import Protector
from .keys import MasterKey
from .metadata import DecryptionError, MetadataCipher, MetadataPolicy
from .transform import OrthogonalTransform

__all__ = [
    "DecryptionError",
    "MasterKey",
    "MetadataCipher",
    "MetadataPolicy",
    "OrthogonalTransform",
    "Protector",
]
__version__ = "0.1.0"
