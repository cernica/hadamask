"""Key management: keep the hadamask master key wrapped by Vault, a cloud KMS or an HSM.

Typical startup code::

    from hadamask.kms import Keyring
    from hadamask.kms.aws import AwsKmsKeyWrapper

    wrapper = AwsKmsKeyWrapper("arn:aws:kms:eu-central-1:111122223333:key/...")
    protector = Keyring.from_wrapped(wrapper, os.environ["HADAMASK_WRAPPED_KEY"]).protector(policy)

Provider modules import their SDK lazily; install the matching extra
(``hadamask[vault]``, ``[aws]``, ``[gcp]``, ``[azure]`` or ``[kms]`` for all).
"""

from .base import (
    KeyManagementError,
    Keyring,
    KeyWrapper,
    LocalKeyWrapper,
    generate_wrapped_key,
    master_key_context,
    parse_wrapped_key,
    unwrap_master_key,
    wrap_master_key,
)

__all__ = [
    "KeyManagementError",
    "KeyWrapper",
    "Keyring",
    "LocalKeyWrapper",
    "generate_wrapped_key",
    "master_key_context",
    "parse_wrapped_key",
    "unwrap_master_key",
    "wrap_master_key",
]
