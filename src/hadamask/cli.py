"""Command line helpers: ``hadamask keygen`` and ``hadamask info``."""

from __future__ import annotations

import argparse
import base64
import os
import sys

from ._deps import missing_extra
from .keys import MasterKey
from .transform import next_power_of_two

WRAP_PROVIDERS = ("local", "vault-transit", "aws-kms", "gcp-kms", "azure-kv")


def _wrapper(args):
    """Build the KeyWrapper selected by ``--wrap`` from the provider options."""
    if args.wrap == "local":
        kek = os.environ.get(args.local_kek_env)
        if not kek:
            raise SystemExit(f"--wrap local needs a base64 32-byte KEK in ${args.local_kek_env}")
        from .kms import LocalKeyWrapper

        return LocalKeyWrapper(base64.b64decode(kek))
    if args.wrap == "vault-transit":
        try:
            import hvac
        except ImportError as exc:
            raise missing_extra("vault", exc) from exc

        from .kms.vault import VaultTransitKeyWrapper

        if not args.transit_key:
            raise SystemExit("--wrap vault-transit needs --transit-key")
        client = hvac.Client(url=args.vault_url, token=os.environ.get(args.vault_token_env))
        return VaultTransitKeyWrapper(client, args.transit_key, args.transit_mount)
    if args.wrap == "aws-kms":
        from .kms.aws import AwsKmsKeyWrapper

        if not args.aws_kms_key_id:
            raise SystemExit("--wrap aws-kms needs --aws-kms-key-id")
        return AwsKmsKeyWrapper(args.aws_kms_key_id, region_name=args.aws_region, endpoint_url=args.aws_endpoint_url)
    if args.wrap == "gcp-kms":
        from .kms.gcp import GcpKmsKeyWrapper

        if not args.gcp_key_name:
            raise SystemExit("--wrap gcp-kms needs --gcp-key-name")
        return GcpKmsKeyWrapper(args.gcp_key_name)
    from .kms.azure import AzureKeyVaultKeyWrapper

    if not args.azure_key_id:
        raise SystemExit("--wrap azure-kv needs --azure-key-id")
    return AzureKeyVaultKeyWrapper(args.azure_key_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hadamask", description="Client-side protection for vector databases.")
    sub = parser.add_subparsers(dest="command", required=True)

    keygen = sub.add_parser("keygen", help="generate a new 256-bit master key")
    keygen.add_argument("--key-id", default="k1", help="public key identifier stored with ciphertexts")
    keygen.add_argument("--env", action="store_true", help="print as environment variable assignments")
    keygen.add_argument(
        "--wrap", choices=WRAP_PROVIDERS, help="wrap the new key with a key manager; the raw key is never printed"
    )
    kms = keygen.add_argument_group("key manager options (with --wrap)")
    kms.add_argument("--local-kek-env", default="HADAMASK_LOCAL_KEK", help="env var holding a base64 32-byte KEK")
    kms.add_argument("--vault-url", default=os.environ.get("VAULT_ADDR", "http://127.0.0.1:8200"))
    kms.add_argument("--vault-token-env", default="VAULT_TOKEN", help="env var holding the Vault token")
    kms.add_argument("--transit-key", help="Vault Transit key name")
    kms.add_argument("--transit-mount", default="transit")
    kms.add_argument("--aws-kms-key-id", help="AWS KMS key id, ARN or alias")
    kms.add_argument("--aws-region")
    kms.add_argument("--aws-endpoint-url", help="e.g. http://127.0.0.1:4566 for LocalStack")
    kms.add_argument("--gcp-key-name", help="projects/<p>/locations/<l>/keyRings/<r>/cryptoKeys/<k>")
    kms.add_argument("--azure-key-id", help="https://<vault>.vault.azure.net/keys/<name>/<version>")

    info = sub.add_parser("info", help="show the stored dimension for a native embedding dimension")
    info.add_argument("dim", type=int, help="native embedding dimension, e.g. 1536")

    args = parser.parse_args(argv)
    if args.command == "keygen" and args.wrap:
        from .kms import generate_wrapped_key

        _, wrapped = generate_wrapped_key(_wrapper(args), args.key_id)
        print(f"HADAMASK_WRAPPED_KEY={wrapped}" if args.env else wrapped)
        print(f"Wrapped with {args.wrap}; only that key manager can unwrap it.", file=sys.stderr)
    elif args.command == "keygen":
        key = MasterKey.generate(args.key_id)
        if args.env:
            print(f"HADAMASK_MASTER_KEY={key.to_base64()}")
            print(f"HADAMASK_KEY_ID={key.key_id}")
        else:
            print(key.to_base64())
        print(
            "Store this key in a secret manager (Vault, KMS, HSM). Losing it makes the data unreadable.",
            file=sys.stderr,
        )
    elif args.command == "info":
        stored = next_power_of_two(args.dim)
        print(f"native dimension {args.dim} -> stored dimension {stored} (create collections with {stored})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
