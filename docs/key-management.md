# Key management

hadamask derives everything (vector transform, metadata AES key, blind-index key) from one
256-bit master key per tenant. Whoever holds that key can read the protected data, so it
should never sit in clear in a file, an environment variable or a container image.

## Envelope encryption

The master key is **wrapped** (encrypted) by a key-encryption key that never leaves your
key manager: Vault Transit, AWS KMS, Google Cloud KMS, Azure Key Vault or Managed HSM. You
store only the wrapped form, a single string:

```
hmk1:<provider>:<key_id>:<base64url(blob)>
```

At startup the application asks the key manager to unwrap it; the clear key exists only in
process memory. The wrapped payload is bound to its `key_id` (and, where the provider
supports it, the binding is also passed as associated data / encryption context), so a blob
cannot be relabelled with another key id.

```bash
pip install "hadamask[aws]"          # or [vault], [gcp], [azure], [kms] for all

hadamask keygen --wrap aws-kms --aws-kms-key-id alias/hadamask --key-id prod-2026 --env
# HADAMASK_WRAPPED_KEY=hmk1:aws-kms:prod-2026:AQICAHh...
```

```python
import os
from hadamask import MetadataPolicy
from hadamask.kms import Keyring
from hadamask.kms.aws import AwsKmsKeyWrapper

wrapper = AwsKmsKeyWrapper("alias/hadamask", region_name="eu-central-1")
protector = Keyring.from_wrapped(wrapper, os.environ["HADAMASK_WRAPPED_KEY"]).protector(
    MetadataPolicy(plaintext=["year"], blind=["client_id"])
)
```

## Providers

### HashiCorp Vault Transit

```bash
vault secrets enable transit
vault write -f transit/keys/hadamask type=aes256-gcm96
hadamask keygen --wrap vault-transit --transit-key hadamask --key-id prod-2026
```

Minimum policy for the application:

```hcl
path "transit/encrypt/hadamask" { capabilities = ["update"] }
path "transit/decrypt/hadamask" { capabilities = ["update"] }
```

```python
import hvac
from hadamask.kms.vault import VaultTransitKeyWrapper
wrapper = VaultTransitKeyWrapper(hvac.Client(url=VAULT_ADDR, token=...), "hadamask")
```

### Vault KV v2 (Vault as a secret store)

If you use Vault to store secrets rather than to wrap them, `VaultKVKeyStore` keeps each
master key as a version of one KV secret. `keyring()` returns the latest version as the
current key and the previous versions as read-only keys.

```python
from hadamask.kms.vault import VaultKVKeyStore
store = VaultKVKeyStore(client, "hadamask/tenant-42")
store.store(MasterKey.generate("prod-2026"))
protector = store.keyring(previous_versions=2).protector(policy)
```

Policy: `read` on `secret/data/hadamask/tenant-42` and `secret/metadata/hadamask/tenant-42`
for the application; `create`/`update` only for the operator who rotates keys.

### AWS KMS

Create a symmetric key (`SYMMETRIC_DEFAULT`). The application role needs `kms:Encrypt` and
`kms:Decrypt` on that key only. The binding is sent as encryption context
`hadamask-context`, which appears in CloudTrail and can be enforced in the key policy:

```json
"Condition": {"StringLike": {"kms:EncryptionContext:hadamask-context": "hadamask/master-key/v1/*"}}
```

### Google Cloud KMS

```bash
gcloud kms keyrings create hadamask --location europe-west3
gcloud kms keys create master-keys --keyring hadamask --location europe-west3 --purpose encryption
```

Grant `roles/cloudkms.cryptoKeyEncrypterDecrypter` (`cloudkms.cryptoKeyVersions.useToEncrypt`
and `useToDecrypt`) on that key. The binding is passed as additional authenticated data.

```python
from hadamask.kms.gcp import GcpKmsKeyWrapper
wrapper = GcpKmsKeyWrapper("projects/p/locations/europe-west3/keyRings/hadamask/cryptoKeys/master-keys")
```

### Azure Key Vault / Managed HSM

Create an RSA key (2048+ bits; RSA-HSM in Premium vaults or Managed HSM). The application
identity needs the `wrapKey` and `unwrapKey` key operations (role "Key Vault Crypto User").
Use a versioned key URL so that rotating the Key Vault key does not break unwrapping.

```python
from hadamask.kms.azure import AzureKeyVaultKeyWrapper
wrapper = AzureKeyVaultKeyWrapper("https://myvault.vault.azure.net/keys/hadamask/<version>")
```

Credentials default to `DefaultAzureCredential` (managed identity, workload identity, ...).

### Local wrapper

`LocalKeyWrapper(kek)` wraps with AES-256-GCM under a 32-byte key you provide. It is meant
for tests, development and air-gapped setups; the KEK is just another local secret, so it
is not a substitute for a KMS or an HSM.

## Rotation

1. Generate a new wrapped key with a new `key_id` (e.g. `prod-2027`).
2. Deploy with the new key as current and the old one as previous:
   `Keyring.from_wrapped(wrapper, new_wrapped, [old_wrapped])`. New writes use the new key;
   metadata written under the old key still decrypts.
3. Rewrite existing records (read with vectors, upsert) so that vectors, ciphertexts and
   blind tokens all move to the new key. Blind filters only match records rewritten under
   the current key.
4. Remove the old key from the previous list, then disable it in the key manager.

Rotating the key-encryption key in the KMS is independent: re-wrap the same master key with
`wrap_master_key(new_wrapper, key)`; no data needs to be rewritten.

## What this does not protect

Envelope encryption keeps the master key out of storage and configuration. It does not
protect against a compromised application host: any process that can call the key manager
with the application's identity can unwrap the key. Limit who can use the KMS key, monitor
decrypt calls (CloudTrail, Vault audit log, Azure diagnostics, Cloud Audit Logs) and do not
log unwrapped keys or decrypted payloads.
