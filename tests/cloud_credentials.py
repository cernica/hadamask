"""Cloud credentials for the managed-service tests.

Read from environment variables (QDRANT_CLOUD_URL, QDRANT_CLOUD_API_KEY, WEAVIATE_CLOUD_URL,
WEAVIATE_CLOUD_API_KEY) or, as a local fallback, from the untracked ``api_keys:`` file in the
project root. That file's labels are matched by host name rather than trusted, because a
*.qdrant.io endpoint is always Qdrant Cloud and a *.weaviate.cloud endpoint is always Weaviate
Cloud. Qdrant Cloud keys are JWTs (start with "eyJ"); Weaviate Cloud keys are not.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_FILE = Path(__file__).resolve().parent.parent / "api_keys:"


def _from_file() -> dict[str, str]:
    if not _FILE.exists():
        return {}
    values = re.findall(r'\w+\s*=\s*"([^"]*)"', _FILE.read_text())
    hosts = [v for v in values if "qdrant.io" in v or "weaviate.cloud" in v]
    keys = [v for v in values if v not in hosts]
    out = {}
    for host in hosts:
        url = host if host.startswith("http") else f"https://{host}"
        if "qdrant.io" in host:
            out["QDRANT_CLOUD_URL"] = url
        else:
            out["WEAVIATE_CLOUD_URL"] = url
    for key in keys:
        out["QDRANT_CLOUD_API_KEY" if key.startswith("eyJ") else "WEAVIATE_CLOUD_API_KEY"] = key
    return out


def get(name: str) -> Optional[str]:
    return os.environ.get(name) or _from_file().get(name)
