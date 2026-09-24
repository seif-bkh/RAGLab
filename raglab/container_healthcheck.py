#!/usr/bin/env python3
"""container_healthcheck.py — Docker HEALTHCHECK for the RAGLab image.

Stdlib only, no network beyond 127.0.0.1: probes GET /health inside the
container and reports healthy when the service answers — 200 (open service)
OR 401 (service up, RAGLAB_SERVICE_TOKEN gate active and this probe, like
any tokenless client, was refused). Both mean the process is serving; only
a connection failure or another status is unhealthy. Works under
`docker run --network none`.

Not part of the HTTP contract — a container-runtime probe, not an endpoint.
"""

import os
import sys
import urllib.error
import urllib.request

PORT = os.environ.get("RAGLAB_PORT", "8000")

try:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except urllib.error.HTTPError as exc:
    sys.exit(0 if exc.code in (200, 401) else 1)   # 401 = up but token-gated = healthy
except (urllib.error.URLError, OSError):
    sys.exit(1)
