#!/usr/bin/env python3
"""offline_smoke.py — runtime proof that the RAGLab image serves with NO
network. Runs INSIDE a container started with `docker run --network none`
(CI: .github/workflows/docker-image.yml) and asserts:

  1. GET /health without a token        -> 401 (X-Service-Token gate active)
  2. GET /health with the right token   -> 200 + the expected profile fields
  3. tiktoken cl100k_base loads offline -> the BPE file was baked at build time
     (no fallback to the chars/4 estimator — chunk boundaries match CI)

Exits 0 when all three hold; prints one PASS line per check. Stdlib only —
the image's python is the only interpreter involved.
"""

import json
import os
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:" + os.environ.get("RAGLAB_PORT", "8000")
TOKEN = os.environ.get("SMOKE_TOKEN", "ci-smoke-token")


def get(path, token=None):
    headers = {"X-Service-Token": token} if token else {}
    req = urllib.request.Request(BASE + path, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


failed = 0


def check(name, ok, detail=""):
    global failed
    print(("PASS  " if ok else "FAIL  ") + name + (f"  ({detail})" if detail else ""))
    failed += 0 if ok else 1


status, _ = get("/health")
check("no token -> 401 (gate active)", status == 401, f"status={status}")

status, body = get("/health", token=TOKEN)
payload = json.loads(body) if status == 200 else {}
check("valid token -> 200 + profile + index count",
      status == 200 and payload.get("status") == "ok"
      and isinstance(payload.get("index", {}).get("count"), int),
      f"status={status}")

import tiktoken  # noqa: E402 — post-import on purpose: proves the wheel is in the venv
enc = tiktoken.get_encoding("cl100k_base")  # offline: served from the baked cache
tokens = enc.encode(" raglab offline smoke ", disallowed_special=())
check("tiktoken cl100k_base loads with --network none (baked cache)",
      len(tokens) >= 3, f"tokens={len(tokens)}")

print(f"[offline-smoke] {'OK' if failed == 0 else f'{failed} FAILED'}")
sys.exit(0 if failed == 0 else 1)
