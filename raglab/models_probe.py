#!/usr/bin/env python3
"""Read-only FREE-MODEL probe for all three answer providers (2026-10-02,
owner request for personal-machine use). No documents, no queries, no
inference — one catalog GET per provider with a configured key.

Honest semantics per provider (a catalog listing proves VISIBILITY, never
price or upstream identity):

- xKiro    GET /v1/models (Bearer) — OpenAI-style with pricing metadata;
           "verified_free" = access_tier 'free' AND every declared price
           exactly 0 (the free_gateway rule, fail closed).
- NVIDIA   GET https://integrate.api.nvidia.com/v1/models (Bearer) — the
           catalog carries NO pricing; listed ids are visible to YOUR key.
           Free usage on NVIDIA is account/credit-based (build.nvidia.com),
           so this probe reports visibility only, never "free".
- Google   GET generativelanguage v1beta/models (API key) — lists models
           with supportedGenerationMethods; the Gemini API free tier is
           rate-limited and account-level, so models usable for chat are
           flagged chat_capable, never "free".

Safety rules inherited from provider_catalog.py: bounded 2 MB reads,
credential-bearing redirects disabled, raw error bodies never exported (a
gateway may echo the key), missing key is a status — not a crash.

Usage:
    python models_probe.py            # table + results/models_probe/probe.json
    python models_probe.py --json     # JSON only
Keys: XKIRO_API_KEY, NVIDIA_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from free_gateway import is_zero
from provider_catalog import NoCredentialRedirects

MAX_BYTES = 2_000_000

PROBE_PROVIDERS = {
    "xkiro": {
        "key_envs": ("XKIRO_API_KEY",),
        "url": "https://api.xkiro.com/v1/models",
        "auth": "bearer",
        "pricing_in_catalog": True,
    },
    "nvidia": {
        "key_envs": ("NVIDIA_API_KEY",),
        "url": "https://integrate.api.nvidia.com/v1/models",
        "auth": "bearer",
        "pricing_in_catalog": False,
    },
    "google": {
        "key_envs": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "url": "https://generativelanguage.googleapis.com/v1beta/models",
        "auth": "query_key",
        "pricing_in_catalog": False,
    },
}

OUTPUT = Path(__file__).resolve().parent / "results/models_probe/probe.json"


def _first_key(envs):
    for name in envs:
        value = os.environ.get(name, "").strip()
        if value:
            return name, value
    return None, ""


def _get_json(url, headers, opener):
    client = opener or urllib.request.build_opener(NoCredentialRedirects())
    request = urllib.request.Request(url, headers=headers)
    with client.open(request, timeout=30) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Oversized catalog")
    return json.loads(raw)


def probe_provider(provider, *, opener=None):
    """One provider -> a report row. Never raises for missing keys/HTTP."""
    settings = PROBE_PROVIDERS[provider]
    key_env, key = _first_key(settings["key_envs"])
    row = {"provider": provider, "key_env": key_env,
           "key_configured": bool(key),
           "catalog_requests": 1, "inference_calls": 0}
    if not key:
        return {**row, "status": "missing_key",
                "hint": f"export {settings['key_envs'][0]}=... to probe this provider"}
    headers = {"Accept": "application/json",
               "User-Agent": "RAGLab-readonly-models-probe/1.0"}
    url = settings["url"]
    if settings["auth"] == "bearer":
        headers["Authorization"] = "Bearer " + key
    else:  # query_key — Google's documented key parameter
        url += "?key=" + key
    try:
        data = _get_json(url, headers, opener)
    except Exception as exc:  # noqa: BLE001 — report, never export bodies
        return {**row, "status": "http_error",
                "error_type": type(exc).__name__}
    return {**row, "status": "listed", **_summarize(provider, data)}


def _summarize(provider, data):
    """Provider-specific honest summarization of a listed catalog."""
    if provider in ("xkiro", "nvidia"):  # OpenAI-style {data: [{id, ...}]}
        entries = [m for m in (data.get("data") or [])
                   if isinstance(m, dict) and isinstance(m.get("id"), str)]
        ids = sorted({m["id"] for m in entries})
        out = {"advertised_model_count": len(ids), "model_ids": ids}
        if provider == "xkiro":
            free, paid = [], []
            for m in entries:
                prices = m.get("pricing") or {}
                zero = (m.get("access_tier") == "free"
                        and all(is_zero(prices.get(k))
                                for k in ("input", "output"))
                        and all(is_zero(prices[k]) for k in ("cache_read", "cache_write")
                                if k in prices))
                (free if zero else paid).append(m["id"])
            out["verified_free"] = sorted(set(free))
            out["free_note"] = ("verified_free = access_tier 'free' AND all "
                                "declared prices exactly 0 (fail closed)")
        else:
            out["free_note"] = ("the NVIDIA catalog carries no pricing; free "
                                "usage is account/credit-based (build.nvidia.com)"
                                " — this probe proves visibility only")
        return out
    # Google: {models: [{name, displayName, supportedGenerationMethods}]}
    entries = [m for m in (data.get("models") or []) if isinstance(m, dict)]
    chat = sorted(m["name"].split("/", 1)[-1] for m in entries
                  if "generateContent" in (m.get("supportedGenerationMethods") or []))
    return {"advertised_model_count": len(entries),
            "chat_capable_models": chat,
            "free_note": ("the Gemini API free tier is rate-limited and "
                          "account-level; the catalog endpoint carries no "
                          "pricing — chat_capable means usable for /answer")}


def collect(*, opener=None):
    return {"generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "catalog_only", "inference_calls": 0, "documents_sent": False,
            "providers": [probe_provider(name, opener=opener)
                          for name in PROBE_PROVIDERS]}


def _render(report):
    lines = []
    for row in report["providers"]:
        lines.append(f"═ {row['provider']} — {row['status']}"
                     + (f" (key: {row['key_env']})" if row.get("key_configured") else ""))
        if row["status"] != "listed":
            lines.append(f"   {row.get('hint', 'no catalog')}")
            continue
        lines.append(f"   advertised models: {row['advertised_model_count']}")
        if "verified_free" in row:
            lines.append(f"   verified_free ({len(row['verified_free'])}): "
                         + (", ".join(row["verified_free"]) or "—"))
        if "chat_capable_models" in row:
            lines.append(f"   chat-capable ({len(row['chat_capable_models'])}): "
                         + (", ".join(row["chat_capable_models"][:12])
                            + (" …" if len(row["chat_capable_models"]) > 12 else "")))
        if "model_ids" in row and row["provider"] == "nvidia":
            lines.append(f"   visible to your key (first 12): "
                         + ", ".join(row["model_ids"][:12])
                         + (" …" if len(row["model_ids"]) > 12 else ""))
        lines.append(f"   {row['free_note']}")
    lines.append(f"\ninference calls: {report['inference_calls']} "
                 f"(read-only probe) | {report['generated_at']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="models_probe",
        description="Read-only free/available model probe: xKiro, NVIDIA, Google")
    parser.add_argument("--json", action="store_true",
                        help="print the JSON report instead of the table")
    args = parser.parse_args(argv)

    report = collect()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        print(_render(report))
        print(f"\nJSON artifact: {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
