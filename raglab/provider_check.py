#!/usr/bin/env python3
"""provider_check.py — ask each provider directly what it says about your key.

`./raglab/run_local.sh --provider-check` (or `.venv/bin/python provider_check.py`)

Why this exists: when the answer provider refuses, the service can only report
what the provider told it — and if the failing thing is the network, the key or
the country, the service is not even the right layer to debug. This asks each
provider the same questions the clients ask (list models, then one minimal
generation / embedding), prints the HTTP status and the provider's own message,
and turns the common messages into an action ("this key is wrong", "this model
is not served for your key", "this country is blocked", "quota").

It talks to the providers directly: it does NOT need the service, and it does
not read or write the index. Keys are printed masked; every message goes
through safe_error, so a key echoed in a provider error is redacted.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nvidia_api import safe_error                                    # noqa: E402
from profiles import (GOOGLE_API_ROOT, NVIDIA_CHAT_BASE_URL,          # noqa: E402
                      GATEWAY_CATALOG, first_set_env, masked)

TIMEOUT = 25


def fetch(url: str, headers: dict, payload=None, *, timeout=TIMEOUT) -> dict:
    """One HTTPS call. HTTP errors are RESULTS here, not exceptions.

    Returns {"code": int|None, "text": str, "error": str}. code=0 means the
    request never reached the provider (DNS/TLS/timeout) — which is a distinct
    verdict from any HTTP status, and the reason this reports it separately.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {"code": response.status, "text": response.read().decode("utf-8", "replace"),
                    "error": ""}
    except urllib.error.HTTPError as exc:
        return {"code": exc.code, "text": exc.read().decode("utf-8", "replace"), "error": ""}
    except Exception as exc:                                         # noqa: BLE001
        return {"code": 0, "text": "", "error": safe_error(exc)}


ADVICE = (
    ("location is not supported",
     "the Gemini API is blocked from this country — no key will fix it here. "
     "Pick the xKiro or NVIDIA answer provider (menu 2) instead."),
    ("user location is not supported",
     "the Gemini API is blocked from this country — switch answer provider (menu 2)."),
    ("api key not valid", "the key itself is wrong or expired — re-paste it with --keys."),
    ("api_key_invalid", "the key itself is wrong or expired — re-paste it with --keys."),
    ("api key expired", "the key has expired — get a new one and re-paste it with --keys."),
    ("resource_exhausted", "quota for this key/model is used up — wait, or pick another model."),
    ("quota", "quota for this key/model is used up — wait, or pick another model."),
    ("rate limit", "rate-limited — wait a minute, or pick another model."),
    ("permission", "the key is valid but not allowed for this model — pick another model."),
    ("not found", "this model ID is not served for this key — pick another model (menu 2)."),
    ("no such model", "this model ID does not exist — pick another model (menu 2)."),
)


def interpret(row: dict) -> list[str]:
    """Plain-word next steps for one probe row (empty when it succeeded)."""
    if row["ok"]:
        return []
    if row["code"] == 0:
        return [f"the request never reached the provider ({row['error'] or 'no detail'}) — "
                f"this is a network/DNS/TLS/blocked-host problem on this machine, "
                f"not a key problem. Quota or an invalid key would have returned an HTTP code."]
    blob = f"{row.get('message', '')} {row.get('error', '')}".lower()
    advice = [text for token, text in ADVICE if token in blob]
    if row["code"] in (401, 403) and not advice:
        advice.append("the provider rejected the key — check it in menu 3 (keep/change/add).")
    if row["code"] == 429 and not advice:
        advice.append("rate-limited/quota — wait, or pick another model.")
    if row["code"] == 404 and not advice:
        advice.append("that endpoint or model does not exist for this key — check the model ID.")
    if not advice:
        advice.append("see the provider message above; if it mentions a parameter, the model ID "
                      "or a request field is the cause, not the key.")
    return advice


def first_line(text: str, limit: int = 200) -> str:
    """Provider messages are JSON; the human line is what a person can act on."""
    text = (text or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            error = parsed.get("error") or parsed.get("detail") or parsed
            if isinstance(error, dict) and error.get("message"):
                text = str(error["message"])
            elif isinstance(error, str):
                text = error
    except ValueError:
        pass
    return " ".join(text.split())[:limit]


def probe(label: str, url: str, headers: dict, payload=None, *, fetch=fetch) -> dict:
    result = fetch(url, headers, payload)
    row = {"label": label, "url": url, "code": result["code"],
           "message": first_line(result["text"]), "error": result["error"],
           "ok": result["code"] == 200}
    return row


def report(rows: list, echo=print) -> tuple[int, int]:
    """Print the table + advice. Returns (passed, failed)."""
    passed = sum(1 for row in rows if row["ok"])
    failed = sum(1 for row in rows if not row["ok"])
    for row in rows:
        status = "OK  " if row["ok"] else "FAIL"
        detail = row["message"] if row["code"] else (row["error"] or "no response")
        echo(f"  [{status}] {row['label']:38} HTTP {row['code'] or '—':>3}  {detail}")
        for line in interpret(row):
            echo(f"         -> {line}")
    return passed, failed


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    echo = print
    echo("=" * 100)
    echo("provider check — the calls the service makes, asked directly (no service, no index)")
    echo("=" * 100)

    nvidia_name, nvidia_key = first_set_env(("NVIDIA_API_KEY",))
    xkiro_name, xkiro_key = first_set_env(("XKIRO_API_KEY",))
    google_name, google_key = first_set_env(("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    rows = []

    if nvidia_key:
        label = f"{nvidia_name} (embeddings list)"
        rows.append(probe(label, f"{NVIDIA_CHAT_BASE_URL}/models",
                          {"Authorization": f"Bearer {nvidia_key}"}))
    if xkiro_key:
        label = f"{xkiro_name} (models list)"
        rows.append(probe(label, f"{GATEWAY_CATALOG['xkiro']['base_url'].rstrip('/')}/models",
                          {"Authorization": f"Bearer {xkiro_key}"}))
    if google_key:
        label = f"{google_name} (Gemini models list)"
        rows.append(probe(label, f"{GOOGLE_API_ROOT}/models?page_size=5",
                          {"x-goog-api-key": google_key}))
        # One minimal generation proves the thing chat actually needs: a key
        # that can LIST models is not automatically allowed to GENERATE.
        from llm_smoke import google_candidates
        try:
            model = (google_candidates(google_key) or ["gemini-2.5-flash-lite"])[0]
        except Exception as exc:                                     # noqa: BLE001
            model = "gemini-2.5-flash-lite"
            echo(f"  [note] could not rank models ({safe_error(exc)}) — trying {model}")
        rows.append(probe(f"{google_name} (generateContent {model})",
                          f"{GOOGLE_API_ROOT}/models/{model}:generateContent",
                          {"x-goog-api-key": google_key, "Content-Type": "application/json"},
                          {"contents": [{"parts": [{"text": "ping"}]}],
                           "generationConfig": {"maxOutputTokens": 8}}))
    if xkiro_key:
        from profiles import SUPPORTED_ANSWER
        model = SUPPORTED_ANSWER["model"]
        rows.append(probe(f"{xkiro_name} (chat {model})",
                          f"{GATEWAY_CATALOG['xkiro']['base_url'].rstrip('/')}/chat/completions",
                          {"Authorization": f"Bearer {xkiro_key}",
                           "Content-Type": "application/json"},
                          {"model": model, "messages": [{"role": "user", "content": "ping"}],
                           "max_tokens": 8}))

    if not rows:
        echo("no provider key is set in this environment — nothing to check.")
        echo("set them with:  ./raglab/run_local.sh --keys")
        return 2
    for name, key, key_name in (("nvidia", nvidia_key, nvidia_name),
                                ("xkiro", xkiro_key, xkiro_name),
                                ("google", google_key, google_name)):
        if key:
            echo(f"  key in use: {key_name} ({masked(key)})")
    echo("-" * 100)
    passed, failed = report(rows, echo=echo)
    echo("-" * 100)
    echo(f"[provider] {passed} ok, {failed} failed")
    if failed:
        echo("[provider] the failing row above is the error the service would report as "
             "status=error/provider_error; its message is the actionable part.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
