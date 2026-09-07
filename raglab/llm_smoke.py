#!/usr/bin/env python3
"""CI-only LLM smoke for the real test: one retrieved, cited, validated answer.

The real-test workflow (`.github/workflows/real-test.yml`) uses this instead of
`main.py answer` because the pinned app answer profile (pipeline_policy:
xkiro/qwen3.8-max:free) kept hitting free-tier capacity 502s. Per the owner's
instruction (2026-09-07) the smoke LLM rotates providers:

  phase A: nvidia/nemotron-3.5-lightning-30b-a3b via integrate.api.nvidia.com
           (NVIDIA_API_KEY), with phase-level retries — the NVIDIA build
           endpoint and free tiers return transient 502/503 capacity errors.
           chat_payload() already disables Nemotron-3 thinking for the
           cited-JSON contract (greedy, no reasoning tokens).
  phase B (only if A fails): GOOGLE_API_KEY and the cheapest available
           free-tier Gemini model (flash-lite preferred; auto-selected from
           the public /models listing so dated/preview IDs self-resolve),
           same grounded-v1 prompt and the same quote-membership validation.

Like harness50.py this is a harness: it drives the app's retrieval and
answer-contract code paths but does not change the pinned profile.

Usage (from raglab/):
    python llm_smoke.py --question "ما هي المرابحة؟" --query-lang ar \
        --output results/harness50/real_answer_murabaha.json

Exit codes: 0 = a validated cited answer (either phase); 2 = calls were made
but nothing validated (invalid output / abstention); 1 = hard failure
(missing key, empty collection, no reachable model).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import config as cfg
from answer import (AnswerGenerator, ERRORS, REFUSALS, answer_messages,
                    build_sources, detect_language, validate_answer)
from artifacts import write_json
from evaluate import prepare_query_text
from main import make_embedder
from nvidia_api import NvidiaClient, safe_error
from retrieval import expand_neighbors, retrieve
from store import get_collection

NVIDIA_SMOKE_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GOOGLE_KEY_ENV = "GOOGLE_API_KEY"
GOOGLE_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
NVIDIA_PHASE_ATTEMPTS = 2
NVIDIA_PHASE_BACKOFF_S = 60
GOOGLE_MAX_TRIES = 3  # distinct model candidates, cheapest first


def smoke_cfg() -> SimpleNamespace:
    ns = SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    ns.ANSWER_PROVIDER = "nvidia"
    ns.ANSWER_MODEL = NVIDIA_SMOKE_MODEL
    return ns


def retrieve_hits(ns, question: str, language: str):
    collection = get_collection(ns)
    if not collection.count():
        raise SystemExit("collection is empty; run the arm ingests first")
    embedder = make_embedder(skip_sanity=True)
    hits, variants = retrieve(ns, embedder, collection, prepare_query_text(question),
                              language=language, translator=None,
                              top_k=ns.ANSWER_TOP_K, variant_strategy=ns.QUERY_VARIANT_STRATEGY)
    hits = expand_neighbors(collection, hits, ns.ANSWER_NEIGHBOR_RADIUS)
    return hits, variants


# ---------------------------------------------------------------------------
# Phase A — NVIDIA build endpoint
# ---------------------------------------------------------------------------

def run_nvidia_phase(ns, question, language, hits):
    client = NvidiaClient(base_url=NVIDIA_BASE_URL, timeout=120, attempts=2,
                          min_interval=3.0, max_retry_delay=45)
    gen = AnswerGenerator(ns, client, approved_models=(NVIDIA_SMOKE_MODEL,))
    last = None
    for attempt in range(1, NVIDIA_PHASE_ATTEMPTS + 1):
        if attempt > 1:
            print(f"[llm_smoke] nvidia phase attempt {attempt}/{NVIDIA_PHASE_ATTEMPTS} "
                  f"after {NVIDIA_PHASE_BACKOFF_S}s backoff (capacity errors are transient)")
            time.sleep(NVIDIA_PHASE_BACKOFF_S)
        last = gen.answer(question, hits, language, use_cache=False)
        status, reason = last.get("status"), last.get("reason")
        print(f"[llm_smoke] nvidia attempt {attempt} -> status={status} "
              f"reason={reason} served={last.get('served_model')} "
              f"error={safe_error(last.get('error'))[:200]}")
        if status == "answered" and last.get("validation_ok"):
            break
    last = dict(last or {})
    last.update(phase="nvidia", provider="nvidia", model=NVIDIA_SMOKE_MODEL,
                api_endpoint=NVIDIA_BASE_URL,
                inference_performed=last.get("status") not in (None, "refused"))
    return last


def is_usable(result: dict) -> bool:
    return bool(result) and result.get("status") == "answered" and result.get("validation_ok")


# ---------------------------------------------------------------------------
# Phase B — Google free tier (fallback only)
# ---------------------------------------------------------------------------

def _model_rank(model_id: str):
    mid = model_id.lower()
    tier = 0 if "flash-lite" in mid else (1 if "flash" in mid else 2)
    preview = 1 if "preview" in mid or "exp" in mid else 0
    dated = 1 if re.search(r"\d{4}-\d{2}", mid) else 0
    return (tier, preview, dated, mid)


def google_candidates(key: str) -> list:
    """Cheapest-first Gemini models available to this key (free tier)."""
    req = urllib.request.Request(GOOGLE_API_ROOT + "/models?page_size=200",
                                 headers={"x-goog-api-key": key,
                                          "User-Agent": "RAGLab-llm-smoke/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    ids = [m.get("name", "").removeprefix("models/")
           for m in data.get("models", [])
           if isinstance(m, dict) and str(m.get("name", "")).startswith("models/")]
    ids = [i for i in ids if re.match(r"^gemini-", i)]
    return sorted(set(ids), key=_model_rank)[:GOOGLE_MAX_TRIES]


def google_call(model: str, key: str, messages: list, max_tokens: int):
    url = f"{GOOGLE_API_ROOT}/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
        "contents": [{"role": "user", "parts": [{"text": messages[1]["content"]}]}, ],
        "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens,
                             "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(url, data=json.dumps(body).ensure_ascii is None and
                                 json.dumps(body).encode("utf-8"),
                                 headers={"x-goog-api-key": key,
                                          "Content-Type": "application/json",
                                          "User-Agent": "RAGLab-llm-smoke/1.0"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        candidate = data["candidates"][0]
        if candidate.get("finishReason") in {"SAFETY", "RECITATION", "PROHIBITED_CONTENT"}:
            return None, f"google filtered the response ({candidate['finishReason']})"
        text = "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", []))
        return text, None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return None, f"google HTTP {exc.code}: {safe_error(raw)[:400]}"
    except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, IndexError) as exc:
        return None, f"google call error: {safe_error(exc)}"


def run_google_phase(ns, question, language, hits, sources, prior_error):
    key = os.environ.get(GOOGLE_KEY_ENV, "").strip()
    if not key:
        return {"provider": "google", "model": None, "phase": "google-fallback",
                "status": "error", "reason": "missing_key", "validation_ok": False,
                "claims": [], "sources": sources,
                "answer": ERRORS.get(language, ERRORS["en"]),
                "error": f"{GOOGLE_KEY_ENV} is not configured; the fallback cannot run",
                "inference_performed": False,
                "nvidia_attempt": safe_error(prior_error)[:400]}
    try:
        cands = google_candidates(key)
    except Exception as exc:
        cands = []
        print(f"[llm_smoke] google /models listing failed: {safe_error(exc)}")
    if not cands:
        return {"provider": "google", "model": None, "phase": "google-fallback",
                "status": "error", "reason": "no_free_model_found", "validation_ok": False,
                "claims": [], "sources": sources,
                "answer": ERRORS.get(language, ERRORS["en"]),
                "error": "no gemini model visible to this key "
                         f"(nvidia_attempt={safe_error(prior_error)[:200]})",
                "inference_performed": False}
    print(f"[llm_smoke] google fallback candidates (cheapest first): {cands}")
    messages = answer_messages(question, language, sources, ns.ANSWER_PROMPT_VERSION)
    last_error = "no model attempted"
    for model in cands:
        text, err = google_call(model, key, messages, ns.ANSWER_MAX_TOKENS)
        if err:
            last_error = err
            print(f"[llm_smoke] google {model} -> {err}")
            continue
        try:
            claims = validate_answer(text, sources)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            last_error = f"{model} invalid_output: {safe_error(exc)}"
            print(f"[llm_smoke] google {model} -> invalid output: {safe_error(exc)[:200]}")
            continue
        status = "answered" if claims else "refused"
        answer = "\n".join(c["text"] + " " + " ".join(
            f"[{s}]" for s in dict.fromkeys(e["source_id"] for e in c["evidence"]))
            for c in claims)
        result = {"provider": "google", "model": model, "phase": "google-fallback",
                  "prompt_version": ns.ANSWER_PROMPT_VERSION,
                  "status": status, "reason": "supported" if claims else "insufficient_evidence",
                  "claims": claims, "sources": sources,
                  "answer": answer if claims else REFUSALS.get(language, REFUSALS["en"]),
                  "validation_ok": True, "inference_performed": True,
                  "nvidia_attempt": safe_error(prior_error)[:400]}
        print(f"[llm_smoke] google {model} -> status={status} claims={len(claims)}")
        return result
    return {"provider": "google", "model": cands[0], "phase": "google-fallback",
            "status": "error", "reason": "all_models_failed", "validation_ok": False,
            "claims": [], "sources": sources,
            "answer": ERRORS.get(language, ERRORS["en"]),
            "error": last_error, "inference_performed": True}


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--question", required=True)
    ap.add_argument("--query-lang", choices=["en", "fr", "ar"], default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ns = smoke_cfg()
    question = args.question.strip()
    language = args.query_lang or detect_language(question)
    hits, variants = retrieve_hits(ns, question, language)
    sources = build_sources(hits, ns.ANSWER_CONTEXT_TOKENS)
    if not sources:
        print("[llm_smoke] no sources retrieved; nothing to ground an answer on")
        write_json(args.output, {"provider": "nvidia", "model": NVIDIA_SMOKE_MODEL,
                                 "phase": "nvidia", "status": "refused",
                                 "reason": "no_context", "validation_ok": True,
                                 "claims": [], "sources": [],
                                 "answer": REFUSALS.get(language, REFUSALS["en"])})
        return 0

    result = run_nvidia_phase(ns, question, language, hits)
    if not is_usable(result):
        prior = result.get("error") or f"status={result.get('status')} reason={result.get('reason')}"
        print(f"[llm_smoke] nvidia phase not usable ({prior}); falling back to google")
        result = run_google_phase(ns, question, language, hits, sources, prior)

    result.update(question=question, language=language,
                  query_variants=variants,
                  prompt_version=getattr(ns, "ANSWER_PROMPT_VERSION", "grounded-v1"))
    write_json(args.output, result)
    ok = is_usable(result)
    print(f"[llm_smoke] final: provider={result.get('provider')} "
          f"model={result.get('model')} phase={result.get('phase')} "
          f"status={result.get('status')} validation_ok={result.get('validation_ok')} "
          f"claims={len(result.get('claims') or [])} -> {args.output}")
    if ok:
        print(result["answer"])
        return 0
    return 2 if result.get("inference_performed") else 1


if __name__ == "__main__":
    sys.exit(main())
