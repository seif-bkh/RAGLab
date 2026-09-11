#!/usr/bin/env python3
"""app.py — the interactive console front-end for everything RAGLab can do.

The RUNTIME half (provider/model registries, the lab-config copy, the chat
clients for every provider, profile-scoped collections, ingest, greeting
detection) lives in profiles.py and is shared with service.py, the standalone
HTTP microservice. This file is only the console: menus, prompts, per-provider
key entry, app_state.json persistence. See raglab/SERVICE.md for the service.

Run it from this folder, inside the same virtualenv as main.py:

    python app.py

You get a menu over the whole lab — corpus inspection, ingestion, retrieval,
grounded answers, a chat REPL, evaluation, embedding sanity checks, offline
diagnostics, a chunk-search tool that says whether any quote lives inside ONE
chunk (the invalid_output diagnostic) — plus the two things that used to mean
editing files by hand:

  * switching providers and models. The embedding slot offers every provider
    registered in embedder.build_embedder (nvidia, gemini, jina, huggingface,
    openai, cohere, voyage) with their registered models; the answer/chat slot
    offers the xKiro gateway (the pinned SKU, plus any custom model ID you
    enter — non-pinned SKUs are labelled experimental and skip the free-price
    gate rather than weaken it), the free NVIDIA build-endpoint chat models
    (the chat.py profile plus the registered ANSWER_MODELS), the Google
    free-tier Gemini path from llm_smoke.py, and the Kira AI OpenAI-compatible
    gateway (kiraai.vn, e.g. glm-5.3-free). Every custom model ID you type is
    remembered per provider in raglab/app_state.json and offered again on the
    next session (menu 2 lists and can remove them).
  * API keys per provider. On startup, and again from the menus, the app asks
    for each selected provider whether to KEEP the key found in raglab/.env,
    CHANGE it, or — when none exists — paste one. The answer is written back
    to raglab/.env and the current process. A key is never printed in full:
    only its first 8 characters, the repo's standing masking rule.

Policy note (this console is a lab surface, same standing as chat.py): the
supported benchmark pipeline is pinned in pipeline_policy.py to NVIDIA
nemotron embeddings + the xKiro qwen/qwen3.8-max:free answerer, and
`main.py answer` refuses anything else. Nothing here weakens that. Every
selection runs through the same chunker, the same store fingerprints, the
same retrieval, the same verbatim-citation validation and the same
private/live-question refusals — but on the app's OWN Chroma collections
(raglab_app_<provider>_<model>_<chunking>), never on main.py's or a harness
index, so switching can never corrupt another entry point's vectors. No
number produced here is a benchmark result; selecting the pinned pair only
labels the session "supported pipeline".

Where things live:
  * provider/model selections -> raglab/app_state.json (gitignored). Never
    .env: config.py deliberately rejects model overrides from .env, so model
    choice must not be smuggled in through it.
  * API keys                  -> raglab/.env (gitignored; never commit it).
  * app caches/index          -> embeddings_cache_app_*.json,
    answers_cache_app_*.json, chroma_db/raglab_app_* (all gitignored).

Non-interactive shortcuts:

    python app.py --status        doctor report, no prompts, then exit
    python app.py --no-keycheck   skip the startup keep/change/add questions
    python app.py --reset-state   forget saved selections, use defaults
    python app.py --ingest        startup checks, then build/rebuild the index
    python app.py --chat          startup checks, then straight into the chat
    python app.py --ask "..."     startup checks, one grounded answer, exit
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import config as cfg
from nvidia_api import safe_error
import chat as chat_mod

# The runtime half lives in profiles.py (shared with service.py); the console
# re-exports it so existing callers and tests keep working unchanged.
from profiles import (ANSWER_PROVIDERS, EMBEDDING_PROVIDERS, KEY_ENV_INFO,
                      KIRA_BASE_URL, SUPPORTED_ANSWER, SUPPORTED_EMBEDDING,
                      build_generator, build_lab_config, collection_count,
                      collection_name, consistent_model, data_dirs,
                      default_state, first_set_env, greeting_language,
                      greeting_reply, ingest, is_greeting, masked,
                      missing_key_slots, pipeline_marker, slot_display,
                      stored_index_info)

PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
STATE_PATH = PROJECT_DIR / "app_state.json"


# ---------------------------------------------------------------------------
# Provider registries — what "registered" means for this console: a provider
# the codebase can actually drive, with the env key(s) its client reads and
# the model IDs its modules name. Exact IDs only; no alias substitution.
# ---------------------------------------------------------------------------



PLACEHOLDER_PATTERNS = ("paste-your", "paste your", "your-key", "your_key",
                        "your_kira", "changeme", "change-me", "xxx", "placeholder",
                        "api_key")   # pasted templates like YOUR_KIRA_API_KEY


# ---------------------------------------------------------------------------
# .env handling — keys are the ONLY thing this app writes there.
# ---------------------------------------------------------------------------

def read_env_file(path=None) -> dict:
    """KEY -> value map of the literal .env file (os.environ may hold more).

    The path resolves at CALL time (a default argument would bind the constant
    at import time and silently ignore a later retarget — tests caught exactly
    that).
    """
    path = ENV_PATH if path is None else path
    assignments: dict[str, str] = {}
    if not path.exists():
        return assignments
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if match:
            assignments[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    return assignments


def write_env_assignment(name: str, value: str) -> None:
    """Update or append one KEY=value in raglab/.env, preserving every other line.

    Comments and ordering are kept: .env is a file humans also edit, and a
    rewrite that dropped their notes would be a regression, not a convenience.
    """
    existed = ENV_PATH.exists()
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if existed else []
    pattern = re.compile(r"^\s*(?:export\s+)?" + re.escape(name) + r"\s*=")
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = f"{name}={value}"
            break
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"# added by app.py {datetime.now(timezone.utc):%Y-%m-%d}")
        lines.append(f"{name}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not existed:
        ENV_PATH.chmod(0o600)   # a fresh .env should not be group/world-readable


def remove_env_assignment(name: str) -> bool:
    """Drop one assignment from .env (and the process env). True if it existed."""
    if not ENV_PATH.exists():
        return False
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(r"^\s*(?:export\s+)?" + re.escape(name) + r"\s*=")
    kept = [line for line in lines if not pattern.match(line)]
    removed = len(kept) != len(lines)
    if removed:
        ENV_PATH.write_text("\n".join(kept) + "\n", encoding="utf-8")
    os.environ.pop(name, None)
    return removed






def looks_like_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in PLACEHOLDER_PATTERNS) or len(value) < 8


def key_flow(env_names, label: str, *, hint: str = "", force_change: bool = False) -> bool:
    """The keep / change / add question for one provider's key.

    Returns True when the provider ends up configured (existing key kept, or a
    new one written to raglab/.env AND the live process env, so the current
    session works without a restart). force_change skips the keep/change
    question (the caller already answered it) and goes straight to the paste.
    """
    primary = env_names[0]
    env, value = first_set_env(env_names)
    if value:
        print(f"\n{label}")
        print(f"  {env} is set ({masked(value)}).")
        if not force_change:
            while True:
                choice = prompt("  keep it (Enter) / c: change it / s: skip: ").lower()
                if choice in ("", "k", "keep"):
                    return True
                if choice == "s":
                    return True    # the key stays configured either way
                if choice in ("c", "change"):
                    break
                print("  answer Enter, k, c or s")
    else:
        print(f"\n{label}")
        print(f"  no key configured (checked: {', '.join(env_names)}).")
        if hint:
            print(f"  hint: {hint}")
    while True:
        try:
            new = getpass.getpass(
                f"  paste the key for {primary} — saved to {ENV_PATH.name}, "
                "input hidden (Enter to cancel): ").strip()
        except EOFError:
            print("  cancelled — no key entered.")
            return bool(value)
        if not new:
            print("  cancelled — nothing written." if not value else "  kept the existing key.")
            return bool(value)
        if re.search(r"[\s\"']", new):
            print("  a key cannot contain spaces or quotes; try again.")
            continue
        if looks_like_placeholder(new):
            print("  that looks like the .env.example placeholder, not a real key; try again.")
            continue
        write_env_assignment(primary, new)
        os.environ[primary] = new
        print(f"  saved {primary} to {ENV_PATH.name} ({masked(new)}) — the value is never shown in full.")
        if primary == "NVIDIA_API_KEY" and not new.startswith("nvapi-"):
            print("  note: NVIDIA build-endpoint keys normally start with 'nvapi-'; "
                  "if calls fail with 401, that is the first thing to check.")
        return True




def ensure_slot_key(state: dict, slot: str) -> bool:
    """Interactive fallback before an action: ask for a missing key once more."""
    registry = EMBEDDING_PROVIDERS if slot == "embedding" else ANSWER_PROVIDERS
    info = registry.get(state[slot]["provider"], {})
    envs = info.get("key_envs", ())
    if not envs or first_set_env(envs)[1]:
        return True
    return key_flow(envs, f"[{slot}] {state[slot]['provider']} — {info['label']}",
                    hint=info.get("key_hint", ""))


def startup_key_check(state: dict, interactive: bool = True) -> None:
    """Ask keep/change/add for every selected provider that needs a key."""
    for slot, registry in (("embedding", EMBEDDING_PROVIDERS), ("answer", ANSWER_PROVIDERS)):
        entry = state[slot]
        info = registry.get(entry["provider"], {})
        envs = info.get("key_envs", ())
        if not envs:
            continue                      # huggingface runs locally, nothing to ask
        if not interactive:
            continue
        key_flow(envs, f"[{slot}] {entry['provider']} — {info['label']}",
                 hint=info.get("key_hint", ""))


# ---------------------------------------------------------------------------
# Saved selection state — raglab/app_state.json, never .env (see module doc).
# ---------------------------------------------------------------------------



def load_state() -> dict:
    state = default_state()
    first_run = not STATE_PATH.exists()
    if not first_run:
        try:
            saved = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise ValueError("not a JSON object")
        except (OSError, ValueError) as exc:
            print(f"[app] WARNING: {STATE_PATH.name} is unreadable ({exc}); using defaults")
            saved = {}
        for key in ("embedding", "answer"):
            if isinstance(saved.get(key), dict):
                state[key] = {**state[key], **saved[key]}
        for key in ("chunking", "retrieval"):
            if isinstance(saved.get(key), dict):
                state[key] = {**state[key], **saved[key]}
        if isinstance(saved.get("data_dirs"), list):
            state["data_dirs"] = [str(d) for d in saved["data_dirs"]]
        custom = saved.get("custom_models")
        if isinstance(custom, dict):
            # The per-provider model-ID memory: keep only sane string lists.
            state["custom_models"] = {
                str(provider): [str(m) for m in models
                                if isinstance(m, str) and m.strip()]
                for provider, models in custom.items() if isinstance(models, list)}
    # A state the registries cannot drive would crash mid-action; reset loudly.
    for slot, registry in (("embedding", EMBEDDING_PROVIDERS), ("answer", ANSWER_PROVIDERS)):
        entry = state[slot]
        if entry.get("provider") not in registry or not str(entry.get("model", "")).strip():
            print(f"[app] NOTE: unknown {slot} selection in {STATE_PATH.name}; "
                  f"reset to {default_state()[slot]}")
            state[slot] = default_state()[slot]
    # xKiro has exactly one legal SKU (free_gateway refuses substitutes), so a
    # hand-edited model id would only fail later, mid-answer, with a policy error.
    if state["answer"]["provider"] == "xkiro" and state["answer"]["model"] != SUPPORTED_ANSWER["model"]:
        print(f"[app] NOTE: xKiro answers are pinned to {SUPPORTED_ANSWER['model']}; "
              f"reset from {state['answer']['model']!r}.")
        state["answer"] = dict(SUPPORTED_ANSWER)
    state["chunking"]["mode"] = str(state["chunking"].get("mode") or "restructure").lower()
    if state["chunking"]["mode"] not in {"size", "manual", "restructure"}:
        state["chunking"]["mode"] = "restructure"
    if first_run:
        print(f"[app] first run — selections default to the supported pipeline pair; "
              f"they are saved in {STATE_PATH.name} as you change them.")
    return state


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


# The per-provider memory of model IDs you typed (see record_custom_model):
# newest last, deduped, bounded so a long history cannot grow without limit.
MAX_SAVED_MODELS = 20


def record_custom_model(state: dict, provider: str, model_id: str) -> None:
    """Remember a custom model ID under its provider, then persist the state.

    This is the 'cache' that makes typed IDs reappear in model selection,
    categorized by provider. Registered catalog models are never stored here —
    only what you typed yourself.
    """
    saved = state.setdefault("custom_models", {})
    entries = [m for m in saved.get(provider, []) if m != model_id]
    entries.append(model_id)
    saved[provider] = entries[-MAX_SAVED_MODELS:]
    save_state(state)






# ---------------------------------------------------------------------------
# Lab configuration — a copy of config with the selected slots swapped in.
# Same pattern as chat.chat_config: a copy, never a mutation of the module,
# because main.py and the harness must keep seeing the pinned values.
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Runtime builders
# ---------------------------------------------------------------------------











def open_or_build_index(local, embedder, state, *, allow_ingest: bool = True):
    """A live, fingerprint-verified collection — or None if the user declined."""
    from store import ensure_fresh_chunks, get_collection
    collection = get_collection(local, reset=False)
    if collection.count():
        try:
            ensure_fresh_chunks(collection, local)
        except RuntimeError as exc:
            print(f"\n[app] the index does not match the current settings: {safe_error(exc)}")
            if not allow_ingest or not confirm("rebuild this collection now (re-embeds every chunk)?"):
                print("[app] cancelled — retrieval over stale chunks would be wrong, so nothing ran.")
                return None
            return ingest(local, embedder, state, reset=True)
        return collection
    print(f"[app] the index for this profile is empty ({local.CHROMA_COLLECTION_NAME}).")
    if not allow_ingest or not confirm("build it now? (embedding calls for every chunk, cached afterwards)"):
        return None
    return ingest(local, embedder, state, reset=False)


def prepare_runtime(state: dict, *, need_generator: bool = True):
    """lab config + embedder + index (+ generator). None when the user backed out."""
    for slot in missing_key_slots(state):
        if not ensure_slot_key(state, slot):
            print(f"[app] the selected {slot} provider still has no key — action cancelled.")
            return None
    local = build_lab_config(state)
    from embedder import build_embedder
    embedder = build_embedder(local)
    collection = open_or_build_index(local, embedder, state)
    if collection is None:
        return None
    generator = build_generator(local) if need_generator else None
    return local, embedder, collection, generator






# ---------------------------------------------------------------------------
# Small console helpers
# ---------------------------------------------------------------------------

def prompt(text: str) -> str:
    return input(text).strip()


def confirm(question: str) -> bool:
    answer = prompt(f"{question} [y/N]: ").lower()
    return answer in ("y", "yes")


def choose_number(count: int, *, default: int = 1) -> int:
    """A 1..count choice with a default; 0 means 'back' at the call site."""
    while True:
        raw = prompt(f"choice [{default}]: ")
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw)
        if raw == "0":
            return 0
        print(f"  enter a number 1-{count} (0 = back)")


def sdk_status(registry_entry) -> str:
    sdk = registry_entry.get("sdk")
    if not sdk:
        return "stdlib HTTPS (no SDK needed)"
    module, pip_name = sdk
    if importlib.util.find_spec(module):
        return f"sdk {pip_name} installed"
    return f"sdk {pip_name} MISSING (pip install {pip_name})"


def key_status_line(envs) -> str:
    if not envs:
        return "no key needed (runs locally)"
    env, value = first_set_env(envs)
    return f"{env} set ({masked(value)})" if value else f"{', '.join(envs)} MISSING"


def print_hits(hits, variants, mode: str) -> None:
    for variant in variants:
        print(f"[query] variant [{variant['label']}]: {variant['text']}")
    for hit in hits:
        meta = hit.get("metadata") or {}
        print("-" * 78)
        print(f"rank       : {hit['rank']}")
        if hit.get("similarity") is not None:
            print(f"similarity : {hit['similarity']:+.4f}  (cosine, 1 - distance)")
        if hit.get("keyword_score") is not None:
            print(f"BM25 score : {hit['keyword_score']:.4f}")
        if mode == "blend" and hit.get("blend_score") is not None:
            print(f"blend score: {hit['blend_score']:.4f}  (lambda*cosine + (1-lambda)*BM25_norm)")
        if mode == "rrf" and hit.get("rrf_score") is not None:
            print(f"RRF score  : {hit['rrf_score']:.4f}  (1/(k+rank) fusion)")
        print(f"language   : {meta.get('language')}")
        print(f"heading    : {meta.get('heading') or '(none)'}")
        print(f"source     : {meta.get('source')} | chunk {meta.get('chunk_index')} | "
              f"model {meta.get('embedding_model')}")
        print(f"id         : {hit['id']}")
        print("text:")
        print(hit["text"])
    if not hits:
        print("[query] no hits (an empty index or a language filter that excludes everything)")


# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------

def action_status(state: dict) -> None:
    print("=" * 78)
    print("RAGLab console — status / doctor (no inference, no completion calls)")
    print("=" * 78)
    assignments = read_env_file()
    print(f".env        : {ENV_PATH}"
          f" — {'present' if ENV_PATH.exists() else 'MISSING'}"
          f" ({len(assignments)} assignment(s) in the file)")
    print(f"state       : {STATE_PATH}"
          f" — {'present' if STATE_PATH.exists() else 'not created yet (defaults in use)'}")
    print(f"chunking    : {state['chunking']['mode']}"
          + (f" ({state['chunking']['size']}/{state['chunking']['overlap']} tokens)"
             if state["chunking"]["mode"] == "size" else ""))
    try:
        from chunker import tokenizer_identity
        tokenizer = tokenizer_identity()
        note = ("" if tokenizer == "cl100k_base" else
                "  — WARNING: fallback estimator; chunk boundaries differ from a real "
                "cl100k_base build (set TIKTOKEN_CACHE_DIR)")
        print(f"tokenizer   : {tokenizer}{note}")
    except Exception as exc:                                 # noqa: BLE001 — report, never crash
        print(f"tokenizer   : unavailable ({safe_error(exc)})")
    print(f"retrieval   : k={state['retrieval']['top_k']} mode={state['retrieval']['mode']}"
          f" lang_filter={state['retrieval']['lang_filter'] or 'none'}"
          f" neighbor_radius={state['retrieval']['neighbor_radius']}")
    try:
        print(f"corpus      : {', '.join(str(p) for p in data_dirs(state))}")
    except ValueError as exc:
        print(f"corpus      : {exc}")

    print("\nSelected slots")
    for slot, registry in (("embedding", EMBEDDING_PROVIDERS), ("answer", ANSWER_PROVIDERS)):
        entry = state[slot]
        info = registry[entry["provider"]]
        print(f"  {slot:9}: {slot_display(entry)}")
        print(f"            key: {key_status_line(info['key_envs'])}"
              + (f" | {sdk_status(info)}" if slot == "embedding" else ""))
    print(f"  profile   : {pipeline_marker(state)}")

    print("\nAPI keys (raglab/.env + environment; masked to 8 characters)")
    for name, description in KEY_ENV_INFO.items():
        value = os.environ.get(name, "").strip()
        print(f"  {name:22} {'set (' + masked(value) + ')' if value else '—'}  {description}")

    print("\nOptional provider SDKs")
    for provider, info in EMBEDDING_PROVIDERS.items():
        if info.get("sdk"):
            print(f"  {provider:12} {sdk_status(info)}")

    saved = {p: ms for p, ms in (state.get("custom_models") or {}).items() if ms}
    if saved:
        print("\nSaved custom model IDs (from your previous sessions)")
        for provider in sorted(saved):
            print(f"  {provider:12} {', '.join(saved[provider])}")

    count = collection_count(state)
    print(f"\nIndex for this profile: {collection_name(state)} — "
          + (f"{count} chunk(s)" if count is not None else "count unavailable (is chromadb installed?)"))
    _count, stored_fp = stored_index_info(state)
    stored_tok = re.search(r"tok([^:]+)", stored_fp or "")
    if stored_tok:
        from chunker import tokenizer_identity
        current_tok = tokenizer_identity()
        match = "matches this machine" if stored_tok.group(1) == current_tok else \
            f"DIFFERS from this machine's {current_tok} — boundaries here are not what the index was built with"
        print(f"index built with tokenizer: {stored_tok.group(1)} ({match})")

    print("\nQuestion sets available")
    for name in ("questions.json", "questions_50.json", "questions_real.json"):
        path = PROJECT_DIR / name
        if path.exists():
            cases = 0
            try:
                cases = len(json.loads(path.read_text(encoding="utf-8")).get("cases", []))
            except (OSError, ValueError):
                pass
            print(f"  {name} — {cases} case(s)")
    print("=" * 78)


def select_model(provider: str, registry_entry, current: str, state: dict) -> str:
    registered = registry_entry["models"]
    registered_ids = {entry["id"] for entry in registered}
    # What you typed before, minus anything now in the registered catalog.
    saved = [model for model in state.get("custom_models", {}).get(provider, [])
             if model not in registered_ids]
    print(f"\nModels registered for {provider}:")
    for index, model in enumerate(registered, 1):
        marker = "  <- current" if model["id"] == current else ""
        print(f"  {index}. {model['id']} — {model['note']}{marker}")
    if saved:
        print("  saved from your previous sessions:")
        for index, model in enumerate(saved, len(registered) + 1):
            marker = "  <- current" if model == current else ""
            print(f"  {index}. {model} — your custom ID{marker}")
    custom_allowed = registry_entry.get("custom", True)
    if not custom_allowed:
        print("  (the policy pins this provider to the listed SKU; no substitutes)")
    total = len(registered) + len(saved)
    if custom_allowed:
        total += 1
        print(f"  {total}. type another exact model ID")
    while True:
        choice = choose_number(total, default=1)
        if choice == 0:
            return current
        if choice <= len(registered):
            return registered[choice - 1]["id"]
        if choice <= len(registered) + len(saved):
            return saved[choice - len(registered) - 1]
        model_id = prompt("exact model ID: ")
        if model_id and not re.search(r"\s", model_id):
            if provider == "xkiro" and model_id != SUPPORTED_ANSWER["model"]:
                print("  note: non-pinned xKiro IDs run as EXPERIMENTAL calls here — the live "
                      f"free-price check and benchmark attribution apply only to "
                      f"{SUPPORTED_ANSWER['model']}.")
            record_custom_model(state, provider, model_id)
            print(f"  remembered {model_id} for {provider} "
                  f"(saved in {STATE_PATH.name}; review with menu 2)")
            return model_id
        print("  a model ID has no spaces; try again.")




def select_provider(slot: str, state: dict) -> None:
    registry = EMBEDDING_PROVIDERS if slot == "embedding" else ANSWER_PROVIDERS
    current = state[slot]["provider"]
    print(f"\nRegistered {slot} providers:")
    names = list(registry)
    for index, name in enumerate(names, 1):
        info = registry[name]
        marker = "  <- current" if name == current else ""
        print(f"  {index}. {name:12} {info['label']}")
        print(f"                 key: {key_status_line(info['key_envs'])}"
              + (f" | {sdk_status(info)}" if slot == "embedding" else ""))
        if marker:
            print(f"                {marker}")
    choice = choose_number(len(names), default=names.index(current) + 1)
    if choice == 0:
        return
    provider = names[choice - 1]
    info = registry[provider]
    if info.get("key_envs"):
        key_flow(info["key_envs"], f"[{slot}] {provider} — {info['label']}",
                 hint=info.get("key_hint", ""))
    else:
        print(f"\n[{slot}] {provider} — {info['label']} (no key to configure)")
    current_model = consistent_model(provider, info, state, state[slot]["model"])
    model = select_model(provider, info, current_model, state)
    state[slot] = {"provider": provider, "model": model}
    save_state(state)
    print(f"\n[app] saved: {slot} = {provider}/{model}")
    if slot == "embedding":
        count = collection_count(state)
        print(f"[app] this profile's collection is {collection_name(state)} — "
              + (f"{count} chunk(s) already indexed" if count else
                 "empty, so the first query/chat will offer to ingest"))
        print("[app] note: every embedding profile keeps its OWN collection; "
              "switching never touches another profile's vectors.")


def action_saved_models(state: dict) -> None:
    """Review/remove the model IDs you typed on previous sessions."""
    saved = {provider: models for provider, models
             in (state.get("custom_models") or {}).items() if models}
    if not saved:
        print("\n[saved] no custom model IDs saved yet — one is recorded each time you type "
              "a model ID in the selection menus.")
        return
    print("\nSaved custom model IDs (categorized by provider, stored in "
          f"{STATE_PATH.name})")
    entries = []
    for provider in sorted(saved):
        for model in saved[provider]:
            entries.append((provider, model))
            print(f"  {len(entries):>2}. {provider}/{model}")
    print("   0. back")
    while True:
        raw = prompt("number to remove (0 = back): ")
        if not raw or raw == "0":
            return
        if raw.isdigit() and 1 <= int(raw) <= len(entries):
            provider, model = entries[int(raw) - 1]
            if confirm(f"remove {provider}/{model} from the saved list?"):
                remaining = [m for m in state["custom_models"].get(provider, [])
                             if m != model]
                if remaining:
                    state["custom_models"][provider] = remaining
                else:
                    state["custom_models"].pop(provider, None)
                save_state(state)
                print(f"[saved] removed {provider}/{model}")
            return
        print("  enter a number from the list, or 0")


def action_providers(state: dict) -> None:
    while True:
        print("\nProviders & models — pick a slot to switch")
        print(f"  1. embedding    : {slot_display(state['embedding'])}")
        print(f"  2. answer/chat  : {slot_display(state['answer'])}")
        saved_total = sum(len(models) for models
                          in (state.get("custom_models") or {}).values())
        print(f"  3. saved custom model IDs ({saved_total} saved, by provider)")
        print(f"  profile: {pipeline_marker(state)}")
        print("  0. back")
        choice = choose_number(3, default=0)
        if choice == 1:
            select_provider("embedding", state)
        elif choice == 2:
            select_provider("answer", state)
        elif choice == 3:
            action_saved_models(state)
        else:
            return


def action_keys(state: dict) -> None:
    while True:
        print("\nAPI keys — values are masked to 8 characters and never printed in full")
        names = list(KEY_ENV_INFO)
        for index, name in enumerate(names, 1):
            value = os.environ.get(name, "").strip()
            print(f"  {index:>2}. {name:22} {'set (' + masked(value) + ')' if value else '—'}")
            print(f"      {KEY_ENV_INFO[name]}")
        print("   0. back")
        choice = choose_number(len(names), default=0)
        if choice == 0:
            return
        name = names[choice - 1]
        value = os.environ.get(name, "").strip()
        if value:
            answer = prompt(f"{name} is set ({masked(value)}) — Enter=keep, c=change, d=remove: ").lower()
            if answer in ("", "k", "keep"):
                continue
            if answer == "d":
                if remove_env_assignment(name):
                    print(f"[app] removed {name} from {ENV_PATH.name} and this session.")
                else:
                    print(f"[app] {name} was not in {ENV_PATH.name} (it came from the shell "
                          "environment); unset for this session only.")
                continue
            if answer not in ("c", "change"):
                continue
            key_flow((name,), f"[keys] {name} — {KEY_ENV_INFO[name]}",
                     hint="see the provider's dashboard or .env.example", force_change=True)
        else:
            key_flow((name,), f"[keys] {name} — {KEY_ENV_INFO[name]}",
                     hint="see the provider's dashboard or .env.example")


def action_inspect(state: dict) -> None:
    import statistics
    from chunker import chunk_all
    from loader import load_all
    print("=" * 78)
    print("INSPECT — load and chunk only, NO embedding, NO API calls")
    print("=" * 78)
    local = build_lab_config(state)
    dirs = data_dirs(state)
    docs = load_all(dirs)
    if not docs:
        print("[inspect] nothing to inspect; put documents in the corpus directories "
              "or change them in lab settings.")
        return
    chunks = chunk_all(docs, local)
    print(f"[inspect] parameters: mode={local.CHUNKING_MODE} | "
          f"size={local.CHUNK_SIZE_TOKENS} | overlap={local.CHUNK_OVERLAP_TOKENS}")
    print(f"[inspect] {len(chunks)} chunk(s) across {len(docs)} document(s)")
    counts = [chunk.token_count for chunk in chunks]
    if counts:
        print(f"[inspect] tokens: total={sum(counts)} min={min(counts)} "
              f"median={statistics.median(counts):.0f} mean={statistics.mean(counts):.1f} max={max(counts)}")
        step = max(1, local.CHUNK_SIZE_TOKENS // 5)
        print("[inspect] token histogram:")
        for low in range(0, max(counts) + 1, step):
            bar = "#" * sum(1 for c in counts if low <= c < low + step)
            if bar:
                print(f"  {low:>4}-{low + step:<4}: {bar}")
    raw = prompt("\nhow many chunks to print in full? [3, 0 = none]: ")
    limit = int(raw) if raw.isdigit() else 3
    for chunk in chunks[:limit]:
        print("-" * 78)
        print(f"chunk #{chunk.index:03d} | source={chunk.source} | language={chunk.language} | "
              f"tokens={chunk.token_count} | section={chunk.section_type}")
        print(f"heading: {chunk.heading or '(none)'}")
        print(chunk.text)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Quote-vs-chunk diagnostic — the "why was my quote rejected" tool
# ---------------------------------------------------------------------------

def _search_rows(state: dict):
    """(description, rows) with rows = [(text, source, index, heading), ...].

    Prefers the STORED chunks of the current profile — they are literally what
    retrieval supplied to the model — and falls back to a fresh offline
    chunking with the current settings when that index is empty.
    """
    local = build_lab_config(state)
    try:
        from store import _client
        client = _client(local)
        collection = client.get_collection(local.CHROMA_COLLECTION_NAME)
        count = collection.count()
    except Exception:                                   # noqa: BLE001 — chroma absent/unbuilt
        count = 0
    if count:
        got = collection.get(include=["documents", "metadatas"])
        rows = [(text, (meta or {}).get("document") or (meta or {}).get("source") or "?",
                 (meta or {}).get("chunk_index"), (meta or {}).get("heading") or "")
                for text, meta in zip(got.get("documents") or [], got.get("metadatas") or [])]
        return (f"{count} stored chunk(s) in {local.CHROMA_COLLECTION_NAME} "
                "(exactly what retrieval supplies)"), rows
    from chunker import chunk_all
    from loader import load_all
    chunks = chunk_all(load_all(data_dirs(state)), local)
    rows = [(chunk.text, chunk.source, chunk.index, chunk.heading) for chunk in chunks]
    return (f"{len(rows)} freshly chunked chunk(s) with the current settings "
            "(the profile index is empty)"), rows


def locate_text(needle: str, rows):
    """Where does a quote/phrase live relative to the chunk boundaries?

    Uses the citation gate's own normalization (answer.normalized_quote), so
    'full' being non-empty means a verbatim quote of `needle` CAN pass the
    gate from that chunk. 'head'/'tail' locate the first/last ~24 characters
    when no single chunk holds the whole text (a quote crossing a chunk
    boundary can never validate, however faithfully the model copies it).
    """
    from answer import normalized_quote
    target = normalized_quote(needle)
    if not target:
        return {"needle": "", "full": [], "head": [], "tail": []}
    normalized = [(normalized_quote(text), row) for row in rows for text in [row[0]]]
    window = 24 if len(target) > 24 else len(target)
    return {
        "needle": target,
        "full": [row for norm, row in normalized if target in norm],
        "head": [row for norm, row in normalized if target[:window] in norm],
        "tail": [row for norm, row in normalized if target[-window:] in norm],
    }


def action_search_chunks(state: dict) -> None:
    print("=" * 78)
    print("SEARCH CHUNKS — is this text inside ONE chunk? (no API calls)")
    print("=" * 78)
    needle = prompt("text to find (a phrase, or the full rejected quote): ")
    if not needle:
        return
    description, rows = _search_rows(state)
    print(f"[search] scanned {description}")
    found = locate_text(needle, rows)
    if found["full"]:
        print(f"[search] the FULL text is inside {len(found['full'])} chunk(s) — a verbatim "
              "quote of it can pass the citation gate:")
        for _text, source, index, heading in found["full"]:
            print(f"  - {source}::chunk_{index:04d} | heading={heading or '(none)'}")
        if confirm("print the full text of the first hit?"):
            text, source, index, heading = found["full"][0]
            print("-" * 78)
            print(f"{source}::chunk_{index:04d} | heading={heading or '(none)'}")
            print(text)
        return
    head, tail = found["head"], found["tail"]
    shared = [row for row in head if row in tail]
    if head and tail and not shared:
        print("[search] the text CROSSES chunk boundaries — no single chunk contains it, so a "
              "one-chunk verbatim quote of the whole text can never validate:")
        for _text, source, index, heading in head:
            print(f"  - starts in {source}::chunk_{index:04d} (heading={heading or '(none)'})")
        for _text, source, index, heading in tail:
            print(f"  - ends in   {source}::chunk_{index:04d} (heading={heading or '(none)'})")
        print("[search] fixes: a larger k or neighbor radius >= 1 (adjacent chunks are supplied "
              "too, and each claim may quote any of them), or a different chunking mode/size.")
        return
    if shared or head or tail:
        print("[search] the text's edges sit in one chunk but the middle differs — the wording "
              "pasted is not the corpus wording (a paraphrase, or characters the lab's "
              "normalization does not fold, e.g. ى vs ي).")
        return
    print("[search] not found anywhere in the chunked corpus — the quote was invented or differs "
          "beyond the lab's normalization.")


def action_embed_test(state: dict) -> None:
    if not ensure_slot_key(state, "embedding"):
        print("[app] no key for the selected embedding provider — cancelled.")
        return
    local = build_lab_config(state)
    from embedder import build_embedder
    embedder = build_embedder(local)
    embedder.startup_report()      # provider/model/batch report + the 3-language sanity check


def action_ingest(state: dict) -> None:
    if not ensure_slot_key(state, "embedding"):
        print("[app] no key for the selected embedding provider — cancelled.")
        return
    local = build_lab_config(state)
    from embedder import build_embedder
    from store import get_collection
    embedder = build_embedder(local)
    collection = get_collection(local, reset=False)
    if collection.count():
        print(f"[app] {local.CHROMA_COLLECTION_NAME} already holds {collection.count()} chunk(s).")
        if not confirm("rebuild it from scratch (reset + re-embed)?"):
            print("[app] kept the existing index.")
            return
        reset = True
    else:
        reset = False
    embedder.startup_report()      # 1 batched call: verifies the key BEFORE the big spend
    ingest(local, embedder, state, reset=reset)


def action_query(state: dict) -> None:
    runtime = prepare_runtime(state, need_generator=False)
    if runtime is None:
        return
    local, embedder, collection, _ = runtime
    from evaluate import prepare_query_text
    from retrieval import retrieve
    from translate import detect_language
    question = prompt("\nquestion (blank to cancel): ")
    if not question:
        return
    q_text = prepare_query_text(question)
    language = detect_language(q_text)
    mode = state["retrieval"]["mode"]
    k = state["retrieval"]["top_k"]
    lang_filter = state["retrieval"]["lang_filter"]
    print(f"[query] k={k} | mode={mode} | lang_filter={lang_filter or 'none'} | "
          f"query_lang={language} | embedder={embedder.provider_name}/{embedder.model}")
    hits, variants = retrieve(local, embedder, collection, q_text, language=language,
                              translator=None, mode=mode, top_k=k, lang_filter=lang_filter,
                              variant_strategy="original")
    print_hits(hits, variants, mode)









def maybe_greeting_reply(question: str, *, language: str = None) -> bool:
    """Print the local greeting reply and return True when input is pure smalltalk."""
    reply = greeting_reply(question, language=language)
    if reply is None:
        return False
    _language, text = reply
    print(text)
    print("[app] greeting handled locally — no retrieval, no model call, nothing asserted "
          "about the documents.")
    return True


def one_turn(state, runtime, question: str, *, show_context: bool = False) -> dict:
    """A single grounded question through chat.ask — shared by menu 8 and --ask."""
    if maybe_greeting_reply(question):
        return {"status": "greeting", "reason": "smalltalk", "validation_ok": True,
                "model": "(none — answered locally)", "claims": [], "sources": [],
                "question": question, "retrieved": 0, "seconds": 0.0}
    local, embedder, collection, generator = runtime
    result = chat_mod.ask(local, embedder, collection, generator, question,
                          top_k=local.ANSWER_TOP_K,
                          neighbor_radius=local.ANSWER_NEIGHBOR_RADIUS,
                          use_cache=True, mode=state["retrieval"]["mode"],
                          lang_filter=state["retrieval"]["lang_filter"])
    print(chat_mod.format_turn(result, show_context=show_context))
    print(f"\n[app] status={result['status']}/{result.get('reason')} · model={result['model']} · "
          f"{result.get('seconds', 0)}s · retrieved={result.get('retrieved', 0)} · "
          f"cached={result.get('cached', False)}"
          + (f" · failed check: {result['error']}" if result.get("error") else ""))
    if result.get("reason") == "invalid_output":
        # The answer text can be factually right and still be refused: the contract
        # requires every claim to quote one retrieved chunk EXACTLY. Say which check
        # failed and what usually causes it, because "invalid_output" alone is opaque.
        print("[app] the citation gate rejected the reply. Common causes: a quote that crosses a "
              "chunk boundary (check with the search-chunks action), a punctuation/character "
              "variant (– vs -, ى vs ي, model 'correcting' spelling), or malformed JSON.")
        if confirm("ask the model again? (hosted models are not bit-deterministic; "
                   "a fresh reply often quotes verbatim)"):
            return one_turn(state, runtime, question, show_context=show_context)
    if result.get("status") == "error" and confirm("provider error — ask again?"):
        return one_turn(state, runtime, question, show_context=show_context)
    return result


def action_answer(state: dict) -> None:
    question = prompt("\nquestion (blank to cancel): ")
    if not question:
        return
    if maybe_greeting_reply(question):      # a greeting needs no index and no model
        return
    runtime = prepare_runtime(state, need_generator=True)
    if runtime is None:
        return
    one_turn(state, runtime, question)


def action_chat(state: dict) -> None:
    runtime = prepare_runtime(state, need_generator=True)
    if runtime is None:
        return
    local, embedder, collection, generator = runtime
    from store import collection_languages
    settings = {"top_k": local.ANSWER_TOP_K, "show_context": False, "log": None,
                "model": local.ANSWER_MODEL, "thinking": False,
                "chunking": local.CHUNK_SIZE_TOKENS, "overlap": local.CHUNK_OVERLAP_TOKENS,
                "mode": state["retrieval"]["mode"], "language": None,
                "neighbor_radius": local.ANSWER_NEIGHBOR_RADIUS,
                "corpus_languages": collection_languages(collection)}
    print("\nAsk about the documents. Ctrl-D or :quit to leave, :help for commands "
          "(:k N, :show, :lang, :mode, :log).")
    print(f"[app] chat: {generator.model} over {collection.count()} chunk(s) | "
          f"k={settings['top_k']} | mode={settings['mode']} | "
          f"languages: {', '.join(settings['corpus_languages']) or '?'}")
    stream = sys.stdin if not sys.stdin.isatty() else chat_mod._prompted_lines()
    for question in chat_mod.read_questions(stream, settings):
        if maybe_greeting_reply(question, language=settings.get("language")):
            continue
        try:
            result = chat_mod.ask(local, embedder, collection, generator, question,
                                  top_k=settings["top_k"],
                                  neighbor_radius=settings.get("neighbor_radius", 0),
                                  use_cache=True, mode=settings["mode"],
                                  language=settings.get("language"),
                                  lang_filter=state["retrieval"]["lang_filter"],
                                  corpus_langs=settings.get("corpus_languages"))
        except (ValueError, RuntimeError) as exc:
            print(f"[app] {safe_error(exc)}")
            continue
        chat_mod.log_turn(settings["log"], result)
        print(chat_mod.format_turn(result, show_context=settings["show_context"]))
        print(f"\n[app] {result['status']}/{result.get('reason')} · {result['model']} · "
              f"{result.get('seconds', 0)}s · k={settings['top_k']}"
              + (f" · failed check: {result['error']}" if result.get("error") else ""))


def action_evaluate(state: dict) -> None:
    sets = ["questions.json", "questions_50.json", "questions_real.json"]
    print("\nWhich question set?")
    for index, name in enumerate(sets, 1):
        path = PROJECT_DIR / name
        cases = 0
        if path.exists():
            try:
                cases = len(json.loads(path.read_text(encoding="utf-8")).get("cases", []))
            except (OSError, ValueError):
                pass
        print(f"  {index}. {name} ({cases} case(s))")
    print("  4. another path")
    choice = choose_number(4, default=1)
    if choice == 0:
        return
    if choice == 4:
        path = Path(prompt("path to the question set JSON: ").expanduser())
    else:
        path = PROJECT_DIR / sets[choice - 1]
    runtime = prepare_runtime(state, need_generator=False)
    if runtime is None:
        return
    local, embedder, collection, _ = runtime
    from evaluate import load_question_set, print_report, run_evaluation, save_run
    cases = load_question_set(path)
    if not cases:
        print(f"[evaluate] no questions in {path}")
        return
    raw = prompt(f"top-k hits to record per question [{cfg.EVAL_TOP_K}]: ")
    top_k = int(raw) if raw.isdigit() and int(raw) > 0 else cfg.EVAL_TOP_K
    run = run_evaluation(local, embedder, collection, cases,
                         mode=state["retrieval"]["mode"], top_k=top_k, translator=None)
    print_report(run)
    save_run(run, local.RESULTS_DIR)
    print(f"[app] note: this is a lab measurement on {local.CHROMA_COLLECTION_NAME}, "
          "not a benchmark result.")


def action_diagnostics(state: dict) -> None:
    while True:
        print("\nDiagnostics")
        print("  1. offline harness50 A/B — BM25 chunking comparison, NO API calls")
        print("  2. xKiro provider catalog — read-only listing, NO inference "
              "(needs XKIRO_API_KEY)")
        print("  0. back")
        choice = choose_number(2, default=0)
        if choice == 1:
            print("[app] running harness50.py in a subprocess "
                  f"({sys.executable})...")
            completed = subprocess.run([sys.executable, "harness50.py"],
                                       cwd=PROJECT_DIR)
            print(f"[app] harness50.py exited with {completed.returncode}"
                  + (" — see results/harness50/comparison.md" if completed.returncode == 0 else ""))
        elif choice == 2:
            try:
                from provider_catalog import collect
                collect()
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                print(f"[app] catalog check failed: {safe_error(exc)}")
        elif choice == 0:
            return


def action_settings(state: dict) -> None:
    while True:
        chunking = state["chunking"]
        retrieval = state["retrieval"]
        dirs = state.get("data_dirs")
        print("\nLab settings (stored in app_state.json)")
        print(f"  1. chunking mode      : {chunking['mode']}  (size / restructure / manual)")
        print(f"  2. chunk size tokens  : {chunking['size']}  (size mode)")
        print(f"  3. chunk overlap      : {chunking['overlap']}  (size mode)")
        print(f"  4. retrieval top-k    : {retrieval['top_k']}")
        print(f"  5. retrieval mode     : {retrieval['mode']}  (vector / rrf / blend)")
        print(f"  6. language filter    : {retrieval['lang_filter'] or 'none'}  (none / ar / fr / en)")
        print(f"  7. neighbor radius    : {retrieval['neighbor_radius']}  (0 / 1 / 2)")
        print(f"  8. corpus directories : {', '.join(dirs) if dirs else 'default (../docs + raglab/data)'}")
        print("  9. reset everything to defaults")
        print("  0. back")
        choice = choose_number(9, default=0)
        if choice == 0:
            return
        if choice == 1:
            mode = prompt("chunking mode (size / restructure / manual): ").lower()
            if mode in {"size", "restructure", "manual"}:
                chunking["mode"] = mode
            else:
                print("  mode must be size, restructure or manual")
        elif choice in (2, 3):
            raw = prompt("value: ")
            if raw.isdigit() and int(raw) > 0:
                chunking["size" if choice == 2 else "overlap"] = int(raw)
            else:
                print("  enter a positive integer")
        elif choice == 4:
            raw = prompt("top-k (1-20): ")
            if raw.isdigit() and 1 <= int(raw) <= 20:
                retrieval["top_k"] = int(raw)
            else:
                print("  enter a number 1-20")
        elif choice == 5:
            mode = prompt("retrieval mode (vector / rrf / blend): ").lower()
            if mode in {"vector", "rrf", "blend"}:
                retrieval["mode"] = mode
            else:
                print("  mode must be vector, rrf or blend")
        elif choice == 6:
            value = prompt("language filter (none / ar / fr / en): ").lower()
            retrieval["lang_filter"] = value if value in {"ar", "fr", "en"} else None
        elif choice == 7:
            raw = prompt("neighbor radius (0 / 1 / 2): ")
            if raw in {"0", "1", "2"}:
                retrieval["neighbor_radius"] = int(raw)
        elif choice == 8:
            raw = prompt("comma-separated directories, or 'default': ")
            if raw.lower() in ("", "default"):
                state["data_dirs"] = None
            else:
                state["data_dirs"] = [part.strip() for part in raw.split(",") if part.strip()]
        elif choice == 9:
            if confirm("reset ALL selections and settings to defaults?"):
                state.clear()
                state.update(default_state())
        if choice in {1, 2, 3}:
            count = collection_count(state)
            print(f"[app] collection for this chunking: {collection_name(state)} — "
                  + (f"{count} chunk(s) (a fingerprint mismatch will offer a rebuild)"
                     if count else "empty"))
        save_state(state)


# ---------------------------------------------------------------------------
# Menu plumbing
# ---------------------------------------------------------------------------

MENU_ACTIONS = [
    ("1", "status / doctor", action_status),
    ("2", "providers & models (switch embedding / answer profiles)", action_providers),
    ("3", "API keys (keep / change / add / remove)", action_keys),
    ("4", "inspect corpus (chunking preview — no API calls)", action_inspect),
    ("5", "search the chunks for a text/quote (no API calls)", action_search_chunks),
    ("6", "embedding sanity check (one batched embedding call)", action_embed_test),
    ("7", "ingest / rebuild the index for this profile", action_ingest),
    ("8", "retrieval query (top-k hits, no chat model)", action_query),
    ("9", "grounded answer (one question, cited)", action_answer),
    ("10", "chat over the documents", action_chat),
    ("11", "evaluate a question set", action_evaluate),
    ("12", "lab settings (chunking, retrieval, corpus)", action_settings),
    ("13", "diagnostics (offline harness50, xKiro catalog)", action_diagnostics),
]


def banner(state: dict) -> None:
    count = collection_count(state)
    index = (f"{count} chunk(s) indexed" if count else "no index yet")
    print("\n" + "=" * 78)
    print(f"RAGLab console — embeddings: {slot_display(state['embedding'])}")
    print(f"                answers:    {slot_display(state['answer'])} | {index}")
    print(f"[{pipeline_marker(state)}]")
    print("=" * 78)


def menu_loop(state: dict) -> None:
    while True:
        banner(state)
        for key, label, _ in MENU_ACTIONS:
            print(f"  {key:>2}  {label}")
        print("   0  quit")
        try:
            choice = prompt("> ")
        except EOFError:
            print("\n[app] input closed — goodbye.")
            return
        if choice in ("", "0", "q", "quit", "exit"):
            print("[app] goodbye.")
            return
        handler = next((fn for key, _, fn in MENU_ACTIONS if key == choice), None)
        if handler is None:
            print("[app] unknown option — pick a number from the list.")
            continue
        try:
            handler(state)
        except KeyboardInterrupt:
            print("\n[app] interrupted — back to the menu (Ctrl-C here quits).")
        except EOFError:
            print("\n[app] input closed mid-action — back to the menu.")
        except SystemExit as exc:                     # embedder's loud missing-key exits
            print(f"[app] action aborted: {safe_error(exc) or exc}")
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"[app] action failed: {safe_error(exc)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="RAGLab console — one interactive entry point for every "
                    "lab function, with provider/model switching and per-provider "
                    "API-key management.")
    parser.add_argument("--status", action="store_true",
                        help="print the doctor report and exit (no prompts)")
    parser.add_argument("--no-keycheck", action="store_true", dest="no_keycheck",
                        help="skip the startup keep/change/add key questions")
    parser.add_argument("--reset-state", action="store_true", dest="reset_state",
                        help="forget saved selections and use the defaults")
    parser.add_argument("--ingest", action="store_true",
                        help="startup checks, then build/rebuild this profile's index")
    parser.add_argument("--chat", action="store_true",
                        help="startup checks, then open the chat REPL")
    parser.add_argument("--ask", metavar="QUESTION", dest="ask",
                        help="startup checks, one grounded answer, then exit")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    state = load_state()
    if args.reset_state:
        state = default_state()
        save_state(state)
        print(f"[app] selections reset to defaults and saved to {STATE_PATH.name}")

    if args.status:
        action_status(state)
        return 0

    print("RAGLab console — everything the lab can do, from one place.")
    print("Keys live in raglab/.env (never committed); provider/model selections in "
          "raglab/app_state.json.")
    print("This is the lab surface next to the pinned pipeline: it uses its own "
          "collections and writes no benchmark numbers.")
    startup_key_check(state, interactive=not args.no_keycheck)

    try:
        if args.ingest:
            action_ingest(state)
            return 0
        if args.ask:
            if maybe_greeting_reply(args.ask):
                return 0
            runtime = prepare_runtime(state, need_generator=True)
            if runtime is None:
                return 2
            result = one_turn(state, runtime, args.ask)
            return 0 if result.get("validation_ok", True) else 2
        if args.chat:
            action_chat(state)
            return 0
    except SystemExit as exc:
        print(f"[app] aborted: {safe_error(exc) or exc}")
        return 2
    except (ValueError, RuntimeError, OSError, EOFError) as exc:
        print(f"[app] failed: {safe_error(exc)}")
        return 2

    try:
        menu_loop(state)
    except KeyboardInterrupt:
        print("\n[app] interrupted — goodbye.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[app] interrupted")
        sys.exit(130)
