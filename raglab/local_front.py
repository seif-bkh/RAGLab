#!/usr/bin/env python3
"""local_front.py — the app.py console experience, driven entirely over REST.

Same menus, same flows as app.py (status/doctor, provider & model switching,
API keys keep/change/add, inspect, chunk search, embedding sanity, ingest,
retrieval, grounded answers, chat, evaluate, settings, diagnostics) — but
every action is an HTTP call to the RAGLab service. This file imports NOTHING
from the lab: it is a pure client (stdlib urllib + JSON), so it exercises
exactly the boundary another microservice in your architecture would.

The service must be running (see raglab/SERVICE.md):

    python -m uvicorn service:app --host 0.0.0.0 --port 8000

then:

    python local_front.py                 the console menu (like python app.py)
    python local_front.py --status        doctor report, no prompts, exit
    python local_front.py --ingest        build/rebuild the index, wait for the job
    python local_front.py --ask "What is Murabaha?"
    python local_front.py --search "ما هي المرابحة؟" -k 5
    python local_front.py --interactive   straight into the chat REPL
    python local_front.py --smoke         endpoint smoke suite (state-aware)
    python local_front.py --no-keycheck   skip the startup key questions
    python local_front.py --base-url http://localhost:8000

Where things live (mirror of app.py, split client/server):
  * provider/model selections -> the SERVICE profile (POST /profile), not a
    local file: the front is stateless except for one thing —
  * custom model IDs you type  -> front_state.json next to this file, so the
    "saved from your previous sessions" list survives restarts, exactly like
    app.py's app_state.json.
  * API keys                  -> the service process env via POST /keys
    (optionally persisted to raglab/.env ON THE SERVICE HOST); never echoed —
    presence and first 8 characters only, the repo's standing masking rule.

Exit codes: 0 ok / 1 failed / 2 service unreachable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = os.environ.get("RAGLAB_SERVICE_URL", "http://localhost:8000")
ANSWER_TIMEOUT = 180          # xKiro's first call includes the live price check
LONG_TIMEOUT = 960            # evaluate / diagnostics can re-chunk the corpus
INGEST_POLL_SECONDS = 1.0
STATE_PATH = Path(__file__).resolve().parent / "front_state.json"
PLACEHOLDER_PATTERNS = ("paste-your", "paste your", "your-key", "your_key",
                        "your_kira", "changeme", "change-me", "xxx", "placeholder",
                        "api_key")
MAX_SAVED_MODELS = 20
QUESTION_SETS = ("questions.json", "questions_50.json", "questions_real.json")


# ---------------------------------------------------------------------------
# HTTP layer (stdlib only)
# ---------------------------------------------------------------------------

class Api:
    """GET/POST/DELETE JSON against the service. Never raises on HTTP errors:
    returns (status, body) so callers can assert on the error contract."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, payload=None, params=None, timeout=None):
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(
                {key: value for key, value in params.items() if value is not None})
        data = None
        headers = {"Accept": "application/json",
                   "User-Agent": "RAGLab-local-front/1.0"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return resp.status, self._read(resp)
        except urllib.error.HTTPError as exc:
            return exc.code, self._read(exc)

    @staticmethod
    def _read(resp) -> object:
        raw = resp.read().decode("utf-8", "replace")
        try:
            return json.loads(raw)
        except ValueError:
            return raw

    def get(self, path: str, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path: str, payload=None, params=None, **kwargs):
        return self.request("POST", path, payload=payload, params=params, **kwargs)

    def delete(self, path: str, **kwargs):
        return self.request("DELETE", path, **kwargs)


def reason_of(body) -> str:
    """The service's error reason, wherever FastAPI put it."""
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict) and detail.get("reason"):
            return str(detail["reason"])
        if isinstance(detail, str):
            return detail
    return ""


def detail_of(body) -> str:
    if isinstance(body, dict) and isinstance(body.get("detail"), dict):
        return json.dumps(body["detail"], ensure_ascii=False)
    return str(body)[:400]


# ---------------------------------------------------------------------------
# Front state: the custom-model-ID memory (the one thing the front keeps)
# ---------------------------------------------------------------------------

def load_front_state() -> dict:
    state = {"custom_models": {}}
    try:
        saved = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(saved, dict) and isinstance(saved.get("custom_models"), dict):
            state["custom_models"] = {
                str(provider): [str(m) for m in models if isinstance(m, str) and m.strip()]
                for provider, models in saved["custom_models"].items()
                if isinstance(models, list)}
    except (OSError, ValueError):
        pass
    return state


def save_front_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def record_custom_model(state: dict, provider: str, model_id: str) -> None:
    saved = state.setdefault("custom_models", {})
    entries = [m for m in saved.get(provider, []) if m != model_id]
    entries.append(model_id)
    saved[provider] = entries[-MAX_SAVED_MODELS:]
    save_front_state(state)


# ---------------------------------------------------------------------------
# Console helpers (same shape as app.py's)
# ---------------------------------------------------------------------------

def prompt(text: str) -> str:
    return input(text).strip()


def confirm(question: str) -> bool:
    return prompt(f"{question} [y/N]: ").lower() in ("y", "yes")


def choose_number(count: int, *, default: int = 1) -> int:
    while True:
        raw = prompt(f"choice [{default}]: ")
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw)
        if raw == "0":
            return 0
        print(f"  enter a number 1-{count} (0 = back)")


def masked(value: str) -> str:
    text = str(value or "")
    return text[:8] + "…" if len(text) > 8 else ("set" if text else "")


def looks_like_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in PLACEHOLDER_PATTERNS) or len(value) < 8


def paste_key() -> str | None:
    try:
        value = input("  paste the key — input hidden (Enter to cancel): ").strip()
    except EOFError:
        return None
    if not value:
        return None
    if re.search(r"[\s\"']", value):
        print("  a key cannot contain spaces or quotes; try again.")
        return paste_key()
    if looks_like_placeholder(value):
        print("  that looks like the .env.example placeholder, not a real key; try again.")
        return paste_key()
    return value


# ---------------------------------------------------------------------------
# The console (every action = one or more endpoint calls)
# ---------------------------------------------------------------------------

class Console:
    def __init__(self, api: Api, front_state: dict):
        self.api = api
        self.state = front_state

    # -- data fetchers --------------------------------------------------------

    def health(self):
        status, body = self.api.get("/health")
        if status != 200:
            raise RuntimeError(f"GET /health -> {status}: {detail_of(body)}")
        return body

    def profile(self):
        status, body = self.api.get("/profile")
        if status != 200:
            raise RuntimeError(f"GET /profile -> {status}: {detail_of(body)}")
        return body["profile"]

    def models(self):
        status, body = self.api.get("/models")
        if status != 200:
            raise RuntimeError(f"GET /models -> {status}: {detail_of(body)}")
        return body

    def keys(self):
        status, body = self.api.get("/keys")
        if status != 200:
            raise RuntimeError(f"GET /keys -> {status}: {detail_of(body)}")
        return body["keys"]

    def key_map(self) -> dict:
        return {row["env"]: row for row in self.keys()}

    def provider_key_status(self, slot_rows, provider) -> str:
        row = next((r for r in slot_rows if r["provider"] == provider), {})
        envs = row.get("key_envs") or []
        if not envs:
            return "no key needed"
        kmap = self.key_map()
        set_envs = [env for env in envs if kmap.get(env, {}).get("status") == "set"]
        if set_envs:
            return f"{set_envs[0]} set ({kmap[set_envs[0]]['masked']})"
        return f"{', '.join(envs)} MISSING"

    # -- startup --------------------------------------------------------------

    def banner(self) -> None:
        health = self.health()
        profile = health["profile"]
        count = health["index"]["count"]
        index = f"{count} chunk(s) indexed" if count else "no index yet"
        print("\n" + "=" * 78)
        print(f"RAGLab front — service {self.api.base_url}")
        print(f"               embeddings: {profile['embedding']['provider']}/{profile['embedding']['model']}")
        print(f"               answers:    {profile['answer']['provider']}/{profile['answer']['model']} | {index}")
        print(f"[{profile['pipeline']}]")
        print("=" * 78)

    def key_flow(self, slot: str, provider_row) -> None:
        """keep / change / add for one provider's key — app.py's flow, via /keys."""
        envs = provider_row.get("key_envs") or []
        if not envs:
            return
        kmap = self.key_map()
        primary = envs[0]
        set_row = next((kmap[e] for e in envs if kmap.get(e, {}).get("status") == "set"), None)
        print(f"\n[{slot}] {provider_row['provider']} — {provider_row['label']}")
        if set_row:
            print(f"  {set_row['env']} is set ({set_row['masked']}).")
            choice = prompt("  keep it (Enter) / c: change it / s: skip: ").lower()
            if choice not in ("c", "change"):
                return
        else:
            print(f"  no key configured (checked: {', '.join(envs)}).")
        value = paste_key()
        if value is None:
            print("  cancelled — nothing sent.")
            return
        persist = confirm("  also persist it to the service's raglab/.env?")
        status, body = self.api.post("/keys", payload={
            "key_env": set_row["env"] if set_row else primary, "value": value,
            "persist": persist})
        if status != 200:
            print(f"  [front] HTTP {status}: {detail_of(body)}")
            return
        print(f"  service now has {body['env']} set ({body['masked']})"
              + (" — written to the service's .env too" if body.get("persisted") else ""))

    def startup_key_check(self) -> None:
        """Ask keep/change/add for the selected providers' keys (app.py parity)."""
        health = self.health()
        models = self.models()
        for slot in ("embedding", "answer"):
            provider = health["profile"][slot]["provider"]
            row = next((r for r in models[slot] if r["provider"] == provider), None)
            if row and row.get("key_envs"):
                self.key_flow(slot, row)

    # -- 1 status / doctor ------------------------------------------------------

    def action_status(self) -> None:
        health = self.health()
        profile = self.profile()
        print("=" * 78)
        print(f"RAGLab front — status / doctor (service {self.api.base_url})")
        print("=" * 78)
        p = health["profile"]
        print(f"service     : {self.api.base_url} (version {health.get('version')})")
        print(f"embedding   : {p['embedding']['provider']}/{p['embedding']['model']}")
        print(f"answer      : {p['answer']['provider']}/{p['answer']['model']}")
        print(f"chunking    : {p['chunking']['mode']}"
              + (f" ({p['chunking']['size']}/{p['chunking']['overlap']} tokens)"
                 if p["chunking"]["mode"] == "size" else ""))
        print(f"retrieval   : k={p['retrieval']['top_k']} mode={p['retrieval']['mode']}"
              f" lang_filter={p['retrieval']['lang_filter'] or 'none'}"
              f" neighbor_radius={p['retrieval']['neighbor_radius']}")
        print(f"corpus      : {', '.join(profile.get('data_dirs') or ['(service default)'])}")
        index = health["index"]
        print(f"index       : {index['collection']} — {index['count']} chunk(s)"
              + (f" | tokenizer match: {index['tokenizer_match']}"
                 if index.get("tokenizer_match") is not None else ""))
        print(f"ingest job  : {health['ingest']['state']}")
        print(f"switching   : {'enabled' if health.get('switching') else 'see /profile'}")
        print(f"pipeline    : {p['pipeline']}")
        print("\nAPI keys on the service (masked to 8 characters)")
        for row in self.keys():
            mark = f"set ({row['masked']})" if row["status"] == "set" else "—"
            print(f"  {row['env']:22} {mark}  {row['description']}")
        saved = self.state.get("custom_models") or {}
        if saved:
            print("\nSaved custom model IDs (this front's memory)")
            for provider in sorted(saved):
                print(f"  {provider:12} {', '.join(saved[provider])}")
        print("=" * 78)

    # -- 2 providers & models -----------------------------------------------------

    def switch_profile(self, payload: dict) -> bool:
        status, body = self.api.post("/profile", payload=payload)
        if status == 403:
            print(f"[front] the service refused: {reason_of(body)} — start it with "
                  "RAGLAB_ALLOW_PROFILE_SWITCH=1 (docker-compose.yml pins 0).")
            return False
        if status != 200:
            print(f"[front] HTTP {status}: {detail_of(body)}")
            return False
        for note in body.get("notes") or []:
            print(f"[front] note: {note}")
        print(f"[front] profile updated — collection: {body['collection']}")
        if body.get("index_note"):
            print(f"[front] {body['index_note']}")
        return True

    def select_model(self, slot: str, provider_row) -> None:
        provider = provider_row["provider"]
        registered = provider_row["models"]
        registered_set = set(registered)
        saved = [m for m in self.state.get("custom_models", {}).get(provider, [])
                 if m not in registered_set]
        print(f"\nModels registered for {provider}:")
        for index, model in enumerate(registered, 1):
            print(f"  {index}. {model}")
        if saved:
            print("  saved from your previous sessions:")
            for index, model in enumerate(saved, len(registered) + 1):
                print(f"  {index}. {model} — your custom ID")
        total = len(registered) + len(saved) + 1
        print(f"  {total}. type another exact model ID")
        while True:
            choice = choose_number(total, default=1)
            if choice == 0:
                return
            if choice <= len(registered):
                model = registered[choice - 1]
            elif choice <= len(registered) + len(saved):
                model = saved[choice - len(registered) - 1]
            else:
                model = prompt("exact model ID: ")
                if not model or re.search(r"\s", model):
                    print("  a model ID has no spaces; try again.")
                    continue
                record_custom_model(self.state, provider, model)
                print(f"  remembered {model} for {provider} (front_state.json)")
            if slot == "answer" and provider == "xkiro" and model != "qwen/qwen3.8-max:free":
                print("  note: non-pinned xKiro IDs run as EXPERIMENTAL calls on the "
                      "service — no live free-price check, no benchmark attribution.")
            self.switch_profile({slot: {"provider": provider, "model": model}})
            return

    def action_providers(self) -> None:
        while True:
            health = self.health()
            models = self.models()
            p = health["profile"]
            print("\nProviders & models — pick a slot to switch (all via POST /profile)")
            print(f"  1. embedding    : {p['embedding']['provider']}/{p['embedding']['model']}")
            print(f"  2. answer/chat  : {p['answer']['provider']}/{p['answer']['model']}")
            saved_total = sum(len(v) for v in self.state.get("custom_models", {}).values())
            print(f"  3. saved custom model IDs ({saved_total} saved, by provider)")
            print(f"  profile: {p['pipeline']}")
            print("  0. back")
            choice = choose_number(3, default=0)
            if choice == 0:
                return
            if choice == 3:
                self.action_saved_models()
                continue
            slot = "embedding" if choice == 1 else "answer"
            rows = models[slot]
            current = p[slot]["provider"]
            print(f"\nRegistered {slot} providers:")
            for index, row in enumerate(rows, 1):
                marker = "  <- current" if row["provider"] == current else ""
                print(f"  {index}. {row['provider']:12} {row['label']}{marker}")
                print(f"                 key: {self.provider_key_status(rows, row['provider'])}")
            pick = choose_number(len(rows), default=next(
                (i for i, r in enumerate(rows, 1) if r["provider"] == current), 1))
            if pick == 0:
                continue
            row = rows[pick - 1]
            self.key_flow(slot, row)
            self.select_model(slot, row)

    def action_saved_models(self) -> None:
        saved = {p: ms for p, ms in (self.state.get("custom_models") or {}).items() if ms}
        if not saved:
            print("\n[saved] no custom model IDs saved yet — one is recorded each time "
                  "you type a model ID in the selection menus.")
            return
        print(f"\nSaved custom model IDs (categorized by provider, {STATE_PATH.name})")
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
                    remaining = [m for m in self.state["custom_models"][provider] if m != model]
                    if remaining:
                        self.state["custom_models"][provider] = remaining
                    else:
                        self.state["custom_models"].pop(provider, None)
                    save_front_state(self.state)
                    print(f"[saved] removed {provider}/{model}")
                return
            print("  enter a number from the list, or 0")

    # -- 3 API keys ---------------------------------------------------------------

    def action_keys(self) -> None:
        while True:
            rows = self.keys()
            print("\nAPI keys on the service — masked to 8 characters, never in full")
            for index, row in enumerate(rows, 1):
                mark = f"set ({row['masked']})" if row["status"] == "set" else "—"
                print(f"  {index:>2}. {row['env']:22} {mark}")
                print(f"      {row['description']}")
            print("   0. back")
            choice = choose_number(len(rows), default=0)
            if choice == 0:
                return
            row = rows[choice - 1]
            if row["status"] == "set":
                answer = prompt(f"{row['env']} is set ({row['masked']}) — "
                                "Enter=keep, c=change, d=remove: ").lower()
                if answer in ("", "k", "keep"):
                    continue
                if answer == "d":
                    persist = confirm("  also remove it from the service's raglab/.env?")
                    status, body = self.api.delete(
                        f"/keys/{row['env']}", params={"persist": "true" if persist else None})
                    print(f"[front] HTTP {status}: {detail_of(body)}"
                          if status != 200 else
                          f"[front] {row['env']} dropped from the service"
                          + (" and its .env" if body.get("removed_from_env_file") else "") + ".")
                    continue
                if answer not in ("c", "change"):
                    continue
            value = paste_key()
            if value is None:
                print("  cancelled — nothing sent.")
                continue
            persist = confirm("  also persist it to the service's raglab/.env?")
            status, body = self.api.post("/keys", payload={
                "key_env": row["env"], "value": value, "persist": persist})
            if status != 200:
                print(f"  [front] HTTP {status}: {detail_of(body)}")
                continue
            print(f"  service now has {body['env']} set ({body['masked']})"
                  + (" — written to the service's .env too" if body.get("persisted") else ""))

    # -- 4 inspect -----------------------------------------------------------------

    def action_inspect(self) -> None:
        raw = prompt("how many chunks to print in full? [3, 0 = none]: ")
        limit = int(raw) if raw.isdigit() else 3
        status, body = self.api.get("/inspect", params={"limit": limit})
        if status != 200:
            print(f"[front] HTTP {status}: {detail_of(body)}")
            return
        print("=" * 78)
        print("INSPECT — chunking preview (no model calls)")
        print("=" * 78)
        for doc in body["documents"]:
            print(f"  {doc['name']} — {doc['language']}, {doc['chars']} chars")
        c = body["chunking"]
        print(f"chunking: {c['mode']}"
              + (f" ({c['size']}/{c['overlap']} tokens)" if c["mode"] == "size" else ""))
        s = body["chunks"]
        print(f"chunks  : {s['count']} | tokens total={s['tokens_total']} "
              f"min={s['tokens_min']} median={s['tokens_median']} mean={s['tokens_mean']} "
              f"max={s['tokens_max']}")
        print("token histogram:")
        for row in s["histogram"]:
            print(f"  {row['bucket']:>9}: {'#' * row['chunks']}")
        for chunk in body["sample"]:
            print("-" * 78)
            print(f"chunk #{chunk['index']:03d} | source={chunk['source']} | "
                  f"language={chunk['language']} | tokens={chunk['tokens']} | "
                  f"section={chunk['section']}")
            print(f"heading: {chunk['heading'] or '(none)'}")
            print(chunk["text"])
        print("=" * 78)

    # -- 5 search the chunks ----------------------------------------------------------

    def action_search_chunks(self) -> None:
        needle = prompt("text to find (a phrase, or the full rejected quote): ")
        if not needle:
            return
        status, body = self.api.post("/chunks/search", payload={"text": needle})
        if status != 200:
            print(f"[front] HTTP {status}: {detail_of(body)}")
            return
        print(f"[search] scanned {body['description']}")
        if body["full"]:
            print(f"[search] the FULL text is inside {len(body['full'])} chunk(s) — a "
                  "verbatim quote of it can pass the citation gate:")
            for row in body["full"]:
                print(f"  - {row['source']}::chunk_{row['chunk_index']:04d} "
                      f"| heading={row['heading'] or '(none)'}")
            if confirm("print the full text of the first hit?"):
                print("-" * 78)
                print(body.get("first_full_text", ""))
            return
        head, tail = body["head"], body["tail"]
        head_ids = {(r["source"], r["chunk_index"]) for r in head}
        shared = any((r["source"], r["chunk_index"]) in head_ids for r in tail)
        if head and tail and not shared:
            print("[search] the text CROSSES chunk boundaries — no single chunk contains "
                  "it, so a one-chunk verbatim quote of the whole text can never validate:")
            for row in head:
                print(f"  - starts in {row['source']}::chunk_{row['chunk_index']:04d}")
            for row in tail:
                print(f"  - ends in   {row['source']}::chunk_{row['chunk_index']:04d}")
            print("[search] fixes: a larger k or neighbor radius >= 1, or a different "
                  "chunking mode/size (service settings, menu 12).")
        elif shared or head or tail:
            print("[search] the text's edges sit in one chunk but the middle differs — "
                  "the wording pasted is not the corpus wording (a paraphrase, or "
                  "characters the lab's normalization does not fold).")
        else:
            print("[search] not found anywhere in the chunked corpus — the quote was "
                  "invented or differs beyond the lab's normalization.")

    # -- 6 embedding sanity --------------------------------------------------------

    def action_embed_test(self) -> None:
        status, body = self.api.post("/embeddings/sanity")
        if status != 200:
            print(f"[front] HTTP {status}: {detail_of(body)}")
            return
        print("=" * 72)
        print("[sanity] provider :", body["provider"])
        print("[sanity] model    :", body["model"])
        print("[sanity] dimension:", body["dimension"])
        print("[sanity] batch    :", body["batch_size"],
              "| cache entries:", body["cache_entries"],
              "| api calls:", body["api_calls"])
        for row in body["cosines"]:
            print(f"[sanity]  cosine({row['pair']}) = {row['cosine']:+.4f}")
        print("[sanity]", body["interpretation"])
        print("=" * 72)

    # -- 7 ingest --------------------------------------------------------------------

    def action_ingest(self, *, reset=None) -> int:
        health = self.health()
        if reset is None:
            count = health["index"]["count"]
            reset = confirm(f"the index holds {count} chunk(s) — rebuild from "
                            "scratch (reset + re-embed)?") if count else False
        status, body = self.api.post("/ingest", params={"reset": "true" if reset else None})
        if status != 200:
            print(f"[front] HTTP {status}: {detail_of(body)}")
            return 1
        print(f"[front] ingest accepted (reset={reset}) — polling /ingest/status…")
        return self.wait_ingest()

    def wait_ingest(self, timeout: float = 900.0) -> int:
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            status, job = self.api.get("/ingest/status")
            if status != 200 or not isinstance(job, dict):
                print(f"[front] lost /ingest/status (HTTP {status})")
                return 1
            if job.get("state") != last:
                print(f"[front] state={job.get('state')}"
                      + (f" stored={job.get('stored')}" if job.get("stored") else "")
                      + (f" error={job.get('error')}" if job.get("error") else ""))
                last = job.get("state")
            if job.get("state") != "running":
                return 0 if job.get("state") == "done" else 1
            time.sleep(INGEST_POLL_SECONDS)
        print("[front] ingest timed out — check GET /ingest/status later")
        return 1

    # -- 8 retrieval query -------------------------------------------------------------

    def action_query(self) -> None:
        question = prompt("\nquestion (blank to cancel): ")
        if not question:
            return
        profile = self.profile()
        default_k = profile["retrieval"]["top_k"]
        raw = prompt(f"k [{default_k}]: ")
        k = int(raw) if raw.isdigit() and 1 <= int(raw) <= 20 else default_k
        status, body = self.api.post("/search", payload={"question": question, "k": k})
        if status != 200:
            print(f"[front] HTTP {status}: {detail_of(body)}")
            return
        show_search(body)

    # -- 9 grounded answer / --ask -------------------------------------------------------

    def ask_once(self, question: str, *, k=None, show=False, query_lang=None,
                 mode=None) -> int:
        payload = {"question": question}
        if k:
            payload["k"] = k
        if show:
            payload["include_excerpts"] = True
        if query_lang:
            payload["query_lang"] = query_lang
        if mode:
            payload["mode"] = mode
        status, body = self.api.post("/answer", payload=payload)
        if status != 200:
            print(f"[front] HTTP {status}: {detail_of(body)}")
            return 1
        show_answer(body)
        if body.get("reason") == "invalid_output":
            print("[front] the citation gate rejected the reply (the failed check is in "
                  "the model's raw preview above, server-side).")
            if confirm("ask the model again? (hosted models are not bit-deterministic)"):
                return self.ask_once(question, k=k, show=show, query_lang=query_lang, mode=mode)
        if body.get("status") == "error" and confirm("provider error — ask again?"):
            return self.ask_once(question, k=k, show=show, query_lang=query_lang, mode=mode)
        return 0 if body.get("validation_ok", True) else 2

    def action_answer(self) -> None:
        question = prompt("\nquestion (blank to cancel): ")
        if question:
            self.ask_once(question)

    # -- 10 chat ---------------------------------------------------------------------------

    def action_chat(self) -> int:
        health = self.health()
        profile = health["profile"]
        k = profile["retrieval"]["top_k"]
        show = False
        query_lang = None
        mode = profile["retrieval"]["mode"]
        print("\nChat over the documents — every turn is a POST /answer. "
              "Ctrl-D or :quit to leave, :help for commands.")
        print(f"[front] answers: {profile['answer']['provider']}/{profile['answer']['model']} | "
              f"index: {health['index']['count']} chunk(s) | k={k} | mode={mode}")
        while True:
            try:
                line = input("\nfront chat> ").strip()
            except EOFError:
                print()
                return 0
            if not line:
                continue
            if line in {":quit", ":q", ":exit"}:
                return 0
            if line in {":help", ":h"}:
                print("commands: :k N | :show | :lang ar|fr|en|auto | :mode vector|rrf|blend"
                      " | :health | :quit   (anything else is asked via POST /answer)")
                continue
            if line.startswith(":k"):
                try:
                    k = max(1, min(20, int(line.split()[1])))
                except (IndexError, ValueError):
                    print("[front] usage: :k 5")
                    continue
                print(f"[front] k={k}")
                continue
            if line == ":show":
                show = not show
                print(f"[front] excerpts {'on' if show else 'off'}")
                continue
            if line.startswith(":lang"):
                value = line.split()[-1].lower() if len(line.split()) > 1 else ""
                if value == "auto":
                    query_lang = None
                elif value in {"ar", "fr", "en"}:
                    query_lang = value
                else:
                    print("[front] :lang takes ar, fr, en or auto")
                    continue
                print(f"[front] claims language {value}")
                continue
            if line.startswith(":mode"):
                value = line.split()[-1].lower() if len(line.split()) > 1 else ""
                if value in {"vector", "rrf", "blend"}:
                    mode = value
                    print(f"[front] retrieval mode {mode}")
                else:
                    print("[front] :mode takes vector, rrf or blend")
                continue
            if line == ":health":
                print(json.dumps(health, ensure_ascii=False, indent=2))
                continue
            try:
                self.ask_once(line, k=k, show=show, query_lang=query_lang, mode=mode)
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                print(f"[front] request failed: {exc}")

    # -- 11 evaluate -----------------------------------------------------------------------

    def action_evaluate(self) -> None:
        print("\nWhich question set? (loaded on the service host)")
        for index, name in enumerate(QUESTION_SETS, 1):
            print(f"  {index}. {name}")
        print(f"  {len(QUESTION_SETS) + 1}. an absolute path on the service host")
        choice = choose_number(len(QUESTION_SETS) + 1, default=1)
        if choice == 0:
            return
        if choice <= len(QUESTION_SETS):
            questions = QUESTION_SETS[choice - 1]
        else:
            questions = prompt("path to the question set JSON: ")
            if not questions:
                return
        status, body = self.api.post("/evaluate", payload={"questions": questions},
                                     timeout=LONG_TIMEOUT)
        if status != 200:
            print(f"[front] HTTP {status}: {detail_of(body)}")
            return
        metrics = body["metrics"]
        overall = metrics.get("overall") or {}
        print("=" * 78)
        print(f"EVALUATION — {questions} | n={overall.get('n')}")
        print(f"hit@1/3/5: {overall.get('hit@1'):.2f} / {overall.get('hit@3'):.2f} / "
              f"{overall.get('hit@5'):.2f}" if overall.get("n") else "no evaluable questions")
        for section in ("by_category", "by_language"):
            for name, row in (metrics.get(section) or {}).items():
                if row.get("n"):
                    print(f"  {section[3:]} {name:12} n={row['n']:<3} "
                          f"hit@1={row['hit@1']:.2f} hit@5={row['hit@5']:.2f}")
        misses = [q for q in body["questions"]
                  if not q.get("is_out_of_scope") and not q.get("hit_at_1")]
        if misses:
            print(f"top-1 misses: {', '.join(q['id'] for q in misses)}")
        print(f"[front] full run saved on the service: {body['saved_to']}")
        print(f"[front] {body['note']}")
        print("=" * 78)

    # -- 12 service settings ------------------------------------------------------------------

    def action_settings(self) -> None:
        while True:
            profile = self.profile()
            chunking, retrieval = profile["chunking"], profile["retrieval"]
            dirs = profile.get("data_dirs")
            print("\nService settings (applied via POST /profile)")
            print(f"  1. chunking mode      : {chunking['mode']}  (size / restructure / manual)")
            print(f"  2. chunk size tokens  : {chunking['size']}  (size mode)")
            print(f"  3. chunk overlap      : {chunking['overlap']}  (size mode)")
            print(f"  4. retrieval top-k    : {retrieval['top_k']}")
            print(f"  5. retrieval mode     : {retrieval['mode']}  (vector / rrf / blend)")
            print(f"  6. language filter    : {retrieval['lang_filter'] or 'none'}  (none / ar / fr / en)")
            print(f"  7. neighbor radius    : {retrieval['neighbor_radius']}  (0 / 1 / 2)")
            print(f"  8. corpus directories : {', '.join(dirs) if dirs else '(service default)'}  (paths on the service host)")
            print("  0. back")
            choice = choose_number(8, default=0)
            if choice == 0:
                return
            if choice == 1:
                value = prompt("chunking mode (size / restructure / manual): ").lower()
                if value in {"size", "restructure", "manual"}:
                    self.switch_profile({"chunking": {"mode": value}})
            elif choice in (2, 3):
                raw = prompt("value: ")
                if raw.isdigit() and int(raw) > 0:
                    self.switch_profile({"chunking": {"size" if choice == 2 else "overlap": int(raw)}})
                else:
                    print("  enter a positive integer")
            elif choice == 4:
                raw = prompt("top-k (1-20): ")
                if raw.isdigit() and 1 <= int(raw) <= 20:
                    self.switch_profile({"retrieval": {"top_k": int(raw)}})
                else:
                    print("  enter a number 1-20")
            elif choice == 5:
                value = prompt("retrieval mode (vector / rrf / blend): ").lower()
                if value in {"vector", "rrf", "blend"}:
                    self.switch_profile({"retrieval": {"mode": value}})
            elif choice == 6:
                value = prompt("language filter (none / ar / fr / en): ").lower()
                self.switch_profile({"retrieval": {"lang_filter": value if value in {"ar", "fr", "en"} else None}})
            elif choice == 7:
                raw = prompt("neighbor radius (0 / 1 / 2): ")
                if raw in {"0", "1", "2"}:
                    self.switch_profile({"retrieval": {"neighbor_radius": int(raw)}})
            elif choice == 8:
                raw = prompt("comma-separated directories on the service host, or 'default': ")
                if raw.lower() in ("", "default"):
                    self.switch_profile({"data_dirs": []})
                else:
                    self.switch_profile({"data_dirs": [d.strip() for d in raw.split(",") if d.strip()]})

    # -- 13 diagnostics -------------------------------------------------------------------------

    def action_diagnostics(self) -> None:
        while True:
            print("\nDiagnostics (run on the service)")
            print("  1. offline harness50 A/B — BM25 chunking comparison, NO provider calls (~10-30s)")
            print("  2. xKiro provider catalog — read-only listing, NO inference (needs the key)")
            print("  0. back")
            choice = choose_number(2, default=0)
            if choice == 1:
                print("[front] POST /diagnostics/harness50 — re-chunks the corpus twice, "
                      "this takes a while…")
                status, body = self.api.post("/diagnostics/harness50", timeout=LONG_TIMEOUT)
                if status != 200:
                    print(f"[front] HTTP {status}: {detail_of(body)}")
                    continue
                print(f"[front] harness50 exit code {body['exit_code']}"
                      + (f" — report: {body['report']}" if body.get("report") else ""))
                for line in body.get("output_tail") or []:
                    print("  " + line)
            elif choice == 2:
                status, body = self.api.post("/diagnostics/catalog", timeout=LONG_TIMEOUT)
                if status != 200:
                    print(f"[front] HTTP {status}: {detail_of(body)}")
                    continue
                for provider in body.get("providers", []):
                    print(f"[front] {provider['provider']}: {provider['status']}"
                          + (f" ({provider.get('advertised_model_count')} models)"
                             if provider.get("advertised_model_count") else ""))
            elif choice == 0:
                return

    # -- menu loop -------------------------------------------------------------------------------

    MENU = [
        ("1", "status / doctor", "action_status"),
        ("2", "providers & models (switch embedding / answer profiles)", "action_providers"),
        ("3", "API keys (keep / change / add / remove — on the service)", "action_keys"),
        ("4", "inspect corpus (chunking preview — no model calls)", "action_inspect"),
        ("5", "search the chunks for a text/quote (no model calls)", "action_search_chunks"),
        ("6", "embedding sanity check (one batched embedding call)", "action_embed_test"),
        ("7", "ingest / rebuild the index for this profile", "action_ingest"),
        ("8", "retrieval query (top-k hits, no chat model)", "action_query"),
        ("9", "grounded answer (one question, cited)", "action_answer"),
        ("10", "chat over the documents", "action_chat"),
        ("11", "evaluate a question set", "action_evaluate"),
        ("12", "service settings (chunking, retrieval, corpus)", "action_settings"),
        ("13", "diagnostics (offline harness50, xKiro catalog)", "action_diagnostics"),
    ]

    def menu_loop(self) -> None:
        while True:
            try:
                self.banner()
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, RuntimeError) as exc:
                print(f"[front] lost the service: {exc}")
                return
            for key, label, _ in self.MENU:
                print(f"  {key:>2}  {label}")
            print("   0  quit")
            try:
                choice = prompt("> ")
            except EOFError:
                print("\n[front] input closed — goodbye.")
                return
            if choice in ("", "0", "q", "quit", "exit"):
                print("[front] goodbye.")
                return
            handler = next((name for key, _, name in self.MENU if key == choice), None)
            if handler is None:
                print("[front] unknown option — pick a number from the list.")
                continue
            try:
                getattr(self, handler)()
            except KeyboardInterrupt:
                print("\n[front] interrupted — back to the menu.")
            except EOFError:
                print("\n[front] input closed mid-action — back to the menu.")
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                print(f"[front] request failed: {exc}")


# ---------------------------------------------------------------------------
# Rendering for /answer and /search (shared by one-shot commands and the console)
# ---------------------------------------------------------------------------

def show_answer(body) -> None:
    if not isinstance(body, dict):
        print(body)
        return
    print(body.get("answer", ""))
    quotes = {}
    for claim in body.get("claims") or []:
        for evidence in claim.get("evidence") or []:
            quotes.setdefault(evidence.get("source_id"), []).append(evidence.get("quote"))
    for source in body.get("sources") or []:
        used = quotes.get(source.get("source_id")) or []
        if used or source.get("text"):
            print(f"\n[{source['source_id']}] {source.get('document')} — {source.get('chunk_id')}")
            if source.get("text"):
                print(source["text"])
            else:
                print("\n".join(f'    "{quote}"' for quote in used))
    print(f"\n[front] status={body.get('status')}/{body.get('reason')}"
          f" · model={body.get('model')} · {body.get('seconds', 0)}s"
          f" · retrieved={body.get('retrieved', 0)} · cached={body.get('cached', False)}"
          + (f" · failed check: {body.get('error')}" if body.get("error") else ""))


def show_search(body) -> None:
    if not isinstance(body, dict):
        print(body)
        return
    print(f"[front] k={body.get('k')} · mode={body.get('mode')} · "
          f"embedder={body.get('embedder', {}).get('provider')}/{body.get('embedder', {}).get('model')} · "
          f"language={body.get('language')}")
    for hit in body.get("hits") or []:
        print("-" * 78)
        print(f"rank       : {hit.get('rank')}")
        if hit.get("similarity") is not None:
            print(f"similarity : {hit['similarity']:+.4f}  (cosine)")
        print(f"source     : {hit.get('source')} | chunk {hit.get('chunk_index')}")
        print(f"heading    : {hit.get('heading') or '(none)'}")
        print("text:")
        print(hit.get("text", ""))
    if not body.get("hits"):
        print("[front] no hits")


# ---------------------------------------------------------------------------
# Smoke suite (state-aware; --smoke and CI) — kept from the previous front
# ---------------------------------------------------------------------------

class Suite:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name: str, cond, detail: str = "") -> None:
        print(("PASS  " if cond else "FAIL  ") + name + (f"  ({detail})" if detail else ""))
        if cond:
            self.passed += 1
        else:
            self.failed += 1

    def note(self, text: str) -> None:
        print("      " + text)


def run_suite(api: Api, *, spend: bool = True) -> tuple[int, int]:
    """Exercise every endpoint; returns (passed, failed).

    State-aware: expectations for /search, /answer and /ingest come FROM
    /health and /models, so a keyless or index-less service is tested just as
    completely as a ready one — its refusal paths ARE its correct behavior.
    """
    suite = Suite()
    print("=" * 78)
    print(f"RAGLab local front — smoke suite against {api.base_url}")
    print("=" * 78)

    status, root = api.get("/")
    suite.check("GET / answers", status == 200 and isinstance(root, dict)
                and root.get("service") == "raglab", f"status={status}")

    status, health = api.get("/health")
    ok = (status == 200 and isinstance(health, dict)
          and health.get("status") == "ok"
          and isinstance(health.get("profile", {}).get("embedding"), dict)
          and isinstance(health.get("index", {}).get("count"), int)
          and isinstance(health.get("keys"), dict))
    suite.check("GET /health reports profile, index and key presence", ok, f"status={status}")
    if ok:
        blob = json.dumps(health, ensure_ascii=False)
        suite.check("GET /health never echoes key values",
                    all(value in ("set", "missing") for value in health["keys"].values())
                    and "nvapi-" not in blob and "sk-" not in blob)

    status, models = api.get("/models")
    answer_rows = models.get("answer", []) if isinstance(models, dict) else []
    providers = {row.get("provider") for row in answer_rows}
    kira_models = {m for row in answer_rows if row.get("provider") == "kira"
                   for m in (row.get("models") or [])}
    suite.check("GET /models lists the registered providers",
                status == 200 and {"xkiro", "nvidia", "google", "kira"} <= providers
                and "glm-5.3-free" in kira_models,
                f"answer providers={sorted(p for p in providers if p)}")

    status, profile = api.get("/profile")
    suite.check("GET /profile reports collection and switching state",
                status == 200 and str(profile.get("collection", "")).startswith("raglab_app_")
                and "switching" in profile, f"status={status}")

    status, body = api.post("/profile", payload={})
    if status == 403 and reason_of(body) == "profile_switching_disabled":
        suite.check("POST /profile refuses switching when disabled (403)", True)
    elif status == 200:
        suite.check("POST /profile accepts a no-op switch (switching enabled)", True)
    else:
        suite.check("POST /profile behaves (403 disabled or 200 no-op)", False,
                    f"status={status} reason={reason_of(body)}")

    status, body = api.post("/answer", payload={"question": "hello"})
    suite.check("POST /answer answers greetings locally (no model call)",
                status == 200 and body.get("status") == "greeting"
                and body.get("inference_performed") is False, f"status={status}")

    status, _ = api.post("/answer", payload={"question": ""})
    suite.check("POST /answer rejects an empty question (422)", status == 422)
    status, _ = api.post("/answer", payload={"question": "hi", "k": 99})
    suite.check("POST /answer rejects k out of range (422)", status == 422)
    status, _ = api.get("/no-such-endpoint")
    suite.check("unknown paths 404", status == 404)
    status, ingest_status = api.get("/ingest/status")
    suite.check("GET /ingest/status answers",
                status == 200 and isinstance(ingest_status, dict) and "state" in ingest_status,
                f"state={ingest_status.get('state') if isinstance(ingest_status, dict) else '?'}")

    # -- keys endpoints --------------------------------------------------------
    status, keys = api.get("/keys")
    key_rows = keys.get("keys", []) if isinstance(keys, dict) else []
    suite.check("GET /keys lists the known key env vars",
                status == 200 and {"NVIDIA_API_KEY", "XKIRO_API_KEY", "KIRA_API_KEY"}
                <= {row.get("env") for row in key_rows}, f"status={status}")
    status, body = api.post("/keys", payload={"key_env": "KIRA_API_KEY",
                                              "value": "YOUR_KIRA_API_KEY"})
    suite.check("POST /keys rejects placeholder values (400)",
                status == 400 and reason_of(body) == "placeholder_value",
                f"status={status} reason={reason_of(body)}")
    status, body = api.post("/keys", payload={"key_env": "NOT_A_KEY", "value": "whatever12345"})
    suite.check("POST /keys rejects unknown env names (400)",
                status == 400 and reason_of(body) == "unknown_key_env")
    kira_row = next((row for row in key_rows if row.get("env") == "KIRA_API_KEY"), {})
    if kira_row.get("status") == "missing":
        status, body = api.post("/keys", payload={"key_env": "KIRA_API_KEY",
                                                  "value": "kira-suite-temp-123456"})
        ok = (status == 200 and body.get("status") == "set"
              and body.get("masked") == "kira-sui…" and not body.get("persisted"))
        status2, body2 = api.delete("/keys/KIRA_API_KEY")
        suite.check("POST+DELETE /keys round-trip (never echoes the value)",
                    ok and status2 == 200 and body2.get("status") == "missing",
                    f"set={status} delete={status2}")
    else:
        suite.note("KIRA_API_KEY is configured on the service — round-trip skipped")

    # -- inspect + chunk search (no model calls) ----------------------------------
    status, body = api.get("/inspect", params={"limit": 1})
    ok = (status == 200 and body.get("documents")
          and body.get("chunks", {}).get("count", 0) > 0 and body.get("sample"))
    suite.check("GET /inspect previews the chunking (no model calls)", ok,
                f"status={status} chunks={body.get('chunks', {}).get('count') if isinstance(body, dict) else '?'}")
    if ok:
        first = body["sample"][0]
        status, body = api.post("/chunks/search", payload={"text": first["text"][:40]})
        suite.check("POST /chunks/search locates text inside one chunk",
                    status == 200 and body.get("full"),
                    f"status={status} full={len(body.get('full') or []) if isinstance(body, dict) else '?'}")

    # -- state-dependent block ------------------------------------------------------
    emb_provider = health["profile"]["embedding"]["provider"]
    ans_provider = health["profile"]["answer"]["provider"]
    emb_row = next((row for row in models.get("embedding", [])
                    if row.get("provider") == emb_provider), {})
    ans_row = next((row for row in answer_rows if row.get("provider") == ans_provider), {})
    emb_key_set = bool(emb_row.get("key_set", True))
    ans_key_set = bool(ans_row.get("key_set", True))
    index_count = health["index"]["count"]
    question = {"question": "What is Murabaha?"}

    if not emb_key_set:
        status, body = api.post("/search", payload=question)
        suite.check("POST /search names the missing embedding key (503)",
                    status == 503 and reason_of(body) == "missing_api_key",
                    f"status={status} reason={reason_of(body)}")
        status, body = api.post("/ingest")
        suite.check("POST /ingest names the missing embedding key (503)",
                    status == 503 and reason_of(body) == "missing_api_key")
        status, body = api.post("/embeddings/sanity")
        suite.check("POST /embeddings/sanity names the missing key (503)",
                    status == 503 and reason_of(body) == "missing_api_key")
        suite.note("the service is correctly refusing work it cannot do; "
                   "set the keys (menu 3) and run: python local_front.py --ingest")
    elif index_count == 0:
        status, body = api.post("/search", payload=question)
        suite.check("POST /search refuses an empty index (409)",
                    status == 409 and reason_of(body) == "empty_index")
        status, body = api.post("/answer", payload=question)
        suite.check("POST /answer refuses an empty index (409)",
                    status == 409 and reason_of(body) == "empty_index")
        suite.note("keys are set but the index is empty; build it: "
                   "python local_front.py --ingest")
    else:
        if not spend:
            suite.note("index ready — live /search and /answer skipped (--no-spend)")
        else:
            suite.note("index ready — making one live /search call (one embedding call)")
            status, body = api.post("/search", payload=question)
            hits = body.get("hits", []) if isinstance(body, dict) else []
            suite.check("POST /search returns ranked chunks",
                        status == 200 and hits and hits[0].get("text") and "rank" in hits[0],
                        f"status={status} hits={len(hits)}")
            if not ans_key_set:
                status, body = api.post("/answer", payload=question)
                suite.check("POST /answer names the missing answer key (503)",
                            status == 503 and reason_of(body) == "missing_api_key")
            else:
                suite.note("making one live /answer call (one model call)")
                status, body = api.post("/answer", payload=question)
                ok = (status == 200 and isinstance(body, dict)
                      and body.get("status") in {"answered", "refused"}
                      and isinstance(body.get("answer"), str)
                      and isinstance(body.get("validation_ok"), bool))
                suite.check("POST /answer returns a grounded result or a safe refusal",
                            ok, f"status={status} status_field={body.get('status') if isinstance(body, dict) else '?'}")

    print("=" * 78)
    print(f"[front] {suite.passed} passed, {suite.failed} failed")
    return suite.passed, suite.failed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local_front.py",
        description="The app.py console experience over the RAGLab REST API "
                    "(pure HTTP client; the service must be running).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"service base URL (default {DEFAULT_BASE_URL}; "
                             "or env RAGLAB_SERVICE_URL)")
    parser.add_argument("--status", action="store_true",
                        help="doctor report over the API, no prompts, exit")
    parser.add_argument("--ask", metavar="QUESTION", dest="ask",
                        help="one grounded answer via POST /answer")
    parser.add_argument("--search", metavar="QUESTION", dest="search",
                        help="retrieval only, via POST /search")
    parser.add_argument("-k", "--top-k", type=int, default=None, dest="k")
    parser.add_argument("--mode", choices=["vector", "rrf", "blend"], default=None)
    parser.add_argument("--lang-filter", choices=["ar", "fr", "en"], default=None,
                        dest="lang_filter")
    parser.add_argument("--show-context", action="store_true", dest="show_context",
                        help="with --ask: also print the retrieved excerpts")
    parser.add_argument("--ingest", action="store_true",
                        help="build/rebuild the index (background job) and wait")
    parser.add_argument("--reset", action="store_true",
                        help="with --ingest: drop and rebuild the collection")
    parser.add_argument("--interactive", action="store_true",
                        help="straight into the chat REPL")
    parser.add_argument("--smoke", action="store_true",
                        help="run the state-aware endpoint smoke suite")
    parser.add_argument("--no-keycheck", action="store_true", dest="no_keycheck",
                        help="skip the startup keep/change/add key questions")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    api = Api(args.base_url, timeout=ANSWER_TIMEOUT)
    try:
        status, _ = api.get("/health")
        if status != 200:
            print(f"[front] /health answered HTTP {status} — not a RAGLab service?")
            return 2
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
        print(f"[front] cannot reach {api.base_url}: {exc}")
        print("[front] start the service first:  python -m uvicorn service:app "
              "--host 0.0.0.0 --port 8000   (or: docker compose up)")
        return 2

    console = Console(api, load_front_state())
    try:
        if args.status:
            console.action_status()
            return 0
        if args.smoke:
            passed, failed = run_suite(api, spend=not args.status)
            return 0 if failed == 0 else 1
        if not args.no_keycheck and not (args.ingest and args.reset):
            console.startup_key_check()
        if args.ingest:
            return console.action_ingest(reset=args.reset)
        if args.ask:
            return console.ask_once(args.ask, k=args.k, show=args.show_context,
                                    mode=args.mode)
        if args.search:
            status, body = api.post("/search", payload={
                "question": args.search, "k": args.k, "mode": args.mode,
                "lang_filter": args.lang_filter})
            if status != 200:
                print(f"[front] HTTP {status}: {detail_of(body)}")
                return 1
            show_search(body)
            return 0
        if args.interactive:
            return console.action_chat()
        console.menu_loop()
        return 0
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
        print(f"[front] request failed: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
