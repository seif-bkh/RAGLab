#!/usr/bin/env python3
"""local_front.py — talk to the RAGLab service over its REST API, to test it.

A pure HTTP client: it imports NOTHING from the lab (no profiles, no service
code) — every capability below goes through the same JSON endpoints another
microservice would call. That is the point: it tests the HTTP boundary, not
the functions behind it.

Usage (the service must be running — see raglab/SERVICE.md):

    python local_front.py                     smoke suite over every endpoint
    python local_front.py --ingest [--reset]  build/rebuild the index, wait for the job
    python local_front.py --ask "What is Murabaha?"
    python local_front.py --search "ما هي المرابحة؟" -k 5
    python local_front.py --interactive       small REPL over the API
    python local_front.py --no-spend          suite without live model calls
    python local_front.py --base-url http://localhost:8000

The smoke suite is state-aware, so it is truthful in every service state:
with no keys it EXPECTS 503 missing_api_key (and passes when the service
refuses correctly), with keys but no index it EXPECTS 409 empty_index, and
with a ready index it makes one real /search and one real /answer call
(skippable with --no-spend).

Exit codes: 0 = all checks passed / command succeeded; 1 = check failures;
2 = cannot reach the service.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE_URL = os.environ.get("RAGLAB_SERVICE_URL", "http://localhost:8000")
ANSWER_TIMEOUT = 180        # xKiro's first call includes the live price check
INGEST_POLL_SECONDS = 1.0


# ---------------------------------------------------------------------------
# Minimal HTTP layer (stdlib only — the front adds no dependency)
# ---------------------------------------------------------------------------

class Api:
    """GET/POST JSON against the service. Never raises on HTTP errors:
    returns (status, body) so callers can assert on the error contract."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, payload=None, params=None):
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
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
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

    def get(self, path: str):
        return self.request("GET", path)

    def post(self, path: str, payload=None, params=None):
        return self.request("POST", path, payload=payload, params=params)


# ---------------------------------------------------------------------------
# Check helpers (same style as tests_offline.py: one PASS/FAIL line each)
# ---------------------------------------------------------------------------

class Suite:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name: str, cond, detail: str = "") -> None:
        line = ("PASS  " if cond else "FAIL  ") + name + (f"  ({detail})" if detail else "")
        print(line)
        if cond:
            self.passed += 1
        else:
            self.failed += 1

    def note(self, text: str) -> None:
        print("      " + text)


def _reason(body) -> str:
    """The service's error reason, wherever FastAPI put it."""
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict) and detail.get("reason"):
            return str(detail["reason"])
        if isinstance(detail, str):
            return detail
    return ""


# ---------------------------------------------------------------------------
# The smoke suite — state-aware, truthful in every service state
# ---------------------------------------------------------------------------

def run_suite(api: Api, *, spend: bool = True) -> tuple[int, int]:
    """Exercise every endpoint; returns (passed, failed).

    State-aware: the expectations for /search, /answer and /ingest come FROM
    /health and /models (keys present? index built?), so a keyless or
    index-less service is tested just as completely as a ready one — its
    refusal paths ARE its correct behavior.
    """
    suite = Suite()
    print("=" * 78)
    print(f"RAGLab local front — smoke suite against {api.base_url}")
    print("=" * 78)

    # -- informational endpoints ---------------------------------------------

    status, root = api.get("/")
    suite.check("GET / answers", status == 200 and isinstance(root, dict)
                and root.get("service") == "raglab", f"status={status}")

    status, health = api.get("/health")
    ok = (status == 200 and isinstance(health, dict)
          and health.get("status") == "ok"
          and isinstance(health.get("profile", {}).get("embedding"), dict)
          and isinstance(health.get("profile", {}).get("answer"), dict)
          and isinstance(health.get("index", {}).get("count"), int)
          and isinstance(health.get("keys"), dict))
    suite.check("GET /health reports profile, index and key presence", ok,
                f"status={status}")
    if ok:
        # Key VALUES must never appear anywhere — presence strings only.
        blob = json.dumps(health, ensure_ascii=False)
        suite.check("GET /health never echoes key values",
                    all(value in ("set", "missing") for value in health["keys"].values())
                    and "nvapi-" not in blob and "sk-" not in blob,
                    f"keys={ {k: v for k, v in health['keys'].items()} }")

    status, models = api.get("/models")
    answer_rows = models.get("answer", []) if isinstance(models, dict) else []
    providers = {row.get("provider") for row in answer_rows}
    kira_models = {model for row in answer_rows if row.get("provider") == "kira"
                   for model in (row.get("models") or [])}
    suite.check("GET /models lists the registered providers",
                status == 200 and {"xkiro", "nvidia", "google", "kira"} <= providers
                and "glm-5.3-free" in kira_models,
                f"answer providers={sorted(p for p in providers if p)}")

    status, profile = api.get("/profile")
    suite.check("GET /profile reports collection and switching state",
                status == 200 and isinstance(profile, dict)
                and str(profile.get("collection", "")).startswith("raglab_app_")
                and "switching" in profile, f"status={status}")

    status, body = api.post("/profile", payload={})
    reason = _reason(body)
    if status == 403 and reason == "profile_switching_disabled":
        suite.check("POST /profile refuses switching by default (403)", True)
    elif status == 200:
        suite.check("POST /profile accepts a no-op switch (switching enabled)", True)
    else:
        suite.check("POST /profile behaves (403 default or 200 no-op)", False,
                    f"status={status} reason={reason}")

    # -- greetings: work with no keys, no index, zero calls -------------------

    status, body = api.post("/answer", payload={"question": "hello"})
    suite.check("POST /answer answers greetings locally (no model call)",
                status == 200 and body.get("status") == "greeting"
                and body.get("inference_performed") is False
                and isinstance(body.get("answer"), str) and body["answer"],
                f"status={status}")

    # -- request validation ------------------------------------------------------

    status, _ = api.post("/answer", payload={"question": ""})
    suite.check("POST /answer rejects an empty question (422)", status == 422,
                f"status={status}")
    status, _ = api.post("/answer", payload={"question": "hi", "k": 99})
    suite.check("POST /answer rejects k out of range (422)", status == 422,
                f"status={status}")
    status, _ = api.get("/no-such-endpoint")
    suite.check("unknown paths 404", status == 404, f"status={status}")

    status, ingest_status = api.get("/ingest/status")
    suite.check("GET /ingest/status answers", status == 200
                and isinstance(ingest_status, dict) and "state" in ingest_status,
                f"status={status} state={ingest_status.get('state') if isinstance(ingest_status, dict) else '?'}")

    # -- state-dependent block: keys and index ----------------------------------

    # Which keys does the SELECTED profile need, and are they set? /models says.
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
                    status == 503 and _reason(body) == "missing_api_key",
                    f"status={status} reason={_reason(body)}")
        status, body = api.post("/ingest")
        suite.check("POST /ingest names the missing embedding key (503)",
                    status == 503 and _reason(body) == "missing_api_key",
                    f"status={status} reason={_reason(body)}")
        suite.note("the service is correctly refusing work it cannot do; "
                   "set the keys and run: python local_front.py --ingest")
    elif index_count == 0:
        status, body = api.post("/search", payload=question)
        suite.check("POST /search refuses an empty index (409)",
                    status == 409 and _reason(body) == "empty_index",
                    f"status={status} reason={_reason(body)}")
        status, body = api.post("/answer", payload=question)
        suite.check("POST /answer refuses an empty index (409)",
                    status == 409 and _reason(body) == "empty_index",
                    f"status={status} reason={_reason(body)}")
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
                        status == 200 and hits and hits[0].get("text")
                        and "rank" in hits[0],
                        f"status={status} hits={len(hits)}")
            if not ans_key_set:
                status, body = api.post("/answer", payload=question)
                suite.check("POST /answer names the missing answer key (503)",
                            status == 503 and _reason(body) == "missing_api_key",
                            f"status={status} reason={_reason(body)}")
            else:
                suite.note("making one live /answer call (one model call; the "
                           "first xKiro call also does the live price check)")
                status, body = api.post("/answer", payload=question,
                                        params=None)
                ok = (status == 200 and isinstance(body, dict)
                      and body.get("status") in {"answered", "refused"}
                      and isinstance(body.get("answer"), str)
                      and isinstance(body.get("claims"), list)
                      and isinstance(body.get("sources"), list)
                      and isinstance(body.get("validation_ok"), bool))
                suite.check("POST /answer returns a grounded result or a safe refusal",
                            ok, f"status={status} status_field={body.get('status') if isinstance(body, dict) else '?'}"
                                 f" reason={_reason(body)}")
                if isinstance(body, dict) and body.get("status") == "answered":
                    suite.note("answered with "
                               f"{len(body.get('claims') or [])} cited claim(s)")

    print("=" * 78)
    print(f"[front] {suite.passed} passed, {suite.failed} failed")
    return suite.passed, suite.failed


# ---------------------------------------------------------------------------
# One-shot commands
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
    line = (f"\n[front] status={body.get('status')}/{body.get('reason')}"
            f" · model={body.get('model')} · {body.get('seconds', 0)}s"
            f" · retrieved={body.get('retrieved', 0)} · cached={body.get('cached', False)}")
    print(line)


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
        print(f"text:")
        print(hit.get("text", ""))
    if not body.get("hits"):
        print("[front] no hits")


def cmd_ask(api: Api, question: str, k: int | None, include_excerpts: bool) -> int:
    payload = {"question": question}
    if k:
        payload["k"] = k
    if include_excerpts:
        payload["include_excerpts"] = True
    status, body = api.post("/answer", payload=payload)
    if status != 200:
        print(f"[front] HTTP {status}: {_reason(body) or body}")
        return 1
    show_answer(body)
    return 0 if body.get("validation_ok", True) else 2


def cmd_search(api: Api, question: str, k: int | None, mode: str | None,
               lang_filter: str | None) -> int:
    payload = {"question": question}
    if k:
        payload["k"] = k
    if mode:
        payload["mode"] = mode
    if lang_filter:
        payload["lang_filter"] = lang_filter
    status, body = api.post("/search", payload=payload)
    if status != 200:
        print(f"[front] HTTP {status}: {_reason(body) or body}")
        return 1
    show_search(body)
    return 0


def cmd_ingest(api: Api, reset: bool, timeout: float) -> int:
    status, body = api.post("/ingest", params={"reset": "true" if reset else None})
    if status != 200:
        print(f"[front] HTTP {status}: {_reason(body) or body}")
        return 1
    print(f"[front] ingest accepted (reset={reset}) — polling status…")
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        status, job = api.get("/ingest/status")
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


HELP = """commands in the interactive front:
  anything else   ask it via POST /answer (a greeting is answered locally)
  :s QUERY        search chunks only (POST /search)
  :k N            excerpts per question (1-20, default: the service profile)
  :show           also print the retrieved excerpts with answers
  :health         GET /health
  :models         GET /models
  :profile        GET /profile
  :ingest [reset] build/rebuild the index and wait (POST /ingest)
  :help           this list
  :quit           leave (Ctrl-D works too)
"""


def cmd_interactive(api: Api) -> int:
    print(f"[front] talking to {api.base_url} — :help lists commands")
    k = None
    show = False
    while True:
        try:
            line = input("\nfront> ").strip()
        except EOFError:
            print()
            return 0
        if not line:
            continue
        if line in {":quit", ":q", ":exit"}:
            return 0
        if line in {":help", ":h"}:
            print(HELP)
            continue
        if line == ":health":
            print(json.dumps(api.get("/health")[1], ensure_ascii=False, indent=2))
            continue
        if line == ":models":
            print(json.dumps(api.get("/models")[1], ensure_ascii=False, indent=2))
            continue
        if line == ":profile":
            print(json.dumps(api.get("/profile")[1], ensure_ascii=False, indent=2))
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
        if line.startswith(":ingest"):
            cmd_ingest(api, reset="reset" in line.split(), timeout=3600)
            continue
        if line.startswith(":s "):
            cmd_search(api, line[3:].strip(), k, None, None)
            continue
        cmd_ask(api, line, k, include_excerpts=show)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local_front.py",
        description="Test the RAGLab service by talking to its REST API "
                    "(pure HTTP client; the service must be running).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"service base URL (default {DEFAULT_BASE_URL}; "
                             "or env RAGLAB_SERVICE_URL)")
    parser.add_argument("--ask", metavar="QUESTION", dest="ask",
                        help="one grounded answer via POST /answer")
    parser.add_argument("--search", metavar="QUESTION", dest="search",
                        help="retrieval only, via POST /search")
    parser.add_argument("-k", "--top-k", type=int, default=None, dest="k",
                        help="excerpts per question (1-20)")
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
                        help="small REPL over the API")
    parser.add_argument("--no-spend", action="store_true", dest="no_spend",
                        help="smoke suite: skip live model calls")
    parser.add_argument("--timeout", type=float, default=INGEST_TIMEOUT_DEFAULT,
                        help="seconds to wait for an ingest job (default 900)")
    return parser


INGEST_TIMEOUT_DEFAULT = 900.0


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

    try:
        if args.ingest:
            return cmd_ingest(api, reset=args.reset, timeout=args.timeout)
        if args.ask:
            return cmd_ask(api, args.ask, args.k, args.show_context)
        if args.search:
            return cmd_search(api, args.search, args.k, args.mode, args.lang_filter)
        if args.interactive:
            return cmd_interactive(api)
        passed, failed = run_suite(api, spend=not args.no_spend)
        return 0 if failed == 0 else 1
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
        print(f"[front] request failed: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
