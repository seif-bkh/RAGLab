#!/usr/bin/env python3
"""service.py — RAGLab as a standalone HTTP microservice.

One service exposing the lab's runtime over REST, for a microservice
architecture where RAGLab is one deployable unit next to your other services.
It imports the SAME runtime the interactive console (app.py) uses —
profiles.py: provider/model registries, the lab-config copy, chat clients for
every provider, profile-scoped Chroma collections, ingest, the verbatim
citation gate, the greeting short-circuit — and adds NO console of its own:
no prompts, no menus, no app_state.json. Configuration is 12-factor:
environment variables at boot, API keys from the environment (never echoed).

Run (from raglab/, with requirements-service.txt installed):

    python -m uvicorn service:app --host 0.0.0.0 --port 8000
    # or: python service.py            # same, via the __main__ block
    # interactive API docs: http://localhost:8000/docs

Endpoints:

    GET  /health          liveness + profile + index + key presence (no secrets)
    GET  /models          registered models per provider
    GET  /profile         the active profile (embedding/answer/chunking/retrieval)
    POST /profile         switch provider/model/chunking/retrieval/corpus at runtime
                          (only when RAGLAB_ALLOW_PROFILE_SWITCH=1)
    GET  /keys            known API-key env vars, descriptions, masked presence
    POST /keys            set a key in the service process (optionally persist to .env)
    DELETE /keys/{env}    drop a key from the process (optionally from .env)
    POST /search          retrieval only: top-k chunks for a question
    POST /answer          grounded, cited answer (or refusal) for a question
    POST /ingest          build/rebuild this profile's index (background job)
    GET  /ingest/status   what the ingest job is doing / last did
    GET  /inspect         chunking preview over the corpus (no model calls)
    POST /chunks/search   is a quote inside ONE chunk? (citation-gate diagnostic)
    POST /embeddings/sanity  one batched embedding call + 3-language cosine report
    POST /evaluate        run a question set against the index (embeddings!)
    POST /diagnostics/harness50  offline BM25 chunking A/B (subprocess, ~10-20s)
    POST /diagnostics/catalog    read-only xKiro model catalog (no inference)

The console-parity surface (local_front.py) drives exactly these endpoints:
same menus as app.py, every action an HTTP call.

Policy: the supported benchmark pipeline stays pinned in pipeline_policy.py.
The default profile IS the supported pair (NVIDIA nemotron embeddings + xKiro
qwen/qwen3.8-max:free with its live free-price check). Alternative providers
and custom model IDs run with the same chunker/retrieval/citation machinery
on their OWN raglab_app_* collections — lab surfaces, no benchmark
attribution, exactly like the console.

Operational notes (read before deploying):
  * No authentication/authorization is built in. Put the service behind your
    own gateway/auth sidecar; do not expose it publicly as-is.
  * Single-replica by design: the index is a local ChromaDB directory and
    ingestion runs in one background thread. Scale reads with your gateway,
    not with replicas sharing nothing.
  * POST /ingest embeds every chunk (provider calls); the embedding cache
    (embeddings_cache_app_<provider>.json) makes re-ingests cheap.
  * Not production-ready for a banking service (see README.md) — same
    standing as the rest of this lab.

Environment (all optional; defaults = the supported pipeline pair):

    RAGLAB_EMBEDDING_PROVIDER, RAGLAB_EMBEDDING_MODEL
    RAGLAB_ANSWER_PROVIDER,   RAGLAB_ANSWER_MODEL
    RAGLAB_CHUNKING_MODE, RAGLAB_CHUNK_SIZE_TOKENS, RAGLAB_CHUNK_OVERLAP_TOKENS
    RAGLAB_TOP_K, RAGLAB_RETRIEVAL_MODE, RAGLAB_LANG_FILTER, RAGLAB_NEIGHBOR_RADIUS
    RAGLAB_DATA_DIRS               comma-separated corpus dirs (default: ../docs + data/)
    RAGLAB_ALLOW_PROFILE_SWITCH    1 to enable POST /profile (default 1 locally;
                                   docker-compose.yml pins 0 — enable consciously)
    RAGLAB_CORS_ORIGINS            comma-separated allowed origins (default *)
    plus the provider keys: NVIDIA_API_KEY, XKIRO_API_KEY, GOOGLE_API_KEY /
    GEMINI_API_KEY, KIRA_API_KEY, JINA_API_KEY, ... (see raglab/.env.example)
"""

from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import profiles
from nvidia_api import NvidiaAPIError, safe_error

SERVICE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Errors that carry an HTTP status
# ---------------------------------------------------------------------------

class ServiceError(Exception):
    """Raise anywhere in a handler path; converted to a JSON error response."""

    def __init__(self, status_code: int, reason: str, **detail: Any):
        super().__init__(reason)
        self.status_code = status_code
        self.payload = {"reason": reason, **detail}


def _redact(detail: dict) -> dict:
    """Never let a provider error echo a key into a response body."""
    return {key: (safe_error(value) if isinstance(value, str) else value)
            for key, value in detail.items()}


# ---------------------------------------------------------------------------
# Profile from environment (boot-time; 12-factor)
# ---------------------------------------------------------------------------

def profile_from_env() -> dict:
    """Build the service profile from RAGLAB_* env vars, validated loudly.

    Raises SystemExit with a plain-English message at boot (container start)
    rather than a stack trace on the fifth request.
    """
    state = profiles.default_state()
    env = os.environ.get

    def take(name: str, apply):
        value = env(name)
        if value not in (None, ""):
            apply(value.strip())

    take("RAGLAB_EMBEDDING_PROVIDER", lambda v: state["embedding"].__setitem__("provider", v.lower()))
    take("RAGLAB_EMBEDDING_MODEL", lambda v: state["embedding"].__setitem__("model", v))
    take("RAGLAB_ANSWER_PROVIDER", lambda v: state["answer"].__setitem__("provider", v.lower()))
    take("RAGLAB_ANSWER_MODEL", lambda v: state["answer"].__setitem__("model", v))
    take("RAGLAB_CHUNKING_MODE", lambda v: state["chunking"].__setitem__("mode", v.lower()))
    take("RAGLAB_CHUNK_SIZE_TOKENS", lambda v: state["chunking"].__setitem__("size", int(v)))
    take("RAGLAB_CHUNK_OVERLAP_TOKENS", lambda v: state["chunking"].__setitem__("overlap", int(v)))
    take("RAGLAB_TOP_K", lambda v: state["retrieval"].__setitem__("top_k", int(v)))
    take("RAGLAB_RETRIEVAL_MODE", lambda v: state["retrieval"].__setitem__("mode", v.lower()))
    take("RAGLAB_LANG_FILTER", lambda v: state["retrieval"].__setitem__("lang_filter", v.lower() or None))
    take("RAGLAB_NEIGHBOR_RADIUS", lambda v: state["retrieval"].__setitem__("neighbor_radius", int(v)))
    take("RAGLAB_DATA_DIRS", lambda v: state.__setitem__("data_dirs", [d.strip() for d in v.split(",") if d.strip()]))

    problems = []
    for slot, registry in (("embedding", profiles.EMBEDDING_PROVIDERS),
                           ("answer", profiles.ANSWER_PROVIDERS)):
        entry = state[slot]
        if entry["provider"] not in registry:
            problems.append(f"{slot} provider {entry['provider']!r} is not registered "
                            f"(registered: {', '.join(sorted(registry))})")
        elif not str(entry["model"]).strip():
            problems.append(f"{slot} model is empty")
    if state["retrieval"]["mode"] not in {"vector", "rrf", "blend"}:
        problems.append(f"RAGLAB_RETRIEVAL_MODE must be vector, rrf or blend "
                        f"(got {state['retrieval']['mode']!r})")
    if problems:
        raise SystemExit("[service] invalid profile from environment:\n  - "
                         + "\n  - ".join(problems))
    return state


# ---------------------------------------------------------------------------
# Runtime: lazily built, cached, thread-safe per profile
# ---------------------------------------------------------------------------

class Runtime:
    """Cached lab config + embedder + generator for the active profile.

    The embedder and (for xKiro) the price-checked generator involve network
    calls to construct, so both are built once per profile and reused. A
    profile switch drops the caches; the next request rebuilds them.
    """

    def __init__(self, profile: dict, *, generator=None, overrides: dict | None = None):
        self.lock = threading.RLock()
        self.profile = profile
        self.overrides = dict(overrides or {})
        self._local = None
        self._embedder = None
        self._generator = generator  # injectable for tests

    def local(self):
        with self.lock:
            if self._local is None:
                local = profiles.build_lab_config(self.profile)
                for key, value in self.overrides.items():
                    setattr(local, key, value)
                self._local = local
            return self._local

    def _require_key(self, slot: str):
        registry = (profiles.EMBEDDING_PROVIDERS if slot == "embedding"
                    else profiles.ANSWER_PROVIDERS)
        envs = registry[self.profile[slot]["provider"]].get("key_envs", ())
        if envs and not profiles.first_set_env(envs)[1]:
            raise ServiceError(
                503, "missing_api_key", slot=slot, key_env=envs[0],
                hint=f"set {envs[0]} in the service environment (see raglab/.env.example)")

    def embedder(self):
        with self.lock:
            if self._embedder is None:
                self._require_key("embedding")
                from embedder import build_embedder
                self._embedder = build_embedder(self.local())
            return self._embedder

    def generator(self):
        with self.lock:
            if self._generator is None:
                self._require_key("answer")
                self._generator = profiles.build_generator(self.local())
            return self._generator

    def collection(self):
        """The profile's collection, refusing the empty and the stale cases."""
        from store import get_collection
        local = self.local()
        collection = get_collection(local, reset=False)
        if not collection.count():
            raise ServiceError(409, "empty_index", collection=local.CHROMA_COLLECTION_NAME,
                               hint='POST /ingest to build this profile\'s index first')
        return collection

    def index_info(self):
        """(count, stored_chunk_fp) for the profile's collection, honoring any
        config overrides (tests redirect CHROMA_DIR; production does not)."""
        from store import _client
        local = self.local()
        try:
            collection = _client(local).get_collection(local.CHROMA_COLLECTION_NAME)
            count = collection.count()
        except Exception:                               # noqa: BLE001 — absent is fine
            return 0, None
        if not count:
            return 0, None
        metas = collection.get(include=["metadatas"], limit=1).get("metadatas") or [{}]
        return count, (metas[0] or {}).get("chunk_fp")

    def switch(self, profile: dict):
        with self.lock:
            self.profile = profile
            self._local = None
            self._embedder = None
            self._generator = None


class IngestJobs:
    """One ingest at a time, in a background thread, with a status endpoint.

    Ingestion embeds every chunk (minutes, provider calls); a synchronous
    endpoint would be a timeout trap. The job state is a plain dict — the
    service is single-process by design (see module docstring).
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.status = {"state": "idle", "started_at": None, "finished_at": None,
                       "collection": None, "stored": None, "error": None}

    def start(self, runtime: Runtime, *, reset: bool):
        with self.lock:
            if self.status["state"] == "running":
                raise ServiceError(409, "ingest_already_running")
            self.status = {"state": "running", "started_at": _now(), "finished_at": None,
                           "collection": profiles.collection_name(runtime.profile),
                           "stored": None, "error": None}
        thread = threading.Thread(target=self._run, args=(runtime, reset), daemon=True)
        thread.start()

    def _run(self, runtime: Runtime, reset: bool):
        try:
            runtime._require_key("embedding")
            embedder = runtime.embedder()
            collection = profiles.ingest(runtime.local(), embedder, runtime.profile,
                                         reset=reset)
            self.status.update(state="done", finished_at=_now(),
                               stored=collection.count())
        except SystemExit as exc:                       # embedder's loud missing-key exit
            self.status.update(state="error", finished_at=_now(), error=str(exc))
        except Exception as exc:                        # noqa: BLE001 — the job must report, not die
            self.status.update(state="error", finished_at=_now(),
                               error=safe_error(exc))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: Optional[int] = Field(None, ge=1, le=20)
    mode: Optional[str] = None   # vector | rrf | blend; validated below
    lang_filter: Optional[str] = None
    query_lang: Optional[str] = None


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: Optional[int] = Field(None, ge=1, le=20)
    include_excerpts: bool = False
    query_lang: Optional[str] = None    # ar | fr | en — write claims in this language
    mode: Optional[str] = None          # vector | rrf | blend
    lang_filter: Optional[str] = None   # ar | fr | en — restrict retrieval


class KeySetRequest(BaseModel):
    key_env: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=4096)
    persist: bool = Field(False,
                          help="also write it to raglab/.env on the service host "
                               "(like the console does); default: process env only")


class ChunkSearchRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class EvaluateRequest(BaseModel):
    questions: str = Field(
        "questions.json",
        description="a known set name (questions.json, questions_50.json, "
                    "questions_real.json) or an absolute path on the service host")
    top_k: Optional[int] = Field(None, ge=1, le=50)


class ProfileSwitchRequest(BaseModel):
    embedding: Optional[dict] = None   # {"provider": ..., "model": ...}
    answer: Optional[dict] = None
    chunking: Optional[dict] = None    # {"mode", "size", "overlap"}
    retrieval: Optional[dict] = None   # {"top_k", "mode", "lang_filter", "neighbor_radius"}
    data_dirs: Optional[list[str]] = None   # corpus dirs on the SERVICE host


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(profile: dict | None = None, *, generator=None,
               allow_profile_switch: bool | None = None,
               cors_origins: list[str] | None = None,
               config_overrides: dict | None = None) -> FastAPI:
    """Build the service app.

    `profile` defaults to the environment (profile_from_env). The remaining
    keyword arguments exist for tests and embedded deployments: an injected
    generator (no provider calls), a switch flag override, CORS origins, and
    lab-config overrides (e.g. redirecting CHROMA_DIR to a temp dir).
    """
    if profile is None:
        profile = profile_from_env()
    if allow_profile_switch is None:
        # Console parity by default for local runs; docker-compose pins "0".
        allow_profile_switch = os.environ.get("RAGLAB_ALLOW_PROFILE_SWITCH", "1") == "1"
    if cors_origins is None:
        raw = os.environ.get("RAGLAB_CORS_ORIGINS", "*")
        cors_origins = [origin.strip() for origin in raw.split(",") if origin.strip()]

    runtime = Runtime(profile, generator=generator, overrides=config_overrides)
    jobs = IngestJobs()
    app = FastAPI(
        title="RAGLab service",
        description=__doc__.split("Endpoints:")[0],
        version=SERVICE_VERSION,
    )
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins or ["*"],
                       allow_methods=["*"], allow_headers=["*"])

    def service_error_handler(request: Request, exc: ServiceError):
        return JSONResponseFromError(exc)

    def JSONResponseFromError(exc: ServiceError):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=exc.status_code,
                            content={"detail": _redact(exc.payload)})

    # FastAPI needs the handler registered for the custom class:
    app.add_exception_handler(ServiceError, service_error_handler)

    # -- informational ------------------------------------------------------

    @app.get("/")
    def root():
        return {"service": "raglab", "version": SERVICE_VERSION,
                "docs": "/docs", "health": "/health",
                "profile": "/profile", "models": "/models",
                "endpoints": ["POST /search", "POST /answer",
                              "POST /ingest", "GET /ingest/status"]}

    @app.get("/health")
    def health():
        """Liveness + configuration truth. Key values are never included —
        only which env var names are set, masked per the repo's standing rule."""
        local = runtime.local()
        count, stored_fp = runtime.index_info()
        index = {"collection": local.CHROMA_COLLECTION_NAME, "count": count}
        if stored_fp:
            import re as _re
            from chunker import tokenizer_identity
            stored_tok = _re.search(r"tok([^:]+)", stored_fp)
            index["chunk_fp"] = stored_fp[:16]
            index["tokenizer_match"] = (stored_tok.group(1) == tokenizer_identity()
                                        if stored_tok else None)
        keys = {}
        for slot, registry in (("embedding", profiles.EMBEDDING_PROVIDERS),
                               ("answer", profiles.ANSWER_PROVIDERS)):
            envs = registry[runtime.profile[slot]["provider"]].get("key_envs", ())
            for env_name in envs:
                keys[env_name] = "set" if profiles.first_set_env((env_name,))[1] else "missing"
        return {"status": "ok", "version": SERVICE_VERSION,
                "profile": {
                    "embedding": runtime.profile["embedding"],
                    "answer": runtime.profile["answer"],
                    "chunking": runtime.profile["chunking"],
                    "retrieval": runtime.profile["retrieval"],
                    "pipeline": profiles.pipeline_marker(runtime.profile)},
                "index": index, "keys": keys, "ingest": jobs.status}

    @app.get("/models")
    def models():
        def slot_models(registry):
            return [{"provider": name,
                     "label": info["label"],
                     "key_envs": list(info.get("key_envs", ())),
                     "key_set": bool(profiles.first_set_env(info.get("key_envs", ()) or ("",))[1])
                     if info.get("key_envs") else True,
                     "models": [model["id"] for model in info["models"]]}
                    for name, info in registry.items()]
        return {"embedding": slot_models(profiles.EMBEDDING_PROVIDERS),
                "answer": slot_models(profiles.ANSWER_PROVIDERS),
                "note": "custom model IDs may be supplied via POST /profile "
                        "(when switching is enabled) or the RAGLAB_* env vars"}

    @app.get("/profile")
    def get_profile():
        return {"profile": runtime.profile,
                "collection": profiles.collection_name(runtime.profile),
                "pipeline": profiles.pipeline_marker(runtime.profile),
                "switching": "enabled" if allow_profile_switch else
                             "disabled (set RAGLAB_ALLOW_PROFILE_SWITCH=1)"}

    # -- profile switching (optional) ----------------------------------------

    @app.post("/profile")
    def switch_profile(request: ProfileSwitchRequest):
        if not allow_profile_switch:
            raise ServiceError(403, "profile_switching_disabled",
                               hint="start the service with RAGLAB_ALLOW_PROFILE_SWITCH=1")
        candidate = {key: (dict(value) if isinstance(value, dict) else value)
                     for key, value in runtime.profile.items()}
        notes = []
        for slot, registry in (("embedding", profiles.EMBEDDING_PROVIDERS),
                               ("answer", profiles.ANSWER_PROVIDERS)):
            update = getattr(request, slot)
            if not update:
                continue
            provider = str(update.get("provider", candidate[slot]["provider"])).strip().lower()
            if provider not in registry:
                raise ServiceError(400, "unknown_provider", slot=slot, provider=provider,
                                   registered=sorted(registry))
            previous = (update.get("model") or candidate[slot]["model"])
            candidate[slot] = {"provider": provider,
                               "model": profiles.consistent_model(
                                   provider, registry[provider], candidate, previous)}
            if slot == "answer" and provider == "xkiro" and \
                    candidate[slot]["model"] != profiles.SUPPORTED_ANSWER["model"]:
                notes.append("non-pinned xKiro SKUs are EXPERIMENTAL here: no live "
                             "free-price check, no benchmark attribution")
        retrieval = request.retrieval or {}
        for key, allowed in (("mode", {"vector", "rrf", "blend"}),
                             ("lang_filter", {None, "ar", "fr", "en"})):
            if key in retrieval and retrieval[key] in allowed:
                candidate["retrieval"][key] = retrieval[key]
        if "top_k" in retrieval and isinstance(retrieval["top_k"], int) \
                and 1 <= retrieval["top_k"] <= 20:
            candidate["retrieval"]["top_k"] = retrieval["top_k"]
        if "neighbor_radius" in retrieval and retrieval["neighbor_radius"] in (0, 1, 2):
            candidate["retrieval"]["neighbor_radius"] = retrieval["neighbor_radius"]
        chunking = request.chunking or {}
        if "mode" in chunking:
            if chunking["mode"] not in {"size", "restructure", "manual"}:
                raise ServiceError(400, "bad_chunking_mode", mode=chunking["mode"],
                                   allowed=["size", "restructure", "manual"])
            candidate["chunking"]["mode"] = chunking["mode"]
        for key in ("size", "overlap"):
            if key in chunking:
                value = chunking[key]
                if not isinstance(value, int) or value <= 0:
                    raise ServiceError(400, "bad_chunking_value", key=key, value=value)
                candidate["chunking"][key] = value
        if request.data_dirs is not None:
            missing = [d for d in request.data_dirs
                       if not Path(d.strip()).expanduser().is_dir()]
            if missing:
                raise ServiceError(400, "bad_data_dirs", missing=missing,
                                   hint="paths must exist on the service host")
            candidate["data_dirs"] = [d.strip() for d in request.data_dirs if d.strip()]
        runtime.switch(candidate)
        jobs.status.update(collection=profiles.collection_name(candidate))
        return {"profile": candidate,
                "collection": profiles.collection_name(candidate),
                "index_note": "the embedding collection changes with provider/model/"
                              "chunking — check GET /health and POST /ingest if needed",
                "notes": notes}

    # -- retrieval -----------------------------------------------------------

    @app.post("/search")
    def search(request: SearchRequest):
        from evaluate import prepare_query_text
        from retrieval import retrieve
        from translate import detect_language
        if request.mode is not None and request.mode not in {"vector", "rrf", "blend"}:
            raise ServiceError(400, "bad_mode", mode=request.mode,
                               allowed=["vector", "rrf", "blend"])
        if request.lang_filter not in (None, "ar", "fr", "en"):
            raise ServiceError(400, "bad_lang_filter", lang_filter=request.lang_filter)
        if request.query_lang not in (None, "ar", "fr", "en"):
            raise ServiceError(400, "bad_query_lang", query_lang=request.query_lang)
        local = runtime.local()
        embedder = runtime.embedder()
        collection = runtime.collection()
        q_text = prepare_query_text(request.question)
        language = request.query_lang or detect_language(q_text)
        mode = request.mode or runtime.profile["retrieval"]["mode"]
        k = request.k or runtime.profile["retrieval"]["top_k"]
        try:
            hits, variants = retrieve(local, embedder, collection, q_text,
                                      language=language, translator=None, mode=mode,
                                      top_k=k, lang_filter=request.lang_filter,
                                      variant_strategy="original")
        except (ValueError, RuntimeError) as exc:
            raise ServiceError(409, "retrieval_refused", error=safe_error(exc)) from None
        return {"question": request.question, "language": language, "k": k, "mode": mode,
                "embedder": {"provider": embedder.provider_name, "model": embedder.model},
                "variants": [{"label": v["label"], "text": v["text"]} for v in variants],
                "hits": [{"rank": hit.get("rank"), "id": hit.get("id"),
                          "similarity": hit.get("similarity"),
                          "language": (hit.get("metadata") or {}).get("language"),
                          "heading": (hit.get("metadata") or {}).get("heading"),
                          "source": (hit.get("metadata") or {}).get("source"),
                          "chunk_index": (hit.get("metadata") or {}).get("chunk_index"),
                          "text": hit.get("text")} for hit in hits]}

    # -- grounded answers ------------------------------------------------------

    @app.post("/answer")
    def answer(request: AnswerRequest):
        import chat as chat_mod
        for field, allowed in (("mode", {"vector", "rrf", "blend"}),
                               ("lang_filter", {None, "ar", "fr", "en"}),
                               ("query_lang", {None, "ar", "fr", "en"})):
            value = getattr(request, field)
            if value is not None and value not in allowed:
                raise ServiceError(400, f"bad_{field}", **{field: value},
                                   allowed=sorted(v for v in allowed if v))
        greeting = profiles.greeting_reply(request.question, language=request.query_lang)
        if greeting is not None:
            language, text = greeting
            return {"status": "greeting", "reason": "smalltalk", "answer": text,
                    "language": language, "inference_performed": False,
                    "model": "(none — answered locally)", "claims": [], "sources": []}
        local = runtime.local()
        embedder = runtime.embedder()
        collection = runtime.collection()
        generator = runtime.generator()
        k = request.k or local.ANSWER_TOP_K
        try:
            result = chat_mod.ask(local, embedder, collection, generator,
                                  request.question, top_k=k,
                                  neighbor_radius=local.ANSWER_NEIGHBOR_RADIUS,
                                  use_cache=True,
                                  mode=request.mode or runtime.profile["retrieval"]["mode"],
                                  lang_filter=(request.lang_filter if request.lang_filter is not None
                                               else runtime.profile["retrieval"]["lang_filter"]),
                                  language=request.query_lang)
        except (ValueError, RuntimeError) as exc:
            raise ServiceError(409, "answer_refused", error=safe_error(exc)) from None
        except NvidiaAPIError as exc:
            raise ServiceError(502, "provider_error", error=safe_error(exc)) from None
        sources = []
        for source in result.get("sources") or []:
            row = {"source_id": source["source_id"], "document": source["document"],
                   "chunk_id": source["chunk_id"], "heading": source.get("heading", "")}
            if request.include_excerpts:
                row["text"] = source["text"]
            sources.append(row)
        return {"status": result["status"], "reason": result.get("reason"),
                "answer": result["answer"], "claims": result.get("claims") or [],
                "sources": sources, "model": result["model"],
                "language": result.get("language"),
                "validation_ok": result.get("validation_ok", True),
                "cached": result.get("cached", False),
                "retrieved": result.get("retrieved", 0),
                "dropped_for_budget": result.get("dropped_for_budget", 0),
                "seconds": result.get("seconds", 0.0),
                "inference_performed": result.get("status") not in (None, "refused", "greeting")}

    # -- ingestion --------------------------------------------------------------

    @app.post("/ingest")
    def start_ingest(reset: bool = False):
        runtime._require_key("embedding")
        jobs.start(runtime, reset=reset)
        return {"state": "accepted", "reset": reset,
                "collection": profiles.collection_name(runtime.profile),
                "status": "/ingest/status"}

    @app.get("/ingest/status")
    def ingest_status():
        return jobs.status

    # -- API keys (the console's key manager, over HTTP) -------------------------
    # Values are accepted (the console needs to paste them) but NEVER echoed:
    # responses carry only presence and the masked first 8 characters. Setting
    # a key affects the service process immediately; persist=true additionally
    # writes it to raglab/.env on the service host, exactly like app.py.

    @app.get("/keys")
    def list_keys():
        keys = []
        for env_name, description in profiles.KEY_ENV_INFO.items():
            value = os.environ.get(env_name, "").strip()
            keys.append({"env": env_name, "description": description,
                         "status": "set" if value else "missing",
                         "masked": profiles.masked(value) if value else None})
        return {"keys": keys}

    @app.post("/keys")
    def set_key(request: KeySetRequest):
        if request.key_env not in profiles.KEY_ENV_INFO:
            raise ServiceError(400, "unknown_key_env", key_env=request.key_env,
                               registered=sorted(profiles.KEY_ENV_INFO))
        if re.search(r"[\s\"']", request.value):
            raise ServiceError(400, "bad_key_value",
                               hint="a key cannot contain spaces or quotes")
        if profiles.looks_like_placeholder(request.value):
            raise ServiceError(400, "placeholder_value",
                               hint="that looks like the .env.example template, "
                                    "not a real key")
        os.environ[request.key_env] = request.value
        if request.persist:
            profiles.write_env_assignment(request.key_env, request.value)
        # A new key must invalidate anything built with the old/missing one.
        with runtime.lock:
            runtime._embedder = None
            runtime._generator = None
        return {"env": request.key_env, "status": "set",
                "masked": profiles.masked(request.value),
                "persisted": bool(request.persist)}

    @app.delete("/keys/{env_name}")
    def delete_key(env_name: str, persist: bool = False):
        if env_name not in profiles.KEY_ENV_INFO:
            raise ServiceError(400, "unknown_key_env", key_env=env_name,
                               registered=sorted(profiles.KEY_ENV_INFO))
        removed_file = profiles.remove_env_assignment(env_name) if persist else False
        os.environ.pop(env_name, None)
        with runtime.lock:
            runtime._embedder = None
            runtime._generator = None
        return {"env": env_name, "status": "missing",
                "removed_from_env_file": removed_file}

    # -- corpus inspection (no model calls) ---------------------------------------

    @app.get("/inspect")
    def inspect(limit: int = 3):
        import statistics
        from chunker import chunk_all
        from loader import load_all
        if not 0 <= limit <= 50:
            raise ServiceError(400, "bad_limit", limit=limit, allowed="0-50")
        local = runtime.local()
        try:
            docs = load_all(profiles.data_dirs(runtime.profile))
        except ValueError as exc:
            raise ServiceError(409, "no_corpus", error=str(exc)) from None
        if not docs:
            raise ServiceError(409, "no_corpus",
                               hint="no loadable documents in the configured data dirs")
        chunks = chunk_all(docs, local)
        counts = [chunk.token_count for chunk in chunks]
        step = max(1, local.CHUNK_SIZE_TOKENS // 5)
        histogram = [{"bucket": f"{low}-{low + step}",
                      "chunks": sum(1 for c in counts if low <= c < low + step)}
                     for low in range(0, max(counts) + 1, step)]
        return {
            "documents": [{"name": doc["name"], "language": doc["language"],
                           "chars": len(doc["text"])} for doc in docs],
            "chunking": {"mode": local.CHUNKING_MODE, "size": local.CHUNK_SIZE_TOKENS,
                         "overlap": local.CHUNK_OVERLAP_TOKENS},
            "chunks": {"count": len(chunks), "tokens_total": sum(counts),
                       "tokens_min": min(counts), "tokens_max": max(counts),
                       "tokens_median": statistics.median(counts),
                       "tokens_mean": round(statistics.mean(counts), 1),
                       "histogram": [row for row in histogram if row["chunks"]]},
            "sample": [{"index": chunk.index, "source": chunk.source,
                        "language": chunk.language, "tokens": chunk.token_count,
                        "section": chunk.section_type, "heading": chunk.heading,
                        "text": chunk.text} for chunk in chunks[:limit]],
        }

    # -- quote-vs-chunk diagnostic (no model calls) --------------------------------

    @app.post("/chunks/search")
    def chunks_search(request: ChunkSearchRequest):
        description, rows = profiles.search_rows(runtime.profile)

        def brief(row):
            return {"source": row[1], "chunk_index": row[2], "heading": row[3] or None}

        found = profiles.locate_text(request.text, rows)
        payload = {"description": description, "needle": found["needle"],
                   "full": [brief(row) for row in found["full"]],
                   "head": [brief(row) for row in found["head"]],
                   "tail": [brief(row) for row in found["tail"]]}
        if found["full"]:
            payload["first_full_text"] = found["full"][0][0]
        return payload

    # -- embedding sanity check (one batched embedding call) ------------------------

    @app.post("/embeddings/sanity")
    def embeddings_sanity():
        # The same three phrases as BaseEmbedder._sanity_check (en/fr/ar), so
        # the report says the same thing the console's sanity check says.
        phrases = [("English", "savings account"),
                   ("French", "compte épargne"),
                   ("Arabic", "حساب التوفير")]
        embedder = runtime.embedder()          # 503 when the key is missing
        from embedder import cosine
        vectors = embedder.embed_texts([phrase for _, phrase in phrases])
        pairs = []
        for i in range(len(phrases)):
            for j in range(i + 1, len(phrases)):
                pairs.append({"pair": f"{phrases[i][0]}/{phrases[j][0]}",
                              "cosine": round(cosine(vectors[i], vectors[j]), 4)})
        return {"provider": embedder.provider_name, "model": embedder.model,
                "dimension": len(vectors[0]) if vectors else None,
                "batch_size": embedder.batch_size,
                "cache_entries": embedder.cache.size,
                "api_calls": embedder.api_calls,
                "phrases": [phrase for _, phrase in phrases],
                "cosines": pairs,
                "interpretation": "clearly positive similarities mean the model places "
                                  "the languages in one shared space; negative/near-zero "
                                  "values mean cross-lingual retrieval is likely to fail"}

    # -- evaluation (embeds every question — provider calls) -------------------------

    QUESTION_SETS = ("questions.json", "questions_50.json", "questions_real.json")

    @app.post("/evaluate")
    def evaluate(request: EvaluateRequest):
        from evaluate import load_question_set, run_evaluation, save_run
        path = Path(request.questions)
        if not path.is_absolute():
            path = profiles.PROJECT_DIR / request.questions
        if request.questions not in QUESTION_SETS and not path.exists():
            raise ServiceError(400, "unknown_question_set", questions=request.questions,
                               known=list(QUESTION_SETS),
                               hint="or pass an absolute path that exists on the service host")
        try:
            cases = load_question_set(path)
        except FileNotFoundError as exc:
            raise ServiceError(400, "unknown_question_set", error=str(exc)) from None
        if not cases:
            raise ServiceError(400, "empty_question_set", questions=request.questions)
        local = runtime.local()
        embedder = runtime.embedder()
        collection = runtime.collection()
        try:
            run = run_evaluation(local, embedder, collection, cases,
                                 mode=runtime.profile["retrieval"]["mode"],
                                 top_k=request.top_k or 20, translator=None)
        except (ValueError, RuntimeError) as exc:
            raise ServiceError(409, "evaluation_refused", error=safe_error(exc)) from None
        saved = save_run(run, local.RESULTS_DIR)
        # The full run carries every question's hits — heavy. Give the caller
        # the aggregates plus per-question outcomes, not the raw hit lists.
        questions = [{"id": q.get("id"), "language": q.get("language"),
                      "category": q.get("category"), "hit_at_1": q.get("hit_at_1"),
                      "hit_at_3": q.get("hit_at_3"), "hit_at_5": q.get("hit_at_5"),
                      "is_out_of_scope": q.get("is_out_of_scope")}
                     for q in run.get("questions", [])]
        return {"metrics": run["metrics"], "config": run["config"],
                "questions": questions, "saved_to": str(saved),
                "note": "lab measurement on " + local.CHROMA_COLLECTION_NAME +
                        " — not a benchmark result"}

    # -- diagnostics ---------------------------------------------------------------

    @app.post("/diagnostics/harness50")
    def diagnostics_harness50():
        import subprocess
        import sys as _sys
        # Offline, deterministic, no provider calls — but it re-chunks the whole
        # corpus twice, so it takes tens of seconds; that is why it is a POST.
        try:
            completed = subprocess.run(
                [_sys.executable, "harness50.py"], cwd=profiles.PROJECT_DIR,
                capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            raise ServiceError(504, "harness50_timeout") from None
        return {"exit_code": completed.returncode,
                "output_tail": (completed.stdout or "").splitlines()[-20:],
                "report": str(profiles.PROJECT_DIR / "results/harness50/comparison.md")
                if completed.returncode == 0 else None}

    @app.post("/diagnostics/catalog")
    def diagnostics_catalog():
        try:
            from provider_catalog import collect
            report = collect()
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            raise ServiceError(502, "catalog_failed", error=safe_error(exc)) from None
        return report

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app",
                host=os.environ.get("RAGLAB_HOST", "0.0.0.0"),
                port=int(os.environ.get("RAGLAB_PORT", "8000")))
