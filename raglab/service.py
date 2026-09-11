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
    POST /profile         switch provider/model at runtime
                          (only when RAGLAB_ALLOW_PROFILE_SWITCH=1)
    POST /search          retrieval only: top-k chunks for a question
    POST /answer          grounded, cited answer (or refusal) for a question
    POST /ingest          build/rebuild this profile's index (background job)
    GET  /ingest/status   what the ingest job is doing / last did

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
    RAGLAB_ALLOW_PROFILE_SWITCH    1 to enable POST /profile (default 0)
    RAGLAB_CORS_ORIGINS            comma-separated allowed origins (default *)
    plus the provider keys: NVIDIA_API_KEY, XKIRO_API_KEY, GOOGLE_API_KEY /
    GEMINI_API_KEY, KIRA_API_KEY, JINA_API_KEY, ... (see raglab/.env.example)
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
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


class ProfileSwitchRequest(BaseModel):
    embedding: Optional[dict] = None   # {"provider": ..., "model": ...}
    answer: Optional[dict] = None
    retrieval: Optional[dict] = None   # {"top_k", "mode", "lang_filter", "neighbor_radius"}


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
        allow_profile_switch = os.environ.get("RAGLAB_ALLOW_PROFILE_SWITCH", "0") == "1"
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
        greeting = profiles.greeting_reply(request.question)
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
                                  use_cache=True, mode=runtime.profile["retrieval"]["mode"],
                                  lang_filter=runtime.profile["retrieval"]["lang_filter"])
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

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app",
                host=os.environ.get("RAGLAB_HOST", "0.0.0.0"),
                port=int(os.environ.get("RAGLAB_PORT", "8000")))
