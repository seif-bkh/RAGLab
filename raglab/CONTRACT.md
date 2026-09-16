# RAGLab service — HTTP contract

**Audience:** the fullstack team building against this service.
**Service version:** `1.2.4` (reported by `GET /health` → `version`).
**Machine-readable schema:** FastAPI generates OpenAPI 3 at `/openapi.json` and
interactive docs at `/docs`. This document is the human contract — semantics,
state, error behavior and integration rules that a schema alone does not carry.
When this doc and `/openapi.json` disagree, `/openapi.json` is truth for field
names; this doc is truth for behavior. **Designing the frontend?** The
screen-by-screen UX blueprint built on this contract is `raglab/FRONTEND.md`.
**Building the settings screens?** The exact-request cookbook (field matrix,
every mutation with side effects, worked sequences) is `raglab/COOKBOOK.md`.

**What the service is:** a grounded Q&A engine over a fixed document corpus.
You send a question; it retrieves chunks from its local vector index, asks a
chat model to answer **as verbatim-cited claims**, validates every citation
quote against the source chunk **and every number in every claim against its
evidence** — then scrubs known PII patterns from the output — and returns
claims + evidence or a safe refusal. It is one stateful worker (one profile,
one index, one ingest at a time) — not a stateless API. Read §2 before wiring
anything.

---

## 1. Ground rules

| Rule | Detail |
|---|---|
| Transport | HTTP/1.1, JSON bodies (`content-type: application/json`), UTF-8 everywhere. The corpus is Arabic + French — never transcode or transliterate. |
| Base URL | One deployment, one port (default `:8000`). All paths are root-relative (`/search`, not `/api/v1/search`). |
| Auth | **Optional shared-secret token.** When `RAGLAB_SERVICE_TOKEN` is set on the service (or its alias `RAGLAB_TOKEN`), every request must carry it in the `X-Service-Token` header (constant-time compare) → `401 unauthorized` otherwise. Unset = open (the local/dev default). This is defense in depth, not user auth — the service still expects to sit behind your gateway (real auth, rate limits, CORS tightening). `RAGLAB_CORS_ORIGINS` configures browser CORS (default: `*`); CORS preflights are exempt from the token check by design. |
| Timestamps | UTC, ISO-8601 with offset, second precision (`2026-09-14T14:18:01+00:00`). |
| Idempotency | `GET`s are safe. `POST /answer` and `POST /search` are read-only w.r.t. state (but may spend provider budget). `POST /ingest`, `POST /profile`, `POST /keys`, `DELETE /keys/{env}` mutate state. Nothing is transactional across calls. |
| Sessions | **There are none.** No cookies, no conversation state, no per-client storage. Every request is judged against the current global profile + index. Chat history is the client's job (§5.4). |

### 1.1 Status codes

| Code | Meaning here |
|---|---|
| `200` | Success — **including a refusal** (§3.10: `status: "refused"` is a valid answer, not an error). |
| `400` | Semantic error — the request is well-formed JSON but violates a rule. Body: error envelope with a `reason` (§4). |
| `401` | Missing/wrong `X-Service-Token` — only on deployments where `RAGLAB_SERVICE_TOKEN` is set (`unauthorized`). |
| `403` | Profile switching is disabled on this deployment (`profile_switching_disabled`). |
| `404` / `405` | Unknown path / wrong method. |
| `409` | State conflict — empty index, stale index, ingest already running, corpus missing, provider refused mid-request. |
| `422` | Request validation (missing/short fields, wrong types) — FastAPI/pydantic format, an **array** (§4.2). |
| `502` | Upstream provider error (`provider_error`, `catalog_failed`). Safe-redacted. |
| `503` | A required API key is not set — the body names the env var (`missing_api_key`). |
| `504` | `harness50_timeout` — the offline diagnostic subprocess exceeded 900 s. |

### 1.2 Error envelope

Every service-level error is `{"detail": {"reason": "<code>", ...context}}` —
`reason` is a stable machine code (full catalog in §4), the extra keys are
context (which slot, which env var, what is registered). Values pass a
redaction filter: **API keys can never appear in an error body.**

```json
{"detail": {"reason": "missing_api_key", "slot": "embedding",
            "key_env": "NVIDIA_API_KEY",
            "hint": "set NVIDIA_API_KEY in the service environment (see raglab/.env.example)"}}
```

### 1.3 Which endpoints spend money

| Endpoint | Embedding calls | Chat calls | Notes |
|---|---|---|---|
| `GET /health`, `/`, `/models`, `/profile`, `/keys`, `/ingest/status`, `/inspect` | — | — | Free. |
| `GET/POST/DELETE /documents*` | — | — | Free (disk only; indexing is the usual `POST /ingest`). |
| `POST /chunks/search` | — | — | Free (re-chunks locally; seconds, CPU only). |
| `POST /keys`, `DELETE /keys/{env}` | — | — | Free. |
| `POST /profile` | — | — | Free. May orphan the active index (§2.3). |
| `POST /search` | ✅ 1 (per request) | — | Cached by query text. |
| `POST /answer` | ✅ 1 | ✅ 1 (unless greeting/refusal/cache hit) | Answers are cached per question. First xKiro-pinned call also does a live free-price check. |
| `POST /ingest` | ✅ 1 per chunk (cache makes re-runs cheap) | — | Background job (§3.8). |
| `POST /embeddings/sanity` | ✅ 1 (batched, 3 phrases) | — | Diagnostic. |
| `POST /evaluate` | ✅ 1 per question | — | Slow; minutes on the 50-question set. |
| `POST /diagnostics/harness50` | — | — | Offline subprocess, up to 900 s, **synchronous**. |
| `POST /diagnostics/catalog` | — | — | One live call to the xKiro gateway. |

---

## 2. State model — read this before wiring the UI

The service is a single stateful worker. Three pieces of state drive every
response:

```
 profile ──determines──▶ collection ──holds──▶ index (chunks + fingerprints)
    │                                                  ▲
    └── keys (process env) ──gates──▶ embedder/generator ─┘
```

### 2.1 The profile

The global configuration. Returned by `GET /profile` / `GET /health`
(`profile` key). Set at boot from `RAGLAB_*` env vars; mutable at runtime via
`POST /profile` **only if** `RAGLAB_ALLOW_PROFILE_SWITCH=1` (docker-compose
pins `0` by default; local runs default to `1`).

```json
{"version": 1,
 "embedding": {"provider": "nvidia", "model": "nvidia/nemotron-3-embed-1b"},
 "answer":    {"provider": "xkiro", "model": "qwen/qwen3.8-max:free"},
 "chunking":  {"mode": "restructure", "size": 220, "overlap": 40},
 "retrieval": {"top_k": 5, "mode": "vector", "lang_filter": null, "neighbor_radius": 0},
 "data_dirs": ["/app/docs", "/app/raglab/data"],
 "custom_models": {}}
```

| Field | Type | Constraints |
|---|---|---|
| `embedding.provider` / `answer.provider` | string | Must be registered (`GET /models` lists them). |
| `embedding.model` / `answer.model` | string | An explicitly sent model ID is honored **verbatim** — custom IDs allowed; one token, no spaces. The registered lists are known-good suggestions, not a whitelist. |
| `chunking.mode` | `size` \| `restructure` \| `manual` | `size` uses `size`/`overlap` (tokens). |
| `chunking.size` / `chunking.overlap` | int > 0 | Used in `size` mode. |
| `retrieval.top_k` | int 1–20 | Default chunks per query. |
| `retrieval.mode` | `vector` \| `rrf` \| `blend` | Retrieval strategy. |
| `retrieval.lang_filter` | `null` \| `ar` \| `fr` \| `en` | Restrict retrieval to one corpus language. |
| `retrieval.neighbor_radius` | 0 \| 1 \| 2 | Widen hits with adjacent chunks. |
| `data_dirs` | string[] | Corpus directories **on the service host** (Docker: inside the container). |
| `version`, `custom_models` | — | Internal bookkeeping; treat as opaque, do not mutate. |

`pipeline` (in `/health` and `/profile` responses) is a display label:
`"supported pipeline — the same model pair main.py answer uses"` for the pinned
NVIDIA + xKiro pair, otherwise `"experimental lab profile — same
chunker/retrieval/citation checks, but no benchmark number belongs to it"`.
Show it when the user picks models.

### 2.2 Collections

Every `(embedding provider+model, chunking mode)` pair gets its own ChromaDB
collection: `raglab_app_<provider>_<model_slug>_<chunking>` (e.g.
`raglab_app_nvidia_nemotron_3_embed_1b_restructure`). Switching profiles never
corrupts another profile's vectors — but each new combination starts with an
**empty index** until you ingest (§2.3).

### 2.3 Index lifecycle

```
 (new profile) ──▶ EMPTY ──POST /ingest──▶ RUNNING ──▶ DONE
                     ▲                                   │
                     └── 409 empty_index on /search,     index usable;
                          /answer, /evaluate             409 retrieval_refused
                                                          if fingerprints went
                                                          stale (settings
                          RUNNING ──▶ ERROR (job.error)   changed vs stored)
```

* `409 empty_index` → `POST /ingest`, poll `GET /ingest/status` until `done`.
* `409 retrieval_refused` → the stored chunks were built with different
  settings (chunker version, tokenizer, size…) than the current profile →
  rebuild with `POST /ingest?reset=true`.
* One ingest at a time; a second `POST /ingest` while running →
  `409 ingest_already_running`. Ingest is **not** tied to a client — any
  client can start it, any client can poll it.
* **While the job runs, the index is sealed** (one writer at a time):
  `POST /search`, `/answer`, `/evaluate`, `POST /profile` and
  `DELETE /documents/{id}` return `409 ingest_in_progress` (poll
  `/ingest/status`, retry when `done`); greetings still work;
  `GET /documents` rows show `status: "indexing"` without reading the store.

### 2.4 Keys

Provider API keys live in the service process environment. `GET /keys` lists
every known env var with masked presence (`"nvapi-me…"`, first 8 chars);
values are **never** returned by any endpoint. `POST /keys` sets one for the
process (optionally persisting to the service host's `raglab/.env`);
`DELETE /keys/{env}` drops it. Setting/deleting a key immediately invalidates
the cached embedder/generator. Missing key → `503 missing_api_key` naming the
env var, on every endpoint that needs it.

### 2.5 Documents (the gateway feed)

Besides the corpus directories in the profile, the service owns a document
store (`RAGLAB_DOCUMENTS_DIR`, default `documents/`) that only the
`/documents` API writes. Pushed documents are versioned by content hash and
are part of **every** profile's corpus — the store's dir is always appended
to `data_dirs`, including after `POST /profile` changes them. Indexing a
pushed document still goes through the §2.3 lifecycle (`POST /ingest`);
deleting one purges its chunks from every collection immediately. Per-document
status (`pending` / `indexed` / `stale`) is derived from the active
collection + the last ingest time — see §3.15.

---

## 3. Endpoint reference

Examples below were captured from **real responses** of the offline test
fixture (a one-file corpus, stubbed embedder); field names and shapes are
exact — numeric values are illustrative where the fixture's are meaningless
(e.g. similarity 0.0 from the stub embedder).

### 3.0 `GET /config` — self-description for consoles/adapters
One call a generic console can render directly: the flat fields it displays
(`chat_model`, `embedding_model`, `vector_dimension` — `null` until an index
exists), editability (`editable.profile` follows the switching flag; keys,
documents and index are always editable), the full profile, and a
`capabilities` map naming the exact call for every mutation. Nothing about
this service needs a CLI — if a console claims "configured via CLI", it is
reading an outdated premise.

```json
{"service": "raglab", "version": "1.2.4",
 "chat_model": "xkiro/qwen/qwen3.8-max:free",
 "embedding_model": "nvidia/nvidia/nemotron-3-embed-1b",
 "vector_dimension": 2048,
 "editable": {"profile": true, "api_keys": true, "documents": true, "index": true},
 "switching": "enabled",
 "profile": {"...": "..."},
 "collection": "raglab_app_nvidia_nemotron_3_embed_1b_restructure",
 "pipeline": "supported pipeline — the same model pair main.py answer uses",
 "index": {"count": 346},
 "keys": {"NVIDIA_API_KEY": "set", "XKIRO_API_KEY": "set"},
 "capabilities": {"switch_models": "POST /profile {embedding|answer: {provider, model}}",
                  "chunking": "POST /profile {chunking: {mode, size, overlap}}",
                  "retrieval": "POST /profile {retrieval: {top_k, mode, lang_filter, neighbor_radius}}",
                  "corpus_dirs": "POST /profile {data_dirs: [...]}",
                  "api_keys": "GET/POST/DELETE /keys",
                  "documents": "GET/POST/DELETE /documents",
                  "reindex": "POST /ingest (reset=true for a clean rebuild)"},
 "docs": {"openapi": "/openapi.json", "interactive": "/docs",
          "contract": "raglab/CONTRACT.md", "cookbook": "raglab/COOKBOOK.md",
          "frontend": "raglab/FRONTEND.md"}}
```

### 3.1 `GET /` — service banner
Returns `{"service": "raglab", "version", "docs", "health", "profile", "models", "endpoints"}`. Useful as a ping.

### 3.2 `GET /health` — liveness + the whole state at a glance
The endpoint your UI polls. No secrets — key values never appear, only
`set`/`missing` per env var.

```json
{"status": "ok", "version": "1.2.4",
 "profile": {"embedding": {"provider": "nvidia", "model": "nvidia/nemotron-3-embed-1b"},
             "answer": {"provider": "xkiro", "model": "qwen/qwen3.8-max:free"},
             "chunking": {"mode": "restructure", "size": 220, "overlap": 40},
             "retrieval": {"top_k": 5, "mode": "vector", "lang_filter": null, "neighbor_radius": 0},
             "pipeline": "supported pipeline — the same model pair main.py answer uses"},
 "index": {"collection": "raglab_app_nvidia_nemotron_3_embed_1b_restructure",
           "count": 346, "chunk_fp": "chunkv4:msize:s6", "tokenizer_match": true},
 "keys": {"NVIDIA_API_KEY": "missing", "XKIRO_API_KEY": "missing"},
 "ingest": {"state": "done", "started_at": "2026-09-14T14:18:01+00:00",
            "finished_at": "2026-09-14T14:18:01+00:00",
            "collection": "raglab_app_nvidia_nemotron_3_embed_1b_restructure",
            "stored": 346, "error": null}}
```

Notes: `keys` covers only the env vars the **current** profile's providers
need (use `GET /keys` for all of them). `index.chunk_fp`/`tokenizer_match`
appear once an index exists; `tokenizer_match: false` → expect
`409 retrieval_refused` → rebuild.

### 3.3 `GET /models` — the registries
```json
{"embedding": [ {"provider": "nvidia",
                 "label": "NVIDIA build endpoint (stdlib HTTPS, no SDK)",
                 "key_envs": ["NVIDIA_API_KEY"], "key_set": false,
                 "models": ["nvidia/nemotron-3-embed-1b"]}],
 "answer":    [ {"provider": "kira",
                 "label": "Kira AI — OpenAI-compatible gateway (kiraai.vn)",
                 "key_envs": ["KIRA_API_KEY"], "key_set": false,
                 "models": ["glm-5.3-free"]}],
 "note": "custom model IDs may be supplied via POST /profile (when switching is enabled) or the RAGLAB_* env vars"}
```
(truncated to one provider per slot — the real response lists every registered
provider: embeddings `nvidia, gemini, jina, huggingface, openai, cohere,
voyage`; answers `xkiro, nvidia, google, kira`)
`key_set: true` with an empty `key_envs` list means "no key needed". Use this
to build the provider/model picker (§5.5). The lists are suggestions — the
user may type any exact model ID.

### 3.4 `GET /profile` / `POST /profile` — read / switch the profile
`GET` returns `{profile, collection, pipeline, switching}` where `switching`
is `"enabled"` or `"disabled (set RAGLAB_ALLOW_PROFILE_SWITCH=1)"`.

`POST /profile` body — every key optional, only sent keys change anything:

| Key | Type | Rules |
|---|---|---|
| `embedding` | `{provider?, model?}` | Provider must be registered (`400 unknown_provider`). Explicit `model` is applied verbatim (`400 bad_model` if it has spaces). Provider-only switch: same provider = no-op; different provider = its first registered model. |
| `answer` | `{provider?, model?}` | Same rules. Non-pinned xKiro models add a `notes[]` warning (§5.5). |
| `chunking` | `{mode?, size?, overlap?}` | `mode` ∈ `size/restructure/manual` (`400 bad_chunking_mode`); `size`/`overlap` positive ints (`400 bad_chunking_value`). |
| `retrieval` | `{top_k?, mode?, lang_filter?, neighbor_radius?}` | Same constraints as §2.1; out-of-range values are **ignored silently**, not errors. |
| `data_dirs` | `string[]` | Each must exist **on the service host** (`400 bad_data_dirs` with the missing list). |

Response: `{profile, collection, index_note, notes[]}`. `collection` is the
new collection name — if it differs from the index in `/health`, it is empty:
offer ingest. Errors: `403 profile_switching_disabled` when the deployment
locks switching.

```json
{"profile": {"…": "…", "answer": {"provider": "kira", "model": "glm-custom-9"},
             "retrieval": {"top_k": 7, "…": "…"}},
 "collection": "raglab_app_huggingface_qwen3_embedding_0_6b_size",
 "index_note": "the embedding collection changes with provider/model/chunking — check GET /health and POST /ingest if needed",
 "notes": []}
```

### 3.5 `GET /keys` / `POST /keys` / `DELETE /keys/{env}` — key management
`GET /keys` → `{keys: [{env, description, status: "set"|"missing", masked}]}` —
`masked` is `"kira-co…"` (first 8 chars + ellipsis) or `null`.

`POST /keys` body: `{key_env, value, persist?}` — `key_env` must be known
(`400 unknown_key_env`), the value cannot contain spaces/quotes
(`400 bad_key_value`) or look like the `.env.example` template
(`400 placeholder_value`). `persist: true` additionally writes it to the
service host's `raglab/.env` (survives a process restart, **not** a container
recreation — in Docker, hand keys to compose `env_file` instead).

```json
{"env": "KIRA_API_KEY", "status": "set", "masked": "kira-con…", "persisted": false}
```

`DELETE /keys/{env}?persist=false` → `{"env", "status": "missing", "removed_from_env_file"}`.

### 3.6 `GET /inspect?limit=3` — chunking preview (free)
Corpus + chunk statistics with sample chunks. `limit` 0–50 (`400 bad_limit`);
`409 no_corpus` if the data dirs are empty/unreadable.

```json
{"documents": [{"name": "note.md", "language": "en", "chars": 110}],
 "chunking": {"mode": "size", "size": 60, "overlap": 10},
 "chunks": {"count": 2, "tokens_total": 28, "tokens_min": 13, "tokens_max": 15,
            "tokens_median": 14.0, "tokens_mean": 14,
            "histogram": [{"bucket": "12-24", "chunks": 2}]},
 "sample": [{"index": 0, "source": "note.md", "language": "en", "tokens": 15,
             "section": "front-matter", "heading": "# Account",
             "text": "# Account\n\nThe Atlas current account has no management fee."}]}
```

### 3.7 `POST /chunks/search` — quote-vs-chunk diagnostic (free)
Input: `{"text": "…"}` (1–8000 chars). Answers: *is this text inside ONE
chunk?* — i.e. could a verbatim quote of it ever pass the citation gate?

```json
{"description": "2 freshly chunked chunk(s) with the current settings …",
 "needle": "the atlas card costs 10 dinars",
 "full":  [{"source": "note.md", "chunk_index": 1, "heading": "## Fees"}],
 "head":  [{"source": "note.md", "chunk_index": 1, "heading": "## Fees"}],
 "tail":  [{"source": "note.md", "chunk_index": 1, "heading": "## Fees"}],
 "first_full_text": "## Fees\n\nThe Atlas card costs 10 dinars per year."}
```

`full` non-empty → a quote of `needle` **can** validate from those chunks.
`full` empty but `head`/`tail` non-empty → the text crosses a chunk boundary —
a faithful quote of it can **never** pass the gate (chunking must change, not
the model). Use this to explain "the model quoted correctly but was refused".

### 3.8 `POST /ingest?reset=false` + `GET /ingest/status` — build the index
`POST /ingest` validates the embedding key (`503 missing_api_key`), refuses a
second concurrent run (`409 ingest_already_running`), then chunks + embeds the
corpus in a background thread. It returns immediately:

```json
{"state": "accepted", "reset": false,
 "collection": "raglab_app_nvidia_nemotron_3_embed_1b_restructure",
 "status": "/ingest/status"}
```

Poll `GET /ingest/status` (1–2 s interval) — the same object also appears in
`/health` as `ingest`:

```json
{"state": "done", "started_at": "2026-09-14T14:18:01+00:00",
 "finished_at": "2026-09-14T14:18:01+00:00",
 "collection": "raglab_app_nvidia_nemotron_3_embed_1b_restructure",
 "stored": 346, "error": null}
```

States: `idle` (never run) → `running` → `done` (with `stored` count) |
`error` (with a safe-redacted `error` string). `reset=true` deletes the
collection first (clean rebuild — required after fingerprint mismatches).
A stale-fingerprint `error` (settings changed since the index was built,
e.g. a different chunk size in a previous session) names the remedy as
`POST /ingest?reset=true`.
Duration: minutes, proportional to corpus size and cache state.

### 3.9 `POST /search` — retrieval only (no chat model)
Request: `{question (1–2000 chars, required), k? (1–20), mode?, lang_filter?,
query_lang?}`. `query_lang` forces the query's language detection
(`ar/fr/en`); `lang_filter` restricts retrieved chunks to one language;
invalid enum values → `400 bad_mode` / `bad_lang_filter` / `bad_query_lang`.

```json
{"question": "What does the Atlas card cost?", "language": "en", "k": 2,
 "mode": "vector",
 "embedder": {"provider": "nvidia", "model": "nvidia/nemotron-3-embed-1b"},
 "variants": [{"label": "en(original)", "text": "What does the Atlas card cost?"}],
 "hits": [{"rank": 1, "id": "note.md::chunk_0001", "similarity": 0.83,
           "language": "en", "heading": "## Fees", "source": "note.md",
           "chunk_index": 1, "text": "## Fees\n\nThe Atlas card costs 10 dinars per year."}]}
```

Errors: `503 missing_api_key` (embedding), `409 empty_index` (ingest first),
`409 retrieval_refused` (stale index — rebuild with `?reset=true`).

`hits[].text` is PII-scrubbed like `/answer` outputs (§3.10): emails, phone
numbers, RIB/IBAN and CIN numbers come back as `[EMAIL]` / `[PHONE]` /
`[RIB]` / `[CIN]` placeholders.

### 3.10 `POST /answer` — the product endpoint
Request:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `question` | string 1–2000 | required | Any language (ar/fr/en); detected automatically. |
| `k` | int 1–20 | profile `top_k` | Chunks retrieved for this question. |
| `include_excerpts` | bool | `false` | Adds `text` to each source (verbatim chunk). |
| `query_lang` | `ar`/`fr`/`en` | auto-detect | Forces claim language + query language. |
| `mode` | `vector`/`rrf`/`blend` | profile mode | Retrieval strategy override. |
| `lang_filter` | `ar`/`fr`/`en` | profile value | Restrict retrieval to one corpus language. |

Three 200-shapes — **all three are successes**:

**(a) greeting** — pure smalltalk (`"bonjour"`, `"السلام عليكم"`, …) is
answered locally, zero provider calls:

```json
{"status": "greeting", "reason": "smalltalk",
 "answer": "Bonjour ! Je suis l'assistant documentaire du laboratoire : …",
 "language": "fr", "inference_performed": false,
 "model": "(none — answered locally)", "claims": [], "sources": []}
```

**(b) refused** — the safe answer. `status: "refused"` with a `reason`:
`private_or_live_request` (asked for personal/live data — decided locally),
`insufficient_evidence` / `no_context` (corpus doesn't support it),
`invalid_output` (the model's reply failed the citation gate's structural or
quote-membership checks), or `unsourced_number` (a claim stated a number its
evidence quotes don't contain — a computed/converted/renamed figure). On the
`invalid_output`/`unsourced_number` paths the response also carries `error`
(the gate's finding, safe-redacted) and `raw_preview` (the model's rejected
reply, PII-scrubbed) for display like "[model said, not accepted]". Render
`answer` as-is; it is a user-safe explanation:

```json
{"status": "refused", "reason": "private_or_live_request",
 "answer": "I cannot answer this from the supplied documents. I do not have access to personal accounts, credentials, or live banking data.",
 "claims": [], "sources": [], "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
 "language": "en", "validation_ok": true, "cached": false, "retrieved": 0,
 "dropped_for_budget": 0, "seconds": 0, "inference_performed": false}
```

**(c) answered** — validated, cited claims:

```json
{"status": "answered", "reason": "supported",
 "answer": "The Atlas card costs 10 dinars per year. [S1]",
 "claims": [{"text": "The Atlas card costs 10 dinars per year.",
             "evidence": [{"source_id": "S1",
                           "quote": "## Fees The Atlas card costs 10 dinars p"}]}],
 "sources": [{"source_id": "S1", "document": "note.md",
              "chunk_id": "note.md::chunk_0001", "heading": "## Fees"}],
 "model": "nvidia/nemotron-3.5-lightning-30b-a3b", "language": "en",
 "validation_ok": true, "cached": false, "retrieved": 1,
 "dropped_for_budget": 0, "seconds": 1.8, "error": null, "raw_preview": null,
 "inference_performed": true}
```

Field contract for (c):

| Field | Meaning / UI rule |
|---|---|
| `answer` | Human-readable summary with `[S#]` markers. Safe to render as text. |
| `claims[].text` | One concise factual claim, written in the **user's language**. Every digit-form number in it is machine-verified to exist in its evidence quotes (else the whole reply refuses as `unsourced_number`). |
| `claims[].evidence[].source_id` | Links to `sources[].source_id` (`S1`, `S2`, …). |
| `claims[].evidence[].quote` | The verbatim supporting words, in the **document's original language** (may differ from the claim's language — do not "fix" this). Quote membership in the cited chunk is machine-verified before you see it. |
| `sources[]` | The retrieved chunks the model could cite. With `include_excerpts: true`, each also has `text` (the full chunk). `source_id`s not cited by any claim were retrieved but unused. |
| `validation_ok` | `true` here **by construction** — invalid outputs come back as `refused/invalid_output` or `refused/unsourced_number`. |
| `cached` | Answer served from cache (fast, free). |
| `retrieved` / `dropped_for_budget` | Hits found / dropped to fit the model's context budget. |
| `seconds` | Provider call duration. |
| `error` / `raw_preview` | `null` on this path; populated only on `invalid_output`/`unsourced_number` refusals (see (b)). |
| `inference_performed` | `false` for greetings and local refusals — use it to mark "no AI call" in your UI. |

**What is machine-verified before you see an `answered` payload:** every
evidence quote is a contiguous verbatim member of its cited chunk, AND every
number in every claim appears in that claim's evidence (normalization covers
French `2,75` vs English `2.75` vs Arabic `٢٫٧٥`, thousands grouping
`50 000` vs `50000`, and Arabic-Indic digits). Claims prose itself is still
model-written — render, don't re-parse.

**Output PII scrubbing (post-gate):** `answer`, `claims[].text`,
`claims[].evidence[].quote`, `sources[].text` (when requested) and `/search`
`hits[].text` pass through a scrubber that replaces emails, phone numbers
(Tunisian and international formats), RIB/IBAN (20-digit forms) and CIN
numbers (8 digits in CIN context) with `[EMAIL]` / `[PHONE]` / `[RIB]` /
`[CIN]`. Amounts, rates, dates and counts are deliberately untouched. The
gate validates the RAW verbatim text; the scrubber rewrites only what you
see. Diagnostics endpoints (`/inspect`, `/chunks/search`) intentionally show
raw text — they are admin-facing truth, not user-facing output.

Errors: `503 missing_api_key` (either slot — body says which), `409 empty_index`,
`409 answer_refused` (mid-request provider/refusal error), `502 provider_error`,
`502 provider_unreachable` (network-layer failure — incl. the first-call xKiro
free-price check; the body carries the underlying network error).

### 3.11 `POST /embeddings/sanity` — one batched embedding call
Embeds 3 phrases (en/fr/ar: "savings account", "compte épargne", "حساب التوفير")
and reports cross-lingual cosines — the check for "is this embedding model
usable for cross-lingual retrieval at all".

```json
{"provider": "nvidia", "model": "nvidia/nemotron-3-embed-1b",
 "dimension": 2048, "batch_size": 8, "cache_entries": 6, "api_calls": 1,
 "phrases": ["savings account", "compte épargne", "حساب التوفير"],
 "cosines": [{"pair": "English/French", "cosine": 0.71},
             {"pair": "English/Arabic", "cosine": 0.66},
             {"pair": "French/Arabic", "cosine": 0.69}],
 "interpretation": "clearly positive similarities mean the model places the languages in one shared space; negative/near-zero values mean cross-lingual retrieval is likely to fail"}
```

### 3.12 `POST /evaluate` — run a question set
Request: `{questions: "questions.json" | "questions_50.json" |
"questions_real.json" | absolute path on the service host, top_k? (1–50,
default 20)}`. Unknown name/path → `400 unknown_question_set`; empty set →
`400 empty_question_set`. Embeds every question (spends budget, minutes).

Response: `{metrics, config, questions, saved_to, note}`. `metrics` has
`overall` and `by_category`/`by_language` (`n`, `hit@1/3/5`), `separation`
and `out_of_scope` blocks; `questions[]` is one row per question
(`{id, language, category, hit_at_1/3/5, is_out_of_scope}`); `config` is the
full run configuration — treat as opaque/diagnostics; `note` reminds you this
is a lab measurement, **not** a benchmark result (only the pinned pair may be
quoted as one).

### 3.13 `POST /diagnostics/harness50` — offline harness (synchronous)
Runs the 50-question offline harness as a subprocess. **Synchronous, up to
900 s** — call with a long client timeout, from a background job, never from
a request handler. `504 harness50_timeout` on overrun. Returns
`{exit_code, output_tail[] (last 20 lines), report (path on the service host,
or null)}`.

### 3.14 `POST /diagnostics/catalog` — live xKiro model catalog
One live call to the xKiro gateway; `502 catalog_failed` on error. Returns
the catalog snapshot as collected by `provider_catalog.collect()`.

### 3.15 `POST /documents` / `GET /documents` / `GET /documents/{id}` / `DELETE /documents/{id}` — the gateway feed target
The service owns a document store (`RAGLAB_DOCUMENTS_DIR`, default
`documents/` next to the code). Pushed documents join the corpus of **every
profile** — the dir is always part of `data_dirs`, whatever `POST /profile`
says — and are indexed by the usual `POST /ingest`. This is the surface an
agent gateway's document feed drives.

**Push** — `POST /documents?id=<doc-id>&index=false`, either body:

* JSON: `{"id"?, "filename", "content", "content_encoding": "text"|"base64"}`
  (`text` = UTF-8 document text; `base64` for PDFs/DOCX binaries)
* multipart/form-data: one file field (the filename's extension picks the
  parser)

Rules: extensions `.txt/.md/.pdf/.docx` only (`400 bad_document_type`);
ids are `[A-Za-z0-9][A-Za-z0-9._-]{0,79}` (`400 bad_document_id` — also your
path-traversal guard); size cap `RAGLAB_MAX_DOCUMENT_BYTES` (default 20 MB →
`413 document_too_large`).

```json
{"result": "created",
 "document": {"id": "rates", "filename": "rates.md", "stored_as": "pushed-rates.md",
              "bytes": 48, "sha256_16": "33ba87bcd627d15b", "version": 1,
              "received_at": "2026-09-15T23:39:20+00:00",
              "updated_at": "2026-09-15T23:39:20+00:00",
              "status": "pending", "chunks_in_index": 0},
 "index_started": false, "notes": [], "ingest": {"state": "idle", "…": "…"},
 "index_hint": "POST /ingest (or push with ?index=true) to (re)build the index"}
```

`201` on create, `200` on re-push. **Versioning is content-based:** same id +
identical bytes → `result: "unchanged"` (idempotent — a feed may retry
freely); different bytes → `version` bumps and the old chunks keep serving
until the next ingest (status `stale`). Stored files are namespaced
`pushed-<id><ext>`, so they can never collide with the repo's own corpus
files; `stored_as` is the `source` name chunks carry (and what you see in
`/inspect`).

**Indexing is explicit:** push a batch, then `POST /ingest` once (re-ingests
are embedding-cache-cheap), or push a single doc with `?index=true` to start
the background job immediately (`index_started` tells you; if a job is
already running or a key is missing, the push still succeeds and `notes[]`
says why indexing did not start).

**List / status:**

```json
{"documents": [{"id": "guide", "…": "…", "version": 1, "status": "indexed",
                "chunks_in_index": 1}],
 "documents_dir": "/app/raglab/documents",
 "index": {"collection": "raglab_app_nvidia_nemotron_3_embed_1b_restructure",
           "indexed_at": "2026-09-15T23:39:21+00:00"}}
```

`status` per document: `pending` (not in the index), `indexed` (chunks
present, ingest newer than the doc), `stale` (doc changed since the last
ingest — old chunks still serving). `GET /documents/{id}` → the same row;
`404 unknown_document` otherwise.

**Delete** — `DELETE /documents/{id}`: removes the stored file AND purges
its chunks from **every** local collection, so the index stays truthful with
no re-ingest:

```json
{"status": "deleted", "id": "guide",
 "chunks_removed": {"raglab_app_nvidia_nemotron_3_embed_1b_restructure": 1},
 "note": "chunks purged from every local collection; no re-ingest needed"}
```

---

## 4. Error catalog

### 4.1 Service errors — `{"detail": {"reason": …}}`

| `reason` | HTTP | Where | Meaning | Client action |
|---|---|---|---|---|
| `unknown_provider` | 400 | `/profile` | Provider not registered (list in body). | Fix the picker. |
| `bad_model` | 400 | `/profile` | Model ID has spaces / empty. | One token, no spaces. |
| `bad_chunking_mode` | 400 | `/profile` | Mode ∉ size/restructure/manual. | — |
| `bad_chunking_value` | 400 | `/profile` | size/overlap not a positive int. | — |
| `bad_data_dirs` | 400 | `/profile` | Dir(s) missing on the service host (list in body). | Paths must exist where the service runs, not in the browser. |
| `bad_document_id` | 400 | `/documents` | Id not in `[A-Za-z0-9][A-Za-z0-9._-]{0,79}` (also the path-traversal guard). | Use a safe id. |
| `bad_document_type` | 400 | `/documents` | Extension not `.txt/.md/.pdf/.docx`. | Convert first. |
| `bad_filename` | 400 | `/documents` | Filename with path separators/control characters, or a multipart push without a named file field. A filename is a name, not a path. | Send a plain name like `rates.md`. |
| `invalid_document_content` | 400 | `/documents` | Empty content, bad base64, or unknown `content_encoding`. | — |
| `invalid_document_push` | 400 | `/documents` | Body is neither valid JSON nor a valid push payload. | — |
| `document_too_large` | 413 | `/documents` | Bytes over `RAGLAB_MAX_DOCUMENT_BYTES` (limit in body). | Split or raise the cap. |
| `unknown_document` | 404 | `/documents/{id}` | No such document id. | `GET /documents` lists the ids. |
| `bad_mode` / `bad_lang_filter` / `bad_query_lang` | 400 | `/search`, `/answer` | Bad enum value. | — |
| `bad_limit` | 400 | `/inspect` | limit ∉ 0–50. | — |
| `unknown_key_env` | 400 | `/keys` | Env var not known (registered list in body). | — |
| `bad_key_value` | 400 | `/keys` | Value has spaces or quotes. | — |
| `placeholder_value` | 400 | `/keys` | Value looks like the `.env.example` template. | Paste the real key. |
| `unknown_question_set` | 400 | `/evaluate` | Unknown name and no such absolute path. | — |
| `empty_question_set` | 400 | `/evaluate` | Set loaded but has no cases. | — |
| `unauthorized` | 401 | every path | Missing/wrong `X-Service-Token` (only when `RAGLAB_SERVICE_TOKEN` is set). | Send the header; CORS preflights are exempt. |
| `profile_switching_disabled` | 403 | `/profile` | This deployment locks switching. | Hide the switcher; configure via env at boot. |
| `empty_index` | 409 | `/search`, `/answer`, `/evaluate` | Collection has no chunks. | `POST /ingest`, poll `/ingest/status`. |
| `retrieval_refused` | 409 | `/search` | Stored chunks ≠ current settings (stale fingerprint). | `POST /ingest?reset=true`. |
| `answer_refused` | 409 | `/answer` | Mid-request ValueError/RuntimeError (safe-redacted `error`). | Show `error`; if persistent, rebuild index. |
| `evaluation_refused` | 409 | `/evaluate` | Same, during evaluation. | — |
| `ingest_already_running` | 409 | `/ingest` | A job is running. | Poll `/ingest/status`. |
| `ingest_in_progress` | 409 | `/search`, `/answer`, `/evaluate`, `/profile`, `/documents/{id}` DELETE | The index is being (re)built — reads and collection mutations are sealed until the job finishes (one writer at a time). | Poll `GET /ingest/status`; retry when `done`. |
| `no_corpus` | 409 | `/inspect` | Data dirs empty/unreadable. | Check `data_dirs` in the profile. |
| `provider_error` | 502 | `/answer` | Upstream provider failed (safe-redacted). | Retry; if persistent, check keys/model. |
| `provider_unreachable` | 502 | `/answer`, `/search`, `/evaluate`, `/embeddings/sanity` | Network-layer failure building or calling a provider — DNS/proxy/timeout/unreachable endpoint. The pinned xKiro path does a live free-price check on first `/answer`, so that is the usual trigger. | Check egress to the provider endpoint; retry once transient issues clear. |
| `catalog_failed` | 502 | `/diagnostics/catalog` | xKiro catalog call failed. | — |
| `missing_api_key` | 503 | `/search`, `/answer`, `/ingest`, `/evaluate`, `/embeddings/sanity` | A key is not set — body carries `slot` + `key_env`. | Key UI (`POST /keys`) or service env. |
| `harness50_timeout` | 504 | `/diagnostics/harness50` | Subprocess exceeded 900 s. | Retry off-peak. |
| `internal_error` | 500 | any | An exception nothing else caught — the safety net. Always JSON (never a plain-text 500), with a safe-redacted `error`; the service log carries the traceback. | Report it with the timestamp; retry idempotent calls. |

### 4.2 Validation errors — `422`
Malformed requests (missing `question`, > 2000 chars, wrong types) return
pydantic's array envelope — `detail` is a **list**, not an object:

```json
{"detail": [{"type": "string_too_short", "loc": ["body", "question"],
             "msg": "String should have at least 1 character", "input": "",
             "ctx": {"min_length": 1}}]}
```

Handle both shapes: `detail.reason` (string) vs `detail` (array).

---

## 5. Integration recipes

### 5.1 Boot sequence (the canonical flow)

```
1. GET /health
   401?                     → the deployment sets RAGLAB_SERVICE_TOKEN:
                              send X-Service-Token on every request
2. keys complete?           ← health.keys / GET /keys      → key UI (§5.2)
3. index.count > 0?         ← health.index                 → else offer ingest
4. POST /ingest             ← background job               → poll /ingest/status
5. enable search + answer   ← ready
```

Derive the UI state from `/health` on every poll:

| `health` says | UI state |
|---|---|
| any needed key `missing` | "Configure API keys" (only if that key's features are offered) |
| `ingest.state == "running"` | progress spinner (+ `stored` when done) |
| `index.count == 0` | "Build the index" button |
| `index.tokenizer_match == false` | "Index is stale — rebuild" (`/ingest?reset=true`) |
| otherwise | search + answer enabled |

### 5.2 Key UI
List from `GET /keys` (env, description, masked presence). Paste → `POST /keys`
(never log the value client-side; the response only ever contains the mask).
Offer "remember" = `persist: true` — with the Docker caveat from §3.5. After
any key change, re-poll `/health`: missing-key errors clear immediately.

### 5.3 Answer rendering
* `status: "answered"` → render `answer`, then claims as a list; under each
  claim, its evidence quotes with a link to the source (document name +
  heading from `sources`). Claim language ≠ quote language is normal.
* `status: "refused"` → render `answer` (a safe, user-facing explanation);
  optionally branch on `reason` for iconography. **Do not retry automatically**
  — refusals are deterministic for the same question + corpus.
* `status: "greeting"` → render `answer`, hide the citation UI.
* `inference_performed: false` → you may badge "answered without AI".
* `cached: true` → instant; fine to re-ask.

### 5.4 Chat
There is **no server-side conversation state**. Build chat by holding the
history in the client and calling `POST /answer` per turn (the service answers
each question against the corpus independently — it does not see prior turns).
For a transcript UI, append each response's claims/sources per turn. The
reference implementation is `raglab/local_front.py` (menu 10); it sends
`X-Service-Token` automatically when `RAGLAB_SERVICE_TOKEN` is set in its
environment.

### 5.5 Provider/model picker
From `GET /models`: group by slot, show `key_set` state per provider, and
allow typing an exact model ID (custom IDs are first-class — see §3.4). After
`POST /profile`: if `collection` ≠ the collection in `/health`, warn that the
index is empty and offer ingest. If `notes[]` is non-empty, display the notes
(experimental-SKU warnings). The pinned pair (NVIDIA nemotron embeddings +
xKiro `qwen/qwen3.8-max:free`) is the only benchmark-attributable pipeline —
`health.profile.pipeline` tells you which one is active.

### 5.6 Long operations
* `POST /ingest` — never await synchronously in a request handler; poll
  `/ingest/status` (1–2 s).
* `POST /diagnostics/harness50` — synchronous up to 900 s; call it from a
  background job with a ≥ 960 s timeout.
* `POST /evaluate` — minutes; same advice.

### 5.7 The document feed (agent-gateway pattern)
The gateway owns its document store and pushes to whichever agent is active.
Against this service that is:

```
for doc in batch:                       # idempotent — retries are free
    POST /documents  {id, filename, content}          # 201/200
POST /ingest                                           # once per batch
poll GET /ingest/status until done                     # 1–2 s interval
GET /documents                                         # statuses: indexed
```

* Use a **stable id** per document (your store's key, sanitized to
  `[A-Za-z0-9][A-Za-z0-9._-]{0,79}`); identical bytes re-pushed are a no-op.
* Push binaries (PDF/DOCX) as multipart file fields or base64 JSON; text
  documents as `content_encoding: "text"`.
* Removal: `DELETE /documents/{id}` — chunks are purged from every
  collection immediately, no re-ingest needed.
* A single late document can go out with `?index=true` to skip the separate
  `/ingest` call.
* In Docker, put `RAGLAB_DOCUMENTS_DIR` (default `/app/raglab/documents`)
  on a volume, or pushed documents are lost when the container is recreated.

---

## 6. Deployment summary

Config is environment-only at boot (12-factor). Full table + Docker notes:
`raglab/SERVICE.md`. The essentials: `RAGLAB_EMBEDDING_PROVIDER/MODEL`,
`RAGLAB_ANSWER_PROVIDER/MODEL`, `RAGLAB_CHUNKING_MODE/SIZE/OVERLAP`,
`RAGLAB_TOP_K`, `RAGLAB_RETRIEVAL_MODE`, `RAGLAB_LANG_FILTER`,
`RAGLAB_NEIGHBOR_RADIUS`, `RAGLAB_DATA_DIRS`,
`RAGLAB_ALLOW_PROFILE_SWITCH` (default `1`; compose ships `1` with the
service token required — keep them together),
`RAGLAB_SERVICE_TOKEN` (require `X-Service-Token` on every request; `RAGLAB_TOKEN` is an accepted alias),
`RAGLAB_DOCUMENTS_DIR` (the pushed-documents store; volume it in Docker),
`RAGLAB_MAX_DOCUMENT_BYTES` (per-push cap, default 20 MB),
`RAGLAB_CORS_ORIGINS` (default `*`), plus provider keys
(`NVIDIA_API_KEY`, `XKIRO_API_KEY`, `GOOGLE_API_KEY`/`GEMINI_API_KEY`,
`KIRA_API_KEY`, …). Invalid combinations fail at boot, not on request 5.

Operational limits that shape the frontend: **one replica** (local ChromaDB +
in-process caches — do not load-balance across replicas), **one ingest at a
time**, no streaming, no webhooks, no multi-tenancy. Not production-ready for
a banking service — supervised pilot standing (see README.md).
