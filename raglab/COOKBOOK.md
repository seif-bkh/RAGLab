# RAGLab config-editing cookbook — exact requests for the app's settings UI

**Audience:** the app (and the AI agent building it) that edits this agent's
configuration: models, chunking, retrieval, corpus, API keys, documents,
indexing. This is the copy-paste companion to `CONTRACT.md` (behavior truth)
and `FRONTEND.md` (screens/UX). Every request below is real; every response
shape is exact. Auth first: if the deployment sets a token, **every** request
needs the `X-Service-Token` header.

**One-call probe:** `GET /config` self-reports everything a console needs —
`chat_model`, `embedding_model`, `vector_dimension`, editability flags and a
`capabilities` map naming the call for every mutation. Render that first.

**Golden rule of this agent's config:** the profile lives in the SERVICE.
The UI never stores config locally — it reads it, edits it via `POST
/profile`, and derives consequences (collection change → offer reindex)
from the responses. A page reload must always resume from the service.

---

## 0. Readiness probe — before rendering any editable control

```bash
curl -s localhost:8000/profile
```
```json
{"profile": {"…": "…"},
 "collection": "raglab_app_nvidia_nemotron_3_embed_1b_restructure",
 "pipeline": "supported pipeline — the same model pair main.py answer uses",
 "switching": "enabled"}
```

Gate the settings UI on `switching`:
* `"enabled"` → render editors normally.
* `"disabled (set RAGLAB_ALLOW_PROFILE_SWITCH=1)"` → render read-only + a
  deployment note. Any `POST /profile` returns `403
  profile_switching_disabled`. Fix is on the SERVICE side, never the UI:
  `RAGLAB_ALLOW_PROFILE_SWITCH=1` (local uvicorn default is enabled;
  docker-compose ships it enabled with the token required).
  If you deployed with an older compose file, redeploy — that pin was `0`.

## 1. Where every field comes from (the read assembly)

| Field (settings screen) | Source | Notes |
|---|---|---|
| current embedding/answer model, chunking, retrieval, data_dirs | `GET /profile` → `profile` | the one call for current values |
| switching enabled? | `GET /profile` → `switching` | gate editors on it |
| collection + pipeline badge | `GET /profile` → `collection`, `pipeline` | also in `/health` |
| available providers + their models + key state | `GET /models` | registries = suggestions, custom IDs allowed |
| key presence + masked values | `GET /keys` | values are NEVER returned |
| index count / stale / ingest state | `GET /health` → `index`, `ingest` | the status bar |
| documents + per-doc status | `GET /documents` | |

`/health` drives the chrome (FRONTEND.md §4.1); `/profile` + `/models` +
`/keys` load when the user opens Settings.

## 2. Mutations — request, response, side effects

All mutations are `POST /profile` with **only the changed keys** (anything
you don't send stays unchanged), except keys (`/keys`) and documents
(`/documents`). All responses return the FULL updated profile — update your
local state from the response, don't patch.

### 2.1 Switch the embedding provider/model — ⚠ index changes
```json
{"embedding": {"provider": "nvidia", "model": "nvidia/nemotron-3-embed-1b"}}
```
Response: `{profile, collection, index_note, notes[]}`. `provider` must be in
`GET /models` (else `400 unknown_provider`); `model` is honored verbatim —
custom IDs allowed, one token, no spaces (`400 bad_model`). Provider-only
(`{"embedding": {"provider": "jina"}}`) keeps the current model if that
provider offers it, else its first registered model.
**Side effect:** `collection` in the response ≠ the one in `/health` → the
new profile's index is EMPTY → show "Build the index for this profile" (§2.8).

### 2.2 Switch the answer provider/model — ✅ no reindex needed
```json
{"answer": {"provider": "kira", "model": "glm-5.3-free"}}
```
Same rules as 2.1. The embedding slot is untouched, so the collection and
index stay valid — **switching the answer model never requires reindexing.**
Non-pinned xKiro models (anything but `qwen/qwen3.8-max:free`) return
`notes: ["non-pinned xKiro SKUs are EXPERIMENTAL here: …"]` — surface them
as a warning banner. Custom IDs are first-class:
```json
{"answer": {"provider": "kira", "model": "glm-5.3-free-0916"}}
```

### 2.3 Chunking — ⚠ fingerprint changes → stale index
```json
{"chunking": {"mode": "size", "size": 320, "overlap": 60}}
```
`mode` ∈ `size|restructure|manual` (`400 bad_chunking_mode`); `size` and
`overlap` positive ints (`400 bad_chunking_value`). The collection NAME only
contains the mode, but the stored fingerprint contains everything — after a
chunking change, the first retrieval returns `409 retrieval_refused` (stale)
until you rebuild with reset (§2.8). Tell the user this BEFORE they save.

### 2.4 Retrieval knobs — ✅ immediate, no rebuild
```json
{"retrieval": {"top_k": 8, "mode": "rrf", "lang_filter": "ar", "neighbor_radius": 1}}
```
`top_k` 1–20, `mode` ∈ `vector|rrf|blend`, `lang_filter` `null|ar|fr|en`,
`neighbor_radius` 0–2. **Out-of-range values are silently ignored by the
API** — validate client-side so the UI never lies. Affects the next query;
nothing to rebuild. (`/search` and `/answer` also accept per-request
`mode`/`lang_filter`/`query_lang` overrides — use them for the "advanced"
disclosure on the Ask screen.)

### 2.5 Corpus directories
```json
{"data_dirs": ["/app/docs", "/app/raglab/data"]}
```
Every path must exist **on the service host** (`400 bad_data_dirs` with the
missing list — paths in the browser don't count). The service's documents
dir is ALWAYS part of the corpus whatever you send — don't try to remove it.
New dirs only join the index after a rebuild.

### 2.6 API keys
```json
POST /keys   {"key_env": "NVIDIA_API_KEY", "value": "nvapi-…", "persist": false}
DELETE /keys/NVIDIA_API_KEY?persist=false
```
`key_env` must be known (`400 unknown_key_env` — the registered list is in
the response); the value cannot contain spaces/quotes (`400 bad_key_value`)
or look like the `.env.example` template (`400 placeholder_value`). Responses
carry only the mask (`"masked": "nvapi-12…"`). `persist: true` also writes
the service host's `raglab/.env` (survives restarts, NOT container
recreations — in Docker, compose `env_file` is the durable path). Setting or
deleting a key resets the cached embedder/generator immediately. Test button:
`POST /embeddings/sanity` (one batched call, 3 phrases, cosine report).

### 2.7 Documents (the feed) — summary
Push: `POST /documents` JSON `{id?, filename, content, content_encoding:
"text"|"base64"}` or multipart file field; client-supplied `id` (charset
`[A-Za-z0-9][A-Za-z0-9._-]{0,79}`) = idempotent retries; same bytes →
`unchanged`, different bytes → version bump. Delete: `DELETE
/documents/{id}` (file + chunks purged from every collection, no rebuild).
Full semantics: CONTRACT.md §3.15.

### 2.8 Indexing — which build do I need?
| Situation | Call |
|---|---|
| new/changed profile collection (embedding or chunking.mode changed) | `POST /ingest` |
| stale fingerprint (`409 retrieval_refused`, chunk size/overlap changed) | `POST /ingest?reset=true` |
| pushed a batch of documents | `POST /ingest` |
| one late document | re-push with `?index=true` |
| cold start / "declaration" rebuild everything | `POST /ingest?reset=true` |

All are background jobs: response `{"state": "accepted", …}` → poll
`GET /ingest/status` every 1–2 s until `done` (or `error` — the error names
the remedy). During the job the app is SEALED (FRONTEND.md §4.3): reads
return `409 ingest_in_progress`.

### 2.9 Understanding & sufficiency layers (1.3.0 — env-gated, additive-only)

Three independent service-side switches; none changes a pre-existing field.
**Owner activation 2026-10-02** («شغلها», after the measured trial in
audits/PHASE7_EXPERIMENTS.md): the first two now default **on** and can be
turned off with `0`; `RELATIONAL_EXPANSION_ENABLED` stays off:

| Env var | Effect when `1` |
|---|---|
| `SUFFICIENCY_FIELDS_ENABLED` | `/answer` gains `evidence_status` (كافٍ/غير كافٍ/متعارض/غير محسوم), `requirements_covered/missing`, `conflicts`, `refusal_reason`. |
| `ANSWER_SUFFICIENCY_COMMITMENT` | Insufficient-and-uncovered questions are refused **before any model call** (`reason: evidence_insufficient` + `referral`); partially covered questions are downgraded to one bounded regeneration over the covered micro-questions (`partial: true`). |
| `RELATIONAL_EXPANSION_ENABLED` | Retrieval expansion along the Phase-4 relations under the declared intent policy (measured neutral on both arms). |

Worked sequence (fields + audit):

```
# 1) ask with the fields on (the default since the 2026-10-02 activation;
#    SUFFICIENCY_FIELDS_ENABLED=0 restores the pre-activation response)
POST /answer  {"question": "ما هي عملية المرابحة على معنى القانون عدد 48 لسنة 2016؟"}
→ {..., "evidence_status": "كافٍ", "requirements_covered": ["definition_or_purpose_unit"],
     "requirements_missing": []}

# 2) an out-of-corpus question names its missing requirement
POST /answer  {"question": "كيف احجز تذكرة طائرة من تونس الى دبي؟"}
→ {..., "evidence_status": "غير كافٍ", "requirements_missing": ["procedural_evidence"],
     "refusal_reason": "evidence absent for: procedural_evidence"}

# 3) the audit trail (always on; questions are PII-scrubbed before storage)
GET /audit?limit=5
→ {"retention": 500, "entries": [{"trace_id": "9f1c0a2b7d3e", ...}]}

# 4) stable unit ids on cited law chunks (always on in 1.3.0)
POST /answer  {"question": "...", "include_excerpts": false}
→ sources[0].unit_id == "loi-2016-48:art052"   (deep-linkable across chunking changes)
```

### 2.10 Which free models does MY key see? (models_probe — 2026-10-02)

A read-only probe of the three answer providers' catalogs — no documents,
no queries, no inference, one catalog GET per configured key:

```
python models_probe.py            # human table + results/models_probe/probe.json
python models_probe.py --json     # JSON only
```

Honest semantics per provider (a listing proves **visibility**, never price):

| Provider | Key env | The probe reports |
|---|---|---|
| xKiro | `XKIRO_API_KEY` | ids + `verified_free` = access_tier `free` AND every declared price exactly 0 (fail closed) |
| NVIDIA | `NVIDIA_API_KEY` | ids visible to your key; the catalog carries **no pricing** — free usage is account/credit-based (build.nvidia.com) |
| Google | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `chat_capable_models` (support `generateContent`); the free tier is rate-limited and account-level |

A missing key is a per-provider status line, not an error; HTTP errors are
reported by type only (a gateway may echo credentials in error bodies — raw
bodies are never exported). `GET /models` on the running service lists the
REGISTERED models per provider plus whether each key is set — the probe is
the live-discovery counterpart.

### 2.11 Practical questions, re-expressed technically (Phase 8 — gated OFF)

A practical, non-technical question ("can i get a financement to open a
pub?") used to be refused outright. With the gate ON it triggers ONE
bounded interrogation call: the model answers descriptive questions about
the REQUEST (never answering it), choosing topics verbatim from the
corpus's own topic map; the technical paraphrase is re-checked by the same
deterministic engines and — if the corpus can answer it — answered with
full disclosure:

```
# start the service with the gate on (or set it in raglab/.env)
REPHRASE_INTERROGATION_ENABLED=1 python -m uvicorn service:app

POST /answer {"question": "can i get a financement to open a pub?"}
→ { "status": "answered",
    "understood_as": "ما هي صيغ التمويل التي يوفرها البنك للمشاريع؟",
    "original_question": "can i get a financement to open a pub?",
    "interrogation": {"topics": [...], "confidence": 0.8, ...},
    "claims": [...cited as always...] }
```

`local_front` renders the disclosure line «فُهم طلبك كـ …». If the corpus
still cannot answer the paraphrase, the refusal stays honest — with a
better referral (the nearest corpus topics). Design, guarantees and the
measurement protocol: `audits/PHASE8_INTERROGATION.md`.

## 3. The two side-effect rules (memorize these)

1. **Answer-model or retrieval change → nothing to rebuild.** Next query
   just uses it.
2. **Embedding-model or chunking change → the index no longer matches.**
   Compare `collection` (or expect `409 retrieval_refused`) → offer the
   rebuild. The embedding cache makes rebuilds cheap, not free.

## 4. Worked sequences

### First-run wizard
```
GET /health        → keys missing?  → Settings → API keys (2.6) → Test (sanity)
                   → index 0?       → Documents → Build index (2.8) → poll → done
GET /profile       → switching enabled → unlock editors
```

### Change the answer model (the common case)
```
POST /profile {"answer": {"provider": "kira", "model": "glm-5.3-free"}}
→ 200 {profile, collection (UNCHANGED), notes}
→ toast "Answer model: kira/glm-5.3-free" (+ experimental banner if notes)
→ nothing else. Ask screen works immediately against the existing index.
```

### Change chunk size (or embedding model)
```
POST /profile {"chunking": {"size": 320}}
→ 200; collection may be unchanged BUT the fingerprint is now different
→ banner: "Chunking changed — the index must be rebuilt"
POST /ingest?reset=true  → poll /ingest/status → done (stored: N chunks)
→ clear banner; statuses on GET /documents go back to "indexed"
```

### Replace a document + reindex in one call
```
POST /documents?id=tarif-2026&index=true   {"filename": "tarif.md", "content": "…new bytes…"}
→ 200 {"result": "replaced", "document": {"version": 2, "status": "stale", …},
       "index_started": true}
→ poll /ingest/status → done → GET /documents/tarif-2026 → "indexed"
```
Note: `?index=true` only starts the job when the push CHANGED something
(`created`/`replaced`) — an `unchanged` push (identical bytes) never
triggers it. If you already pushed without the flag, just `POST /ingest`.

## 5. Editing-time failures the UI must handle

| Error | Cause | UI action |
|---|---|---|
| `401 unauthorized` | missing/wrong `X-Service-Token` | session/gateway problem — re-auth |
| `403 profile_switching_disabled` | deployment locked switching | read-only mode + deployment note (§0) |
| `400 unknown_provider` | provider not in `/models` | fix the picker |
| `400 bad_model` | spaces/empty model ID | inline validation: one token |
| `400 bad_chunking_mode` / `bad_chunking_value` | bad mode / non-positive int | inline validation |
| `400 bad_data_dirs` | path missing on the SERVICE host | show the missing list |
| `409 ingest_in_progress` | a job is running | enter sealed mode; poll; retry after `done` |
| `409 retrieval_refused` | index stale vs settings | offer `POST /ingest?reset=true` |

Cross-check anything not here against CONTRACT.md §4 (the full error
catalog) — and `/openapi.json` is always the field-name truth.
