# RAGLab as a standalone service

`raglab/service.py` exposes the lab's runtime as one HTTP microservice — the
unit you deploy next to your other services (gateway, auth, frontend) in a
microservice architecture. It runs the SAME runtime as the interactive console
(`app.py`): both import `raglab/profiles.py`, so the console and the service
can never disagree about which providers, models, collections and validation
rules exist.

**Building a client against this service?** The full HTTP contract for client
teams — request/response schemas with real examples, the state model, the
error catalog, and integration recipes — is `raglab/CONTRACT.md`. This file
is the operator's/deployer's doc. **Designing the app's UI?** The
screen-by-screen frontend/UX blueprint (with the permission matrix) is
`raglab/FRONTEND.md`. **Building the settings screens?** The exact-request
cookbook for every config mutation is `raglab/COOKBOOK.md`.

```
            ┌──────────────┐
            │ your gateway │  (auth, rate limits, CORS — NOT included here)
            └──────┬───────┘
                   │ HTTP/JSON
          ┌────────▼─────────┐     ┌──────────────────────────────┐
          │  raglab service   │────▶│ NVIDIA / xKiro / Google /    │
          │  service.py       │     │ Kira (embeddings + chat)     │
          │  ├ profiles.py    │     └──────────────────────────────┘
          │  ├ chunker/store  │──▶ local ChromaDB index + caches (volume)
          │  └ retrieval/     │
          │    answer (gate)  │
          └───────────────────┘
```

What is deliberately NOT in the service: the console's interactive UX
(that now lives in `local_front.py`, which drives THIS service over HTTP),
`app_state.json` persistence, and ad-hoc model plumbing. Configuration is
12-factor — environment at boot, keys from the environment or `POST /keys`,
no interactive anything inside the service process.

## Quick start

```bash
cd raglab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-service.txt
cp .env.example .env            # fill NVIDIA_API_KEY / XKIRO_API_KEY / ...
python -m uvicorn service:app --host 0.0.0.0 --port 8000
# interactive OpenAPI docs: http://localhost:8000/docs
```

Test it from another terminal — `local_front.py` is the console's twin over
REST: same 13 menus as `app.py` (status/doctor, providers & models, API keys,
inspect, chunk search, sanity, ingest, retrieval, answer, chat, evaluate,
settings, diagnostics), every action an endpoint call. It imports nothing
from the lab (pure stdlib HTTP), so it exercises exactly the boundary another
microservice would:

```bash
python local_front.py                   # the console — menus, prompts, everything
python local_front.py --smoke           # state-aware smoke suite over every endpoint
python local_front.py --status          # one-shot doctor (profile, index, keys)
python local_front.py --ingest          # build the index, wait for the job
python local_front.py --ask "What is Murabaha?"
python local_front.py --search "murabaha"   # retrieval only, no chat model
python local_front.py --base-url http://raglab:8000   # against the compose stack
```

The front is stateless except `front_state.json` (its custom model-ID memory,
capped at 20 IDs per provider). The profile — providers, models, chunking,
retrieval, corpus — lives in the service; keys set from the front live in the
service process (`persist=true` also writes the service host's `raglab/.env`).

Docker (from the repo root):

```bash
cp raglab/.env.example raglab/.env   # fill keys; the image never bakes them
docker compose up --build            # http://localhost:8000/docs
```

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | liveness, active profile, index count + fingerprint/tokenizer match, which keys are set (presence only, never values) |
| GET | `/models` | registered models per provider (embedding + answer slots, + the key env each one needs) |
| GET | `/profile` | the active profile (embedding/answer/chunking/retrieval/corpus), its collection name, switching state |
| GET | `/config` | self-description for consoles: chat/embedding models, vector dimension, editability + capabilities map — the probe a generic adapter renders |
| POST | `/profile` | switch provider+model per slot (a model you name explicitly is used verbatim — custom IDs included; a provider-only switch preselects and never carries a model across providers), chunking (`mode`/`size`/`overlap`), retrieval knobs, corpus dirs — **only when `RAGLAB_ALLOW_PROFILE_SWITCH=1`** |
| GET | `/keys` | the known key env vars, what each unlocks, masked presence (first 8 chars, never the value) |
| POST | `/keys` | set a key in the service process (`{key_env, value, persist?}`; placeholders/quotes/spaces rejected; `persist=true` also writes the service host's `raglab/.env`) |
| DELETE | `/keys/{env}` | drop a key from the process (and from `raglab/.env` with `?persist=true`) |
| POST | `/documents` | push one document into the service's own store (multipart or JSON; content-based versioning; `?index=true` to ingest right away) — the gateway feed target |
| GET | `/documents` | the pushed documents + per-doc index status (`pending`/`indexed`/`stale`) |
| GET | `/documents/{id}` | one document's row |
| DELETE | `/documents/{id}` | remove a document and purge its chunks from every collection |
| POST | `/search` | retrieval only: `{question, k?, mode?, lang_filter?, query_lang?}` → ranked chunks with scores |
| POST | `/answer` | grounded answer: `{question, k?, mode?, lang_filter?, query_lang?, include_excerpts?}` → claims with verbatim-cited evidence, or a safe refusal. Greetings ("bonjour", "السلام عليكم") are answered locally with zero calls |
| POST | `/ingest?reset=false` | build/rebuild this profile's index as a background job (embeds every chunk; the cache makes re-runs cheap) |
| GET | `/ingest/status` | what the ingest job is doing / last did |
| GET | `/inspect?limit=` | the chunking preview: documents, chunk/token totals, sample chunks (no model calls) |
| GET | `/chunks` | the stored-chunk browser: paginated chunk listing with a per-document rollup (chunk counts, token ranges, languages) — exactly what retrieval supplies, for rating the chunking strategy |
| GET | `/chunks/{chunk_id}` | one stored chunk in full: text, metadata, both neighbors and the character overlap with the previous chunk |
| POST | `/chunks/search` | quote-vs-chunk diagnostic: is this text inside ONE chunk (`full` — a verbatim quote can pass the citation gate) or does it cross a boundary (`head`/`tail` in different chunks — it can never validate). No model calls |
| POST | `/embeddings/sanity` | one batched embedding call, 3 phrases (en/fr/ar), cosine-similarity report |
| POST | `/evaluate` | run a question set (`questions.json` / `questions_50.json` / `questions_v2.json` / `questions_real.json` or an absolute host path): metrics + per-question outcomes; the full run is saved under `results/` |
| POST | `/diagnostics/harness50` | the offline 50-question harness as a subprocess job (slow — minutes) |
| POST | `/diagnostics/catalog` | the xKiro provider catalog snapshot (read-only, live) |

Semantics worth knowing before you integrate:

* **A refusal is a valid answer.** `/answer` returns HTTP 200 with
  `status=refused` and a `reason` (`insufficient_evidence`,
  `invalid_output`, `unsourced_number`, `private_or_live_request`, …) when
  the corpus does not support the question or the reply failed the citation
  gate. `invalid_output`/`unsourced_number` responses also carry `error` and
  `raw_preview` (the model's rejected reply, PII-scrubbed) as diagnostics.
  That is the product working, not an error.
* **The gate verifies numbers, not just quotes.** Every digit-form number in
  a claim must appear in that claim's evidence quotes (French/English/Arabic
  decimal and thousands forms normalize to the same digits) — a model that
  computes, converts or renames a figure is refused as `unsourced_number`.
* **Outputs are PII-scrubbed (post-gate).** `/answer` fields and `/search`
  hit texts replace emails, phone numbers, RIB/IBAN and CIN numbers with
  `[EMAIL]`/`[PHONE]`/`[RIB]`/`[CIN]` placeholders; the gate still validates
  the raw verbatim text. Diagnostics (`/inspect`, `/chunks/search`) show raw
  text on purpose (admin-facing).
* **One writer at a time, strictly.** While the ingest job runs, `/search`,
  `/answer`, `/evaluate`, `POST /profile` and `DELETE /documents/{id}` return
  `409 ingest_in_progress` (poll `/ingest/status`; greetings still work).
  No request ever races the job's writes.
* **Errors**: `401 unauthorized` (missing `X-Service-Token` when
  `RAGLAB_SERVICE_TOKEN` is set), `409 empty_index` (POST `/ingest` first),
  `409 ingest_in_progress` (the job is running — poll, then retry),
  `409 retrieval_refused` (stale index vs current settings — rebuild it),
  `500 internal_error` (the safety net: an unhandled exception still comes
  back as JSON, never FastAPI's plain-text 500; the log has the traceback),
  `502 provider_error`, `502 provider_unreachable` (network-layer failure —
  DNS/proxy/timeout, incl. the live free-price check on the first pinned-xKiro
  `/answer`), `503 missing_api_key` (says which env var),
  `422` request validation.
* **First `/answer` on the xKiro profile** performs the live free-price check
  the supported pipeline requires — expect slightly higher latency once. If
  the gateway cannot be reached, the response is `502 provider_unreachable`
  (never a bare 500) with the underlying network error in the body.
* Every response is redacted: provider errors pass through `safe_error`, keys
  are never echoed.

### Examples

```bash
curl -s localhost:8000/health | jq
curl -s -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"question": "What is Murabaha?", "k": 5}' | jq '.hits[].source'
curl -s -X POST localhost:8000/answer \
  -H 'content-type: application/json' \
  -d '{"question": "ما هي المرابحة؟"}' | jq '{status, reason, answer}'
curl -s -X POST 'localhost:8000/ingest?reset=true' ; curl -s localhost:8000/ingest/status
curl -s localhost:8000/inspect?limit=2 | jq '.chunks'
curl -s -X POST localhost:8000/keys -H 'content-type: application/json' \
  -d '{"key_env": "NVIDIA_API_KEY", "value": "nvapi-...", "persist": false}' | jq
```

## Environment variables

Profile (all optional — defaults are the supported pipeline pair:
NVIDIA `nvidia/nemotron-3-embed-1b` embeddings + xKiro
`qwen/qwen3.8-max:free` answers):

| Variable | Meaning | Default |
|---|---|---|
| `RAGLAB_EMBEDDING_PROVIDER` / `RAGLAB_EMBEDDING_MODEL` | embedding slot | `nvidia` / `nvidia/nemotron-3-embed-1b` |
| `RAGLAB_ANSWER_PROVIDER` / `RAGLAB_ANSWER_MODEL` | answer slot | `xkiro` / `qwen/qwen3.8-max:free` |
| `RAGLAB_CHUNKING_MODE` | `size` / `restructure` / `manual` | `restructure` |
| `RAGLAB_CHUNK_SIZE_TOKENS` / `RAGLAB_CHUNK_OVERLAP_TOKENS` | size-mode chunking | `220` / `40` |
| `RAGLAB_TOP_K` | chunks retrieved per question | `5` |
| `RAGLAB_RETRIEVAL_MODE` | `vector` / `rrf` / `blend` | `vector` |
| `RAGLAB_LANG_FILTER` | restrict retrieval to `ar`/`fr`/`en` | none |
| `RAGLAB_NEIGHBOR_RADIUS` | widen hits with adjacent chunks (0–2) | `0` |
| `RAGLAB_DATA_DIRS` | comma-separated corpus dirs | `../docs + raglab/data/` |
| `RAGLAB_DOCUMENTS_DIR` | the pushed-documents store (always part of the corpus; volume it in Docker) | `raglab/documents/` |
| `RAGLAB_MAX_DOCUMENT_BYTES` | per-push size cap | `20971520` (20 MB) |
| `RAGLAB_ALLOW_PROFILE_SWITCH` | enable `POST /profile` | `1` (compose ships `1` with the token required — the app console needs it) |
| `RAGLAB_SERVICE_TOKEN` | require `X-Service-Token` on every request (constant-time check, `401 unauthorized` otherwise); `RAGLAB_TOKEN` is an accepted alias | unset = open (local/dev) |
| `RAGLAB_CORS_ORIGINS` | comma-separated allowed origins | `*` |
| `RAGLAB_CACHE_DIR` | relocate embedding/answer caches (Docker volume) | next to the code |
| `RAGLAB_HOST` / `RAGLAB_PORT` | used by `python service.py` | `0.0.0.0` / `8000` |

Keys (which ones are needed depends on the profile): `NVIDIA_API_KEY`,
`XKIRO_API_KEY`, `GOOGLE_API_KEY`/`GEMINI_API_KEY`, `KIRA_API_KEY`,
`JINA_API_KEY`, `OPENAI_API_KEY`, `COHERE_API_KEY`, `VOYAGE_API_KEY` — see
`.env.example` for what each unlocks.

Invalid combinations fail at boot with a plain-English message (container
start), not on the fifth request.

## Deployment notes — read before exposing this

1. **No user auth is built in — but there is a shared-secret token.** Setting
   `RAGLAB_SERVICE_TOKEN` requires every request to carry it in the
   `X-Service-Token` header (constant-time compare; CORS preflights exempt).
   That removes the footgun of `POST /ingest`, `POST /profile` and
   `/keys` being reachable by anyone who can reach the port — it is NOT user
   authentication. Still put the service behind your own gateway / auth
   sidecar; `POST /ingest` and (if enabled) `POST /profile` spend money and
   change behavior.
2. **Single replica by design.** The index is a local ChromaDB directory and
   ingestion runs in one background thread per process. Run one replica per
   profile; scale reads at the gateway, not by sharing nothing between
   replicas. If you need horizontal scale, front N single-profile replicas
   (one per collection) with routing, or move to a shared vector store — that
   is a different architecture than this lab ships.
3. **Each profile owns its collection** (`raglab_app_<provider>_<model>_<chunking>`),
   exactly like the console: switching providers never corrupts another
   profile's vectors, and a fingerprint guard refuses retrieval over stale
   chunks (`409 retrieval_refused`, rebuild with `/ingest?reset=true`).
4. **Non-pinned answer models are experimental surfaces**: the live free-price
   check and benchmark attribution belong to the pinned xKiro SKU only.
5. **`POST /keys` with `persist=true` inside Docker** writes the container's
   own `raglab/.env` — not the bind-mounted one you copied keys from — so it
   is lost when the container is recreated. Give compose the keys via
   `env_file` and treat `/keys` as a process-level convenience.
6. **Same standing as the rest of the lab**: not production-ready for a
   banking service (see README.md) — suitable for a supervised pilot.

## Tests

`python -m unittest -v test_service` (offline: stubbed embeddings + injected
chat clients; no network, no keys). It is part of CI via `run_tests.sh`. Its
last test case boots a REAL uvicorn server on an ephemeral port and runs
`local_front.py`'s state-aware smoke suite against it over actual HTTP — with
no keys and an empty index, so the tested behavior is the full refusal
contract (503/403/422) plus greetings, exactly the state a fresh deployment
is in — and a second case does the same behind an `X-Service-Token`. The rest
of the cases cover the console-parity endpoints directly (keys round-trip and
redaction, inspect, chunk-search verdicts, embedding sanity, evaluate,
profile switching with chunking/corpus validation), the documents API
(push/versioning/statuses/purge, multipart + JSON + base64), the output
guards (PII scrub after the gate, `unsourced_number` refusals) and the auth
middleware.
