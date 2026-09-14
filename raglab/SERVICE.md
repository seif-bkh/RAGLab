# RAGLab as a standalone service

`raglab/service.py` exposes the lab's runtime as one HTTP microservice — the
unit you deploy next to your other services (gateway, auth, frontend) in a
microservice architecture. It runs the SAME runtime as the interactive console
(`app.py`): both import `raglab/profiles.py`, so the console and the service
can never disagree about which providers, models, collections and validation
rules exist.

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
| GET | `/health` | liveness, active profile, index count + fingerprint/tokenizer match + **`index.stale`** (true when the stored chunks were built with other chunking inputs than the profile produces, with both fingerprints and `index.rebuild`), which keys are set (presence only, never values) |
| GET | `/models` | registered models per provider (embedding + answer slots, + the key env each one needs) |
| GET | `/profile` | the active profile (embedding/answer/chunking/retrieval/corpus), its collection name, switching state |
| POST | `/profile` | switch provider+model per slot, chunking (`mode`/`size`/`overlap`), retrieval knobs, corpus dirs — **only when `RAGLAB_ALLOW_PROFILE_SWITCH=1`** |
| GET | `/keys` | the known key env vars, what each unlocks, masked presence (first 8 chars, never the value) |
| POST | `/keys` | set a key in the service process (`{key_env, value, persist?}`; placeholders/quotes/spaces rejected; `persist=true` also writes the service host's `raglab/.env`) |
| DELETE | `/keys/{env}` | drop a key from the process (and from `raglab/.env` with `?persist=true`) |
| POST | `/search` | retrieval only: `{question, k?, mode?, lang_filter?, query_lang?}` → ranked chunks with scores |
| POST | `/answer` | grounded answer: `{question, k?, mode?, lang_filter?, query_lang?, include_excerpts?, include_diagnostics?}` → claims with verbatim-cited evidence, or a safe refusal. Provider failures come back as `status=error` **with the diagnosis** (`error`, `http_status`, `retry_after_s`, `provider_ok`, `served_model`); `include_diagnostics=true` adds `raw_preview` (the rejected model reply), which production callers can leave off. Greetings ("bonjour", "السلام عليكم") are answered locally with zero calls |
| POST | `/ingest?reset=false` | build/rebuild this profile's index as a background job (embeds every chunk; the cache makes re-runs cheap) |
| GET | `/ingest/status` | what the ingest job is doing / last did |
| GET | `/inspect?limit=` | the chunking preview: documents, chunk/token totals, sample chunks (no model calls) |
| POST | `/chunks/search` | quote-vs-chunk diagnostic: is this text inside ONE chunk (`full` — a verbatim quote can pass the citation gate) or does it cross a boundary (`head`/`tail` in different chunks — it can never validate). No model calls |
| POST | `/embeddings/sanity` | one batched embedding call, 3 phrases (en/fr/ar), cosine-similarity report |
| POST | `/evaluate` | run a question set (`questions.json` / `questions_50.json` / `questions_real.json` or an absolute host path): metrics + per-question outcomes; the full run is saved under `results/` |
| POST | `/diagnostics/harness50` | the offline 50-question harness as a subprocess job (slow — minutes) |
| POST | `/diagnostics/catalog` | the xKiro provider catalog snapshot (read-only, live) |

Semantics worth knowing before you integrate:

* **A refusal is a valid answer.** `/answer` returns HTTP 200 with
  `status=refused` and a `reason` (`insufficient_evidence`,
  `invalid_output`, `private_or_live_request`, …) when the corpus does not
  support the question or the reply failed the verbatim-citation gate.
  That is the product working, not an error.
* **Errors**: `409 empty_index` (POST `/ingest` first), **`409 stale_index`**
  (the index holds chunks built with a different chunk size / overlap / mode
  than the active profile asks for, so every hit would come from the wrong
  segmentation). A stale refusal is structured, never prose to parse:
  `{"reason": "stale_index", "collection": …, "stored": "chunkv4:…",
  "current": "chunkv4:…", "rebuild": "POST /ingest?reset=true …"}`, and
  `GET /health` reports the same mismatch as `index.stale` *before* you ask —
  the front warns and offers the rebuild instead of failing mid-chat. Other
  refusals: `409 answer_refused` / `409 retrieval_refused` /
  `409 evaluation_refused`, `502 provider_error`, `503 missing_api_key` (says
  which env var), `422` request validation.
* **A provider failure is diagnosed, not just reported.** `status=error` with
  `reason=provider_error` (HTTP 200 — the request was valid, the provider
  refused) carries `http_status` and `retry_after_s` when the provider gave
  one, and `error` holds its message. That is what distinguishes a 429 quota
  (wait, or use another model) from a 404/403 (that model ID or key cannot
  generate) — `local_front` prints the matching advice. A `2xx` answer that the
  citation gate rejected is `reason=invalid_output` instead, and its
  `raw_preview` is returned only with `include_diagnostics=true`.
* **The Google profile tries the other free-tier candidates.** The selected
  model is attempted first; if it fails, the next cheapest chat model visible to
  the key is tried (llm_smoke's phase-B behaviour, logged) and `served_model`
  reports which one actually answered — `model` stays the selected ID. Nothing
  is substituted silently.
* **First `/answer` on the xKiro profile** performs the live free-price check
  the supported pipeline requires — expect slightly higher latency once.
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
| `RAGLAB_ALLOW_PROFILE_SWITCH` | enable `POST /profile` | `1` (docker-compose pins `0`) |
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

1. **No auth is built in.** Put the service behind your own gateway / auth
   sidecar. `POST /ingest` and (if enabled) `POST /profile` spend money and
   change behavior — they must not be publicly reachable as-is.
2. **Single replica by design.** The index is a local ChromaDB directory and
   ingestion runs in one background thread per process. Run one replica per
   profile; scale reads at the gateway, not by sharing nothing between
   replicas. If you need horizontal scale, front N single-profile replicas
   (one per collection) with routing, or move to a shared vector store — that
   is a different architecture than this lab ships.
3. **Each profile owns its collection** (`raglab_app_<provider>_<model>_<chunking>`),
   exactly like the console: switching providers never corrupts another
   profile's vectors, and a fingerprint guard refuses retrieval over stale
   chunks (`409 stale_index` with both fingerprints, visible in advance as
   `index.stale` in `/health`; rebuild with `/ingest?reset=true`). Note the
   collection name does NOT carry the size/overlap: changing chunk size in
   `POST /profile` (or `RAGLAB_CHUNK_SIZE_TOKENS` on restart) invalidates the
   existing index on purpose — that is the guard doing its job, not a bug.
4. **Non-pinned answer models are experimental surfaces**: the live free-price
   check and benchmark attribution belong to the pinned xKiro SKU only. A
   provider error is never allowed to read as an answer: the response says
   `provider_ok=false` and the console prints the provider's own message —
   a green-looking front on a dead key is exactly what this guards against.
5. **`POST /keys` with `persist=true` inside Docker** writes the container's
   own `raglab/.env` — not the bind-mounted one you copied keys from — so it
   is lost when the container is recreated. Give compose the keys via
   `env_file` and treat `/keys` as a process-level convenience.
6. **Same standing as the rest of the lab**: not production-ready for a
   banking service (see README.md) — suitable for a supervised pilot.

## Tests

`python -m unittest -v test_service` (offline: stubbed embeddings + injected
chat client; no network, no keys). It is part of CI via `run_tests.sh`. One
test case boots a REAL uvicorn server on an ephemeral port and runs
`local_front.py`'s state-aware smoke suite against it over actual HTTP — with
no keys and an empty index, so the tested behavior is the full refusal
contract (503/403/422) plus greetings, exactly the state a fresh deployment
is in. A second one boots the service against an index built at another chunk
size (the reported "STALE" case) and requires the stale branch of the same
suite to pass over HTTP: `409 stale_index` with both fingerprints and the
rebuild, and no live call. `StaleIndexTest` pins the contract directly —
`index.stale` flips with the profile, `/search`, `/answer` and `/evaluate`
refuse with the structured reason, the front's helpers turn it into the right
command, and a rebuild heals it. `AnswerProviderFailureTest` and
`test_nvidia_pipeline.GoogleFreeTier` pin the provider-failure contract with a
stubbed Gemini path: a 429 reaches the client as `http_status=429` +
`retry_after_s` (with the front's advice), a dead model falls back to the next
free-tier candidate and reports it as `served_model`, and `raw_preview` appears
only when asked for. The rest of the cases cover the
console-parity endpoints (keys round-trip and redaction, inspect, chunk-search
verdicts, embedding sanity, evaluate, profile switching with chunking/corpus
validation).
