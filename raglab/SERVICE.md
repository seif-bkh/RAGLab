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

What is deliberately NOT in the service: the console's menus and prompts,
`app_state.json` persistence, the benchmark harnesses, the diagnostics
utilities. Configuration is 12-factor — environment at boot, keys from the
environment, no interactive anything.

## Quick start

```bash
cd raglab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-service.txt
cp .env.example .env            # fill NVIDIA_API_KEY / XKIRO_API_KEY / ...
python -m uvicorn service:app --host 0.0.0.0 --port 8000
# interactive OpenAPI docs: http://localhost:8000/docs
```

Test it from another terminal (pure HTTP client — imports nothing from the
lab, so it exercises exactly the boundary another microservice would):

```bash
python local_front.py                   # state-aware smoke suite over every endpoint
python local_front.py --ingest          # build the index, wait for the job
python local_front.py --ask "What is Murabaha?"
python local_front.py --interactive     # small REPL over the API
```

Docker (from the repo root):

```bash
cp raglab/.env.example raglab/.env   # fill keys; the image never bakes them
docker compose up --build            # http://localhost:8000/docs
```

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | liveness, active profile, index count + fingerprint/tokenizer match, which keys are set (presence only, never values) |
| GET | `/models` | registered models per provider (embedding + answer slots) |
| GET | `/profile` | the active profile, its collection name, switching state |
| POST | `/profile` | switch embedding/answer provider+model (or retrieval knobs) at runtime — **only when `RAGLAB_ALLOW_PROFILE_SWITCH=1`** |
| POST | `/search` | retrieval only: `{question, k?, mode?, lang_filter?, query_lang?}` → ranked chunks with scores |
| POST | `/answer` | grounded answer: `{question, k?, include_excerpts?}` → claims with verbatim-cited evidence, or a safe refusal. Greetings ("bonjour", "السلام عليكم") are answered locally with zero calls |
| POST | `/ingest?reset=false` | build/rebuild this profile's index as a background job (embeds every chunk; the cache makes re-runs cheap) |
| GET | `/ingest/status` | what the ingest job is doing / last did |

Semantics worth knowing before you integrate:

* **A refusal is a valid answer.** `/answer` returns HTTP 200 with
  `status=refused` and a `reason` (`insufficient_evidence`,
  `invalid_output`, `private_or_live_request`, …) when the corpus does not
  support the question or the reply failed the verbatim-citation gate.
  That is the product working, not an error.
* **Errors**: `409 empty_index` (POST `/ingest` first), `409 retrieval_refused`
  (stale index vs current settings — rebuild it), `502 provider_error`,
  `503 missing_api_key` (says which env var), `422` request validation.
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
| `RAGLAB_ALLOW_PROFILE_SWITCH` | enable `POST /profile` | `0` |
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
   chunks (`409 retrieval_refused`, rebuild with `/ingest?reset=true`).
4. **Non-pinned answer models are experimental surfaces**: the live free-price
   check and benchmark attribution belong to the pinned xKiro SKU only.
5. **Same standing as the rest of the lab**: not production-ready for a
   banking service (see README.md) — suitable for a supervised pilot.

## Tests

`python -m unittest -v test_service` (offline: stubbed embeddings + injected
chat client; no network, no keys). It is part of CI via `run_tests.sh`, and
its last test case boots a REAL uvicorn server on an ephemeral port and runs
`local_front.py`'s smoke suite against it over actual HTTP — with no keys and
an empty index, so the tested behavior is the full refusal contract
(503/403/422) plus greetings, exactly the state a fresh deployment is in.
