# RAGLab agent console — frontend & UX blueprint

**Audience:** the fullstack team building the app that drives this agent.
**Behavior truth:** `raglab/CONTRACT.md` (endpoints, states, errors). This doc
maps every capability to screens, components and UX rules — the reference
client is `raglab/local_front.py` (its 14 menus = your screens).

The agent is **fully controllable over HTTP**: models, chunking, retrieval,
corpus, API keys, document feed, indexing, evaluations. Nothing below needs a
backend change — it is all `POST /profile`, `/keys`, `/documents`, `/ingest`,
`/evaluate` + the read endpoints. What the UI must respect is the **state
model** (§4) — that is where RAG consoles usually go wrong.

**Companion docs:** `raglab/CONTRACT.md` = behavior truth (endpoints, states,
errors). `raglab/COOKBOOK.md` = every settings mutation as an exact request
with its side effects — feed that to whoever builds the Settings screens.

---

## 1. Capability map (what the console can do)

| local_front menu | Screen in your app | API |
|---|---|---|
| status / doctor | **Global status bar** (always visible) | `GET /health` |
| providers & models | **Settings → Models** | `GET /models`, `POST /profile` |
| service settings (chunking, retrieval, corpus) | **Settings → Retrieval & corpus** | `GET/POST /profile` |
| API keys | **Settings → API keys** | `GET/POST/DELETE /keys` |
| documents (push/list/remove) | **Documents** | `/documents*` |
| ingest / rebuild | **Documents → Index** (button + job banner) | `POST /ingest?reset=`, `GET /ingest/status` |
| retrieval query | **Retrieval lab** | `POST /search` |
| grounded answer / chat | **Ask** (the main screen) | `POST /answer` |
| inspect corpus | **Retrieval lab → Corpus** | `GET /inspect` |
| search the chunks for a quote | **Retrieval lab → Quote check** | `POST /chunks/search` |
| embedding sanity check | **Settings → API keys → Test** | `POST /embeddings/sanity` |
| evaluate a question set | **Evaluations** | `POST /evaluate` |
| diagnostics | **Evaluations → Diagnostics** | `/diagnostics/*` |

## 2. Shell & navigation

Sidebar (icon + label) + a persistent **status bar** across the top. The
status bar is the doctor: it renders the `/health` state machine (§4) so the
user always knows what the agent can do right now.

```
┌────────────────────────────────────────────────────────────────────────┐
│ ● Ask      │  RAGLab · nvidia/nemotron-3-embed-1b · xkiro/qwen3.8-max │
│ ○ Documents│  ──────────────────────────────────────────────────────── │
│ ○ Retrieval│  index: 346 chunks ✓   keys: 2/2 ✓   ingest: idle        │
│ ○ Settings │  [experimental profile ▾]        (pipeline marker)        │
│ ○ Evals    │                                                          │
└────────────────────────────────────────────────────────────────────────┘
  sidebar            status bar = live /health (poll 5–15 s, or after actions)
```

Three pills render the whole truth: **profile** (embedding + answer model,
with the pipeline marker as a badge), **index** (chunk count, collection,
stale ⚠ when `tokenizer_match=false`), **ingest** (idle / running spinner /
last error). During an ingest the whole app enters "sealed" mode (§4.3).

## 3. Screen specs

### 3.1 Ask (the product)
* Question box (autodetects ar/fr/en — offer a language override dropdown =
  `query_lang`), `k` and `mode` as an "advanced" disclosure.
* **No streaming exists** — the citation gate validates the full reply before
  anything is served. Show an honest indeterminate loader ("searching the
  corpus… verifying citations…"), never a fake progress bar.
* Render the three 200-shapes differently: **answered** (claims as cards;
  `[S#]` markers become citation chips linking to the source row; evidence
  quotes in a monospace block with a copy button), **refused** (calm info
  styling, never red/error — show `answer` + the reason taxonomy, and the
  collapsible "what the model said, not accepted" from `raw_preview`), 
  **greeting** (plain bubble, hide the citation UI).
* Badges: `cached` ("served from cache"), `inference_performed=false`
  ("answered without an AI call"), `seconds`.
* Chat = client-side history; each turn is an independent `/answer` call.
* `dir="auto"` on every message and quote — the corpus is Arabic/French;
  claims may be Arabic while quotes are French. Never "fix" the script.
* PII placeholders (`[EMAIL] [PHONE] [RIB] [CIN]`) render as chips, verbatim.

### 3.2 Documents (the feed)
* **Upload**: dropzone → a visible per-file queue (state per row:
  `queued → uploading → done|failed`, bounded ~3 parallel, bytes-weighted
  aggregate progress, retry scoped to failed rows only — the standard bulk-
  upload pattern). Each row sends `POST /documents` with **your document id
  as `id`** (idempotent retries are then free) and shows the response
  (`created v1` / `unchanged` / `replaced vN`).
* **List** (`GET /documents`): table of id, filename, version, bytes, status
  chip, chunks. Status vocabulary maps 1:1 to the API:
  `pending` (grey "waiting for index") → `indexing` (blue, during a job) →
  `indexed` (green) → `stale` (amber "newer version waiting — rebuild").
  State comes from the server, so a reload resumes truthfully.
* Row actions: **Replace** (re-upload → version bump), **Delete** (confirm
  dialog; copy: "removes the file and purges its chunks from every index —
  immediate, no rebuild needed").
* **Index controls**: primary button "Build / rebuild index"
  (`POST /ingest`), danger-zone "Reset collection & rebuild"
  (`?reset=true`) with an explanation of the fingerprint guard, and the
  batch callout: *push a batch first, index once* — the embedding cache
  makes re-indexing cheap but not free. Single late doc → re-push with
  `?index=true`.

### 3.3 Retrieval lab (trust, made visible)
* **Search**: same box as Ask but returns hits only — ranked list with
  similarity, heading, language, source, chunk text (PII chips as-is).
  Controls: `k`, `mode` (vector/rrf/blend), `lang_filter`.
* **Chunk browser**: `GET /chunks` + `GET /chunks/{id}` — the STORED
  chunks (what retrieval actually supplies), browsable per document and
  readable one by one with neighbors and the overlap landing; the tool
  for rating/changing the chunking strategy. local_front menu 15 is the
  reference interaction.
* **Corpus**: `GET /inspect` — document table + a chunk-size histogram
  (great "is my chunking sane" visual) + sample chunks.
* **Quote check**: paste a quote → `POST /chunks/search` → verdict UI:
  ✔ "inside one chunk — a verbatim quote can validate" (`full`) vs
  ⚠ "crosses a chunk boundary — no faithful quote of it can ever pass the
  gate" (`head`/`tail` in different chunks). This is the tool that explains
  refusals to users; give it good empty-state copy.

### 3.4 Settings → Models
* Provider cards from `GET /models` (two groups: embedding, answer): label,
  key state (`key_set` — "key configured" / "needs key" → deep-link to API
  keys), registered models as selectable chips.
* **Custom model ID**: a "type an exact model ID" input on every provider
  (client-side validation: one token, no spaces — mirrors the 400). Remember
  everything the user ever typed, grouped by provider (the local_front does
  exactly this in `front_state.json`).
* Switching = `POST /profile {slot: {provider, model}}`. **After every
  switch, compare `collection` in the response with the index pill**: if it
  changed, show "This profile uses a different collection — build its index"
  with a CTA to Documents → Build.
* The pinned pair (NVIDIA nemotron + xKiro `qwen/qwen3.8-max:free`) shows a
  "supported pipeline" badge; everything else an "experimental" badge from
  `health.profile.pipeline` — plus `notes[]` from the switch response as a
  warning banner.

### 3.5 Settings → Retrieval & corpus
* Chunking: mode radio (`restructure` default / `size` / `manual`); when
  `size` is active, enable numeric inputs for `size` and `overlap` (positive
  ints). Changing either means the current index is stale — say so inline
  ("chunk texts change → the index must be rebuilt").
* Retrieval: `top_k` slider (1–20), `mode` select, `lang_filter` select
  (none/ar/fr/en), `neighbor_radius` (0–2). Corpus: `data_dirs` list editor
  (paths on the **service host** — validate with the 400s, don't guess).
* Save = one `POST /profile` with only the changed keys; toast the returned
  collection. Note: out-of-range retrieval values are silently ignored by
  the API — validate client-side so the UI never lies.

### 3.6 Settings → API keys
* Table from `GET /keys`: env name, human description, status, **masked
  value** (`nvapi-12…`). Never request or display a full value — the API
  will not return one either.
* Set/replace: password input with reveal toggle, validation feedback for
  the 400s (`placeholder_value` = "that's the template, not a key"),
  "remember on the service host" checkbox = `persist:true` (with the Docker
  caveat: recreated containers lose it — env_file is the durable path).
* Delete = `DELETE /keys/{env}` with confirm ("features using it stop
  working immediately").
* **Test connection** button → `POST /embeddings/sanity` (shows dimension +
  the three cross-lingual cosines; a great first-run setup wizard step).

### 3.7 Evaluations
* Run a set (`POST /evaluate`): pick `questions*.json` (or absolute path),
  show an honest long-job loader (minutes; it embeds every question), then
  the metrics table — `hit@1/3/5` overall / by category / by language, with
  the "lab measurement, not a benchmark" note. Only the pinned pair's
  numbers may be called a benchmark — badge accordingly.
* Diagnostics: harness50 (offline, up to 900 s — background job in YOUR
  backend with a long timeout; show `output_tail` lines as a console) and
  the xKiro catalog viewer.

## 4. Cross-cutting rules (where RAG consoles go wrong)

### 4.1 `/health` drives the chrome
| health says | UI |
|---|---|
| needed key missing | pill turns amber; Ask disabled with "configure keys" CTA |
| `index.count == 0` | "Build the index" empty state on Ask/Documents |
| `tokenizer_match == false` | "Index is stale — rebuild" warning + reset CTA |
| `ingest.state == running` | sealed mode (§4.3) |
| otherwise | everything enabled |

### 4.2 Refusals are 200s — never render them as errors
`status:"refused"` with a reason (`insufficient_evidence`, `no_context`,
`invalid_output`, `unsourced_number`, `private_or_live_request`) is the
product working. Copy per reason, calm styling, no auto-retry (they are
deterministic for the same question + corpus).

### 4.3 Single writer: sealed mode during ingest
While `ingest.state == "running"`: disable Ask, Retrieval search, Settings
save, profile switches and document deletes; show a global banner
("Indexing… started 14:02 · the corpus is sealed"). Expect `409
ingest_in_progress` from anything you missed disabling — treat it as a
signal to enter sealed mode, not an error toast. Poll `/ingest/status`
every 1–2 s; on `error` show `error` (it names the remedy, e.g.
`POST /ingest?reset=true`).

### 4.4 Error envelope switch
Every error is `{"detail":{"reason":…}}` (or the 422 array). Route on
`reason`: `missing_api_key` → keys CTA · `empty_index`/`ingest_in_progress`/
`stale` → index CTA · `unauthorized` → session/gateway problem ·
`provider_unreachable`/`provider_error` → "upstream provider" notice with
retry · `internal_error` → report banner with timestamp. One handler, one
component — never scatter this logic.

### 4.5 Long-op honesty
No fake progress. Indeterminate loaders with staged copy; async pattern is
always create-job → poll; a reload must resume from server state, never
browser memory.

## 5. Permissions & deployment

* **Auth**: every request carries `X-Service-Token` (service env:
  `RAGLAB_SERVICE_TOKEN`, alias `RAGLAB_TOKEN`; 401 without it). Compose now
  ships switching **enabled** (`RAGLAB_ALLOW_PROFILE_SWITCH: "1"`) **and**
  the token required — keep both.
* **Where to call from — two supported shapes**:
  1. **Recommended: your backend proxies the agent** (BFF). The browser
     never holds the service token; your gateway's user auth (admin/editor/
     viewer roles) fronts everything. Suggested roles: *viewer* → ask/
     search/inspect; *editor* → + documents & index; *admin* → + profile,
     keys, evaluations.
  2. Browser → agent direct: set `RAGLAB_CORS_ORIGINS` to your origin and
     accept that the shared token is visible to the client — pilot only.
* Mutating vs read-only, for your role checks: mutating = `/profile`,
  `/keys*`, `/documents` POST/DELETE, `/ingest`, `/evaluate`,
  `/diagnostics/*`. Everything else is read.

## 6. Design system suggestions

* **Stack**: shadcn/ui + Tailwind is the pragmatic default for this exact
  shape (dashboard + tables + forms + toasts); dark mode from day one
  (console users live in it).
* Components to build once: StatusPill, DocStatusChip, CitationChip,
  QuoteBlock (mono, dir=auto, copy), RefusalCard, SealedBanner,
  MaskedKeyField, UploadQueueRow, MetricsTable.
* Typography: a font with good Arabic coverage (IBM Plex Sans Arabic or
  Noto Kufi Arabic) — half the corpus and many answers are Arabic.
* Motion: minimal; loaders and toasts only. Nothing animates while sealed.

## 7. References (patterns this blueprint draws on)

* Bulk-upload queues, per-file states, partial success, scoped retry —
  Filestack engineering blog, "Bulk Upload UI That Supports Queues, Retries,
  and Partial Success" (blog.filestack.com/bulk-upload-ui-queues-partial-success/).
* Async job pattern (create → poll; pending/processing/done; honest
  progress; idempotent retries) — dev.to "Designing Asynchronous APIs with
  a Pending, Processing, and Done Workflow".
* Loading-indicator anatomy (indicator, status text, busy target, retry
  path, completion handoff; don't overstate certainty) — uxpatterns.dev
  "Loading Indicator".
* LLM settings-panel patterns (per-provider cards, masked keys, reveal
  toggle, test-connection, admin-gated writes, AES-GCM at rest) —
  cybersecurity-maturity-platform PR #53 (github.com/Ahmed-Althamari/…/pull/53);
  boss-agent-mobile issue #159 (settings sections + save-all + toasts).
* Secrets UX (never log/display, mask form `tok-1234****cdef`, rotation) —
  sofatutor llm-proxy security guide; LiteLLM security best practices.
* RAG product surface inventory (documents UI, citation deep-linking,
  document viewer, search-vs-chat separation, explicit state design) —
  advanced-rag-knowledge-assistant issue #5 (github.com/vigneshrao1723-lab/…).
* Kotaemon (open-source RAG UI): configurable retrieval/generation settings
  in-app, citations with in-browser document preview — the closest
  open-source reference product.
