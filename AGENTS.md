# AGENTS.md — working contract for this repository

If you are an agent (or human) resuming work on RAGLab: **read this file first, follow it,
and keep it updated whenever you learn something that changes the truth.** It is the
contract between sessions. Facts here were verified by running things, not guessed.

Last verified: 2026-09-07 (branch `arena/01a07b8c-raglab`).

---

## 1. What this repo is

A RAG lab for Islamic-banking documents (Arabic/French/English): load → chunk → embed
(NVIDIA `nemotron-3-embed-1b`, 2048 dims) → hybrid retrieval → cited answers
(xKiro gateway, `qwen/qwen3.8-max:free`). No LangChain/LlamaIndex; stdlib HTTPS only.

Current state of the work:

- **Two chunking modes** in `raglab/semantic_chunking.py` + `raglab/restructure.py`,
  selected by `CHUNKING_MODE` env var: `size` (legacy recursive 220/40) and
  `restructure` (the new strategy).
- **The new strategy is a 3-step workflow the user designed** — follow it, don't redesign:
  1. semantic normalization → hierarchy-explicit Markdown;
  2. context breadcrumbs above sub-headings/tables;
  3. recursive structural chunking over
     `["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "]` at the usual 220/40 token budget.
- **Benchmark set**: `raglab/questions_50.json` — 50 questions I generated (17 ar /
  17 fr / 16 en), categories `verbatim` (12), `paraphrase` (13), `cross-lingual` (20),
  `out-of-scope` (5, must have NO expected match). Never copy from
  `raglab/questions.json`, `questions_real.json`, or `benchmarks/retrieval_dev.json`.
- **BM25-only A/B harness**: `raglab/harness50.py` (no API calls; deterministic).
- **Real-model A/B**: `.github/workflows/real-test.yml` (manual trigger, spends API
  calls, reads repo secrets) + `raglab/real_report.py` (table builder).
- **CI**: `.github/workflows/ci.yml` — API-free, runs on every push, must stay green.
- **Interactive console** (added 2026-09-08, session `arena/01a08139-raglab`):
  `raglab/app.py` — one menu over every lab function (inspect/ingest/query/
  answer/chat/evaluate/sanity/diagnostics) that also lets the user switch the
  embedding provider/model (all `build_embedder` providers) and the answer/chat
  provider/model (xKiro pinned SKU, NVIDIA build-endpoint chat models, Google
  free-tier Gemini via `llm_smoke`, Kira AI `kiraai.vn` OpenAI-compatible
  gateway e.g. `glm-5.3-free`), and manages API keys per provider
  (keep/change/add → written to `raglab/.env`, masked to 8 chars). Custom model
  IDs typed in the selection menus are remembered per provider in
  `app_state.json` (`record_custom_model`, capped at 20/provider) and
  re-offered next session; menu 2 → 3 reviews/removes them. Non-pinned xKiro
  SKUs run through `GatewayChatClient` (NvidiaClient on
  `api.xkiro.com/v1`) as EXPERIMENTAL calls — the pinned SKU alone keeps
  `build_answer_generator`'s live free-price check, and no benchmark number is
  attributed to experimental SKUs. Greetings (fr/en/ar) are answered locally,
  zero calls; invalid_output prints the failed check, offers a re-ask, and the
  search-chunks action (citation-gate normalization) tells boundary-crossing
  quotes from paraphrases. It is a lab
  surface like `chat.py`: it builds its OWN collections
  (`raglab_app_<provider>_<model>_<chunking>`) via a `SimpleNamespace` copy of
  config (`build_lab_config`), never mutates the module config, and the pinned
  `pipeline_policy` checks still gate `main.py`. Selections live in
  `raglab/app_state.json` (gitignored) — never in `.env`. Offline coverage:
  `tests_offline.py` drives a stubbed HF+nvidia-chat profile end-to-end
  (state → lab config → ingest → retrieve → cited answer) plus env-writer,
  state round-trip, greeting, locate_text, model-memory and gateway-wiring
  checks. Known subtlety: `config.active_embedding_model()` is
  a closure over module globals, so the lab copy MUST override it with a lambda
  returning the selected model or store.py mislabels chunks.
- **HTTP microservice** (added 2026-09-11, same session): `raglab/service.py`
  (FastAPI+uvicorn, `requirements-service.txt`) exposes the SAME runtime as the
  console over REST: `/health`, `/models`, `/profile` (+ optional
  `POST /profile` behind `RAGLAB_ALLOW_PROFILE_SWITCH=1`), `/search`,
  `/answer` (greetings answered locally, refusals are HTTP 200 + reason),
  `/ingest` as a background thread + `/ingest/status`. Profile from
  `RAGLAB_*` env vars at boot (validated loudly, SystemExit on bad combos);
  keys from env only; Dockerfile + docker-compose.yml at the repo root
  (`RAGLAB_CACHE_DIR` relocates caches into a volume). Single replica by
  design (local ChromaDB); no built-in auth — must sit behind a gateway.
  **The refactor to know about**: the runtime half of app.py moved to
  `raglab/profiles.py` (registries, SUPPORTED_*, build_lab_config,
  Gateway/Google chat clients, build_generator, collection naming, ingest,
  greeting logic, collection_count/stored_index_info); app.py re-exports it
  (existing callers/tests unchanged) and keeps only the console (menus, key
  prompts, app_state.json, model-ID memory). profiles.py imports chat.py;
  service.py imports profiles.py and NOTHING from app.py. Offline coverage:
  `test_service.py` (19 tests, stubbed HF embedder + injected fake generator,
  TestClient) added to run_tests.sh and ci.yml; requirements-benchmark.txt
  gained fastapi+httpx for it. Full docs with the env-var table:
  `raglab/SERVICE.md`.
- **`raglab/CONTRACT.md`** (same session): the HTTP contract for client/fullstack
  teams — ground rules (status ladder, error envelope, spend table), the state
  model (profile → collection → index lifecycle, keys, switching), full
  endpoint reference with real example payloads (captured from the offline
  fixture via a throwaway script), a 24-reason error catalog, and integration
  recipes (boot sequence, answer rendering incl. refused/greeting, chat =
  client-side history, provider picker, long-op polling). MACHINE-truth for
  field names is /openapi.json; CONTRACT.md is the behavior contract.
- **`raglab/local_front.py`** (same session): the console's twin over REST —
  same 13 menus as app.py (status/doctor, providers & models, keys, inspect,
  chunk search, sanity, ingest, retrieval, answer, chat, evaluate, settings,
  diagnostics), every action an endpoint call. Imports NOTHING from the lab,
  only urllib + JSON; stateless except front_state.json (custom model-ID
  memory, 20/provider). Also one-shot flags: --status, --ask, --search,
  --ingest (polls /ingest/status), --smoke, --interactive, --no-keycheck,
  --base-url. `--smoke` runs the state-aware suite (20 checks keyless) that
  reads /health + /models to decide what /search, /answer and /ingest SHOULD
  do in the current state: keyless → expects 503 missing_api_key, keys +
  empty index → expects 409 empty_index, ready → one live /search + /answer.
  `test_service.py::LocalFrontOverHttp` boots a real uvicorn on port 0 and
  requires the suite to pass over actual HTTP in CI; `ConsoleEndpointsTest`
  covers the keys/inspect/chunks-search/sanity/evaluate/profile-switch
  endpoints directly (stubbed). Exit codes 0/1/2 (ok / failed / unreachable).

## 2. Hard constraints (never violate)

1. The hard harness pins chunking at **640/40 in `raglab/benchmarks/hard_harness_plan.json`**
   — do not touch that file or its behavior.
2. `ci.yml` + `raglab/run_tests.sh` reference **flat files in `raglab/`**
   (`tests_offline.py`, `test_nvidia_pipeline.py`, `test_hard_harness.py`,
   `main.py inspect`, …) — never move or rename them.
3. Question-set format follows `raglab/evaluate.py`
   (`load_question_set`): categories `{verbatim, paraphrase, cross-lingual, out-of-scope}`;
   out-of-scope questions must have no expected match.
4. **Never commit `raglab/.env` or any API key value.** The keys exist as GitHub Actions
   repo secrets named `NVIDIA_API_KEY`, `XKIRO_API_KEY` (and `GOOGLE_API_KEY` for the
   LLM fallback) — added by the user 2026-09-07. Reference them by name in workflows;
   the dev token cannot read or list secrets (403) — that is normal, not an error.
5. `raglab/results/` is gitignored ("generated data — never commit these"). Benchmark
   artifacts there are **regenerable**: `python harness50.py` rebuilds the BM25 A/B
   artifacts offline. If a workspace is rebuilt, re-run it rather than treating missing
   files as a loss.
6. Work only on the session branch `arena/01a07b8c-raglab` (push only there). The other
   `arena/*` branches and the four older workflows (`free-models.yml`,
   `hard-harness.yml`, `provider-catalogs.yml`, `retrieval-judge.yml`) belong to earlier
   sessions — do not touch them.

## 3. Network reality (check before promising anything)

From the dev sandbox, **only PyPI and api.github.com are reachable.** TLS-blocked
(verified by probe, do not re-probe pointlessly): huggingface.co,
download.pytorch.org, Azure blob hosts, the tiktoken CDN, `integrate.api.nvidia.com`,
`api.xkiro.com`. Consequences:

- **Live model calls cannot run in the sandbox. Ever.** They run in CI (full network
  there) via `real-test.yml`. Pasting keys into chat does not unblock anything here.
- `cl100k_base.tiktoken` cannot be downloaded locally; CI fetches it. CI therefore
  tokenizes with **real cl100k_base** while local runs use a chars/4 estimator —
  token-dependent behavior (chunk boundaries, orphan-start checks) can differ, and
  only CI is the arbiter for that class of tests.
- Local test gate before pushing: `./raglab/run_tests.sh --offline` (exits 0 when green).
- The venv at `raglab/.venv` is gitignored and does NOT survive workspace rebuilds:
  `python3 -m venv raglab/.venv && raglab/.venv/bin/pip install -r raglab/requirements-benchmark.txt`
  (a few minutes; PyPI works).

## 4. Reading CI from the sandbox (logs are NOT downloadable)

Verified dead — do not retry:
- `gh api .../jobs/{id}/logs` → 302 to Azure blob → connection EOF.
- `gh run view --log(-failed)` → results-receiver EOF.
- `gh run download` (artifacts) → 302 to `*.blob.core.windows.net` → EOF
  (confirmed dead 2026-09-07 on run 34140795660).

Channels that WORK (use in this order):
1. **Status**: `gh run list --limit 5` / `gh run view <run_id>` (conclusion, headSha).
   Note: `--limit 1` can return a stale top entry — always watch the run whose
   `headSha` == `git rev-parse HEAD`.
2. **Annotations**: `gh api repos/seif-bkh/RAGLab/actions/runs/<run_id>/check-runs`
   for ids, then
   `gh api repos/seif-bkh/RAGLab/check-runs/<check_run_id>/annotations`.
   Both workflows emit their diagnostics as annotations on purpose:
   - `ci.yml` failure step re-runs the offline commands and emits the tail as
     `::error::` (carries the reproduction).
   - `real-test.yml` emits the full key-number summary as a `::warning::` annotation
     (readable on success too) and its failure step emits the newest step-log tail as
     `::error::`.
3. **For the full 50-question tables**: they are in the run's stdout (visible in the
   GitHub Actions UI for the user) and in the uploaded artifact
   `real-test-results` (the user can download it; I cannot).
4. GitHub HTML log pages render client-side — `fetch_page` on them gets nothing.
5. The ANNO `::warning::` line is self-sufficient by design (overall/by-category/
   by-language/separation/OOS/flips/misses + per-miss top-1 detail + answer status) —
   it has been verified twice end-to-end (runs 34140795660, 34141551444) and is the
   primary interpretation channel.

## 5. Commit/push discipline (standing user instruction)

- The GitHub token **might expire**: commit + push every finished meaningful step
  without waiting for the next one.
- **After every push, monitor the CI run to completion.** On failure: read the
  annotations (§4), diagnose, fix, re-push — iterate until green.
- Keep `./raglab/run_tests.sh --offline` green locally before pushing (it is what CI
  re-runs; a local red guarantees a CI red).
- Do not run paid/live commands locally — the sandbox can't reach the endpoints anyway.

## 6. Corpus facts (do not re-diagnose)

- `docs/` real corpus is **all Arabic** despite French filenames:
  `Circulaire_BCT_2019-08.pdf` (logical order, corrupted digits),
  `Guide_Interne_Operations_Bancaires_Islamiques.docx`,
  `Loi_2016-48.pdf` (gazette, visual order, ~232 chunks),
  `Madkhal_Sayrafa_Islamiya.docx` (clean).
- PDF text comes out in **visual order with corrupted digits** → repaired only by the
  logical-order bigram scorer in `restructure.py`. **Never "repair" digits** and never
  re-derive facts from the raw digit soup; take authoring substrings from the repaired
  output.
- The Guide stores some decimals inverted — kept as-is (matches the source).
- `raglab/data/` sheets are a **fictional "Banque Atlas"**, parallel FR/AR, each with a
  fictional-disclosure section. Questions may quote their numbers (4.50€/mo, 2.75% brut,
  50 000€ cap, …) — that is by design.
- Pipeline final stats (restructure mode, 346 chunks total): Circulaire 15, Guide 61,
  Loi 232, Madkhal 40, ar sheet 10, fr sheet 11. `main.py inspect` exits 0 on all six.

## 7. Key results (update this section when numbers change)

- **CI green**: run 35090005268 on HEAD `61243b0` (offline suite: 224 unittests
  + 96 checks + inspect + pip check; docs-only — COOKBOOK.md added).
- **COOKBOOK.md** (61243b0): exact-request companion for whoever builds the
  settings UI — readiness probe (GET /profile → switching; old compose pins
  0 → 403), the read assembly, every mutation with side effects (answer
  switch NEVER needs a reindex; embedding/chunking does; retrieval
  out-of-range is silently ignored → validate client-side), which-build
  table (?reset=true vs not), worked sequences, editing-time failure table.
  Gotcha documented: ?index=true only starts the job on created/replaced —
  never on an unchanged push. Feed order for AI agents: CONTRACT.md
  (behavior) → COOKBOOK.md (exact requests) → FRONTEND.md (screens/UX).
- **FRONTEND.md + app-control unlock** (224749d): docker-compose now ships
  RAGLAB_ALLOW_PROFILE_SWITCH=1 WITH RAGLAB_SERVICE_TOKEN required
  (change-me placeholder) — the app console can drive everything
  (models/chunking/retrieval/corpus/keys/documents/ingest/evals) like
  local_front. raglab/FRONTEND.md is the UI/UX blueprint for the fullstack
  team: capability map, shell + health-driven status bar, screen specs with
  API mappings, cross-cutting rules (refusals-are-200s, sealed mode during
  ingest, error-envelope handler, no fake progress, RTL), permission matrix
  (BFF proxy recommended; viewer/editor/admin roles), research references.
- **RAGLAB_TOKEN alias** (69fba34, service 1.2.3): RAGLAB_SERVICE_TOKEN takes
  precedence, RAGLAB_TOKEN is the fallback — a gateway deployment that plumbed
  the shorter name no longer silently runs open. The gateway team's claim that
  RAGLAB_TOKEN was "plumbed through .env.example, .env.docker.example and
  docker-compose.yml but never read" was factually wrong for THIS repo
  (.env.docker.example does not exist here; no file contained RAGLAB_TOKEN;
  the backend has read RAGLAB_SERVICE_TOKEN since 48bb30d) — their plumbing
  lives in their own repo. .env.example gained an optional service section
  documenting the token + header + alias.
- **Stale-hint + filename hygiene** (b5c8948, service 1.2.2): ingest job
  errors translate the store layer's CLI remedy to
  "POST /ingest?reset=true (this service)"; POST /documents rejects
  filenames with path separators/control chars (400 bad_filename, both JSON
  and multipart). Field forensics that shaped it: the user's stale error was
  an index built with chunk size 440 in an earlier session vs the restarted
  default 220 (size is NOT part of the collection name, so the same
  collection is reused and the fingerprint guard fires — correct); the
  "[tarif.md](http://tarif.md)" in their paste was chat-display
  linkification only (content hash proved the request was clean) — but
  copy-from-chat corruption is real on that device: prefer single-line,
  hand-typed commands or local_front. Also: POST /profile drops the injected
  test generator (caches reset on switch) — tests that switch profiles and
  then /answer need a real key or must assert on job outcomes instead.
- **Index sealed during ingest + never a plain-text 500** (2ce40b4, service
  1.2.1, from a field report): while the ingest job runs, /search, /answer,
  /evaluate, POST /profile and DELETE /documents/{id} return
  409 ingest_in_progress (greetings still work; GET /documents rows say
  status=indexing without reading the store) — reads used to race the job's
  sqlite writes and could surface as a bare 500. Plus a safety net:
  add_exception_handler(Exception) → 500 JSON {reason: internal_error,
  safe error} — no unhandled exception can ever be FastAPI plain text
  (Starlette sends the handler response then re-raises; TestClient needs
  raise_server_exceptions=False to assert the envelope). SERVICE_VERSION is
  now bumped on every service change (1.2.0 was reused once and the field
  could not tell builds apart). Concurrency-test gotcha: the embedding cache
  makes re-ingests instant — force an UNCACHED unique chunk (push a fresh
  doc) if the test needs the job to stay running.
- **Network failures never 500** (0d9a2a9, found on a real device): the
  first /answer on the pinned xKiro profile runs the live free-price check
  (free_gateway.load_pricing) inside Runtime.generator(), which was OUTSIDE
  all try/excepts — URLError/DNS/timeout escaped as a bare 500. Now:
  Runtime.embedder()/generator() wrap construction (OSError → 502
  provider_unreachable with slot + safe error + hint; ValueError/RuntimeError
  → 502 provider_error), and the ask/retrieve/evaluate/sanity handlers catch
  OSError mid-request the same way. Guarded by ProviderFailureTest.
- **Documents API = the gateway feed target** (698a30b, service 1.2.0): the
  service OWNS a document store (docstore.py, RAGLAB_DOCUMENTS_DIR, default
  raglab/documents/). POST /documents (multipart or JSON text/base64,
  ?index=true), GET /documents + /{id} (status pending/indexed/stale +
  chunks), DELETE /{id} (file + chunks purged from EVERY collection via
  store.purge_source — no re-ingest). Content-hash versioning (same bytes
  no-op, different bytes version bump); ids [A-Za-z0-9][A-Za-z0-9._-]{0,79}
  (traversal-safe); extensions .txt/.md/.pdf/.docx; size cap
  RAGLAB_MAX_DOCUMENT_BYTES (413). The documents dir is part of EVERY
  profile's corpus (appended at boot + after POST /profile; data_dirs=None
  means defaults — materialize, never drop). Gotchas learned: (1)
  DocumentStore.save->find under one non-reentrant Lock deadlocks — RLock;
  (2) async endpoints must keep sqlite/disk work off the event loop
  (run_in_threadpool); (3) doc status timestamps are second-precision —
  tests that need a stale transition must sleep past the second boundary.
  python-multipart==0.0.32 pinned in both requirements files (multipart
  push). Front: menu 14 + 3 smoke checks (23 total, keyless, self-cleaning).
  Feed contract documented in CONTRACT.md §3.15/§2.5/§5.7.
- **Service 1.1.0 output guards** (48bb30d, from the fullstack team's review):
  (1) `RAGLAB_SERVICE_TOKEN` → every request needs `X-Service-Token`
  (constant-time; CORS preflights exempt; local_front sends it from env);
  (2) `scrub.py` — post-gate PII scrub of /answer + /search outputs
  ([EMAIL]/[PHONE]/[RIB]/[CIN]; diagnostics stay raw on purpose);
  (3) numeric half of the citation gate in answer.py — every digit-form
  number in a claim must exist in its evidence quotes (FR/EN/AR decimal +
  thousands normalization; `unsourced_numbers`/`UnsourcedNumber`); violation
  refuses as `unsourced_number`, and /answer now forwards `error` +
  `raw_preview` (scrubbed) on refusal paths. Known test gotcha: the stub
  embedder is hash-based and Python str hashes are per-process random, so
  retrieval ORDER varies between runs — never assert on hits/sources[0].
- **Service profile-switch contract** (eb32c74): an explicit
  `{provider, model}` in POST /profile is applied VERBATIM — custom model IDs
  included (the registries are known-good suggestions, not a whitelist). The
  bug it fixes: consistent_model silently swapped unregistered IDs for the
  provider's first registered model while answering 200 — the front saved the
  custom ID but the service never used it. Provider-only switches: same
  provider = no-op; different provider = first registered model, never a
  cross-provider carry-over. One token, no spaces → else 400 bad_model.
- **BM25-only A/B** (45 evaluable of 50; k=20; full table in
  `raglab/results/harness50/comparison.md`, regenerable via `harness50.py`):
  - overall hit@1/3/5: size-220/40 `40/69/80` → restructure `47/78/89`
  - verbatim (12) `42/58/75 → 75/92/100`; paraphrase (13) `46/100/100 → 31/92/100`
    (paraphrase hit@1 drop is expected — BM25 has no word overlap; hit@5 is 100 both);
    cross-lingual (20) `35/55/70 → 40/60/75`
  - by language ar(15) `53/87/87 → 47/93/100`, en(15) `47/73/93 → 53/80/100`,
    fr(15) `20/47/60 → 40/60/67`
  - the 5 restructure misses are q26–q30, all fr→ar cross-lingual (documented lexical
    limit; the embedding arm should cover them)
  - OOS max top-1 BM25: base 17.5 vs restructure 18.1
- **Real-model A/B (NVIDIA embeddings + LLM answer smoke)**:
  - LLM role per owner instruction (2026-09-07, standing): the real-test answer smoke
    uses **nvidia/nemotron-3.5-lightning-30b-a3b via integrate.api.nvidia.com with
    NVIDIA_API_KEY**; if that fails, **GOOGLE_API_KEY with the cheapest free-tier
    Gemini model** (flash-lite preferred). Implemented in `raglab/llm_smoke.py`
    (retries + fallback + auto model selection); the app's pinned profile
    (pipeline_policy: xkiro/qwen) is deliberately NOT changed. The old xKiro/Qwen
    smoke was abandoned after repeated free-tier 502 "temporarily at capacity"
    (runs 34140795660, 34141551444, 34142147484).
  - Retrieval A/B completed four times (same code); stable picture at k=20,
    overall hit@1/3/5 — size `67/89/100` vs restructure `73–76/84–87/89–91`:
    restructure wins hit@1 (+6–9 pp) driven by paraphrase (+15 pp), cross-lingual
    (+10–15 pp), fr (+27 pp), en (+7–14 pp); it loses a few recalls at top-20 that
    size always hits (size = 0 misses in all runs). Green run: **34144251576**.
  - Persistent restructure misses (all runs): q01, q02, q03 (ar, Circulaire
    murabaha/salam definitions), q38 (en, same circular) — top-1 lands on
    chapter-heading chunks ("الفصل4"/"الفصل 7", scores 0.49–0.69) or the Guide's
    murabaha section instead of the definition sub-chunk; q44 (en, Madkhal 1963,
    top-1 0.151) is borderline and jitters between runs.
  - **Run-to-run jitter exists**: hosted NVIDIA embeddings are not bit-reproducible,
    so borderline questions move ±1 place between runs (q44 hit/miss varies;
    cross-lingual 90→95→90). Treat differences smaller than ~1 question as noise.
  - Answer smoke (green run): the NVIDIA phase called
    nvidia/nemotron-3.5-lightning-30b-a3b successfully (no capacity error this time)
    but its output failed the verbatim-quote contract
    ("Evidence quote is not in the cited source") → rejected by design → Google
    fallback **gemini-3.1-flash-lite** produced a fully validated cited answer
    (3 claims, 5 sources). So: the fallback chain works end-to-end; the NVIDIA
    model's known weakness is verbatim quoting, not reachability.
  - Per-question full tables: CI run stdout / `real-test-results` artifact
    (run 34144251576); the compact numbers live in the run's check-run annotation
    and in this section.

## 8. Gotchas (learned the hard way — do not relearn)

- `gh run view --log` / job-log / artifact downloads all dead from the sandbox (§4).
- **Fuzzy `edit_file` can silently misapply — or report success without applying** (one
  "successful" edit lost a whole function; another appended duplicates at EOF; a third
  was reported successful yet the buggy line was still in the file at commit time and
  shipped to CI as an AttributeError). After every edit: `grep` for both the absence
  of the old text AND the presence of the new text before moving on, and exercise the
  changed path (compile/compileall is not enough for semantics).
- `except Exception` does not catch `SystemExit` — retrieval code raises
  `SystemExit("collection is empty")`; catch-all diagnostics need `BaseException`.
- **Nemotron-3.5 Lightning (30b-a3b) via the NVIDIA build endpoint can violate the
  verbatim-quote contract** (paraphrased evidence quotes) — the strict validation
  rejects it and the Google fallback takes over. That is the intended behavior of
  the smoke, not a bug: a "green" real-test run may legitimately be served by the
  fallback (the ANNO line says which: `phase google-fallback` + `nvidia_attempt=…`).
- `GOOGLE_API_KEY` is a confirmed-working repo secret (the /models listing succeeded
  in run 34143586968); `nvidia/nemotron-3.5-lightning-30b-a3b` on
  integrate.api.nvidia.com has not yet completed a call — watch its first run.
- `_repair_orphan_starts` in `semantic_chunking.propose` exists because real cl100k can
  close a packing run exactly after an Arabic article marker ("الفصل 2"), leaving a
  chunk opening on ":" — a rule the checker (ORPHAN_TAIL) rejects. The fix merges the
  orphan into its previous chunk (refusing if it would swallow a second subject).
  Under the local chars/4 estimator the exact boundary never appears, which is why this
  class of bug is only CI-visible. The checker itself must stay unchanged.
- `sacrebleu` is a test dependency — it is pinned in `requirements-benchmark.txt`
  (2.5.1); the plain `requirements.txt` does not carry it.
- `gh run list --limit 1` once returned a stale top entry; match `headSha` instead.
- **GitHub wraps workflow `run:` steps in `bash -e -o pipefail`.** A failing command
  aborts the script before any `status=$?` capture, so capturing exit codes needs the
  `status=0; cmd ... || status=$?` idiom. (Bite: first real-test run swallowed the
  answer-smoke failure this way — step "succeeded", no diagnostics posted.)
- The annotations API also carries GitHub's own annotations: Node-version deprecation
  notices and `Process completed with exit code N.` for `continue-on-error` steps —
  filter them when reading a run.
- A workspace rebuild resets the git checkout to the branch's ORIGINAL base while
  leaving working-tree files: detect via `git status` (everything "modified" + the
  session files "untracked"), then `git ls-remote` + fetch the session branch and
  `git reset --hard` to it. Verify untracked files against the commit with
  `git show <sha>:<path> | cmp -s - <path>` before resetting.
