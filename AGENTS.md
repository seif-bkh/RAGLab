# AGENTS.md — working contract for this repository

If you are an agent (or human) resuming work on RAGLab: **read this file first, follow it,
and keep it updated whenever you learn something that changes the truth.** It is the
contract between sessions. Facts here were verified by running things, not guessed.

Last verified: 2026-09-14 (branch `arena/01a09f30-raglab`).

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
- **Stale-index handling, fixed 2026-09-14 (session `arena/01a09f30-raglab`)**,
  after a user's `front chat>` question came back as
  `HTTP 409 {"reason": "answer_refused", "error": "[store] collection is STALE
  — … Rebuild with:  raglab ingest --reset"}`. The refusal was correct (an
  index built at s220 was being asked for s440 chunks — different collection
  size/overlap, same `raglab_app_*` name), but nothing said so usefully and the
  hint named a command that does not exist. Now:
  `store.StaleCollectionError(RuntimeError)` carries `.stored`/`.current`/
  `.rebuild` (module constant `store.REBUILD_HINT` = `python main.py ingest
  --reset`, the real CLI); `ensure_fresh_chunks(collection, cfg, *,
  rebuild_hint=None)` still satisfies every `except RuntimeError` caller;
  `service.py` adds `REBUILD_HINT`, `stale_index_error()`,
  `Runtime.index_state()`, reports `index.stale` (+ `current_chunk_fp`,
  `rebuild`) in `/health`, and maps staleness to `409 reason="stale_index"`
  with both fingerprints in `/search`, `/answer` and `/evaluate` (generic
  `*_refused` reasons unchanged); `local_front.py` has `index_freshness()` +
  `failure_lines()`, warns in the banner/status/chat prompt, offers the
  rebuild and re-asks the same question, and `--smoke` gained a stale branch
  (21 checks instead of the ready-state live calls, zero model calls).
  `tests_offline.py`'s staleness checks now pin the real command and the typed
  attributes; `test_service.StaleIndexTest` + `LocalFrontStaleIndexOverHttp`
  pin the whole contract over TestClient and real HTTP.
- **Provider-failure diagnosis, fixed 2026-09-14 (same session)**, after a chat
  where the greeting and retrieval worked (20 chunks) and the answer came back
  as `[front] status=error/provider_error · model=gemini-2.5-flash-lite ·
  0.453s` with NO reason. Cause: `AnswerGenerator.answer` returns `error`,
  `http_status`, `retry_after_s`, `provider_ok`, `raw_preview` and
  `served_model` for exactly this, `app.py`/`chat.py` print them — and
  `service.py`'s `/answer` response dropped every one of them, so the front
  (and any REST caller) could not tell a 429 quota from a 404 model ID. Fixed:
  the response now carries them (`raw_preview` only with the new
  `include_diagnostics=true`), `nvidia_api.ProviderCallError` is the shared
  base for provider transport errors (status_code/retry_after; `NvidiaAPIError`
  subclasses it, service.py catches the base), `llm_smoke.google_call` returns
  `GoogleCallError` (a `str` subclass, so phase B's logging/JSON is unchanged)
  with `.status_code`/`.retry_after` parsed from the Retry-After header or
  Gemini's own "retry in 33.5s" wording, names a blocked prompt instead of
  leaking `KeyError: 'candidates'`, reports an empty MAX_TOKENS reply as a
  provider outcome, and filters candidates on `supportedGenerationMethods`
  containing `generateContent`; `profiles.GoogleChatClient` raises
  `ProviderCallError`, and when the selected model fails it tries the remaining
  free-tier candidates (llm_smoke phase B behaviour) reporting `served_model`;
  `local_front.provider_hint()` classifies the status into an action (429 →
  wait/other model, 401/403 → menu 3 keys, 404/400 → menu 2 model), one-shot
  `--ask`/`--search` never prompt. Tests: `test_service.AnswerProviderFailureTest`
  (stubbed Gemini through the real client) + `test_nvidia_pipeline.GoogleFreeTier`
  (7 checks: 429/Retry-After classification, blocked prompt, MAX_TOKENS
  ceiling, candidate filtering, fallback + all-fail reporting).

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
- **Every reply to the user ends with a "run it locally" block** (standing
  instruction, 2026-09-14): pull → install → run → verify, copy-pasteable on Linux,
  reflecting what actually changed in that reply. Canonical commands: §9.

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

- **CI green**: run 34838102099 on HEAD `c421749` (provider-failure
  diagnosis: 211 offline unittests — 201 + 7 GoogleFreeTier + 3
  AnswerProviderFailureTest — + 91 checks + inspect + pip check), after a local
  `run_tests.sh --offline` at `EXIT=0`. Stale-index series: 34827911512
  (`0cb7be6`) and 34827605040 (`f9c9f53`), 34827746328 (`59c072d`). All follow
  a green local gate.
- **Offline gate, this session's HEAD (`arena/01a09f30-raglab`, 2026-09-14)**:
  `PYTHON=/home/user/RAGLab/raglab/.venv/bin/python ./run_tests.sh --offline`
  → EXIT=0: 201 unittests (180 + 19 service + 2 new stale-index cases) + 91
  checks + `main.py inspect` (21 chunks over `data/`) + `pip check`, log
  `raglab/logs/test_run_20260914_092005.log`. The venv at `raglab/.venv` was
  rebuilt this session (PyPI reachable; ~2 min for
  `requirements-benchmark.txt`); `run_tests.sh` honors `PYTHON=`.

- **CI green**: run 34582141711 on HEAD `f73c2bd` (offline suite: 199 unittests
  — 180 + 19 service, incl. the local_front-over-real-HTTP case and the new
  ConsoleEndpointsTest — + 89 checks
  + inspect + pip check).
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
- **Every "how to fix it" string must name a command that RUNS.** The stale
  refusal said `raglab ingest --reset`; no entry point in this repo is called
  `raglab` (`main.py`'s parser is `prog="main.py"`, the chat REPL is
  `./raglab/chat.sh --reset --ingest`, the front is
  `python local_front.py --ingest --reset`, the service is
  `POST /ingest?reset=true`) — the user could not have followed it. Any new
  hint: check it against a real parser/endpoint before shipping.
- **The collection name carries provider/model/mode but NOT size/overlap.**
  Changing the chunk size — the front's settings menu (menu 12), `POST
  /profile`, or `RAGLAB_CHUNK_SIZE_TOKENS` at boot — therefore lands on the
  SAME collection and invalidates the index by design. That is the guard
  working, not a bug; `/health.index.stale` and `409 stale_index` are how you
  see it before asking.
- **The console is the reference for what a REST response must carry.** Two
  bugs in a row came from `service.py` dropping fields the console already
  printed (`raw_preview`, `error`, `http_status`, `retry_after_s`,
  `provider_ok`, `served_model`): the front showed `status=error/provider_error`
  with nothing else. When adding a result field in answer.py/chat.py, check
  the service response the same day.
- **A provider failure must never read as an answer.** `status=error` +
  `provider_ok=false` is the contract; the answer TEXT in that case is a
  localized "service temporarily unavailable" placeholder (ERRORS in
  answer.py). A test that only asserts HTTP 200 on `/answer` passes on a dead
  model — assert the status field too.
- **`llm_smoke.google_call` returns its error as a `str` SUBCLASS**
  (`GoogleCallError`) with `.status_code`/`.retry_after`: phase B prints it and
  stores it in the smoke JSON, so it must stay string-shaped. `GoogleChatClient`
  is the only code that reads the attributes.
- **`Runtime.switch()` and `POST`/`DELETE /keys` set `_generator = None`** (a
  new key/profile must invalidate clients built with the old one), so a test
  that injects `generator=object()` loses it the moment it switches the
  profile or triggers the front's `/keys` round-trip → `/answer` turns into
  `503 missing_api_key`. Build a stale index with a SECOND app over the same
  `CHROMA_DIR` (two `chromadb.PersistentClient`s on one path work in-process),
  or keep a dummy `KIRA_API_KEY` set so the suite skips its round-trip.
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
  **With UNCOMMITTED work in the tree (the usual case on turn 2+ of a session),
  that `git reset --hard` destroys it** — realign like this instead:
  1. `cp -a . /tmp/<name>_backup` (cheap insurance);
  2. verify the files you did NOT touch today are byte-identical to the remote
     commit (`git show FETCH_HEAD:<path> | cmp -s - <path>`) — that is what
     proves the tree is "remote + today's edits";
  3. `git stash push -m wip` → `git reset --hard FETCH_HEAD` → `git stash pop`;
  4. the pop CAN conflict (the stash was diffed against the OLD base, and those
     files may have been committed in between): restore each conflicted file from
     the step-1 backup, then `git reset -q` to clear the index's unmerged state,
     then confirm `git diff --stat` lists only today's intended edits and
     `cmp -s` each restored file against the backup.
  A rebuild also deletes `raglab/.venv` (gitignored, outside the snapshot):
  recreate it (`python3 -m venv .venv && .venv/bin/pip install -q -r
  requirements-benchmark.txt`, PyPI is reachable, ~2 min) before trusting any
  local gate.

## 9. Runbook — the user's own Linux machine (verified, keep current)

The user runs this locally and reports transcripts (`front chat>` = the service +
`local_front.py`). Verified end-to-end on a **fresh clone of this branch**
(`249977c`) on 2026-09-14: clone → venv → `main.py inspect` → `run_tests.sh
--offline` (EXIT=0) → `uvicorn service:app` → `local_front.py --status`.

```bash
# --- 1. get the code (first time) -----------------------------------------
git clone -b arena/01a09f30-raglab https://github.com/seif-bkh/RAGLab.git RAGLab
cd RAGLab

# --- 1b. ...or switch an EXISTING clone (older/shallow/single-branch) ------
git remote set-branches origin '*' && git fetch origin && git switch arena/01a09f30-raglab
git pull --ff-only

```

**`fatal: invalid reference: arena/01a09f30-raglab` is expected on a clone made
before this branch existed, or on any `--single-branch` / `--depth 1` clone**:
`git switch` only reaches refs the clone already has, and a shallow clone maps
`origin/main` alone (`remote.origin.fetch` is that single branch) — fetching the
branch is not enough there, git even refuses `--track` with "starting point
'origin/...' is not a branch" until the refspec covers it. `git remote
set-branches origin '*'` widens the refspec, `git fetch origin` then creates the
remote-tracking ref, and the DWIM switch (with tracking) works. Verified on this
branch against a normal clone, a clone with the branch ref removed, a
`--single-branch` clone and a `--depth 1` clone: all four land on `f983d95`
with upstream set and `git pull --ff-only` clean. If `git fetch origin` answers
`couldn't find remote ref`, `origin` is not seif-bkh/RAGLab — check
`git remote -v` and add it: `git remote add upstream
https://github.com/seif-bkh/RAGLab.git`.

```bash
# --- 2. Python env (3.11; the versions CI pins) ---------------------------
python3 -m venv raglab/.venv
raglab/.venv/bin/pip install -U pip
raglab/.venv/bin/pip install -r raglab/requirements-benchmark.txt   # tests+service
#   core CLI only:  -r raglab/requirements.txt
#   service only:   -r raglab/requirements-service.txt
#   hard harness:   -r raglab/requirements-harness.txt

# --- 3. keys (never committed; masked to 8 chars everywhere) --------------
cp raglab/.env.example raglab/.env        # fill NVIDIA_API_KEY / XKIRO_API_KEY / GOOGLE_API_KEY
#   or ask the console: menu 3, or POST /keys on a running service

# --- 4. no-key smoke (works with zero keys and no network) ----------------
cd raglab
.venv/bin/python main.py inspect
PYTHON="$PWD/.venv/bin/python" ./run_tests.sh --offline     # must print EXIT=0

# --- 5. run the service (what `front chat>` talks to) ---------------------
.venv/bin/python -m uvicorn service:app --host 0.0.0.0 --port 8000
#   docs UI: http://localhost:8000/docs

# --- 6. drive it (second terminal, same folder) ---------------------------
.venv/bin/python local_front.py --status          # doctor: profile, index state, keys
.venv/bin/python local_front.py --ingest          # first run: build the index
.venv/bin/python local_front.py --ingest --reset  # rebuild (stale index / new chunk size)
.venv/bin/python local_front.py --interactive     # the chat REPL (front chat>)
.venv/bin/python local_front.py --ask "ما هي المرابحة؟"
.venv/bin/python local_front.py --smoke           # state-aware endpoint suite

# --- 7. console without a service (same runtime, direct) ------------------
.venv/bin/python app.py                           # the 13-menu console
./chat.sh --ingest                                # chat.sh: its OWN collection (raglab_chat, 640/40)
```

Facts that save round trips:
- `local_front.py` defaults to `http://localhost:8000` (override `--base-url`, env
  `RAGLAB_SERVICE_URL`); exit codes 0 ok / 1 failed / 2 unreachable.
- The **service profile must match the index**: `RAGLAB_CHUNK_SIZE_TOKENS`,
  `RAGLAB_CHUNKING_MODE` and provider/model decide the fingerprint. `/health` shows
  `index.stale`; a mismatch refuses with `409 stale_index` and the front offers the
  rebuild (§1, §8). Console and service sharing one `chroma_db/` is the usual trap.
- Docker path (`docker compose up --build`, repo root, port 8000) is documented in
  README/docker-compose.yml but **was not verified in this sandbox** (no docker binary).
- `tiktoken` downloads `cl100k_base` on first use; without network the chunker falls
  back to the chars/4 estimator (marked in its logs, different boundaries — CI uses
  the real BPE).
