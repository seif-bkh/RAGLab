# AGENTS.md — working contract for this repository

If you are an agent (or human) resuming work on RAGLab: **read this file first, follow it,
and keep it updated whenever you learn something that changes the truth.** It is the
contract between sessions. Facts here were verified by running things, not guessed.

Last verified: 2026-09-24 (branch `arena/01a0d2f5-raglab`).

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
- **Benchmark set**: `raglab/questions_50.json` — **50 cases** (2026-09-30, complete:
  owner approved the 2.1 matrix with «أنجز» and batches أ–د were authored the same
  session; ids q01–q50 with no gaps) over the four `docs/` documents only:
  `verbatim` (10), `paraphrase` (20), `cross-lingual` (15 = fr 6 + en 9),
  `out-of-scope` (5, must have NO expected match). Languages: 32 ar / 8 fr / 10 en;
  per document Loi 12 / Circulaire 10 / Guide 11 / Madkhal 12. Every
  expected_substring is a verbatim span of the ADOPTED corrected codices
  (post-2026-09-30 adoption), machine-verified present in its expected document's
  adopted-arm chunks AND distinctive corpus-wide (shared concepts like murabaha
  carry document-distinctive evidence); the design + execution record lives in
  `raglab/audits/QUESTIONS_MATRIX.md`. The five target-state categories and the
  multi-evidence schema will be RE-INTRODUCED as individually reviewed steps (2.4).
  All pre-2026-09-28 numbers are historical (their corpora/cases no longer match).
  Never copy from `raglab/questions.json`, `questions_real.json`, or
  `benchmarks/retrieval_dev.json`.
- **BM25-only A/B harness**: `raglab/harness50.py` (no API calls; deterministic).
- **Model-independence tool** (re-introduced 2026-10-01, step 2.4-ج):
  `raglab/answer_ab.py` — ONE retrieval per question fed identically to N answer
  models (`--model provider/model-id`, first is the reference); pairwise status
  agreement + source-Jaccard, substance-divergent question lists, per-model
  refusal/gate counts; offline-testable with injected factories. Running it
  (step 3.3) spends answer-model calls and happens via CI exclusively. Plans:
  `raglab/audits/PHASE2_MACHINERY.md`.
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

## 1b. Target-state transition: the gap analysis (2026-09-24)

The user delivered a **target-state design report** (external, Arabic) for evolving this
lab into a governed "banking RAG decision system": intent understanding + evidence plan,
sufficiency checks, conflict precedence, document governance, controlled vocabulary,
knowledge units with functional types, relations, structured numeric data, per-user
permissions, audit logging. Its transition plan's Phase 1 is exactly: match the
current-state spec against the target report, converting its assumptions into facts.

**`RAGLAB_GAP_ANALYSIS.md`** (repo root, Arabic, evidence-cited like RAGLAB_SPEC.md) is
that deliverable. Key outcomes (all verified in code this session):

- The target's assumed "simple RAG" current state is wrong in BOTH directions. Ahead of
  assumptions: post-generation citation gate (verbatim quote + numeric membership,
  `answer.py:100-219`), explicit output contract (`grounded-v1/v2`), structured refusal
  taxonomy, fingerprint staleness refusal, docstore content-hash versioning, structural
  chunking with breadcrumbs + Arabic visual-order repair, full hybrid machinery
  (BM25+RRF/blend), Arabic normalization shared by docs AND queries
  (`evaluate.prepare_query_text`), disciplined measured evaluation (NOT impressionistic).
- Behind assumptions (start-from-zero gaps): intent layer, evidence plan + deterministic
  sufficiency, conflict precedence, relations ("exception always retrieved with its rule"
  unguaranteed), document governance axes (effectivity/audience/authority), structured
  numeric path, per-user permissions (single shared service token only), audit log,
  partial/clarification answer states, reranker, controlled lexicon, stable unit ids.
- Decisive architectural fact: **the answer model never touches the query** (no rewriting;
  policy raises on translation: `pipeline_policy.py:16-17`), so retrieval is
  model-independent BY CONSTRUCTION — the target's "fix context, swap model" decisive test
  is implementable today via profile switching + the citation gate as automated judge.
- Transition-plan rescoping recorded in the doc: Phase 3 "add hybrid search" becomes
  "enable existing hybrid as default + measure"; permission/version mandatory filters
  depend on metadata that doesn't exist yet (Phase 4 must partially precede); Phase 6 is
  half-built already. Recommended next step (Phase 2): extend `VALID_CATEGORIES` in
  `evaluate.py` with the target's symptom categories (colloquial/synonyms/implicit/
  compound/ambiguous) — backward-compatibly — plus a `questions_v2.json` with expected
  evidence per question, then baseline vector vs rrf per category.
- Implementation constraints the target plan must respect here: endpoint freeze
  (additive only, §2.7), pinned-pipeline policy (no number attribution to other models),
  sandbox network (live runs only via CI `real-test.yml`), hard-harness files untouchable,
  offline gate green before every push.
- **Phase 2 RE-PLANNED on the docs/ corpus (owner direction, 2026-09-28)** —
  `RAGLAB_ROADMAP.md` (Arabic) remains the execution contract. The owner directed that
  the working corpus is `docs/` ONLY; the web-compiled Al Baraka round (and everything
  built on it) was fully reverted in one forward commit: sheets + chunk maps deleted,
  `questions_v2.json`/`answer_ab.py`/migration tool deleted, all code files restored to
  their pre-round state (evaluate 4 categories, harness50 without flags, service back to
  1.2.5 with the original /evaluate whitelist), q50 trimmed to its 30 docs/-based cases
  and the legacy set to 4. Three standing decisions recorded: (1) fictional/web corpora
  stay OUT of the repository; (2) the Phase-2 machinery (5 categories, multi-evidence
  schema, model-independence tool) is re-introduced ONLY as reviewed steps; (3) q50 is
  rebuilt to 50 docs/-based cases batch by batch with the owner verifying each batch.
  WORK PROTOCOL from now on: every step gets its own published schema
  (objective/inputs/actions/outputs/automatic checks/how the owner tests it/transition
  criterion/rollback) — no step runs inside another, no number is adopted unreviewed.
  The step sequence: docs audit (per document) → question-set plan → batches →
  baselines (BM25 local, then the live CI arm which decides the hybrid default BY
  DATA). ONE necessary coherence fix shipped with the revert: `main.py inspect` and
  `ingest` now default to the REAL corpus (../docs + raglab/data via
  `main.default_data_dirs`, mirroring chat.data_dirs) instead of data/ alone — the
  old default only exited 0 because the deleted sample sheets lived there, and CI
  runs bare `python main.py inspect`. Verified: 4 documents, 325 chunks, exit 0. The pre-2026-09-28 round remains in git history for archaeology only.


---

## 2. Hard constraints (never violate)

1. The hard harness pins chunking at **640/40 in `raglab/benchmarks/hard_harness_plan.json`**
   — do not touch that file or its behavior.
2. `ci.yml` + `raglab/run_tests.sh` reference **flat files in `raglab/`**
   (`tests_offline.py`, `test_nvidia_pipeline.py`, `test_hard_harness.py`,
   `main.py inspect`, …) — never move or rename them.
3. Question-set format follows `raglab/evaluate.py`
   (`load_question_set`): categories `{verbatim, paraphrase, cross-lingual, out-of-scope}`
   plus the five target-state categories re-introduced 2026-10-01 (step 2.4-أ,
   owner «طيب واصل المرحلة الثانية»): `{colloquial, synonyms, implicit, compound,
   ambiguous}` — `ambiguous` cases must carry a non-empty `ambiguity_note`;
   out-of-scope questions must have no expected match. Multi-evidence cases
   (step 2.4-ب) may carry `expected_substrings` (non-empty list of non-empty
   strings) scored by requirement completion. Plans + execution records:
   `raglab/audits/PHASE2_MACHINERY.md`.
4. **Never commit `raglab/.env` or any API key value.** The keys exist as GitHub Actions
   repo secrets named `NVIDIA_API_KEY`, `XKIRO_API_KEY` (and `GOOGLE_API_KEY` for the
   LLM fallback) — added by the user 2026-09-07. Reference them by name in workflows;
   the dev token cannot read or list secrets (403) — that is normal, not an error.
5. `raglab/results/` is gitignored ("generated data — never commit these"). Benchmark
   artifacts there are **regenerable**: `python harness50.py` rebuilds the BM25 A/B
   artifacts offline. If a workspace is rebuilt, re-run it rather than treating missing
   files as a loss.
6. Work only on the session branch (current session: `arena/01a08139-raglab`,
   pushed only there; the older `arena/01a07b8c-raglab` and other `arena/*`
   branches plus the four older workflows (`free-models.yml`,
   `hard-harness.yml`, `provider-catalogs.yml`, `retrieval-judge.yml`) belong
   to earlier sessions — do not touch them).
7. **Service endpoint freeze (user directive, 2026-09-23): never alter an
   existing HTTP endpoint** — not its path, method, request schema, response
   schema or semantics. The application integrating this agent depends on
   them staying exactly as documented in `raglab/CONTRACT.md`. Changes are
   **additive only**: new endpoints are welcome when a capability is missing;
   a new optional response field is acceptable only when a conforming client
   cannot break — when in doubt, add a new endpoint instead.
   `raglab/local_front.py` is the integration testbed: every capability must
   be exercisable through it over HTTP, exactly the way the integrating app
   will call it. Bump `SERVICE_VERSION` on every service change.

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
  `Loi_2016-48.pdf` (gazette, REVERSED word order, reliable digits — see audit),
  `Madkhal_Sayrafa_Islamiya.docx` (clean).
- PDF text comes out in **visual order with corrupted digits** → repaired only by the
  logical-order bigram scorer in `restructure.py`. **Never "repair" digits** in the
  pipeline arms; never re-derive facts from the raw digit soup; take authoring
  substrings from the repaired output. (2026-09-30, FULLY RESOLVED: ALL FOUR documents
  now run on their language-model-repaired corrected codices, owner-adopted — Loi
  «لا اعتراض، واصل» then the three others «طيب واصل» after the packages were presented
  at the adoption gate. Owner directive: «قم بالإصلاح اللغوي لباقي الملفات، واصل
  على نفس المنوال السابق، بدون استشارة». `restructure.py`'s `_ADOPTED_CODEX` maps
  all four docs to `audits/*_corrected.md` — Loi 215 entries / 29 batches;
  Circulaire 35 (all 28 corrupted digit tokens fixed against the JORT-anchored audit
  table; identity «عدد 80» kept per the standing owner decision); Guide 23 (all
  reversed hierarchical numbering corrected, context-anchored: stored
  «2.4-الوكالة»→4.2 vs stored «4.2-السلم»→2.4); Madkhal 21 («(000 59)»→«(59 000)»,
  «5اهم»→«5-اهم»). Each log is machine-checked by `audits/llm_repair_check.py`
  (fidelity/conservation/coverage — generalized: free-form location field + bare
  page-number skip class); 4 regression tests in `AdoptedCodexRepair` pin canaries
  and heading counts. AUTHORING RULE now: expected_substring for ALL documents
  matches the ADOPTED (corrected) corpus form; vetoes go through the repair logs.)
- **Madkhal AUDITED (step 1.4, 2026-09-28 — `raglab/audits/Madkhal_Sayrafa_Islamiya.md`,
  owner review pending)**: internal TRAINING material (an introduction to Islamic
  banking for Al Baraka Tunisia staff) — one of the four ORIGINAL docs/ files, it
  stays. Extraction CLEAN. It carries the corpus's richest RELIABLE numbers
  (1962 Tabung Haji, 1963/1967 Mit Ghamr, 1971 Nasser, 1975 Jeddah, 1977-1985
  wave, 15-20% growth, 500->1166 bn USD 2007->2012, 716 = 511 + 205 institutions)
  — it is the home of history/quantity questions (q13-15, q26-27, q40-41, q44 all
  live on it, all evidence verified present). ONE numeric defect found: 59,000 is
  stored as «(000 59)» (reversed digit groups) — quote as stored or avoid.
  Spelling without hamzas in places («القران الكريم», «تاسيس») — evidence uses the
  stored spelling. The document itself uses internal «» around proper names —
  never quote a span containing them (checker ambiguity). Structure: 5 sections;
  the 5th heading («5اهم منتجات الصيرفة الاسلامية») was not extracted (no
  separator after the digit) — its deposits/financing content sits in chunks
  #30-39. Quote check 31/31 verbatim.
- **Guide AUDITED (step 1.3, 2026-09-28 —
  `raglab/audits/Guide_Interne_Operations_Bancaires_Islamiques.md`, owner review
  pending)**: it is Al Baraka Bank Tunisia's INTERNAL sharia-compliance manual (Jan
  2022, authored by its compliance officer) — one of the four ORIGINAL docs/ files,
  it stays regardless of the deleted web-compiled sheets. Extraction is CLEAN
  (logical order). The historically noted "inverted decimals" is now precise:
  hierarchical section numbers are stored with their components REVERSED (stored
  «1.2-» = true 2.1) consistently in TOC and body — evidence quotes the STORED form.
  Digits and percentages are RELIABLE here (30%/5% screening, 3-year car age, SPOT
  = two business days, sharia-board decisions cited with numbers). Duplication
  found: share-screening rules appear twice (murabaha controls + participation
  controls — either hit is a correct answer), and the mudaraba wording is
  near-duplicated with the Circulaire (q04's evidence) — Guide mudaraba questions
  must use Guide-distinct evidence. Machinery notes: murabaha spans 9 chunks
  (#10-18), the cards section has no own heading (content inside chunk #49 under
  FX), the TOC itself is indexed (#01-03). The 5 existing questions are unaffected
  (evidence verified present). Quote check 45/45 verbatim.
- **Loi AUDITED AND OWNER-APPROVED (step 1.2, 2026-09-28 — `raglab/audits/Loi_2016-48.md`;
  the mandatory authoring rules in its section 13 are binding)**: identity clean (48-2016, 11 July 2016, JORT 58). 198 chapters in 10
  titles, content complete (no gaps); 6 chapter markers cosmetically garbled
  (1/39/99/133/144/195 — content present between neighbours). **Digits are
  RELIABLE in this document** (verified against the official French text — the
  exact OPPOSITE of the Circulaire). The dominant defect was REVERSED WORD ORDER
  body-wide — TREATED 2026-09-28 (owner decision, then REFINED after the owner's
  review of the first diff rejected whole-line flipping as insufficient): the
  extractor scrambles each PHYSICAL rendered line as ZONES — pure-Arabic runs
  arrive word-reversed, digit runs keep logical order, zones themselves are in
  logical order, sentence-terminal punctuation is re-emitted at the raw line's
  end, and words fuse with their numbers at zone boundaries. restructure now
  keeps PDF rendered lines un-reflowed in the loader (normalize_text
  reflow=False) and, when a document passes the dominance gate (>=20 lines
  scoring clearly better reconstructed, >=4:1), zone-reconstructs each rendered
  line: letters fused to digits split BEFORE zoning, Arabic runs un-reversed,
  digit islands kept, chapter-marker prefixes and trailing periods preserved,
  then paragraphs are re-formed. The law-49 tail after the president's
  signature stays in stored form (different extraction order; foreign law).
  Full before/after diff (2334 lines) + 485 residual-fragment proposals for
  owner decision: `raglab/audits/Loi_2016-48_repair_diff.md` (adoption gated
  on that review). Known residual: the title and signature blocks are
  quasi-logical zones that violate the model (listed in the diff). **RESOLVED
  2026-09-30: the algorithmic reconstruction is comparison material only — the
  owner adopted the language-model repair instead** (see `_ADOPTED_CODEX` in
  `restructure.py`; with the codex in place the dominance gate sees logical
  order and the zone reconstruction no-ops). AUTHORING RULE (post-adoption):
  expected_substring must match the ADOPTED corpus form —
  the 4 existing Loi questions were migrated and re-validated (30/30). Gazette
  page headers polluted 74/218 chunks — TREATED: whole line-initial span
  stripping removes 31/31 headers, with a corroboration guard so short body
  lines citing both عدد and الرائد are never dropped; remaining gazette-word
  hits are legitimate publication clauses (arts. 26/36/144-157). The PDF tail
  contains the start of ANOTHER law (49-2016, Raiffeisen loan) in chunks
  #216-217 — not our law, avoid in questions. One confirmed quantity loss: art.
  11's circular deadline stored as شهر where the official text says شهران (two
  months) — no quantity questions on that phrase. Islamic-finance cluster fully
  pinned from the French (arts 4, 11-16, 22, 23, 43, 54 committee+auditor,
  63-64 supervision, 75 Moucharaka exception, 76 investment-deposit obligations,
  169-170 sanctions, 198) with a chapter->chunk map and a clean stored-form
  evidence matrix in audit sections 9/12. Quote check 36/36 verbatim.
- **Circulaire AUDITED AND OWNER-APPROVED (step 1.1, 2026-09-28 —
  `raglab/audits/Circulaire_BCT_2019-08.md`)**. Two standing decisions: (1) the
  in-set identity stays 80/2019 (stored verbatim; official 8/2019 documented in the
  audit); (2) corruption handling = KEEP THE TEXT VERBATIM, the audit's corrections
  table becomes governed Phase-4 metadata, and question evidence must avoid corrupted
  spans (audit section 9 matrix complies). official identity is منشور عدد 8 لسنة 2019 of 14 Oct 2019
  (JORT 2019-092; BCT's Arabic 2019 list says 08). Our copy's header reads «عدد 80
  لسنة2019» — unresolved (extraction artifact or BCT Arabic-version numbering); kept as
  the conventional in-set identity because the stored corpus carries it verbatim.
  **28 digit tokens are corrupted in the STORED text**, including ALL chapter numbers
  14-20 (they render as «الفصل 11» four times + «الفصل81» + «الفصل 02») and every
  law-reference number in the visa. True values pinned from the JORT text: laws 89-1994
  (26 July), 35-2016, 48-2016 (11 July), chapters 11/42/54, committee opinion 8 of
  2 Oct 2019, circular chapters 14-20 by position. Body quantities are spelled in words,
  so chapter CONTENT stays usable. Question evidence MUST avoid corrupted spans — audit
  §9 lists one clean candidate evidence span per chapter. Convention: in audit reports,
  every «...» span is verbatim stored-corpus text, machine-checked by
  `raglab/audits/check_audit_quotes.py <report.md> <doc-name>` (run it after editing
  any report; it is the reusable verifier for steps 1.2-1.4).
- The Guide stores some decimals inverted — kept as-is (matches the source).
- `raglab/data/` is EMPTY by owner decision (2026-09-28): both fictional/web bank
  sheets (Banque Atlas, then a web-compiled Al Baraka set) were deleted with their
  chunk maps — the corpus is the four REAL documents in `docs/` and nothing else.
  `loader.load_all` tolerates the empty/missing dir (warning + continue), so a fresh
  clone stays green.
- Pipeline stats (restructure mode, LOCAL chars/4 estimator — CI's real cl100k is the
  arbiter, expect drift): 339 chunks total: Circulaire 13, Guide 54, Loi 233, Madkhal
  39 — ALL FOUR documents run on their owner-adopted corrected codices (2026-09-30)
  via `restructure.py`'s `_ADOPTED_CODEX` hook (Loi: 198/198 article heads + 33
  chapters; Circulaire: 20/20 article heads + 4 titles; Guide: 24 true-numbering
  heads; Madkhal: section-5 heading now extracted; every substantive codex line
  verified present in each markdown, 0 real loss). Adoption addenda: §18 (Loi), §12.1
  (Circulaire), §13.1 (Guide), §11.1 (Madkhal) in their audits; the master chunk
  locations live in `raglab/audits/MASTER_INDEX.md`. `RESTRUCTURE_RTL_REPAIR=0`
  reverts to the raw stored arm. The zone-reconstruction diff at
  `raglab/audits/Loi_2016-48_repair_diff.md` is comparison material only.
  `main.py inspect` exits 0 on all four. (Historical: Atlas-era 346, Al
  Baraka-era 367/369, pre-treatment docs/ 325, first-fix 326.)
- ACTIVE TRACK (owner decision 2026-09-28, supersedes the v2 adoption gate):
  `raglab/audits/Loi_2016-48_llm_repair.md` — language-model repair of the Loi,
  sentence by sentence, with per-entry notes (fix codes ح1-ح9, unresolved ش⚠) and a
  machine guard `raglab/audits/llm_repair_check.py` (source fidelity of every quoted
  line + word-inventory conservation vs documented fixes + algorithm-agreement stat).
  Batch 1 (pages 1-4, 32 entries) done and owner-approved ("عمل ممتاز");
  the E18 unresolved "اجل" was resolved by the owner (شراء اجل — deferred
  purchase, the essence of Salam). The strict-algorithm derivation was
  ABANDONED by owner decision (2026-09-28, "طرح الخوارزمية أي التخلي
  عنها") — derive_algorithm.py deleted; the track is language-model only.
  Batches 2-7 continue with the same per-entry format and the same machine
  guard (source fidelity + conservation). No baselines until this track
  completes.
- Loader: PDFs keep their physical rendered lines (normalize_text reflow=False)
  because the RTL zone reconstruction must un-reverse each rendered line
  separately; restructure re-forms paragraphs afterwards. DOCX/TXT reflow as
  before. The audit-quote checker whitespace-normalizes both sides, so quote
  checks are insensitive to the line-structure change (verified 90/36/45/31).
- General chunk-map gotcha (mechanism remains in repo): chunk-map offsets live in the
  NORMALIZED text domain (what `load_all` returns) — writing a map against the raw
  file text trips the source fingerprint guard in `semantic_chunking.load_map`.

## 7. Key results (update this section when numbers change)

- **CI green**: run 36395791517 on HEAD `d2e3f0c` (offline suite: 227 unittests
  — 180 core + 47 service — + 96 checks + inspect + pip check, under CI's REAL
  cl100k tokenizer). That commit is Step 0 of the re-planned Phase 2: full revert
  to the docs/-only corpus (§1b). Prior green (historical, reverted round):
  36000876551/35997175104/35997051758 on `d13267e`/`8b2f8c1`/`8a6f94a` (234
  tests, Al Baraka era); 35989037732/35988879011 on `742feb5`/`fa1c09f` (Phase‑1
  gap analysis).
- **Reverse-engineering spec** (961afe9): repo-root `RAGLAB_SPEC.md` —
  evidence-based spec of the whole system (pipelines, 25 routes, security,
  gap analysis), every claim cited file/symbol/line, tagged
  Observed/Inferred/Unknown. Investigation was read-only; no code changed.
- **Endpoint freeze directive recorded** (896dd5b, §2.7): existing HTTP
  endpoints never change (path/method/schemas/semantics); additive only;
  local_front is the integration testbed over HTTP; bump SERVICE_VERSION on
  every service change.
- **GET /config = console self-description** (392fae2, 1.2.4): chat_model +
  embedding_model (slot_display), vector_dimension (null until an index
  exists), editable flags, capabilities map, docs pointers — the one-call
  probe a generic adapter renders. Fixed a gateway console's "configures
  itself via CLI / non communiqué" display-only page premise.
- **Chunk browser** (32904f4, service 1.2.5): GET /chunks (paginated stored-
  chunk listing + per-document rollup + chunking_now) and GET /chunks/{id}
  (full chunk + neighbors + character overlap with the previous chunk,
  min 8 chars). Reads the STORED collection (what retrieval supplies), not
  the /inspect live preview. Sealed during ingest; 409 empty_index; 404
  unknown_chunk; bad_limit/bad_offset 400. local_front menu 15 (doc picker →
  paged rows → one-by-one reader); smoke 25 checks. Demo gotcha: POST /keys
  resets the injected generator — a dummy key on a fake-provider instance
  makes /answer attempt a REAL provider call (smoke "spend" check fails in
  the no-network sandbox; keyless runs pass 25/25).
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
- **RETRACTED 2026-09-28**: the Al Baraka-era baselines (q50 49/80/82, questions_v2
  52/67/67 etc.) were measured on a now-deleted corpus with now-deleted machinery —
  do not cite them. The Atlas-era block below is likewise historical. The CURRENT
  baseline is NONE until Phase 2 re-measures it on the docs/-only set (step 3.1 in
  RAGLAB_ROADMAP.md); until then treat every hit@k number in this file as history.
- **BM25-only A/B on the FULLY-ADOPTED corpus (2026-09-30, INTERIM — the current
  30-case set, not the final 50; regenerable via `harness50.py`)**: overall hit@1/3/5
  size-220/40 `44/76/80` → restructure (all four corrected codices) `60/92/100`;
  validation OK — every expected_substring present in the restructure arm. (With only
  the Loi adopted it was 64/96/100; the delta is Circulaire re-chunking 15→13 on a
  30-case set — chunk-boundary churn, not text quality; formal 3.1 baselines will be
  measured on the rebuilt 50-case set.)
- **BM25-only A/B (ATLAS-ERA, historical, set revised 2026-09-24)** (45 evaluable of- **BM25-only A/B (ATLAS-ERA, historical, set revised 2026-09-24)** (45 evaluable of
  50; k=20; full table in `raglab/results/harness50/comparison.md`, regenerable via
  `harness50.py`):
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
