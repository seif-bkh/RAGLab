# RAGLab — Full Project Report (First Commit to Latest Merge)

| Document control | |
|---|---|
| **Title** | RAGLab — complete development and state report |
| **Author** | AMIRA AYARI |
| **Date** | 2026-10-23 |
| **Repository** | RAGLab — transparent Arabic/French/English document-grounded QA lab |
| **History covered** | `73210f7` “Initial commit” (2026-09-04) → `79952d4` merge of pull request #21 into `main` (2026-10-23) |
| **Latest service version** | 1.5.0 (`SERVICE_VERSION` in `raglab/service.py`) |
| **Main branch tip** | `79952d4` |
| **Status** | Working tree clean; continuous integration green; not yet production-ready for unsupervised banking use |

> All statements below are evidence-based: commit hashes, dates, file paths and
> measured numbers come from the repository’s own history, source code, audit
> logs and frozen benchmark artifacts. No external or anecdotal claims are
> made. Personal identifiers are deliberately omitted; decision-makers are
> referred to by role (“the owner”, “the development team”).

---

## 1. Executive summary

RAGLab is a retrieval-augmented generation laboratory built around four real
Tunisian Islamic-banking documents. Over **50 calendar days (2026-09-04 →
2026-10-23)** and **376 integrated commits** (plus one merge commit), it grew
from a minimal local retrieval experiment into a governed, service-deployable
system with:

- a deterministic, auditable RAG pipeline (restructure chunking, hybrid
  retrieval, evidence-sufficiency gate, citation gate, fail-closed refusals);
- a frozen two-model stack — NVIDIA `nemotron-3-embed-1b` embeddings and
  xKiro `qwen/qwen3.8-max:free` answers — with no provider fallback;
- an HTTP microservice (`service.py`, v1.5.0) with a frozen client contract,
  optional shared-secret auth, Docker and native Windows packaging;
- an extensive measurement culture: a 50-question trilingual harness, a
  469-family hard harness, eight CI workflows and 472 unit tests plus 133
  offline checks;
- a corpus-governance programme that repaired the scanned Arabic legal text
  through reviewed, batch-by-batch linguistic corrections;
- explicit privacy boundaries, including output PII scrubbing and a
  customer-data boundary verification audit.

The latest merged work adds two governance audits, repository cleanup, and a
softer, referral-aware refusal wording while preserving every machine-readable
safety field.

---

## 2. Scope, sources and method

This report reconstructs the project from:

1. the full first-parent history of the merged line (376 commits, root
   `73210f7`);
2. the repository tree at `79952d4` — 249 tracked files, 86 Python modules,
   68 Markdown documents, 8 GitHub Actions workflows;
3. in-repo audit folders (`raglab/audits/`, 54 tracked items) and frozen
   benchmark artifacts (`raglab/benchmarks/`, `results/`);
4. the governed specification set: `RAGLAB_SPEC.md`,
   `RAGLAB_GAP_ANALYSIS.md`, `RAGLAB_ROADMAP.md`, `AGENTS.md`.

**History topology note.** One segment of the work was integrated as a single
squashed snapshot (`365820d`, 231 files) of development performed through
2026-10-05; earlier fine-grained commits remain reachable on sibling branches.
Where this report cites pre-squash work, it names the phase and the surviving
artifact rather than relying on the squashed diff alone. The pull request #21
merge combined two historically unrelated root lines (the original `main`
README commit and the long-lived development line); this is visible in the
graph and is intentional.

---

## 3. System overview at the final state

### 3.1 What RAGLab is

A trilingual (Arabic/French/English) document-grounded question-answering lab
over a **baked corpus of four real documents** in `docs/`:

| Document | Domain | Restructure chunks |
|---|---|---:|
| `Loi_2016-48.pdf` | Tunisian banking law | 233 |
| `Guide_Interne_Operations_Bancaires_Islamiques.docx` | internal Islamic-banking operations guide | 54 |
| `Madkhal_Sayrafa_Islamiya.docx` | Islamic finance primer | 39 |
| `Circulaire_BCT_2019-08.pdf` | BCT circular 80/2019 | 13 |
| **Total** | | **339** |

### 3.2 The supported pipeline (no fallback by design)

1. **Load & normalize** (`loader.py`, `restructure.py`): semantic
   normalization to hierarchy-explicit Markdown, page/gazette cleanup,
   visual-order Arabic repair, context breadcrumbs, then recursive structural
   splitting at a 220-token / 40-overlap budget. The adopted corrected codex
   is the only text indexed; raw `docs/` files are never modified.
2. **Embed & store** (`embedder.py`, `store.py`): native 2048-dimension
   NVIDIA embeddings in a local ChromaDB cosine collection, protected by a
   chunk fingerprint that rejects stale spaces.
3. **Retrieve** (`retrieval.py`): original-query vector retrieval by default,
   with BM25 keyword search, RRF fusion and score-blend modes; a deterministic
   reranker and bounded guided rounds; candidate pool of 20, answer window of 5.
4. **Gate** (`sufficiency.py`, `evidence_plan.py`, `decompose.py`): a
   deterministic evidence-sufficiency decision — *كافٍ / غير كافٍ / متعارض* —
   using declared anchor rules, df-scaled guards, cross-script bridges,
   field-vocabulary bridges and a colloquial concept vocabulary.
5. **Generate & validate** (`answer.py`): one bounded model call, then a
   citation gate enforcing verbatim quote membership, source allow-listing,
   and unsourced-number rejection. Anything failing the gate is withheld.
6. **Serve** (`service.py`): 12-factor HTTP microservice with a frozen
   contract (`CONTRACT.md`), request audit trail and output PII scrubbing.

### 3.3 Repository layout (249 tracked files)

| Area | Contents |
|---|---|
| `raglab/` (207 entries) | pipeline modules, service, CLI, tests, benchmarks, reports, audits |
| `raglab/audits/` (54 entries) | phase dossiers, corrected codices, repair diffs, decision logs, the two 2026-10-23 audits |
| `raglab/benchmarks/` | frozen chunk maps, retrieval dev/holdout sets, 469 accepted hard-harness families |
| `.github/workflows/` (8) | CI, real-model tests, hard harness, retrieval judge, answer A/B, free-model and provider catalogs, Docker image |
| `docs/` | the four baked corpus documents |
| `docker/`, `Dockerfile`, `Dockerfile.prod`, `docker-compose*.yml` | local and production packaging |
| `internship_report/` | 25-page bilingual report (Markdown/LaTeX/PDF) with architecture and timeline figures |
| `RAGLAB_SPEC.md`, `RAGLAB_GAP_ANALYSIS.md`, `RAGLAB_ROADMAP.md`, `AGENTS.md` | governed specification, gap analysis, binding execution roadmap, agent rules |

---

## 4. Chronological development report

### Phase 0 — Laboratory bootstrap (2026-09-04 → 2026-09-08)

The root commit `73210f7` “Initial commit” (2026-09-04) was followed the same
day by `13c72b7` “Add RAGLab: local multilingual RAG retrieval laboratory”.
The first week established the skeleton that still defines the project:

- chunking v2 with sentence-aware overlap and boilerplate exclusion;
- cross-lingual query translation (initially via Google Gemini free tier) and
  language-normalized variant fusion;
- the first GitHub Actions CI with offline inspection and real-API
  integration, failure annotations and artifact-based progress capture;
- friendly provider-SDK error handling and `.env`-driven configuration;
- score-blend hybrid retrieval as a third mode alongside vector and RRF, with
  CI A/B comparisons (vector vs hybrid) and a resumable translation cache to
  survive API quotas.

By 2026-09-08 the team could run the real `docs/` corpus end to end and
publish measured fusion verdicts (including the adopted `lambda=0.70`
blend parameter).

### Phase 1 — Provider benchmarking and model selection (2026-09-11 → 2026-09-16)

A measurement-heavy period produced the free-model and NVIDIA benchmark
reports (`FREE_MODELS_REPORT.md`, `NVIDIA_REPORT.md`) and the provider
catalog machinery. Candidate embedders and answer models were scored on
identical frozen sets, which later justified the pin to exactly two models
and the explicit removal of retired providers. A read-only model discovery
probe (`models_probe.py`) catalogued free/available offerings without
spending inference budget.

### Phase 2 — Packaging, governance protocol and corpus repair (2026-09-23 → 2026-09-30)

**Packaging (2026-09-23/24).** Native Windows setup without Docker
(`9417648`: PowerShell/CMD setup, server and front scripts), a multi-stage
production image with GHCR/Release publishing and offline tarball export
(`fa8e02b`, `Dockerfile.prod`, `docker/`), and dual-platform image export
(`c9e9abd`). Working rules were codified in `AGENTS.md` §9 — every handoff
carries copy-pasteable Windows and Linux command blocks (`b0f961f`,
`1576508`).

**Governance protocol.** `RAGLAB_ROADMAP.md` became the binding execution
contract: a strict step-and-gate protocol, owner-approved decision log, and
the rule that the four `docs/` files are the sole reference corpus (fictional
and web-assembled corpora were permanently removed).

**Linguistic repair of the Arabic codex.** The scanned law exhibited
reversed word order, corrupted numbers and broken headings. Through
twenty-nine reviewed batches (2026-09-28 → 2026-09-30) the team:

- healed 198/198 chapter headings and 233 sections of `Loi_2016-48`;
- settled a 98-item warning decision list
  (`Loi_2016-48_sh_decisions.md`), with the owner delegating full sign-off;
- adopted the corrected law codex, then repaired the remaining three
  documents — 35 entries (all 28 corrupted digits) for the circular, 23 for
  the guide, 21 for the primer;
- kept an automated guard (`llm_repair_check.py`) and per-document repair
  dossiers under `raglab/audits/`.

The operational layer applies repairs only inside `restructure.py`; the raw
source files remain untouched.

### Phase 3 → Phase 7 — The governed RAG machinery (2026-10-01)

A concentrated delivery day implemented roadmap Phases 3–7:

- **Phase 3 interventions:** governed deterministic query lexicon
  (`lexicon.py`, default off, owner-gated) and the deterministic feature-based
  reranker (`rerank.py`, later default-on after measured gains
  62/82/87 → 62/89/89 offline).
- **Phase 4:** governance-axis registry (`governance.py`) with fingerprint
  tests proving every registry entry maps to a real corpus document.
- **Phase 5:** question decomposition (`decompose.py`), the deterministic
  sufficiency and conflict engine (`sufficiency.py`), intent-gated relational
  expansion, and sufficiency fields on `/answer`.
- **Phase 6:** the delivery contract — stable unit identifiers on citations,
  the sufficiency commitment (refuse before any model call when evidence is
  absent), a bounded 500-entry request audit trail, and service **v1.3.0**.
- **Phase 7:** seven owner-directed experiments, recorded in
  `PHASE7_EXPERIMENTS.md` with a decision brief; neutral and positive results
  were documented rather than silently adopted.

### Phase 8 — Cross-script bridges and demand interrogation (2026-10-02)

Two owner-activated capabilities landed:

1. **Declared cross-script bridges** — curated Latin→Arabic equivalences
   (murabaha→مرابحة, sukuk→صكوك, …) plus field-vocabulary phrases, each
   df-checked so a bridge onto boilerplate can never anchor evidence. The
   owner’s live symptom (“what is murabaha?” refused despite 17 defining
   chunks) was closed, with a remote diagnostic (`diag_bridge.py`) proving
   zero regression.
2. **Demand interrogation (service v1.4.0, default ON)** — when nothing
   covers a practical question, one bounded model call re-expresses it as the
   nearest technical question the corpus covers; the paraphrase is
   re-evaluated by the same deterministic engines and answered only with full
   disclosure (`understood_as`, `scope_note`, interrogation seconds), or
   refused with an improved referral. Fail-closed by construction.

The same day brought Docker/local-front integration for the deployed server
and owner activation of the sufficiency fields and commitment gates.

### Phase 9 — Structural diagnostics and hardening (2026-10-04 → 2026-10-05)

- Stage-1 retrieval measurement probes and live topic-map comparisons under
  `raglab/audits/PHASE9_STRUCTURE/`;
- activation of the cross-script arm, revival of the guided rescue round on
  the deployed `/answer` path (it had been dead in production while measured
  offline), and df caps rescaled from counts to corpus shares;
- declared Arabic colloquial concept vocabulary (حانة/خمارة/بار/ملهى→خمر,
  قمار/كازينو/رهان→ميسر, …) with scope disclosure for re-aimed answers;
- refusal text split per reason (the previous single string made corpus
  refusals read like account-access refusals);
- the **duplicate-push 409 guard** (`document_already_in_corpus`): a
  content-hash check that prevented a measured production incident (859 of
  1713 chunks were duplicated re-pushes, inflating term frequencies) from
  recurring;
- the internship report was finalized at 25 pages (`365820d`).

### Phase 10 — Final audits, cleanup and humane refusals (2026-10-23)

Three commits (`33e2aae`, `7ab0acb`, `a19feed`) then the merge `79952d4`:

1. **Retrieval investigation** (`retrieval_probe_casino.py` +
   `RETRIEVAL_CASINO_PROBE.md`): a model-free, chromadb-free BM25 probe over
   the adopted 339 chunks documenting six findings — zero lexical overlap for
   Latin-only questions, two conflicting df realities (prefix-stripped
   sufficiency layer vs surface-form BM25), bare/prefixed rescue-term
   blindness, the gate’s coverage-vs-specificity behavior, a positive
   control, and a documentation drift in `rerank.py` headers.
2. **Customer-data boundary verification**
   (`CUSTOMER_DATA_BOUNDARY_VERIFICATION.md`): RAGLab-side verification of an
   upstream banking platform report — raw question text egresses to NVIDIA
   and xKiro before output scrubbing; the embedding cache retains an
   unscrubbed 120-character preview; four gated remediation items (H-1…H-4).
3. **Repository cleanup:** removal of three root-level chat-session HTML
   exports (3.5 MB, 2,116 lines), recoverable from history.
4. **Soft refusal wording (service v1.5.0 behavior):** `compose_refusal()`
   now builds the human-facing refusal for all six reasons — quoting the
   referral when present, otherwise using the “if you meant …” concept form,
   otherwise returning the legacy sentence byte-identically. Reason codes,
   referral payloads, audit fields and tests are unchanged;
   `SOFT_REFUSAL_ENABLED=0` restores the previous wording exactly.

Pull request #21 merged this work into `main` with all five CI checks green.

---

## 5. Version and milestone table

| Date | Milestone | Marker |
|---|---|---|
| 2026-09-04 | Repository root; local multilingual RAG lab | `73210f7`, `13c72b7` |
| 2026-09-04/05 | CI, hybrid retrieval, real-corpus runs | phase-0 commits |
| 2026-09-11/16 | Provider benchmarking; eventual two-model pin | `FREE_MODELS_REPORT.md`, `NVIDIA_REPORT.md` |
| 2026-09-23/24 | Windows-native + production image packaging | `9417648`, `fa8e02b`, `c9e9abd` |
| 2026-09-28/30 | Corrected Arabic codex adopted on all four documents | repair batches 1–29, `llm_repair_check.py` |
| 2026-10-01 | Phases 3–7 machinery; service **v1.3.0** | `PHASE5…7` dossiers |
| 2026-10-02 | Cross-script bridges; interrogation; service **v1.4.0** | owner-activated gates |
| 2026-10-04/05 | Phase 9 probes; concept vocabulary; 409 dedupe guard | phase-9 folder, `service.py` guard |
| 2026-10-05 | Internship report finalized (25 pages) | `365820d` |
| 2026-10-23 | Two audits, cleanup, soft refusals; PR #21 merged; service **v1.5.0** | `33e2aae`, `7ab0acb`, `a19feed`, `79952d4` |

---

## 6. Quality assurance and measured performance

### 6.1 Automated quality gates

- **472 unit tests** across `test_nvidia_pipeline.py`, `test_hard_harness.py`
  and `test_service.py`, plus **133 offline checks** in `tests_offline.py` —
  all green at the merge; CI runs both on every push.
- **Eight GitHub Actions workflows**, including a manual-only real-model
  workflow (spends API budget only on explicit dispatch or `real-test-*`
  tags) and a Docker image build/import smoke test.
- Frozen benchmark artifacts make historical numbers reproducible without
  re-spending inference calls; caches are content-addressed and gitignored.

### 6.2 Retrieval measurements (adopted 50-question set)

| Arm | hit@1 | hit@3 | hit@5 | hit@20 |
|---|---:|---:|---:|---:|
| BM25 offline (restructure) | 62% | 82% | 87% | 91% |
| Live vector run | 71% | 80% | 87% | — |
| Real-model A/B (restructure arm) | 73–76% | — | — | — |

The deterministic reranker lifted offline hit@3/5 from 82/87 to 89/89 without
regressing verbatim or out-of-scope behavior. Out-of-scope questions are
refused by the deterministic anchor gate even when retrieval returns
non-zero-score fragments — fail-closed behavior verified in testing.

### 6.3 Answer-level evidence

The selected-pipeline regression recorded 13/14 dev rubric passes, 18/18
held-out answer checks, 3/3 synthetic source-injection fixtures, and 9/9
private/live-data refusals performed by local guards without model calls.
The hard harness accepts 469 expert-reviewed trilingual question families
with expected behavior, required facts and forbidden claims.

---

## 7. Security, privacy and data boundaries

- **Output PII scrubbing** (`scrub.py`): emails, phone numbers, Tunisian
  RIB/IBAN and context-bound CIN numbers become `[LABEL]` placeholders at the
  response boundary, after the citation gate validates raw text. Amounts,
  dates and counts are deliberately untouched.
- **Capability guard** (`answer.py`): requests for personal balances,
  credentials or live rates are refused locally, with zero inference; the
  module explicitly documents that it is a conservative guard, not a security
  classifier.
- **Prompt-injection defense**: the generator prompt treats questions and
  sources as untrusted data and forbids role changes, tool use and
  background knowledge.
- **Service protection**: optional constant-time `X-Service-Token`
  authentication; masked key-status reporting; audit trail capped at 500
  PII-scrubbed entries in a gitignored directory.
- **2026-10-23 boundary audit findings (open, owner-gated)**: H-1 input-side
  scrubbing/400 rejection before external calls; H-2 stop storing the raw
  120-character embedding-cache preview in service deployments; H-3 make the
  service token mandatory when customer-facing platforms connect; H-4
  sub-processor disclosure (NVIDIA, xKiro) in the client contract.
- **Readiness statement** (`raglab/READINESS.md`): suitable for a supervised
  pilot on approved non-confidential documents; **not** ready for
  unsupervised customer-facing banking advice.

---

## 8. Deployment and operations

- `service.py` exposes the same runtime as the interactive console through
  HTTP, with the frozen `CONTRACT.md` (request/response schemas, error
  catalog, integration recipes), operator docs in `SERVICE.md`, and UI
  blueprints in `FRONTEND.md` / `COOKBOOK.md`.
- Docker Compose for local stacks; `Dockerfile.prod` for multi-stage
  production images with GHCR/Release publishing and offline tarball
  export/import scripts for both Linux and Windows.
- Native Windows path (no Docker): `setup.ps1`, `run_server.ps1`,
  `run_front.ps1` (plus CMD twins), documented in `WINDOWS.md`.
- A pure-stdlib REST console (`local_front.py`) provides the full operator
  experience — status doctor, key management, ingest, search, chat, smoke
  suites — against any running service.
- Container health check (`container_healthcheck.py`) and offline smoke
  script complete the delivery package.

---

## 9. Known limitations and open backlog

Carried forward from the audits and readiness assessment:

1. **Latin-only conceptual blind spot** — general English/French questions
   without a declared bridge term or in-corpus named entity can be refused
   even when Arabic evidence exists (remediation options R-1…R-4 in
   `RETRIEVAL_CASINO_PROBE.md`).
2. **Two tokenizer/df realities** — the sufficiency layer strips Arabic
   prefixes while BM25 matches surface forms, making bare guided rescue terms
   score zero in hybrid modes; alignment is proposed but not yet implemented.
3. **Coverage vs specificity** — the gate accepts general procedural
   evidence once the shape requirement is met, so the most specific chunk may
   stay outside the answer window; a quality condition for rare bridged
   concepts is proposed.
4. **Hybrid-default decision pending** — rrf/blend as the default remains an
   owner decision gated on live A/B evidence, per the 2026-09-24 roadmap rule.
5. **Customer-data items H-1…H-4** from the boundary verification await
   approval and implementation.
6. **No retention/deletion lifecycle inside RAGLab beyond the 500-entry audit
   cap**; upstream platform retention and cross-border transfer agreements
   remain legal-team work items.
7. The small, correlated evaluation set is not a security, legal-correctness
   or availability guarantee.

---

## 10. Repository hygiene record

- Generated artifacts (`.env`, virtual environments, `chroma_db/`, caches,
  `results/`, logs, `__pycache__`) are comprehensively gitignored; the working
  tree at report time contains no untracked or ignored files.
- The 3.5 MB chat-session HTML exports were removed in commit `7ab0acb` and
  remain recoverable from history if ever needed.
- The duplicate-document 409 guard prevents corpus inflation through
  re-pushes.
- The merged `main` tip (`79952d4`) holds 249 curated files; no large binary
  blobs beyond the four legitimate corpus PDFs/DOCX and the internship PDF.

---

## 11. Appendix A — Commit-to-phase mapping (integrated line)

| Phase | Period | Representative commits |
|---|---|---|
| 0 Bootstrap & CI | 2026-09-04 → 09-08 | `73210f7`, `13c72b7`, `428f419`, hybrid/CI series |
| 1 Benchmarks | 2026-09-11 → 09-16 | free-model/NVIDIA benchmark series |
| 2 Packaging & corpus | 2026-09-23 → 09-30 | `9417648`, `fa8e02b`, `b0f961f`, `1576508`, `c9e9abd`, repair batches 1–29 |
| 3–7 Core machinery | 2026-10-01 | phase 3–7 feature and experiment commits |
| 8 Bridges & interrogation | 2026-10-02 | bridge, activation, Docker and v1.4.0 commits |
| 9 Structural hardening | 2026-10-04 → 10-05 | PHASE9 probes, concept vocabulary, 409 guard, `365820d` |
| 10 Audits & merge | 2026-10-23 | `33e2aae`, `7ab0acb`, `a19feed`, merge `79952d4` |

## 12. Appendix B — Key numerical facts

- 376 integrated commits + 1 merge commit; 50-day span; 249 tracked files.
- 86 Python modules; ≈40.8k Python lines including tests; 68 Markdown documents.
- 4 baked corpus documents; 339 restructure chunks (13 / 54 / 233 / 39).
- 472 unit tests + 133 offline checks; 8 CI workflows; 469 hard-harness families.
- Service version 1.5.0; answer window k=5; candidate pool 20; audit cap 500.
- Two pinned models, zero fallback; one bounded interrogation call; one
  bounded partial regeneration per request.

---

*Prepared by AMIRA AYARI — 2026-10-23. This report reflects the repository
state at merge commit `79952d4` and is intended as a standalone historical
and technical record of the RAGLab project.*
