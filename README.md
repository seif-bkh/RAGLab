# RAGLab

A transparent Arabic/French/English document-grounded QA lab. The supported
pipeline now uses **only two models**:

- **Embeddings:** `nvidia/nemotron-3-embed-1b`, native 2048 dimensions.
- **Answers:** `qwen/qwen3.8-max:free` through xKiro, with live free-price checks.

Retrieval uses the original query, local ChromaDB/cosine, and no separate chat
translation model. There is no model/provider fallback. Retired provider choices
and stale translation-enabled configuration fail before model calls.

Chunking defaults to the **restructure** strategy (`raglab/restructure.py`):
semantic normalization to hierarchy-explicit Markdown (page/gazette cleanup,
section-marker extraction, visual-order Arabic repair), context-breadcrumb
enrichment above every heading and table, then recursive structural splitting
over `["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "]` at the usual 220/40
budget. The old token-window mode remains as `CHUNKING_MODE=size`, and the
chunk fingerprint carries the mode so the two corpora can never be mixed.
`raglab/harness50.py` benchmarks the two strategies over a 50-question
ar/fr/en set (`raglab/questions_50.json`) with BM25-only retrieval:
restructure leads 47% vs 40% hit@1 (75% vs 42% on verbatim cases); see
`results/harness50/comparison.md` and `raglab/README.md` for the full table
and caveats.

The same A/B also runs on the **real models** — pinned NVIDIA nemotron
embeddings for both arms plus one retrieved, cited, validated answer (LLM
role: `nvidia/nemotron-3.5-lightning-30b-a3b` via the NVIDIA build
endpoint, with a Google free-tier fallback) — via
`.github/workflows/real-test.yml`. It spends API calls, so it is
**manual-only**: Actions → "RAGLab real test" → Run workflow, or push a
tag `real-test-*` (the dispatch API only sees the default branch). The
full tables land in the run's stdout and the `real-test-results`
artifact, and a compact key-number summary is posted as a check-run
annotation (readable via the annotations API). Green run so far:
34144251576 — restructure leads hit@1 (73–76% vs 67% for the legacy
220/40 arm, +15 pp on paraphrase, +27 pp on French queries) while the
legacy arm keeps perfect top-20 recall (see AGENTS.md §7).
`raglab/run_real_test.sh` runs the same two-arm evaluation locally
wherever the endpoints are reachable and `raglab/.env` is filled in.

```bash
cd raglab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # configure only NVIDIA_API_KEY and XKIRO_API_KEY
python app.py          # OR the interactive console: menus, provider/model
                       # switching, key keep/change/add, chat — everything
python -m uvicorn service:app   # OR the standalone HTTP microservice
                                # (pip install -r requirements-service.txt;
                                #  docs at /docs — see raglab/SERVICE.md)
python local_front.py           # the same console, driving the service over
                                # REST (menus, keys, ingest, chat, evaluate)
python main.py inspect --data-dir ../docs
python main.py ingest --reset --data-dir ../docs
python main.py query "What is Murabaha?" --query-lang en
python main.py answer "What is Murabaha?" --query-lang en
```

**Not production-ready for a banking service.** The measured Qwen profile passed
13/14 development rubric checks, 18/18 held-out answer checks and three synthetic
source-injection fixtures. That small, correlated test set is not a security,
legal-correctness or availability guarantee. Suitable for a supervised pilot with
approved nonconfidential documents, not unsupervised customer banking advice.

See [the CLI guide](raglab/README.md), [readiness assessment](raglab/READINESS.md),
and [measured model comparison](raglab/FREE_MODELS_REPORT.md). Historical reports
and fixtures remain as evidence; their earlier providers are not active choices.
The sample bank in `data/` is fictional; `docs/` contains the four real evaluation
documents. No UI, account access, cloud vector database, or orchestration framework.
