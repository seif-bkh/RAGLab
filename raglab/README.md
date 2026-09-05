# RAGLab — selected Qwen / Nemotron pipeline

The supported runtime is deliberately narrow:

| Role | Provider / model |
|---|---|
| Embeddings | NVIDIA `nvidia/nemotron-3-embed-1b`, native 2048 dimensions |
| Retrieval | Local persistent ChromaDB, cosine, original query, top 5 |
| Answers | xKiro `qwen/qwen3.8-max:free`, `grounded-v1`, cited evidence |

No separate query translation, provider fallback, or alternate answer model is
active. Historical experiments and immutable measurement JSON remain for audit;
they are not supported runtime choices. See [readiness](READINESS.md) and the
[measured comparison](FREE_MODELS_REPORT.md).

## Setup

Python 3.11 is tested. No LangChain/LlamaIndex or provider SDK is needed.

```bash
cd raglab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set NVIDIA_API_KEY and XKIRO_API_KEY locally. Never commit/paste their values.
```

**Existing `.env` migration:** set `ANSWER_PROVIDER=xkiro`,
`ANSWER_MODEL=qwen/qwen3.8-max:free`, `QUERY_TRANSLATION_ENABLED=0`, and
`QUERY_VARIANT_STRATEGY=original`. Remove obsolete provider settings. Old values
are rejected rather than silently causing calls to a retired model. The old
unused repository secret is not deleted automatically; it can be removed in
GitHub Settings if no other application needs it.

NVIDIA embeddings accept `passage` for documents, `query` for questions, float
vectors and `truncate=NONE`. Use dimension `0` (native) or `2048`, not a reduced
Matryoshka dimension. Invalid/nonfinite/zero vectors and stale spaces fail before
storage or retrieval. Embedding model changes require a collection reset.

## Inspect → ingest → retrieve → answer

```bash
python main.py inspect --data-dir ../docs
python main.py ingest --reset --data-dir ../docs
python main.py query "What are investment deposits?" --query-lang en
python main.py answer "Quelle est la définition du financement Salam ?" --query-lang fr
python main.py answer "متى تأسس بنك البركة تونس؟" --query-lang ar --show-context
python main.py evaluate --questions benchmarks/retrieval_dev.json
```

`query`/`evaluate` stop at retrieval. `answer` uses Qwen by default; explicit
`--provider xkiro --model 'qwen/qwen3.8-max:free'` remains accepted, but no other
provider/model is allowed. `--no-translation` is now redundant. Local vector/RRF/
blend retrieval controls remain diagnostic options; the measured default is
original-query vector retrieval, not hybrid search.

`inspect` makes no model calls and prints chunks, sources, language and token
counts. The real `cl100k_base` tokenizer downloads its public BPE file on first
use; set `TIKTOKEN_CACHE_DIR` for an explicit cache. An estimator may be used for
inspection with a warning, but regression measurements require the real tokenizer.

Ingestion is not additive: a nonempty collection requires `--reset`, checked
before model calls. A reset removes the selected local collection, not documents
or resumable embedding caches. Cache/index identities include model, endpoint,
embedding task, dimensions and tokenizer/chunk settings. Same-sized vectors from
different models are never treated as interchangeable.

## Grounding, free pricing and credentials

- The model receives only the question and retrieved source context, not gold
  answers or evaluation labels. Context is token-bounded; optional neighboring
  chunks are from the same source, never selected by ground truth.
- Rendered claims need source IDs and contiguous evidence quotes that match after
  normalization. Malformed JSON, invented citations and unmatched quotes fail
  closed. **Quote membership is not semantic entailment.**
- Private-account/live-data questions are refused locally before provider or index
  access. The system has no banking accounts, passwords, transaction tools or live
  exchange-rate feed. The regex guard is not a complete security classifier.
- Before Qwen calls, xKiro's live catalog must list the **exact** SKU at free access
  tier and zero USD input/output/cache prices. Paid siblings and model substitutions
  are rejected. If pricing/availability changes, fail closed—do not choose another
  model. This is an advertised-price check, not an independent billing audit.
- Each service receives only its own environment credential. Gateway redirects
  are disabled; errors are redacted. No credentials or reasoning deltas are exported
  in measurements. Bounded pacing/retries and explicit provider errors remain.
- xKiro is a gateway and reports the requested model across routing. Its response
  label does not independently verify an upstream engine/version. Free service
  carries no demonstrated production SLA.

Answer/evidence JSON is saved in `results/`; use `--output PATH` to choose a file.
Translation and answer caches retain their historical implementation tests, but
only the answer cache is used by the selected chat path. Cache writes are atomic
and protected across instances **within one process**; simultaneous writer
processes are unsupported. Local persistence is not a production backup strategy.

## Tests and selected-pipeline regression

```bash
pip install -r requirements-benchmark.txt  # includes reference-metric test dependency
./run_tests.sh --offline --no-push
python -m pip check
```

Ordinary pushes run compile/offline checks only. Legacy multi-model workflow and
runner entry points are retired. Historical low-level provider implementations
remain as offline test fixtures/utilities, not active integrations or defaults.

The **Qwen and Nemotron pipeline regression** Actions workflow is explicitly
triggered by `benchmarks/free_provider_plan.json` or manual dispatch. Its plan is
validated to contain only xKiro Qwen, original-query retrieval and grounded-v1.
It reuses a hash-verified native Nemotron retrieval artifact from the four real
Arabic PDF/DOCX files: 836 chunks, size 220 / overlap 40, 2048 dimensions. It makes
no new embedding/translation calls. All selected-model development, holdout and
injection checks can make fresh client calls; cache usage is disclosed.

With the frozen artifact already downloaded to `results/frozen_native/`, run:

```bash
python main.py benchmark
# or ./run_tests.sh --benchmark
```

The workflow downloads the artifact automatically. Artifacts expire, so future
reproductions may need a newly versioned source snapshot. A separate read-only
Qwen catalog diagnostic uses only `XKIRO_API_KEY`; no retired provider key is read.

Results are saved under `results/free_models/` and as neutral GitHub Checks with
actual claims/evidence, configurations and errors. A completed job is not a
production certificate. Declared quality gates and `production_ready=false` are
separate. No test labels or historical measurements are rewritten after failures.

## Production boundary

This remains a **local evaluation lab / supervised pilot**. Before a banking
service: approved model hosting/retention and version guarantees, independent
Arabic/French banking/legal evaluation, broader adversarial tests, robust PDF/OCR
and structural metadata, deployment access controls/document isolation, load/SLA
validation, backups and operational monitoring are needed. No UI, authentication,
transaction integration or cloud vector database is added by this model cleanup.

See [READINESS.md](READINESS.md) for the prioritized blockers. Only approved
nonconfidential material should be sent to a trial/free gateway.
