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

## Chat: ask a question, get a cited answer

```bash
./raglab/chat.sh --check                      # config, model, chunking, index size — no completion call
./raglab/chat.sh --ingest                      # first run only: embeds docs/ + data/ into ChromaDB
./raglab/chat.sh                               # the REPL
./raglab/chat.sh "ما رأس مال بنك التمويل العائلي؟" -k 8 --json
./raglab/chat.sh --show-context "quel est le délai de recours ?"
```

It answers with `nvidia/nemotron-3.5-lightning-30b-a3b` through `NVIDIA_API_KEY` (a free endpoint on
build.nvidia.com), thinking off by default — greedy, with the reply ceiling spent on the answer instead
of reasoning tokens; `--thinking` switches reasoning on and raises the ceiling to fit it. Retrieval, the
verbatim-evidence check and the private/live-question refusal are this file's own machinery, unchanged:
the answer arrives as claims carrying quotes that must really appear in the retrieved excerpts, so an
invented number comes back as `refused/invalid_output` with the model's raw reply shown rather than as
fluent prose.

A refusal is printed with the excerpts the model was handed, so it is diagnosable rather than
mysterious. `I cannot answer this from the supplied documents` is true whether the corpus is silent,
retrieval missed, or the question asks for personal or live data — so the chat says which of those it
has evidence for, previews the excerpts it supplied, and names what would likely change the outcome. Two
settings follow from measurements in this repo: the chat indexes at the **pinned 640/40** chunking
(read from `benchmarks/hard_harness_plan.json`, `--chunk-tokens` overrides) because the retrieval sweep
measured 82% whole-document recall there against 11% at the application's 220 default; and the context
ceiling is sized from `-k` (5 × 680 tokens), because `build_sources` drops an entire excerpt rather than
truncating it, so a fixed 3000-token ceiling would quietly answer from 4 of 5 hits. Retrieval runs in
its own collection (`raglab_chat`) so the app's index and the chat's never argue over one store.

Two labels matter. This is **not** the supported answerer above: `main.py answer` and the benchmark keep
xKiro Qwen and reject any substitution, so no score in this repo belongs to the chat model. And its
questions are not the harness's: the frozen sample and its numbers stay the harness's, while this path is
for reading the documents. In `chat` commands: `:k 8`, `:show`, `:lang ar|fr|en|auto`,
`:mode vector|rrf|blend`, `:log chat_turns.jsonl`, `:context`, `:quit`. `--data-dir PATH` replaces the
default corpus outright.

### Why an English question gets refused here, measured

The language hint a refusal prints is not folklore. Scoring the same corpus with BM25 over the pinned
640-token chunks (lexical only, no model, no API call): `i don't have a bank account and i have zero
money, can i buy a gamer pc worth 4K TND?` peaks at 9.48 on a passage of `Madkhal_Sayrafa_Islamiya`
stating that a certain development bank `لا يتعامل مع الافراد` — does not deal with individuals at all;
`what are the procedures of buying a vehicle?` peaks at 9.54 on an unrelated preamble. Asked in Arabic,
`ما هي شروط وإجراءات تمويل شراء سيارة او اجهزة اعلامية بالمرابحة؟` peaks at 14.72/14.42 on
`Guide_Interne_Operations_Bancaires_Islamiques` chunks #003 and #005, the ones carrying
`يجوز تمويل شراء عقار او سيارة او سلع...` and the rule that no final sale contract may be concluded
before the bank holds the goods. French lands in between, on the French product sheet rather than the
Arabic guide.

That is the vocabulary gap, which the embedding arm alone is asked to bridge — and two honest limits on
reading it as a verdict. Lexical overlap is not entailment: a zero-overlap English question can still
retrieve correctly, which is what the harness's 82% whole-document recall at 640 tokens measured for its
own English questions (formal, source-anchored ones, not conversational phrasing). And a refusal can be
*right* regardless of retrieval: no document in `docs/` states whether a person with no account and no
income may buy anything, so the model abstaining on that framing is the citation contract working, not a
miss. `:show` or `--show-context` prints what was supplied, which is how the two cases are told apart.

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
no new embedding/translation calls. The post-cleanup run completed 36 fresh Qwen
calls and passed the declared gates; no answer cache was replayed. This is a
regression over the same small fixtures, not fresh independent validation.

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
