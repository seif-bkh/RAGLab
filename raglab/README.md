# RAGLab — multilingual retrieval and grounded-answer lab

RAGLab keeps raw, inspectable provider calls and a local ChromaDB index. It has
no web UI, authentication, transaction tools, or cloud vector store. Generation
is now **optional**; `query` and `evaluate` still stop at retrieval.

**[Current measured results and production blockers](NVIDIA_REPORT.md).** The
experimental defaults are Nemotron embeddings, Kimi/banking-v2 translation, and
DeepSeek/grounded-v1 answers (the only completed development answer profile so
far). Held-out generation and security testing are incomplete; none is a
production-certified choice.

## Setup

Python 3.11 is the tested interpreter.

```bash
cd raglab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-benchmark.txt
cp .env.example .env
# Set NVIDIA_API_KEY in .env. Never commit it or paste it into logs/chat.
```

`requirements.txt` is sufficient for the CLI. `requirements-benchmark.txt` adds
sacreBLEU for chrF++ reference-translation diagnostics. Gemini, Jina, local
Hugging Face, OpenAI, Cohere, and Voyage embedding implementations remain
available in `config.py`; optional providers need their corresponding keys/SDKs.
NVIDIA uses the standard library's HTTPS client, not a new orchestration SDK.

### Exact NVIDIA model contracts

| Role | Model ID | Contract |
|---|---|---|
| Embedding | `nvidia/nemotron-3-embed-1b` | `input_type=passage` for documents, `query` for questions; `encoding_format=float`; `truncate=NONE`; native 2048 dimensions |
| Translator / answerer | `moonshotai/kimi-k3` | `reasoning_effort=low`, temperature 0, fixed seed; only final content consumed |
| Translator / answerer | `deepseek-ai/deepseek-v4-pro-0813` | `chat_template_kwargs={"thinking": false}`; final content only |
| Translator | `nvidia/riva-translate-4b-instruct-v2` | supported language-pair system message, e.g. `en-ar`, then raw source text; French↔Arabic uses an explicit English pivot; never sent the general LLM translation prompt |

The hosted embedding endpoint does **not** accept reduced dimensions for this
model: use `NVIDIA_EMBEDDING_DIM=0` (native default) or `2048`. Model-card claims
about locally slicing vectors are not the hosted API contract. Oversized inputs
fail rather than being silently truncated.

Official contracts:
[embedding](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-embed-1b-infer),
[Kimi](https://docs.api.nvidia.com/nim/reference/moonshotai-kimi-k3-infer),
[DeepSeek](https://build.nvidia.com/deepseek-ai/deepseek-v4-pro-0813),
[Riva](https://docs.api.nvidia.com/nim/reference/nvidia-riva-translate-4b-instruct-v2).

No catalog-based substitution is performed. In particular, results for an older
Kimi/DeepSeek model must not be labeled as results for these IDs. Translation
fallbacks are empty by default and forcibly disabled in benchmarks.

## Inspect, ingest, retrieve

Choose **one** corpus for an experiment:

```bash
# Real documents (four Arabic PDF/DOCX files)
python main.py inspect --data-dir ../docs
python main.py ingest --reset --data-dir ../docs
python main.py query "What are investment deposits?" --query-lang en
python main.py evaluate --questions questions_real.json

# Fictional French/Arabic bank product sheets (separate index rebuild)
python main.py inspect
python main.py ingest --reset
python main.py evaluate
```

`inspect` prints every chunk, source, language, heading, token count and notes.
It makes no model API calls. On first use tiktoken downloads its public BPE file;
set `TIKTOKEN_CACHE_DIR` if you want an explicit cache location. Inspection can
fall back to an estimator with a warning; **benchmarks require the actual
cl100k_base tokenizer**. The tokenizer identity is part of the chunk fingerprint
so estimator-built and real-BPE-built indexes cannot be silently mixed.

Ingestion caches vectors after every batch. Changes to chunk parameters,
embedding model, endpoint or dimension require `ingest --reset`. A reset deletes
the selected local collection, not your documents or resumable embedding cache.
A nonempty collection requires an explicit reset; additive re-ingests are refused
before model calls so old document versions cannot silently remain under reused IDs.
Queries and passages have distinct cache entries; same-dimensional vectors from
different models are not interchangeable. Invalid, zero-norm, nonfinite,
misindexed or wrong-dimensional NVIDIA vectors fail before storage.

### Retrieval variants

```bash
python main.py query "What is Salam financing?" --query-lang en --no-translation
python main.py query "What is Salam financing?" --query-lang en --variant-strategy translated
python main.py query "What is Salam financing?" --query-lang en --hybrid
python main.py query "What is Salam financing?" --query-lang en --hybrid-blend
```

- `original`: original query only (cross-lingual embedding baseline).
- `best`: original plus corpus-language translations, fused by the legacy
  per-variant relative-score method. This is a ranking heuristic, not confidence.
- `translated`: use corpus-language queries when available; a query already in
  the corpus language is left alone.
- `--hybrid`: vector/BM25 reciprocal rank fusion.
- `--hybrid-blend`: `lambda × cosine + (1-lambda) × normalized BM25`.
- `--lang ar|fr|en`: filter both vector and keyword results by document language.

CLI, evaluation and answer generation share `retrieval.py`. Retrieval scores
cannot be treated as probabilities or used as an uncalibrated refusal threshold.

## Query translation

Change `NVIDIA_TRANSLATION_MODEL` in `.env` to any exact translator ID above.
`QUERY_TRANSLATION_PROMPT=basic-v1` is a direct translation baseline;
`banking-v2` adds domain terminology and entity-preserving Riva few-shots.
No benchmark answer facts or expected substrings appear in either prompt.

Kimi and DeepSeek translate numbered batches. Riva translates one source query
per request on English-centric pairs. Its official chat template does not
recognize `fr-ar`/`ar-fr`: these pairs use two requests (`fr-en` then `en-ar`,
or `ar-en` then `en-fr`) with the **same Riva model**. The route and intermediate
English text are cached and recorded. The first retrieval run caught this
adapter bug through the wrong-script guard; no such outputs entered retrieval. Output count, numbering, nonempty text, script and numeric
preservation are validated. Model, endpoint, source/target language and prompt
version scope the cache. Fallback results, if you explicitly enable fallbacks,
are stored under the **actual** model and retain provenance.

For an ordinary CLI query, translation failure can degrade to the original
query, with a warning and recorded failure. Set `QUERY_TRANSLATION_STRICT=1` to
fail instead. Benchmarks always use strict mode and zero fallback models.

HTTP retries are bounded, use pacing, respect Retry-After, and never retry
401/403/404 as transient errors. A long Retry-After is deferred rather than
ignored. Kimi/DeepSeek can use SSE streaming to avoid waiting for a buffered
response. Incomplete streams, output truncation and model substitutions are
errors, not successful results. No reasoning content is used as translation or
an answer. Free/trial endpoints have no production latency guarantee.

## Optional grounded answers

```bash
python main.py answer "Which year was Bank Al Baraka Tunisia founded?" --query-lang en
python main.py answer "Quelle est la définition du financement Salam ?" \
  --query-lang fr --model moonshotai/kimi-k3 --neighbor-radius 1 --show-context
```

`answer.py` asks the selected Kimi/DeepSeek model for JSON claims, each with a
source ID and a contiguous verbatim evidence quote. It renders only validated
claims and citations, and saves the complete answer/evidence record in `results/`.
A token budget bounds context. Neighbor expansion adds adjacent chunks from the
same source, never a ground-truth-selected passage.

The generator refuses unsupported questions, empty context, private-account or
live-data requests. It has no account access, tools, passwords or transaction
capabilities. Invalid JSON, invented citations/quotes, empty claims or truncated
provider output fail closed. API errors are reported separately from a genuine
"not in the documents" refusal. Use `--output PATH` to select the saved JSON.

**Limits:** quote membership is not semantic entailment. A real quote can still
be misinterpreted or cited for the wrong claim. Prompt-injection defenses and
capability regexes are safeguards, not a security proof. Expert review and a
larger adversarial test set remain necessary.

## Reproducible comparison

```bash
# One index, three exact translators, two prompts, original-query baseline,
# limited retrieval tuning on development, then validation on frozen holdout.
python main.py benchmark --stage retrieval

# Complete the basic cited-answer comparison for both requested answer models.
# Successful exact-request caches are reused; this is the bounded resume scope.
python main.py benchmark --stage all --answer-profiles grounded-v1

# Also compare grounded-v1 vs grounded-v2 (+ adjacent context).
# This is a larger experiment, not implied by a basic-profile-only result.
python main.py benchmark --stage all
```

See [`benchmarks/README.md`](benchmarks/README.md) for datasets, constraints,
selection rules and limitations. Important artifacts:

- `results/nvidia/REPORT.md`: readable scorecard, errors, selection and gates.
- `results/nvidia/benchmark.json`: exact configuration, hashes and measurements.
- `results/nvidia/chunks.json`: immutable inspected chunk snapshot.
- `results/nvidia/dev_*.json`, `holdout_*.json`: full rankings, text and variants.
- `results/nvidia/answers_*.json`: generated claims, evidence and rubric checks.
- `benchmark_cache/`: resumable embeddings/translations/answers, ignored by Git.

The benchmark uses its **own** Chroma path and never resets the ordinary CLI
index. Repeat runs reuse exact-input caches. A cached answer is labeled cached;
its recorded first-call latency is not counted as a new live-call latency.
Caches use atomic writes but are local process artifacts, not a shared production
cache service. Do not run simultaneous writer processes against the same files.

### Metrics

- hit@1/3/5: whether a matching, source-constrained evidence chunk appears in the
  top 1/3/5. MRR@10 also rewards earlier correct ranks.
- chrF++: similarity to authored translation references. It is not a semantic
  or expert quality score; valid translations can have different surface forms.
- translation constraints: numbers, required entity names, script and selected
  negation checks. Failure examples remain inspectable.
- answer rubric: approximate required-concept substring checks plus the named
  source constraint, separate from citation validity and refusal rates.
- out-of-scope retrieval scores remain observable; high similarity is not proof
  that a source contains the requested fact.

An incomplete comparison exits nonzero. **Completed** means the experiment
finished, not that its quality gates passed or the application is production-ready.

## Tests and GitHub Actions

```bash
python -m compileall -q .
python tests_offline.py
python -m unittest -v test_nvidia_pipeline
python -m pip check
./run_tests.sh --offline --no-push
```

Regular pushes/PRs run offline checks only. The old Gemini/Jina integration is
explicitly opt-in. The NVIDIA workflow needs only the repository's
`NVIDIA_API_KEY` secret; it never exports that key to the sandbox or artifacts.

Run **NVIDIA model comparison** manually in Actions with mode `probe` or
`benchmark`, or update the versioned `benchmarks/run_plan.json` on the session
branch. The latter is useful when a connected GitHub app can push but cannot
dispatch workflows. Ordinary code pushes do not spend model quota.

Results and caches are retained as workflow artifacts/Actions caches. Small
measurement records are also published as neutral GitHub Checks because some
sandboxes cannot download GitHub's redirected log/artifact blobs. Neutral
measurement checks are not passing quality gates. `report` mode republishes an
existing run without making model calls.

## Additional provider keys: catalog checks only

`provider_catalog.py` can inspect the published xKiro and KiosAPI model catalogs
using `XKIRO_API_KEY` and `KIOSAPI_API_KEY` from the environment (or `.env` when
python-dotenv is installed). The separate **Additional provider catalogs** Actions
workflow uses repository secrets of those names. Its versioned trigger is
`benchmarks/provider_catalog_plan.json`.

This makes **only GET /models requests**: no documents, embeddings or chat calls.
Each key goes only to its own documented HTTPS endpoint; credentialed redirects
are refused. Raw gateway error bodies/headers are not published. Results are
saved under `results/provider_catalog/` and in a neutral, inspectable Check.

A catalog listing—even HTTP 200—does not prove key validity, inference availability,
model quality, or the actual upstream engine. Family/alias matches are reported
separately, **never substituted** for the exact requested model IDs. xKiro's docs
say that its response reports the requested model across routing, so checking
that response field alone cannot establish exact upstream identity. These are
not active pipeline providers and are not folded into the NVIDIA benchmark.

Published endpoints/docs: [xKiro](https://docs.xkiro.com/),
[KiosAPI](https://kiosapi.mintlify.app/getting-started/quickstart).

## Production boundary

This is a stronger **evaluation lab**, not a deployed banking product. Outstanding
work includes approved model hosting/data governance, Arabic/legal expert review,
robust PDF extraction and DOCX/PDF heading metadata, independently authored and
larger holdout suites, load/SLA testing, authentication/authorization, ingestion
lifecycle/versioning, observability and adversarial security testing. The NVIDIA
trial endpoints and a small benchmark cannot establish production readiness.

`CI_REPORT.md` preserves the previous session's historical Gemini/Jina/Qwen
experiments; do not confuse those measurements with the new NVIDIA comparison.
All bank names, fees and rates in `data/` are fictional sample data; the four
files in `../docs` are the separate user-supplied real-document corpus.
