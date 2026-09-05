# RAGLab — local, transparent multilingual RAG retrieval lab

A small local Python laboratory to watch one document travel through
**cleaning → chunking → embedding → storage in local ChromaDB → retrieval
evaluation**, for a fictional multilingual banking assistant (Arabic, French,
English).

**Not a product.** No UI, no authentication, no LLM answer generation, no
cloud vector database. The only network calls are hosted **embedding** API
calls — by default **Google Gemini (`gemini-embedding-2`)**, which works with a
**free Google AI Studio API key** (no credit card). The sample documents are
**invented — they contain no real bank, no real rates, and no real data.**

> ⚠️ The `data/` product sheets describe a fictional "Banque Atlas".
> The bank, its products, fees (4,50 €, 9,00 €, 2,75 %, …), thresholds,
> phone number, email and addresses are all invented for testing only.

---

## 1. Setup

```bash
cd raglab
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your key, e.g. GEMINI_API_KEY=
#   (create one at https://aistudio.google.com/apikey — the free tier works)
```

`requirements.txt` is intentionally minimal:

- `chromadb` — local persistent vector store (runs on disk, `chroma_db/`)
- `python-dotenv` — reads `.env`
- `pypdf` — PDF text extraction
- `tiktoken` — token counting (`cl100k_base`)
- `google-genai` — official Google Gemini client (default embedding provider)

No LangChain, no LlamaIndex, no orchestration framework: every call is raw.

If you switch the provider (section 4), uncomment the matching line in
`requirements.txt` (`openai`, `cohere` or `voyageai`) and keep the others.

## 2. Quick start — the order of commands

```bash
# 1. Inspect: load + chunk only. NO API calls. Iterate on chunking here.
python main.py inspect

# 2. Ingest: chunk -> embed -> store. Clears the collection with --reset.
python main.py ingest --reset

# 3. Query: embed one question, retrieve top-k, print everything.
python main.py query "Which Atlas account is for independents?" --k 5 --lang fr
python main.py query "Combien coûte le compte courant Atlas ?" --k 5 --hybrid

# 4. Evaluate: run questions.json, print metrics, save results/eval_*.json
python main.py evaluate
python main.py evaluate --hybrid          # vector + BM25 fused with RRF
```

Every command that embeds starts by printing the **provider, model name,
vector dimension** and runs a **sanity check**: it embeds `"savings account"`,
`"compte épargne"` and `"حساب التوفير"` and prints the three pairwise cosine
similarities, so you can verify immediately that all three languages live in
a shared space. Pass `--skip-sanity-check` to skip it (saves one small API
call; not recommended).

## 2. Continuous integration (GitHub Actions)

`.github/workflows/ci.yml` runs automatically on push/PR. It has two jobs:

1. **compile-offline** — installs deps, byte-compiles every module, and runs
   `python main.py inspect` (load + chunk only, no API calls, no key needed).
2. **integration-gemini** — needs the repository secret **`GEMINI_API_KEY`**
   (`GOOGLE_API_KEY` is accepted as an alias, as in `.env`). Create it at
   Settings → Secrets and variables → Actions, scoped to the repository (not
   an environment, not a variable), with the exact name. The workflow writes
   it to `.env` (gitignored, never logged) and runs `raglab/ci_test.py`, which drives
   the real pipeline: sanity check → `ingest --reset` → vector + hybrid
   `query` → `evaluate` (vector-only) → `evaluate --hybrid` — asserting
   mechanics (dimension, chunk counts, metadata fields, results JSON shape).
   The final two runs are an A/B: same questions with query translation
   enabled, vector-only vs vector + BM25 RRF, and the compact annotations
   report both plus the delta. Retrieval *quality* is not a test
   failure criterion: the evaluation metrics (hit rates, separation,
   out-of-scope max) are printed to the log as the report, and the results
   JSON for both runs is uploaded as a build artifact. Trigger it at any time
   with **Run workflow** (Actions tab).

To skip the API-backed job locally or on a fork, just don't configure the
secret — the workflow fails with a clear message naming the expected secret.

## 3. What each module does

| file | purpose |
|---|---|
| `config.py` | all knobs: chunk size, overlap, provider, model, paths |
| `loader.py` | read `.txt`/`.md`/`.pdf`/`.docx` from `data/` (or `--data-dir`), clean, normalize Arabic, detect language |
| `chunker.py` | headings → paragraphs → size; prepends heading; table rows → sentences |
| `embedder.py` | provider interface + caching; batching, retries, sanity check |
| `store.py` | local ChromaDB (cosine), BM25 fallback, RRF merge |
| `evaluate.py` | run `questions.json`, metrics, timestamped results JSON |
| `answer.py` | **STUB** for answer generation — deliberately not implemented |
| `main.py` | argparse CLI: `inspect`, `ingest`, `query`, `evaluate` |

### Cleaning (`loader.py`)

- `.txt`/`.md` read as UTF-8; `.pdf` extracted with `pypdf` per page
  (each page keeps a `[page N]` marker so extraction is inspectable);
  `.docx` extracted with the **standard library only** (zip + XML walk of
  `word/document.xml`: paragraphs and tables as `| cell | cell |` rows) —
  no new dependency.
- `inspect` / `ingest` accept `--data-dir PATH` (repeatable) to load extra
  corpora (e.g. `--data-dir ../docs` for the real documents next to the
  repo); `evaluate` accepts `--questions PATH` for another question set.
- **Arabic PDF caveat:** some official Arabic PDFs are stored in visual
  order and with presentation-form glyphs. NFKC (first step of
  `normalize_arabic`) maps the glyphs to base letters; whole-line visual
  reordering cannot be fixed without an RTL pass, so some paragraphs of such
  PDFs stay word-order-jumbled (`inspect` shows the extracted text so you
  can see it). DOCX is unaffected.
- Whitespace normalized: CRLF → LF, non-breaking spaces → spaces, paragraph
  reflow (no glued line breaks inside paragraphs), exactly one blank line
  between blocks. Headings (`#`), table rows (`|`), lists and rules keep
  their own lines.
- Light **Arabic normalization**: NFKC (presentation forms → base letters,
  e.g. Arabic PDFs), unify `أ إ آ ٱ` → `ا`, remove tatweel `ـ`, strip
  diacritics. **No translation, no stemming.**
- Language detected by a tiny stopword heuristic (`fr` / `ar` / `en` /
  `unknown`) and attached as metadata.

### Chunking (`chunker.py`)

Parameters in `config.py`:

- `CHUNK_SIZE_TOKENS = 220` (token budget, `tiktoken` `cl100k_base`)
- `CHUNK_OVERLAP_TOKENS = 40`
- `SPLIT_ON_HEADINGS_FIRST = True`

The chunker splits on headings first, then on paragraph boundaries, and only
falls back to word-boundary hard splits for paragraphs longer than the
budget (every hard-split piece is flagged with a note in `inspect`). Every
chunk is prefixed with its nearest heading (e.g.
`1.3 Frais de tenue de compte et exonération`), so no chunk loses context.
A chunk may exceed `CHUNK_SIZE_TOKENS` by up to `CHUNK_OVERLAP_TOKENS`,
because overlap text is *reused* in the next chunk rather than re-counted
against its budget — that is deliberate, so nothing is lost between chunks.

The overlap is **sentence-aware** (`CHUNK_OVERLAP_SENTENCE_AWARE = True`):
the re-used tail is a set of *whole trailing sentences* (Latin and Arabic
boundaries `. ! ? … ؟ ؛`), never a mid-sentence word fragment. This keeps
chunks that straddle a boundary readable and avoids pollution like a chunk
opening with `شهري لا يقل عن 1 500…`. Set it to `False` to A/B against
word-level overlap.

Every chunk is tagged with a `section_type`: **content**, **front-matter**
(first H1: document title + preamble) or **legal** (general conditions,
terms, disclaimers — detected per-language, but *not* eligibility conditions
like "Conditions d'éligibilité"/"شروط الأهلية"). With
`INDEX_EXCLUDE_BOILERPLATE = True` (default) boilerplate chunks are still
chunked and visible in `inspect`, but they are **not stored**: they were
retrieval magnets — the title chunk names every product, and the disclaimer
literally says "aucun des taux, frais, seuils…" — so they outranked real
answers. Exclusion is printed per chunk at ingest; set to `False` to index
everything.
Markdown tables are converted into full sentences *before* chunking, e.g.
the fee table row becomes:

```text
Pour le produit Compte Courant Atlas, les frais mensuels s'élèvent à 4,50 € ;
ils sont exonérés si Salaire mensuel d'au moins 1 500 € versé sur le compte …
```

(English and Arabic templates exist too; see `convert_table_rows`.)

### Embeddings (`embedder.py`)

- Key from `.env` only — never hardcoded. Default: `GEMINI_API_KEY` from a
  Google AI Studio key (the `google-genai` SDK also accepts `GOOGLE_API_KEY`).
- Provider behind a tiny interface: `BaseEmbedder` + one class per provider.
  Switch by changing **two strings in `config.py`**:
  - `EMBEDDING_PROVIDER = "gemini"` → `"openai"`, `"cohere"` or `"voyage"`
  - `EMBEDDING_MODEL = "gemini-embedding-2"` →
    e.g. `"gemini-embedding-001"`, `"text-embedding-3-large"`,
    `"embed-multilingual-v3.0"` or `"voyage-multilingual-2"`
- Batch calls (`EMBEDDING_BATCH_SIZE`, default 16), retries with exponential
  backoff on rate limits / 5xx / connection errors
  (`EMBEDDING_MAX_RETRIES`) — but **fails fast** on non-retryable errors
  (bad request, wrong key, unknown model).
- **Embedding cache**: `embeddings_cache.json`, keyed by
  `sha256(model + "\n" + input_type + "\n" + text)`. Re-running ingestion
  after changing only metadata (e.g. chunk index, timestamp) makes **zero**
  API calls; the cache holds a text preview per key so you can eyeball it.

#### Gemini specifics (`config.py`)

| knob | default | meaning |
|---|---|---|
| `EMBEDDING_MODEL` | `gemini-embedding-2` | current model, 100+ languages, works on free tier; `gemini-embedding-001` (older, text-only) also supported |
| `GEMINI_OUTPUT_DIMENSIONALITY` | `768` | MRL truncation; Google recommends 768/1536/3072. Keep queries and documents on the same value |
| `GEMINI_USE_TASK_PROMPTS` | `True` | `gemini-embedding-2` does **not** accept `task_type`; with this on, the adapter prefixes documents with `title: none | text: …` and queries with `task: search result | query: …` (Google's recommended prompt format). Off = raw symmetric text |

`gemini-embedding-001` instead uses `task_type=RETRIEVAL_DOCUMENT` for
chunks and `RETRIEVAL_QUERY` for questions automatically. The startup report
prints which mode is active.

**Free tier**: embedding models are included in the Google AI Studio free
tier (per-project rate limits, see the AI Studio rate-limit page). This lab
makes ~2–4 batched requests per ingest and a handful per query/evaluation —
well inside free quota. If you hit a 429, the retry/backoff handles it and
prints every retry; if you hit it repeatedly, check your per-project limit.

**Important**: embedding spaces are model- and dimension-specific. Changing
`EMBEDDING_MODEL` or `GEMINI_OUTPUT_DIMENSIONALITY` means stored vectors are
incompatible with new ones — re-run `python main.py ingest --reset`. The
cache keys include the model, so old cache entries are skipped with a
printed warning rather than mixed in.

### Storage & retrieval (`store.py`)

- Local **persistent ChromaDB** collection `raglab_docs`, **cosine distance**,
  stored in `chroma_db/`.
- Every record carries the chunk text, its embedding, and metadata:
  `document`, `language`, `heading`, `chunk_index`, `source`, `origin`,
  `ingested_at` (UTC ISO), `embedding_model`, `token_count`.
- `python main.py query "question" --k 5 --lang fr`: embeds the (cleaned)
  question, optionally filters by language, prints rank, similarity
  (= `1 − distance`), language, heading, source, and **full chunk text**.
- `--hybrid`: runs a plain **BM25** implementation (`store.keyword_search`)
  over stored chunk texts alongside the vector search and merges the two
  rankings by **reciprocal rank fusion** (`1/(k + rank)` per list, summed).
- `--hybrid-blend`: the alternative fusion — **weighted score blend**
  `score = λ·cosine + (1−λ)·BM25_normalized` (`HYBRID_BLEND_LAMBDA`, default
  0.7). BM25 has no upper bound, so its scores are normalized by their own
  max per query variant; dense similarity stays the primary signal and BM25
  is a boost. RRF is rank-only (it lifts token-matching chunks above the
  fact chunk); the blend aims to keep both. Use `--hybrid-blend` anywhere
  `--hybrid` works (`query` and `evaluate`).

### Query translation — cross-lingual retrieval (experimental, `translate.py`)

The harness diagnostics showed the remaining failures are **language
routing**: an Arabic question clusters on Arabic chunks even though the same
fact exists in the French document (and vice-versa). Chunking cannot fix
that, so by default (`QUERY_TRANSLATION_ENABLED = True`) every query is also
translated into each corpus language and retrieval runs **per variant**:

1. `translate.py` translates each question with `QUERY_TRANSLATION_MODEL`
   (default `gemini-3.5-flash-lite`) — **the same Google AI Studio key and
   google-genai SDK as the embedder**: no new dependency, no new secret.
   Whole batches go in a single numbered-lines request (at most 2 API calls
   per run), and translations are cached in `translations_cache.json`
   (sha256 of model + target + text).
2. Each variant is embedded and retrieved like any other query.
3. `store.best_variant_merge` fuses the variant rankings by
   **language-normalized best-score fusion**: raw scores are not comparable
   across variants (the original same-language query always scores its
   chunks ~0.80 vs ~0.76 for a translated query), so each variant's scores
   are first normalized by that variant's own best match; every chunk then
   keeps its best *relative* score, the variant that produced it, and all
   (variant, rank) pairs. Ties break toward the original variant, so
   same-language questions keep their ranking while a French chunk can win
   for an Arabic question on its French-variant score.

Transparency: `query` prints every variant and, per hit, the `best variant`
and all `variant ranks`; `evaluate` records `query_variants`, `top_variant`
and `correct_variant` per question in the results JSON — you can see exactly
which language of the query found the chunk.

Notes:
- **Retrieval-side only** — nothing generates answers; `answer.py` stays a
  stub.
- On any translation failure the lab **degrades to the original query** and
  records it (never a blocker); use `--no-translation` (or
  `QUERY_TRANSLATION_ENABLED = False`) for baseline comparisons.
- Cost: a few translation calls (cached) + one extra embedding per translated
  variant per question. Re-ingest is NOT needed — this is query-side only.

### Evaluation (`evaluate.py`)

`questions.json` holds the test cases. Each case: `question`, `language`,
`category` (`verbatim`, `paraphrase`, `cross-lingual`, `out-of-scope`) and
either `expected_chunk_index` or `expected_substring` (optionally
`expected_lang` to pin the document language). Out-of-scope cases have **no**
expected match.

The harness embeds every question, records top-k results, and reports:

| metric | meaning |
|---|---|
| **hit@1 / @3 / @5** | how often the correct chunk ranked that high |
| **hit rate by category** | where retrieval fails: wording, language or topic |
| **hit rate by query language** | per-language asymmetry (cross-lingual) |
| **mean correct vs best-incorrect score** | *separation*: is the right chunk meaningfully closer, or just ranked first? A small/negative gap = the model is not actually discriminating. Scores are cosine similarity in vector-only mode and RRF scores in `--hybrid` mode |
| **max score on out-of-scope** | what an irrelevant question still retrieves; pick a refusal threshold above this value |
| **any-language hit on a miss** | for a strict `expected_lang` case that missed, the rank at which a chunk containing the expected fact was found *in another language* — a miss tagged this way is a **language-routing** observation (the answer exists in the corpus, the query just clustered on its own language), not a chunking/retrieval failure |

It prints a per-question table (question, language, category, hit/no, rank of
the correct chunk, its score), **flags every miss**, and saves the full run to
`results/eval_<YYYYMMDD_HHMMSS>.json` including the chunking parameters,
provider, model and top-k so runs are comparable after you change settings.

## 4. Changing settings

| I want to… | do this |
|---|---|
| Change chunk size | edit `CHUNK_SIZE_TOKENS` in `config.py` |
| Change overlap | edit `CHUNK_OVERLAP_TOKENS` in `config.py` |
| Disable heading-first splitting | `SPLIT_ON_HEADINGS_FIRST = False` |
| Switch embedding provider | set `EMBEDDING_PROVIDER` + `EMBEDDING_MODEL` in `config.py`, uncomment the matching `requirements.txt` line, `pip install -r requirements.txt`, set the provider key in `.env` |
| Change Gemini model / dimensions | edit `EMBEDDING_MODEL` / `GEMINI_OUTPUT_DIMENSIONALITY`, then `ingest --reset` (embedding spaces are incompatible) |
| A/B task prompts on `gemini-embedding-2` | flip `GEMINI_USE_TASK_PROMPTS`, then `ingest --reset` |
| Tune retrieval | `RETRIEVAL_TOP_K`, `RRF_RANK_CONSTANT`, `EVAL_TOP_K`, `HYBRID_BLEND_LAMBDA` in `config.py` |
| Toggle query translation | `QUERY_TRANSLATION_ENABLED` in `config.py`, or `--no-translation` on `query` / `evaluate` |
| Change translation model | edit `QUERY_TRANSLATION_MODEL` (e.g. `gemini-3.6-flash`); `QUERY_TRANSLATION_FALLBACK_MODELS` is tried in order when the primary errors |

Typical iteration loop: `inspect` → tweak chunking → `inspect` again →
`ingest --reset` (cached embeddings make re-ingest free) → `query` →
`evaluate` → compare `results/*.json`.

## 5. Files & artifacts

```
raglab/
├── .env.example          # copy to .env
├── config.py             # every knob
├── requirements.txt
├── main.py               # CLI
├── loader.py             # cleaning + language detection
├── chunker.py            # heading/paragraph/table-aware chunking
├── embedder.py           # provider interface + cache
├── store.py              # ChromaDB + BM25 + RRF
├── evaluate.py           # metrics + results JSON
├── translate.py          # query translation (cross-lingual experiment)
├── answer.py             # STUB (no generation)
├── questions.json        # 17 evaluation cases (fictional Atlas sheets)
├── questions_real.json   # 16 cases for the REAL docs/ corpus (Arabic)
├── data/                 # FICTIONAL sample documents (FR + AR)
├── ../docs/              # REAL documents (BCT circular, law 2016-48, guides)
├── chroma_db/            # local ChromaDB (gitignored, generated)
├── results/              # evaluation runs (gitignored, generated)
├── embeddings_cache.json # embedding cache (gitignored, generated)
└── translations_cache.json # query translations (gitignored, generated)
```

## 5.5 Troubleshooting

| symptom | cause / fix |
|---|---|
| `ImportError: cannot import name 'genai' from 'google' (unknown location)` when running `ingest`/`query`/`evaluate` | `google-genai` is not installed in the interpreter you ran (this is exactly what the error means: the `google` namespace exists but the SDK package doesn't). The CLI now prints the exact interpreter and command — run `pip install -r requirements.txt` **inside the same venv**, or the printed `python -m pip install ...`. Then re-run the command. |
| `GEMINI_API_KEY is not set` | copy `.env.example` to `.env` and paste your key from https://aistudio.google.com/apikey |
| anything about `chromadb`/`tiktoken`/`pypdf` missing | same fix: `pip install -r requirements.txt` in the active venv |
| `429` rate limit during ingest | normal on the free tier; the embedder retries with printed backoff. If it persists, wait a minute or lower `EMBEDDING_BATCH_SIZE` in `config.py`, or check your per-project quota in AI Studio. |

## 6. Notes / caveats

- The sanity check and every query cost a small number of API calls; the
  cache makes full re-ingests free once the chunk texts are unchanged.
  The cache key now includes the input type, so a text embedded as a
  document and the same text embedded as a query never collide (relevant
  for the asymmetric Gemini prompt format).
- tiktoken downloads its `cl100k_base` BPE file on first use (once, then
  cached). If that download is blocked (offline/proxy), the chunker prints a
  warning and falls back to a transparent estimator `max(words, chars/4)`;
  you can pre-download and cache the BPE file via `TIKTOKEN_CACHE_DIR`.
  Chunk sizes are then approximations — the estimator is only a lab crutch.
- Text in the documents was written to be read as plain text; PDFs extracted
  with `pypdf` can be messy — that is part of what this lab is for.
- The evaluation is intentionally simple: a correct chunk is defined by
  substring/chunk-index matching, not by LLM judgement.
- No LLM answer generation is implemented. The clearly marked stub lives in
  `answer.py`; add a generator there only if you decide to go further.
