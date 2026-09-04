# RAGLab — local, transparent multilingual RAG retrieval lab

A small local Python laboratory to watch one document travel through
**cleaning → chunking → embedding → storage in local ChromaDB → retrieval
evaluation**, for a fictional multilingual banking assistant (Arabic, French,
English).

**Not a product.** No UI, no authentication, no LLM answer generation, no
cloud vector database. The only network calls are hosted **embedding** API
calls (OpenAI by default). The sample documents are **invented — they contain
no real bank, no real rates, and no real data.**

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
# edit .env and paste your embedding API key (OPENAI_API_KEY by default)
```

`requirements.txt` is intentionally minimal:

- `chromadb` — local persistent vector store (runs on disk, `chroma_db/`)
- `python-dotenv` — reads `.env`
- `pypdf` — PDF text extraction
- `tiktoken` — token counting (`cl100k_base`)
- `openai` — official OpenAI client (default embedding provider)

No LangChain, no LlamaIndex, no orchestration framework: every call is raw.

If you switch the provider (section 4), uncomment the matching line in
`requirements.txt` (`cohere` or `voyageai`) and keep the others.

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

## 3. What each module does

| file | purpose |
|---|---|
| `config.py` | all knobs: chunk size, overlap, provider, model, paths |
| `loader.py` | read `.txt`/`.md`/`.pdf` from `data/`, clean, normalize Arabic, detect language |
| `chunker.py` | headings → paragraphs → size; prepends heading; table rows → sentences |
| `embedder.py` | provider interface + caching; batching, retries, sanity check |
| `store.py` | local ChromaDB (cosine), BM25 fallback, RRF merge |
| `evaluate.py` | run `questions.json`, metrics, timestamped results JSON |
| `answer.py` | **STUB** for answer generation — deliberately not implemented |
| `main.py` | argparse CLI: `inspect`, `ingest`, `query`, `evaluate` |

### Cleaning (`loader.py`)

- `.txt`/`.md` read as UTF-8; `.pdf` extracted with `pypdf` per page
  (each page keeps a `[page N]` marker so extraction is inspectable).
- Whitespace normalized: CRLF → LF, non-breaking spaces → spaces, paragraph
  reflow (no glued line breaks inside paragraphs), exactly one blank line
  between blocks. Headings (`#`), table rows (`|`), lists and rules keep
  their own lines.
- Light **Arabic normalization**: unify `أ إ آ ٱ` → `ا`, remove tatweel
  `ـ`, strip diacritics. **No translation, no stemming.**
- Language detected by a tiny stopword heuristic (`fr` / `ar` / `en` /
  `unknown`) and attached as metadata.

### Chunking (`chunker.py`)

Parameters in `config.py`:

- `CHUNK_SIZE_TOKENS = 220` (token budget, `tiktoken` `cl100k_base`)
- `CHUNK_OVERLAP_TOKENS = 40`
- `SPLIT_ON_HEADINGS_FIRST = True`

The chunker splits on headings first, then on paragraph boundaries, and only
falls back to word-boundary hard splits for paragraphs longer than the
budget. Every chunk is prefixed with its nearest heading (e.g.
`1.3 Frais de tenue de compte et exonération`), so no chunk loses context.
A chunk may exceed `CHUNK_SIZE_TOKENS` by up to `CHUNK_OVERLAP_TOKENS`,
because overlap text is *reused* in the next chunk rather than re-counted
against its budget — that is deliberate, so nothing is lost between chunks.
Markdown tables are converted into full sentences *before* chunking, e.g.
the fee table row becomes:

```text
Pour le produit Compte Courant Atlas, les frais mensuels s'élèvent à 4,50 € ;
ils sont exonérés si Salaire mensuel d'au moins 1 500 € versé sur le compte …
```

(English and Arabic templates exist too; see `convert_table_rows`.)

### Embeddings (`embedder.py`)

- Key from `.env` only — never hardcoded.
- Provider behind a tiny interface: `BaseEmbedder` + one class per provider.
  Switch by changing **two strings in `config.py`**:
  - `EMBEDDING_PROVIDER = "openai"` → `"cohere"` or `"voyage"`
  - `EMBEDDING_MODEL = "text-embedding-3-large"` →
    `"embed-multilingual-v3.0"` or `"voyage-multilingual-2"`
- Batch calls (`EMBEDDING_BATCH_SIZE`, default 16), retries with exponential
  backoff on rate limits/connection errors (`EMBEDDING_MAX_RETRIES`).
- **Embedding cache**: `embeddings_cache.json`, keyed by
  `sha256(model + "\n" + text)`. Re-running ingestion after changing only
  metadata (e.g. chunk index, timestamp) makes **zero** API calls; the
  cache holds a text preview per key so you can eyeball it.

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
| Tune retrieval | `RETRIEVAL_TOP_K`, `RRF_RANK_CONSTANT`, `EVAL_TOP_K` in `config.py` |

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
├── answer.py             # STUB (no generation)
├── questions.json        # 17 evaluation cases
├── data/                 # FICTIONAL sample documents (FR + AR)
├── chroma_db/            # local ChromaDB (gitignored, generated)
├── results/              # evaluation runs (gitignored, generated)
└── embeddings_cache.json # embedding cache (gitignored, generated)
```

## 6. Notes / caveats

- The sanity check and every query cost a small number of API calls; the
  cache makes full re-ingests free once the chunk texts are unchanged.
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
