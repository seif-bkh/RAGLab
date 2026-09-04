"""config.py — every knob of the lab lives here, with no hidden defaults.

Loads .env (if present) so embedding keys are available to the whole process.
"""

import os
from pathlib import Path

# Load .env from this project folder. Missing .env is fine: we just fail
# later, loudly, when a command actually needs a key.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    print("[config] python-dotenv not installed; ignoring .env (install requirements.txt)")


# ---------------------------------------------------------------------------
# Paths (relative to this project folder)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
RESULTS_DIR = PROJECT_DIR / "results"
QUESTIONS_FILE = PROJECT_DIR / "questions.json"

CHROMA_COLLECTION_NAME = "raglab_docs"

# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------
# Change these two lines to switch providers (see .env.example and README).
# Default: Google Gemini API with a Google AI Studio key (free tier OK).
# `gemini-embedding-2` is the current multilingual model (100+ languages) and
# works with AI Studio keys. `gemini-embedding-001` (text-only, older) also
# works; the embedder handles both automatically.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")

# Gemini-specific knobs ------------------------------------------------------
# Both Gemini embedding models default to 3072 dimensions. MRL lets you
# truncate; Google recommends 768 / 1536 / 3072 (lower = smaller/faster,
# nearly identical quality). Set to 0/None to use the model default (3072).
# NOTE: embeddings produced with different dimensions are NOT comparable —
# keep queries and documents on the same setting, and re-ingest after a change.
GEMINI_OUTPUT_DIMENSIONALITY = 768

# `gemini-embedding-2` does NOT accept task_type; Google recommends putting
# task instructions in the prompt instead. When True the embedder prefixes:
#   documents -> "title: none | text: ..."
#   queries    -> "task: search result | query: ..."
# Set False to send raw text (then documents and queries embed symmetrically).
# `gemini-embedding-001` always uses task_type=RETRIEVAL_DOCUMENT/QUERY.
GEMINI_USE_TASK_PROMPTS = True

# Batch size sent to the provider per API call, and retry policy for rate limits.
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_MAX_RETRIES = 5
EMBEDDING_RETRY_BASE_DELAY = 2.0  # seconds, doubled after each failed attempt

# Cache file for embeddings, keyed by sha256(model + chunk text), JSON on disk.
EMBEDDING_CACHE_PATH = PROJECT_DIR / "embeddings_cache.json"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE_TOKENS = 220
CHUNK_OVERLAP_TOKENS = 40
SPLIT_ON_HEADINGS_FIRST = True
# Overlap is applied at SENTENCE boundaries (never mid-sentence): the tail of
# a chunk is a list of whole trailing sentences, so the next chunk never opens
# with an unreadable fragment. Set False to fall back to word-level overlap.
CHUNK_OVERLAP_SENTENCE_AWARE = True

# Tokenizer used by the chunker for counting ("cl100k_base" is the standard
# choice for text-embedding-3-large; it handles Arabic, French and English).
TOKENIZER_BACKEND = "tiktoken"
TOKENIZER_MODEL = "cl100k_base"

# ---------------------------------------------------------------------------
# Storage / retrieval
# ---------------------------------------------------------------------------
# Boilerplate sections (document front-matter/title and legal/conditions
# sections, tagged section_type by chunker.py) are retrieval magnets — they
# mention product names, "fees", "rates" etc. without answering anything.
# When True they are chunked (visible in `inspect`) but NOT stored, so they
# cannot outrank real content. Set False to index everything.
INDEX_EXCLUDE_BOILERPLATE = True
STORE_BATCH_SIZE = 64          # records per chroma add() call
RETRIEVAL_TOP_K = 5            # default for `query`
RRF_RANK_CONSTANT = 60         # k used in reciprocal rank fusion: 1 / (k + rank)
EVAL_TOP_K = 20                # how many hits evaluation records per question

# ---------------------------------------------------------------------------
# Query translation (cross-lingual retrieval) — EXPERIMENTAL
# ---------------------------------------------------------------------------
# The remaining retrieval failures are language routing: an Arabic question
# clusters on Arabic chunks even though the fact also exists in the French
# document. When enabled, every query is translated into each corpus language
# (translate.py) and each chunk is ranked by its best score across variants
# (store.best_variant_merge). Translation uses the SAME Google AI Studio key
# and google-genai SDK as the embedder — no new dependency, no new secret —
# and is retrieval-side only (answer.py remains a stub).
#
# Costs: 1-2 batched translation calls per run + 1 extra embedding per
# translated variant per question. Translations are cached in
# QUERY_TRANSLATION_CACHE_PATH. On any failure the lab degrades to the
# original query and records it — translation is an enhancement, never a
# blocker. Set False (or `--no-translation` on query/evaluate) for baseline
# comparisons.
QUERY_TRANSLATION_ENABLED = os.getenv(
    "QUERY_TRANSLATION_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
QUERY_TRANSLATION_MODEL = os.getenv(
    "QUERY_TRANSLATION_MODEL", "gemini-2.5-flash"
)  # free AI Studio tier: 2.5 Flash / 2.5 Flash-Lite family
# Tried in order when the primary model errors (e.g. not enabled for the
# key/project); the switch is printed and recorded in the results.
QUERY_TRANSLATION_FALLBACK_MODELS = os.getenv(
    "QUERY_TRANSLATION_FALLBACK_MODELS",
    "gemini-2.5-flash-lite,gemini-2.0-flash",
)
QUERY_TRANSLATION_CACHE_PATH = PROJECT_DIR / "translations_cache.json"
