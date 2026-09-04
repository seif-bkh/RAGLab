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
# Default: OpenAI text-embedding-3-large — multilingual (Arabic/French/English).
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

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

# Tokenizer used by the chunker for counting ("cl100k_base" is the standard
# choice for text-embedding-3-large; it handles Arabic, French and English).
TOKENIZER_BACKEND = "tiktoken"
TOKENIZER_MODEL = "cl100k_base"

# ---------------------------------------------------------------------------
# Storage / retrieval
# ---------------------------------------------------------------------------
STORE_BATCH_SIZE = 64          # records per chroma add() call
RETRIEVAL_TOP_K = 5            # default for `query`
RRF_RANK_CONSTANT = 60         # k used in reciprocal rank fusion: 1 / (k + rank)
EVAL_TOP_K = 20                # how many hits evaluation records per question
