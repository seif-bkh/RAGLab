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

# Jina embeddings — EMBEDDING_PROVIDER = "jina" -------------------------------
# Hosted multilingual embeddings via https://api.jina.ai/v1/embeddings
# (needs JINA_API_KEY; raw HTTPS call, no SDK dependency). The model supports
# 100+ languages incl. Arabic (and text+image; we use text inputs only).
# Jina v5 tasks map to our inputs automatically: retrieval.passage for
# documents, retrieval.query for questions. Output is L2-normalized by the API
# (normalized=true) and always an object per input (never truncated).
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL",
                                 "jina-embeddings-v5-omni-small")
# Known embedding dimension for the sanity check. 0 = auto-detect from the
# first real embedding (recommended).
JINA_EMBEDDING_DIM = int(os.getenv("JINA_EMBEDDING_DIM", "0") or 0)
# API base (override only for proxies/mirrors).
JINA_EMBEDDING_BASE_URL = os.getenv("JINA_EMBEDDING_BASE_URL",
                                    "https://api.jina.ai/v1/embeddings")
# Batch inputs per request (Jina accepts many texts in one call; a bigger
# batch means fewer requests per ingest, so a per-minute quota suffocates
# less often).
JINA_EMBEDDING_BATCH_SIZE = int(os.getenv("JINA_EMBEDDING_BATCH_SIZE", "64"))
# Provider-scoped cache: Jina vectors live in a different space than Gemini,
# so the A/B runs never clobber each other's resumable caches.
JINA_EMBEDDING_CACHE_PATH = PROJECT_DIR / os.getenv(
    "JINA_EMBEDDING_CACHE_FILE", "embeddings_cache_jina.json")

# NVIDIA NIM embeddings + LLM — EMBEDDING_PROVIDER = "nvidia" -----------------
# Free hosted endpoints at https://integrate.api.nvidia.com/v1 (OpenAI-
# compatible; key from NVIDIA_API_KEY in .env, NEVER in code). Two roles:
#   EMBEDDER   NVIDIA NeMo Retriever family (multilingual/cross-lingual: an
#              English query can hit Arabic docs without a translation layer),
#              asymmetric input_type=passage|query. NOTE: the older
#              nvidia/llama-3.2-nv-embedqa-1b-v2 is END OF LIFE on the hosted
#              API (HTTP 410); the current family is llama-nemotron-embed.
#              CI discovers the live /v1/models catalog and overrides this
#              default when the exact id is absent.
#   TRANSLATOR QUERY_TRANSLATION_PROVIDER=nvidia + NVIDIA_TRANSLATION_MODEL
#              (moonshotai/kimi-k2.6 or deepseek-ai/deepseek-v4-pro; kimi-k3
#              is not currently in the NIM catalog) — used for query
#              translation only; answer generation stays a stub by design.
#              Free NIM endpoints are rate-limited (~40 RPM): keep batches
#              small, translations are cached.
NVIDIA_EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL",
                                   "nvidia/llama-nemotron-embed-1b-v2")
# Matryoshka dimension: 0 = server default, or a documented truncation size
# (e.g. 512/768/1024) to shrink vectors; the family supports embedding_type
# int8 via the API but we keep float for cosine comparability.
NVIDIA_EMBEDDING_DIM = int(os.getenv("NVIDIA_EMBEDDING_DIM", "0") or 0)
NVIDIA_EMBEDDING_BASE_URL = os.getenv(
    "NVIDIA_EMBEDDING_BASE_URL",
    "https://integrate.api.nvidia.com/v1/embeddings")
NVIDIA_EMBEDDING_BATCH_SIZE = int(os.getenv("NVIDIA_EMBEDDING_BATCH_SIZE",
                                            "32"))
NVIDIA_EMBEDDING_CACHE_PATH = PROJECT_DIR / os.getenv(
    "NVIDIA_EMBEDDING_CACHE_FILE", "embeddings_cache_nvidia.json")
# Query-translation LLM (NVIDIA NIM free endpoints).
NVIDIA_TRANSLATION_MODEL = os.getenv("NVIDIA_TRANSLATION_MODEL",
                                     "moonshotai/kimi-k2.6")
NVIDIA_TRANSLATION_FALLBACK_MODELS = os.getenv(
    "NVIDIA_TRANSLATION_FALLBACK_MODELS",
    "deepseek-ai/deepseek-v4-pro,deepseek-ai/deepseek-v4-flash")
NVIDIA_TRANSLATION_BASE_URL = os.getenv(
    "NVIDIA_TRANSLATION_BASE_URL",
    "https://integrate.api.nvidia.com/v1/chat/completions")

# Local multilingual embeddings — EMBEDDING_PROVIDER = "huggingface" ----------
# Fully offline: no API key, no daily quota, no cost. The model runs on YOUR
# machine (CPU is fine). Recommended model: Qwen3-Embedding-0.6B (Apache-2.0,
# 1024 dims, 32K context, 100+ languages including Arabic, ~640MB download;
# multilingual-MTEB leader-class at laptop size, +8% over BGE-M3 at the same
# parameter count). Battle-tested alternative: BAAI/bge-m3 (MIT, 1024 dims,
# 100+ languages). Requires `pip install sentence-transformers` (see
# requirements.txt — commented, like the other optional providers).
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL",
                               "Qwen/Qwen3-Embedding-0.6B")
# Device: "" = let sentence-transformers pick (CPU / CUDA / MPS / auto).
HF_EMBEDDING_DEVICE = os.getenv("HF_EMBEDDING_DEVICE", "").strip()
# Local inference batch size during ingestion (lower = less RAM).
HF_EMBEDDING_BATCH_SIZE = int(os.getenv("HF_EMBEDDING_BATCH_SIZE", "8"))
# Known embedding dimension, for the CI sanity check. 0 = auto-detect from
# the loaded model (recommended).
HF_EMBEDDING_DIM = int(os.getenv("HF_EMBEDDING_DIM", "0") or 0)
# Query/document discriminative prompts (task instructions). Family-aware by
# default and only when True: Qwen3 uses its built-in "query" prompt, BGE-M3
# the retrieval instruction, E5 the "query:" / "passage:" prefixes. Set to
# "false" to embed raw text (symmetric, sometimes better for short chunks).
HF_EMBEDDING_USE_PROMPTS = os.getenv("HF_EMBEDDING_USE_PROMPTS",
                                     "true").strip().lower() in \
    {"1", "true", "yes", "on"}
# Optional explicit overrides (applied when USE_PROMPTS=True and non-empty).
HF_EMBEDDING_QUERY_PROMPT = os.getenv("HF_EMBEDDING_QUERY_PROMPT", "")
HF_EMBEDDING_DOC_PROMPT = os.getenv("HF_EMBEDDING_DOC_PROMPT", "")


def active_embedding_model() -> str:
    """The model string actually in use for the configured provider.

    Used everywhere an embedding model name is recorded (ingest metadata,
    evaluation run config, cache identity) so a provider switch never labels
    gems as the wrong model's vectors — embedding spaces are provider-specific
    and switching requires a full re-ingest (--reset).
    """
    if (EMBEDDING_PROVIDER or "").strip().lower() in {
            "huggingface", "hf", "sentence_transformers",
            "sentence-transformers"}:
        return HF_EMBEDDING_MODEL
    if (EMBEDDING_PROVIDER or "").strip().lower() == "jina":
        return JINA_EMBEDDING_MODEL
    if (EMBEDDING_PROVIDER or "").strip().lower() == "nvidia":
        return NVIDIA_EMBEDDING_MODEL
    return EMBEDDING_MODEL

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
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "220"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))
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

# Score-blend hybrid (`--hybrid-blend`): score = LAMBDA * cosine +
# (1 - LAMBDA) * normalized BM25 (BM25 normalized by its own max per query
# variant). 1.0 = pure vector; 0.0 = pure keyword. Dense similarity stays the
# primary signal, BM25 boosts chunks sharing query tokens — the experiment
# after RRF showed RRF trades hit@1 for hit@3/5, so the blend aims to keep
# both.
HYBRID_BLEND_LAMBDA = float(os.getenv("HYBRID_BLEND_LAMBDA", "0.7"))

# BM25 sees ONLY the chunk text by default. The question often names the
# source document ("BCT circular 2019-08", "the internal guide") while the
# chunk text does not (the title lives in metadata/source), so keyword hits
# miss the right document. When True, keyword_search appends the source file
# name + heading to each corpus text — a cheap, powerful document cue.
KEYWORD_SEARCH_INCLUDE_METADATA = os.getenv(
    "KEYWORD_SEARCH_INCLUDE_METADATA", "true").strip().lower() in \
    {"1", "true", "yes", "on"}

# Tie-break policy for best_variant_merge when two DIFFERENT chunks each top
# their own query variant (both relative_score == 1.0):
#   - "variant_order": original-language variant first (old behavior; the
#      translated champion is demoted even when it holds the answer);
#   - "raw"          : highest raw score wins (fixed cross-lingual top-1 but
#      reintroduced the same-language score bias and hurt verbatim hit@1);
#   - "same_lang_margin": prefer the champion whose query language matches the
#      chunk's language, then the champion with the biggest relative margin
#      over its variant's second-best (a sharp answer beats a broad match).
# CI measures all three with cached embeddings and keeps the best (reported).
FUSION_TIE_BREAK = os.getenv("FUSION_TIE_BREAK", "same_lang_margin").strip()

# ---------------------------------------------------------------------------
# Query translation (cross-lingual retrieval) — EXPERIMENTAL
# ---------------------------------------------------------------------------
# The remaining retrieval failures are language routing: an Arabic question
# clusters on Arabic chunks even though the fact also exists in the French
# document. When enabled, every query is translated into each corpus language
# (translate.py) and each chunk is ranked by its best LANGUAGE-NORMALIZED score
# across variants (store.best_variant_merge: each variant's scores are divided
# by its own best match, because raw scores are not comparable across
# languages). Translation uses the SAME Google AI Studio key
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
# which backend translates queries: "gemini" (default) or "nvidia" (NIM
# free LLM endpoints: moonshotai/kimi-k3 or deepseek-ai/deepseek-v4-pro).
# Both only translate queries — answer generation stays a stub by design.
QUERY_TRANSLATION_PROVIDER = os.getenv("QUERY_TRANSLATION_PROVIDER",
                                       "gemini").strip().lower()
QUERY_TRANSLATION_MODEL = os.getenv(
    "QUERY_TRANSLATION_MODEL", "gemini-3.5-flash-lite"
)  # GA flash model (free tier friendly; 2.5-flash IDs are retired for new projects)
QUERY_TRANSLATION_FALLBACK_MODELS = os.getenv(
    "QUERY_TRANSLATION_FALLBACK_MODELS",
    "gemini-3.6-flash",
)
QUERY_TRANSLATION_CACHE_PATH = PROJECT_DIR / "translations_cache.json"
