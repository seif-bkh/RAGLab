"""embedder.py — hosted embedding API behind a tiny interface + local cache.

One class per provider (Gemini / OpenAI / Cohere / Voyage), all inheriting
from BaseEmbedder so the rest of the lab only sees:

    embedder = build_embedder(config)
    embedder.embed_texts(["some text", ...])   -> list[list[float]]
    embedder.embed_query("question")           -> list[float]
    embedder.startup_report()                  -> prints provider/model/dimension
                                                  + runs the 3-language sanity check

Behaviour (all of it explicit and printable):
- keys come from .env, never from code;
- batch embedding (EMBEDDING_BATCH_SIZE per API call);
- retry with exponential backoff on rate limits (429) / 5xx / connection errors,
  fail fast on non-retryable 4xx errors;
- embeddings cached in embeddings_cache.json, keyed by
  sha256(model + input_type + text), so metadata-only re-ingests cost nothing
  and document/query embeddings of identical text never collide.

Gemini specifics (default provider, Google AI Studio key, free tier):
- `gemini-embedding-2`: does NOT support `task_type`. Per Google docs, task
  instructions go in the prompt (see GEMINI_USE_TASK_PROMPTS in config.py),
  and each input must be wrapped in a Content object, otherwise multiple
  inputs are AGGREGATED into a single embedding.
- `gemini-embedding-001`: supports task_type=RETRIEVAL_DOCUMENT / _QUERY and
  a plain list of strings (one embedding per string).
- Both default to 3072 dims; we truncate to GEMINI_OUTPUT_DIMENSIONALITY (768,
  recommended). Embedding spaces are model-specific: switching models or
  dimensions requires a full re-ingest (use --reset).
"""

import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """Tiny JSON cache. Shape: {"model": ..., "entries": {key: {...}}}.

    Each entry stores the embedding plus a short preview of the text so the
    JSON file can be inspected by hand.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: dict = {}
        self.model = ""
        self.stale_model = False
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = data.get("entries", {})
            self.model = data.get("model", "")
            print(f"[embedder] cache loaded: {len(self.entries)} entries "
                  f"from {self.path.name}")
        except Exception as exc:  # noqa: BLE001 — a broken cache must never block work
            print(f"[embedder] WARNING: could not read cache {self.path}: {exc}")
            self.entries = {}

    def note_expected_model(self, model: str):
        """Flag entries cached for a different model (spaces are incompatible)."""
        self.stale_model = bool(self.model and self.model != model and self.entries)

    def save(self):
        payload = {
            "model": self.model,
            "entries": self.entries,
            "note": "key = sha256(model + '\\n' + input_type + '\\n' + text); "
                    "preview shows the text tail",
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def get(self, key: str):
        return self.entries.get(key)

    def put(self, key: str, text: str, embedding: list):
        self.entries[key] = {
            "embedding": embedding,
            "preview": text[-120:],  # tail, enough to identify the text by eye
        }

    @property
    def size(self) -> int:
        return len(self.entries)


def require_provider_sdk(module_name: str, pip_name: str):
    """Import a provider SDK, failing with an actionable message if missing.

    "cannot import name 'genai' from 'google'" means the `google` namespace
    exists (usually via protobuf) but google-genai itself is not installed —
    usually because `pip install -r requirements.txt` was never run, or it was
    run with a different Python. This function reports the exact interpreter
    and the exact command to fix it, instead of a raw traceback.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"[embedder] MISSING DEPENDENCY: '{pip_name}' is not importable from\n"
            f"           this Python interpreter ({sys.executable}).\n"
            f"           Fix:   {sys.executable} -m pip install -r requirements.txt\n"
            f"           (or:   {sys.executable} -m pip install {pip_name})\n"
            f"           Make sure you activate the same virtualenv you installed\n"
            f"           into (e.g. `.venv`), then retry.\n"
            f"           Original error: {exc}"
        )


def cache_key(model: str, text: str, input_type: str = "") -> str:
    """Deterministic key: sha256 over model name, input type and exact text.

    input_type is included because some providers embed documents and queries
    differently (Gemini 001 task_type, Gemini 2 prompt headers); the same raw
    text embedded as a document and as a query must not share a cache entry.
    """
    payload = (model + "\n" + (input_type or "none") + "\n" + text).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseEmbedder:
    """Common machinery: caching, batching, retrying. Subclasses implement
    _make_client() and _embed_batch() with the official provider SDK."""

    provider_name = "base"

    def __init__(self, cfg):
        self.model = cfg.EMBEDDING_MODEL
        self.batch_size = cfg.EMBEDDING_BATCH_SIZE
        self.max_retries = cfg.EMBEDDING_MAX_RETRIES
        self.retry_base_delay = cfg.EMBEDDING_RETRY_BASE_DELAY
        self.cache = EmbeddingCache(cfg.EMBEDDING_CACHE_PATH)
        self.cache.model = self.model
        self.cache.note_expected_model(self.model)
        self._client = None
        self._dimension = None
        self.cache_hits = 0
        self.api_calls = 0

    # -- to implement per provider ------------------------------------------
    def _make_client(self):
        raise NotImplementedError

    def _embed_batch(self, texts: list[str], input_type: str = "search_document"):
        raise NotImplementedError

    # -- shared machinery -----------------------------------------------------
    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """True for transient errors worth retrying (rate limits, 5xx, network).

        Non-retryable 4xx (bad request, wrong key, unknown model) fail fast
        instead of burning max_retries * backoff on a mistake.
        """
        code = getattr(exc, "status_code", None)
        if code is None:
            code = getattr(exc, "code", None)
        if code is not None:
            try:
                return int(code) in (408, 429, 500, 502, 503, 504)
            except (TypeError, ValueError):
                pass  # code may be an enum/string; fall through to text match
        text = f"{type(exc).__name__} {exc}".lower()
        return any(tok in text for tok in (
            "429", "resource_exhausted", "rate limit", "quota",
            "unavailable", "deadline", "timeout", "connection", "server error",
        ))

    def _call_with_retry(self, texts: list[str], input_type: str) -> list:
        """Call _embed_batch with bounded exponential-backoff retries.

        Only transient errors are retried; every retry is printed with the
        exception type so rate limits are never silent.
        """
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._embed_batch(texts, input_type)
            except Exception as exc:  # noqa: BLE001 — providers raise many SDK-specific types
                if not self._is_retryable(exc):
                    print(f"[embedder] NON-RETRYABLE error ({type(exc).__name__}): {exc}")
                    raise
                last_error = exc
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                message = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_retries:
                    print(f"[embedder] FAILED after {attempt} attempt(s): {message}")
                else:
                    print(f"[embedder] retry {attempt}/{self.max_retries - 1} "
                          f"after {delay:.1f}s ({message})")
                time.sleep(delay)
        raise RuntimeError(
            f"embedding failed after {self.max_retries} attempts: {last_error}"
        )

    def embed_texts(self, texts: list[str], input_type: str = "search_document") -> list:
        """Embed a list of texts. Cache-first, deduplicated, batched.

        Returns embeddings in the same order as `texts` (duplicates reuse the
        first embedding of that text).
        """
        results = [None] * len(texts)
        todo: list[tuple[int, str, str]] = []
        seen_keys: dict[str, int] = {}

        for i, text in enumerate(texts):
            key = cache_key(self.model, text, input_type)
            entry = self.cache.get(key)
            if entry is not None:
                results[i] = entry["embedding"]
                self.cache_hits += 1
            elif key in seen_keys:
                results[i] = results[seen_keys[key]]  # duplicate inside this call
            else:
                seen_keys[key] = i
                todo.append((i, text, key))

        for start in range(0, len(todo), self.batch_size):
            batch = todo[start : start + self.batch_size]
            texts_batch = [item[1] for item in batch]
            embs = self._call_with_retry(texts_batch, input_type)
            self.api_calls += 1
            print(f"[embedder] embedded {len(batch)} text(s) "
                  f"({start + 1}-{start + len(batch)} of {len(todo)}) in 1 API call")

            for (i, text, key), emb in zip(batch, embs):
                results[i] = emb
                self.cache.put(key, text, emb)

        self.cache.save()
        return results

    def embed_query(self, text: str) -> list:
        """Embed one question. Uses input_type='search_query' where the SDK
        distinguishes document vs query inputs (Cohere / Voyage)."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty query text")
        return self.embed_texts([text], input_type="search_query")[0]

    def _provider_notes(self) -> list[str]:
        """Extra lines printed in startup_report (provider-specific)."""
        return []

    def startup_report(self):
        """Print provider/model/dimension, then run the multilingual sanity check.

        Also sets self._dimension from the first real embedding.
        """
        print("=" * 72)
        print("[embedder] provider          :", self.provider_name)
        print("[embedder] model             :", self.model)
        print("[embedder] dimension         :",
              self._dimension if self._dimension is not None else "(unknown until first embed)")
        print("[embedder] batch size        :", self.batch_size)
        print("[embedder] cache entries     :", self.cache.size)
        print("[embedder] cache file        :", self.cache.path.name)
        if self.cache.stale_model:
            print(f"[embedder] WARNING: cache has {self.cache.size} embedding(s) for model "
                  f"'{self.cache.model}' but the current model is '{self.model}'. "
                  "Embedding spaces are incompatible, so those entries are ignored; "
                  "re-run `ingest --reset` to re-embed everything.")
        for line in self._provider_notes():
            print(line)
        print("[embedder] multi-language sanity check follows (this may hit the API once):")
        self._sanity_check()
        print("=" * 72)

    def _sanity_check(self):
        """Embed three phrases, one per language, and print pairwise cosine
        similarity so you can see whether they live in one shared space.

        Also stores self.sanity_lines so callers (e.g. the CI test) can pass
        the results on to GitHub Actions ::notice:: annotations.
        """
        phrases = [
            ("English", "savings account"),
            ("French", "compte épargne"),
            ("Arabic", "حساب التوفير"),
        ]
        vectors = self.embed_texts([p[1] for p in phrases])
        self._dimension = len(vectors[0]) if vectors else None

        lines = [f"dimension: {self._dimension}"]
        for i in range(len(phrases)):
            for j in range(i + 1, len(phrases)):
                sim = cosine(vectors[i], vectors[j])
                lines.append(f"cosine({phrases[i][0]:<7}, {phrases[j][0]:<7}) = {sim:+.4f}")
        lines.append("interpretation: if all three similarities are clearly positive, "
                     "the model places the languages in a shared space; negative/near-zero "
                     "values mean cross-lingual retrieval is likely to fail.")
        self.sanity_lines = lines
        for line in lines:
            print(f"[sanity]  {line}")
        return vectors


def cosine(a: list, b: list) -> float:
    """Plain Python cosine similarity (no numpy needed for a lab)."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Concrete providers — official SDKs only, no wrappers
# ---------------------------------------------------------------------------


class GeminiEmbedder(BaseEmbedder):
    """Google Gemini API embeddings (official `google-genai` SDK).

    Works with a Google AI Studio key (free tier). Handles both model families:
    - `gemini-embedding-2`  (current, multilingual, 100+ languages):
        task_type is NOT supported -> optional prompt-header instructions
        (config GEMINI_USE_TASK_PROMPTS) and each input must be wrapped in a
        Content object so the API returns one embedding per input.
    - `gemini-embedding-001` (older, text-only):
        task_type=RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY, plain string list.
    Both default to 3072 dims; we request GEMINI_OUTPUT_DIMENSIONALITY (768).
    """

    provider_name = "gemini"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.output_dimensionality = (
            int(getattr(cfg, "GEMINI_OUTPUT_DIMENSIONALITY", 768) or 0) or None
        )
        self.use_task_prompts = bool(getattr(cfg, "GEMINI_USE_TASK_PROMPTS", True))
        self.model_is_v2 = self.model.startswith("gemini-embedding-2")
        self._types = None

    def _make_client(self):
        if self._client is not None:
            return self._client
        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        ).strip()
        if not api_key:
            raise SystemExit(
                "[embedder] GEMINI_API_KEY is not set (GOOGLE_API_KEY also accepted). "
                "Create a key at https://aistudio.google.com/apikey, then copy "
                ".env.example to .env and paste it."
            )
        genai = require_provider_sdk("google.genai", "google-genai")
        types = require_provider_sdk("google.genai.types", "google-genai")

        self._types = types
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=120_000),  # milliseconds
        )
        return self._client

    def _embed_batch(self, texts: list[str], input_type: str = "search_document") -> list:
        client = self._make_client()  # also populates self._types
        types = self._types

        if self.model_is_v2:
            # gemini-embedding-2: no task_type parameter; per Google docs, put
            # the task instruction in the prompt (only when enabled).
            if self.use_task_prompts:
                if input_type == "search_query":
                    texts = [f"task: search result | query: {t}" for t in texts]
                else:
                    texts = [f"title: none | text: {t}" for t in texts]
            # Multiple raw inputs would be AGGREGATED into one embedding, so
            # wrap each input in a Content object -> one embedding per input.
            contents = [
                types.Content(parts=[types.Part.from_text(text=t)]) for t in texts
            ]
            config = (
                types.EmbedContentConfig(output_dimensionality=self.output_dimensionality)
                if self.output_dimensionality
                else None
            )
        else:
            # gemini-embedding-001: task_type + plain string list, one embedding
            # per string (documented behaviour).
            task_type = (
                "RETRIEVAL_QUERY" if input_type == "search_query"
                else "RETRIEVAL_DOCUMENT"
            )
            contents = list(texts)
            config = (
                types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.output_dimensionality,
                )
                if self.output_dimensionality
                else types.EmbedContentConfig(task_type=task_type)
            )

        response = client.models.embed_content(
            model=self.model, contents=contents, config=config
        )
        return [list(item.values) for item in response.embeddings]

    def _provider_notes(self) -> list[str]:
        notes = []
        if self.model_is_v2:
            notes.append(
                "[embedder] gemini-embedding-2: task_type is NOT supported; "
                f"prompt-header task instructions = "
                f"{'ON' if self.use_task_prompts else 'OFF'} "
                "(config GEMINI_USE_TASK_PROMPTS)"
            )
            if self.use_task_prompts:
                notes.append("[embedder]   documents -> 'title: none | text: ...'")
                notes.append("[embedder]   queries    -> 'task: search result | query: ...'")
        else:
            notes.append(
                "[embedder] gemini-embedding-001: task_type = RETRIEVAL_DOCUMENT "
                "(chunks) / RETRIEVAL_QUERY (questions)"
            )
        notes.append(
            f"[embedder] output dimensionality = "
            f"{self.output_dimensionality or 'model default (3072)'} "
            "(config GEMINI_OUTPUT_DIMENSIONALITY)"
        )
        return notes


class OpenAIEmbedder(BaseEmbedder):
    provider_name = "openai"

    def _make_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "[embedder] OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and paste your key."
            )
        openai_mod = require_provider_sdk("openai", "openai")

        self._client = openai_mod.OpenAI(api_key=api_key, timeout=60.0, max_retries=0)
        return self._client

    def _embed_batch(self, texts: list[str], input_type: str = "search_document"):
        client = self._make_client()
        response = client.embeddings.create(model=self.model, input=list(texts))
        # response.data is ordered like the input list.
        return [item.embedding for item in response.data]


class CohereEmbedder(BaseEmbedder):
    provider_name = "cohere"

    def _make_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("COHERE_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "[embedder] COHERE_API_KEY is not set. "
                "Copy .env.example to .env and paste your key."
            )
        cohere = require_provider_sdk("cohere", "cohere")

        # ClientV2 is the current client (cohere>=5.10); fall back for older SDKs.
        if hasattr(cohere, "ClientV2"):
            self._client = cohere.ClientV2(api_key=api_key, timeout=60)
        else:
            self._client = cohere.Client(api_key=api_key, timeout=60)
        return self._client

    def _embed_batch(self, texts: list[str], input_type: str = "search_document"):
        client = self._make_client()
        if hasattr(client, "v2"):
            response = client.embed(model=self.model, inputs=list(texts),
                                    input_type=input_type)
            return [list(item) for item in response.embeddings]
        # Legacy client (cohere < 5): no v2 method, embeddings are flat lists.
        response = client.embed(model=self.model, texts=list(texts),
                                input_type=input_type)
        return [list(item) for item in response.embeddings]


class VoyageEmbedder(BaseEmbedder):
    provider_name = "voyage"

    def _make_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "[embedder] VOYAGE_API_KEY is not set. "
                "Copy .env.example to .env and paste your key."
            )
        voyageai = require_provider_sdk("voyageai", "voyageai")

        self._client = voyageai.Client(api_key=api_key, timeout=60)
        return self._client

    def _embed_batch(self, texts: list[str], input_type: str = "search_document"):
        client = self._make_client()
        response = client.embed(texts=list(texts), model=self.model,
                                input_type=input_type)
        return [list(item) for item in response.embeddings]


def build_embedder(cfg) -> BaseEmbedder:
    """Factory: instantiate the provider named in config.py."""
    providers = {
        "gemini": GeminiEmbedder,
        "openai": OpenAIEmbedder,
        "cohere": CohereEmbedder,
        "voyage": VoyageEmbedder,
    }
    name = (cfg.EMBEDDING_PROVIDER or "").strip().lower()
    if name not in providers:
        raise SystemExit(
            f"[embedder] unknown EMBEDDING_PROVIDER {cfg.EMBEDDING_PROVIDER!r}. "
            f"Supported: {', '.join(sorted(providers))}. Edit config.py."
        )
    print(f"[embedder] building provider '{name}' (model: {cfg.EMBEDDING_MODEL})")
    return providers[name](cfg)
