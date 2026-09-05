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

Legacy Gemini specifics (explicit provider selection):
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

from artifacts import fingerprint, write_json

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
        write_json(self.path, payload)

    def get(self, key: str):
        return self.entries.get(key)

    def put(self, key: str, text: str, embedding: list):
        # Providers (especially local sentence-transformers) can return numpy
        # float32 scalars / torch tensors, which json.dumps cannot serialize.
        # Coerce once here so EVERY provider writes a JSON-safe cache.
        try:
            embedding = [float(x) for x in embedding]
        except (TypeError, ValueError):
            pass  # unexpected non-numeric row: leave as returned
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


def embedding_fingerprint(cfg):
    """Identity of the vector space, separate from the chunker fingerprint."""
    provider = str(getattr(cfg, "EMBEDDING_PROVIDER", "unknown")).strip().lower()
    fields = {"provider": provider, "model": cfg.active_embedding_model(), "contract": "retrieval-v2"}
    prefixes = {"nvidia": "NVIDIA", "gemini": "GEMINI", "jina": "JINA",
                "huggingface": "HF"}
    prefix = prefixes.get(provider, provider.upper())
    for suffix in ("EMBEDDING_DIM", "EMBEDDING_BASE_URL", "OUTPUT_DIMENSIONALITY",
                   "USE_TASK_PROMPTS", "EMBEDDING_USE_PROMPTS", "EMBEDDING_QUERY_PROMPT",
                   "EMBEDDING_DOC_PROMPT"):
        fields[suffix] = getattr(cfg, prefix + "_" + suffix, None)
    # Omitted and explicit native dimensions are the same hosted vector space.
    if provider == "nvidia" and fields["model"] == "nvidia/nemotron-3-embed-1b":
        fields["EMBEDDING_DIM"] = fields["EMBEDDING_DIM"] or 2048
    return fingerprint(fields)


class BaseEmbedder:
    """Common machinery: caching, batching, retrying. Subclasses implement
    _make_client() and _embed_batch() with the official provider SDK."""

    provider_name = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg.active_embedding_model()
        self.cache_identity = embedding_fingerprint(cfg)
        self.batch_size = cfg.EMBEDDING_BATCH_SIZE
        self.max_retries = cfg.EMBEDDING_MAX_RETRIES
        self.retry_base_delay = cfg.EMBEDDING_RETRY_BASE_DELAY
        self.cache = EmbeddingCache(cfg.EMBEDDING_CACHE_PATH)
        self.cache.note_expected_model(self.model)
        self.cache.model = self.model
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
                return int(code) in (0, 408, 429, 500, 502, 503, 504)
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
                self.api_calls += 1
                return self._embed_batch(texts, input_type)
            except Exception as exc:  # noqa: BLE001 — providers raise many SDK-specific types
                if not self._is_retryable(exc):
                    print(f"[embedder] NON-RETRYABLE error ({type(exc).__name__}): {exc}")
                    raise
                last_error = exc
                delay = getattr(exc, "retry_after", None)
                if delay is None:
                    delay = self.retry_base_delay * (2 ** (attempt - 1))
                if delay > float(getattr(self.cfg, "NVIDIA_MAX_RETRY_DELAY", 30)):
                    raise  # defer, never retry before a long Retry-After
                message = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_retries:
                    print(f"[embedder] FAILED after {attempt} attempt(s): {message}")
                else:
                    print(f"[embedder] retry {attempt}/{self.max_retries - 1} "
                          f"after {delay:.1f}s ({message})")
                if attempt < self.max_retries:
                    time.sleep(delay)
        raise RuntimeError(
            f"embedding failed after {self.max_retries} attempts: {last_error}"
        )

    def _call_with_bisect(self, texts: list[str], input_type: str) -> list:
        """Embed a batch, bisecting on a degenerate (NaN/inf/zero) vector.

        One bad provider response currently fails the WHOLE batch with no
        hint of which input caused it. Split the batch in halves until the
        offending input is a singleton; the InvalidVectorError then carries
        that exact input's preview. Healthy halves keep their results.
        """
        try:
            return self._call_with_retry(texts, input_type)
        except InvalidVectorError:
            if len(texts) == 1:
                raise
            mid = len(texts) // 2
            left = self._call_with_bisect(texts[:mid], input_type)
            right = self._call_with_bisect(texts[mid:], input_type)
            return left + right

    def embed_texts(self, texts: list[str], input_type: str = "search_document") -> list:
        """Embed a list of texts. Cache-first, deduplicated, batched.

        Returns embeddings in the same order as `texts` (duplicates reuse the
        first embedding of that text).
        """
        results = [None] * len(texts)
        todo: list[tuple[int, str, str]] = []
        seen_keys: dict[str, int] = {}

        duplicates = {}
        for i, text in enumerate(texts):
            identity = getattr(self, "cache_identity", self.model)
            key = cache_key(identity, text, input_type)
            entry = self.cache.get(key)
            if entry is not None:
                cached = entry["embedding"]
                # Guard against corrupted/NaN cache entries (e.g. from an
                # interrupted provider response): chroma rejects NaN vectors
                # with a cryptic numpy error, so fail here with the input.
                if not _finite_vector(cached):
                    raise RuntimeError(
                        f"embedding cache has an invalid vector for "
                        f"{text[:120]!r} (cache invalid; delete "
                        f"{self.cache.path.name} and re-ingest with --reset)"
                    )
                expected_dim = getattr(self, "expected_dimension", None)
                if expected_dim and len(cached) != expected_dim:
                    raise RuntimeError(f"embedding cache dimension mismatch: expected {expected_dim}, got {len(cached)}")
                results[i] = cached
                self._dimension = len(cached)
                self.cache_hits += 1
            elif key in seen_keys:
                duplicates[i] = seen_keys[key]  # fill AFTER the unique batch returns
            else:
                seen_keys[key] = i
                todo.append((i, text, key))

        for start in range(0, len(todo), self.batch_size):
            batch = todo[start : start + self.batch_size]
            texts_batch = [item[1] for item in batch]
            embs = self._call_with_bisect(texts_batch, input_type)
            if len(embs) != len(batch):
                raise RuntimeError(f"Provider returned {len(embs)} vectors for {len(batch)} texts")
            print(f"[embedder] embedded {len(batch)} text(s) "
                  f"({start + 1}-{start + len(batch)} of {len(todo)}) in 1 API call")

            for (i, text, key), emb in zip(batch, embs):
                results[i] = emb
                self.cache.put(key, text, emb)
            # Save after EVERY batch, not only at the end: if the run later
            # dies on a quota/rate limit, the batches already embedded stay
            # cached and a rerun continues from where it stopped (used by the
            # CI to split a large ingest across days on the free tier).
            self.cache.save()

        for index, original in duplicates.items():
            results[index] = results[original]
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


def _vector_issue(v) -> str | None:
    """Describe why a vector is unusable by chroma, or None when it is fine.

    chroma L2-normalizes every vector it stores: an all-zero vector becomes
    NaN (0/0) and an `inf` component poisons every cosine — both produce the
    cryptic "Expected a 1-dimensional array, got a 0-dimensional array nan"
    crash at add() time. NaN is caught right away; infinity and zero-norm
    must be rejected too or they hide inside chroma.
    """
    if not isinstance(v, list):
        # numpy rows (local sentence-transformers) arrive as ndarray.
        if hasattr(v, "tolist"):
            try:
                v = v.tolist()
            except Exception:  # noqa: BLE001 — scalar/other: reject below
                return "empty"
        else:
            return "empty"
    if not v:
        return "empty"
    try:
        vals = [float(x) for x in v]
    except (TypeError, ValueError):
        return "non-numeric"
    for f in vals:
        if f != f:
            return "NaN"
        if f in (float("inf"), float("-inf")):
            return "infinite"
    if all(f == 0.0 for f in vals):
        return "zero-norm"
    return None


def _finite_vector(v) -> bool:
    """True if v is a non-empty list of finite, non-degenerate floats."""
    return _vector_issue(v) is None


class InvalidVectorError(RuntimeError):
    """A provider returned a vector that chroma cannot store (NaN/inf/zero).

    Non-retryable: re-requesting the same input reproduces the same vector.
    The message carries the exact input preview so the offending chunk is
    identifiable; `embed_texts` bisects large batches down to the bad input.
    """

    status_code = None


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

class JinaHTTPError(RuntimeError):
    """HTTP failure from the Jina API with a status_code for retry decision."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class NvidiaHTTPError(JinaHTTPError):
    """HTTP failure from the NVIDIA NIM API (same status_code semantics)."""


class JinaEmbedder(BaseEmbedder):
    """Hosted multilingual embeddings via https://api.jina.ai/v1/embeddings.

    Raw HTTPS (stdlib urllib — no SDK dependency, per the lab's minimal-dep
    rule). Jina has a per-request `task` field; we map our inputs the way the
    docs recommend:
      - documents  -> task="retrieval.passage"
      - questions  -> task="retrieval.query"
    normalized=true is always set (we rely on cosine anyway, but normalized
    vectors keep scores comparable across providers). The response is an
    object per input, ordered by `index`; dimension is auto-detected from the
    first real embedding.
    """

    provider_name = "jina"

    def __init__(self, cfg):
        super().__init__(cfg)
        # Provider-scoped cache (Jina space != Gemini space; A/Bs must not
        # clobber each other's resumable caches).
        self.cache = EmbeddingCache(
            getattr(cfg, "JINA_EMBEDDING_CACHE_PATH", cfg.EMBEDDING_CACHE_PATH))
        self.cache.note_expected_model(self.model)
        self.cache.model = self.model
        self.batch_size = int(getattr(cfg, "JINA_EMBEDDING_BATCH_SIZE",
                                      self.batch_size) or self.batch_size)

    def _make_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("JINA_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "[embedder] JINA_API_KEY is not set. "
                "Copy .env.example to .env and paste your key "
                "(https://jina.ai — Settings > API Keys)."
            )
        self._client = {"api_key": api_key, "base_url": self._base_url()}
        return self._client

    def _base_url(self) -> str:
        return getattr(self.cfg, "JINA_EMBEDDING_BASE_URL",
                       "https://api.jina.ai/v1/embeddings")

    def _task_for(self, input_type: str) -> str:
        return ("retrieval.query" if input_type == "search_query"
                else "retrieval.passage")

    def _embed_batch(self, texts: list[str], input_type: str = "search_document"):
        client = self._make_client()
        payload = {
            "model": self.model,
            "task": self._task_for(input_type),
            "normalized": True,
            "input": [{"text": t} for t in texts],
        }
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            client["base_url"],
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {client['api_key']}",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:300]
            except Exception:  # noqa: BLE001
                pass
            raise JinaHTTPError(
                exc.code,
                f"Jina API HTTP {exc.code}: {detail or exc.reason}",
            ) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise JinaHTTPError(0, f"Jina API connection error: {exc}") from exc

        data = json.loads(body)
        # Keep a small preview for diagnostics (CI annotates it; never the key).
        self.last_response = data
        self.last_response_preview = {
            "api_model": data.get("model"),
            "object": data.get("object"),
            "n_items": len(data.get("data") or []),
            "first_dim": (len((data.get("data") or [{}])[0].get("embedding")
                              or []) if data.get("data") else 0),
        }
        if data.get("object") == "list" and isinstance(data.get("data"), list):
            ordered = sorted(data["data"], key=lambda item: item.get("index", 0))
            embs = []
            for idx, item in enumerate(ordered):
                emb = item.get("embedding")
                if not isinstance(emb, list):
                    raise RuntimeError(
                        f"Jina API item {item.get('index')} has no embedding "
                        f"list (got {type(emb).__name__}); input preview: "
                        f"{texts[idx][:120]!r}"
                    )
                vals = []
                for v in emb:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"Jina API item {item.get('index')} embedding "
                            f"value {v!r} not numeric; input preview: "
                            f"{texts[idx][:120]!r}"
                        ) from exc
                    vals.append(fv)
                issue = _vector_issue(vals)
                if issue is not None:
                    raise InvalidVectorError(
                        f"Jina API item {item.get('index')} embedding "
                        f"contains {issue}; input preview: "
                        f"{texts[idx][:120]!r}"
                    )
                embs.append(vals)
            if len(embs) != len(texts):
                raise RuntimeError(
                    f"Jina API returned {len(embs)} embeddings for "
                    f"{len(texts)} inputs"
                )
            if embs:
                self._dimension = len(embs[0])
            return embs
        raise RuntimeError(
            f"Jina API unexpected response: {json.dumps(data)[:300]}"
        )

    def _provider_notes(self) -> list[str]:
        return [
            "task mapping: docs->retrieval.passage | queries->retrieval.query",
            "normalized=true (L2-normalized vectors) | dimension auto-detected",
        ]


class NvidiaEmbedder(BaseEmbedder):
    """NVIDIA NIM query/passage embeddings, exact model and strict validation.

    The hosted nemotron-3-embed-1b API returns native 2048-dimensional floats.
    Explicit truncate=NONE prevents invisible loss of source text. Cache keys
    include endpoint, dimensions, model and task contract, not just model text.
    """
    provider_name = "nvidia"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cache = EmbeddingCache(getattr(cfg, "NVIDIA_EMBEDDING_CACHE_PATH", cfg.EMBEDDING_CACHE_PATH))
        self.cache.note_expected_model(self.model)
        self.cache.model = self.model
        self.batch_size = int(getattr(cfg, "NVIDIA_EMBEDDING_BATCH_SIZE", self.batch_size))
        self._dim = int(getattr(cfg, "NVIDIA_EMBEDDING_DIM", 0) or 0)
        self.expected_dimension = 2048 if self.model == "nvidia/nemotron-3-embed-1b" else (self._dim or None)
        if self.model == "nvidia/nemotron-3-embed-1b" and self._dim not in {0, 2048}:
            raise ValueError("Hosted nemotron-3-embed-1b only supports 2048 dimensions; set NVIDIA_EMBEDDING_DIM=0 or 2048")
        self.cache_identity = embedding_fingerprint(cfg)
        if self.batch_size <= 0:
            raise ValueError("NVIDIA_EMBEDDING_BATCH_SIZE must be positive")

    def _make_client(self):
        if self._client is None:
            from nvidia_api import NvidiaClient
            if not os.environ.get("NVIDIA_API_KEY", "").strip():
                raise SystemExit("[embedder] NVIDIA_API_KEY is not set. Configure it in .env or GitHub Actions secrets.")
            self._client = NvidiaClient(
                timeout=getattr(self.cfg, "NVIDIA_API_TIMEOUT", 120), attempts=1,
                min_interval=getattr(self.cfg, "NVIDIA_MIN_INTERVAL", 1.6))
        return self._client

    def _base_url(self):
        return getattr(self.cfg, "NVIDIA_EMBEDDING_BASE_URL", "https://integrate.api.nvidia.com/v1/embeddings")

    def _task_for(self, input_type):
        if input_type not in {"search_document", "search_query"}:
            raise ValueError(f"Unknown embedding input type: {input_type}")
        return "query" if input_type == "search_query" else "passage"

    def _embed_batch(self, texts, input_type="search_document"):
        from nvidia_api import NvidiaAPIError
        payload = {"model": self.model, "input": list(texts),
                   "input_type": self._task_for(input_type),
                   "encoding_format": "float", "truncate": "NONE"}
        if self._dim:
            payload["dimensions"] = self._dim
        try:
            data = self._make_client().request(self._base_url(), payload)
        except NvidiaAPIError as exc:
            error = NvidiaHTTPError(exc.status_code, str(exc))
            error.retry_after = exc.retry_after
            raise error from exc
        self.last_response_preview = {"api_model": data.get("model"), "object": data.get("object")}
        if data.get("model") and data["model"] != self.model:
            raise RuntimeError(f"Embedding model mismatch: requested {self.model}, got {data['model']}")
        items = data.get("data")
        if not isinstance(items, list):
            raise RuntimeError("NVIDIA embedding response has no data list")
        if len(items) != len(texts):
            raise RuntimeError(f"NVIDIA API returned {len(items)} embeddings for {len(texts)} inputs")
        indices = [item.get("index") for item in items if isinstance(item, dict)]
        if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(texts))):
            raise RuntimeError("NVIDIA embedding indices must be unique and cover every input")
        vectors = []
        for item in sorted(items, key=lambda item: item["index"]):
            index = item["index"]
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise RuntimeError(f"NVIDIA API item {index} has no embedding list")
            issue = _vector_issue(vector)
            if issue:
                raise InvalidVectorError(f"NVIDIA API item {index} embedding contains {issue}; input preview: {texts[index][:120]!r}")
            if self.expected_dimension and len(vector) != self.expected_dimension:
                raise RuntimeError(f"NVIDIA embedding dimension: expected {self.expected_dimension}, got {len(vector)}")
            if vectors and len(vector) != len(vectors[0]):
                raise RuntimeError("NVIDIA returned inconsistent vector dimensions")
            vectors.append([float(x) for x in vector])
        if vectors:
            self._dimension = len(vectors[0])
        self.last_response_preview.update(n_items=len(vectors), first_dim=self._dimension)
        return vectors

    def _provider_notes(self):
        return ["docs->passage | queries->query | encoding=float | truncate=NONE",
                "Exact model ID; no automatic fallback; native dimensions validated"]


class HuggingFaceEmbedder(BaseEmbedder):
    """Local multilingual embeddings via `sentence-transformers` (free).
    Default model (config.HF_EMBEDDING_MODEL): Qwen3-Embedding-0.6B —
    Apache-2.0, 1024 dims, 32K context, 100+ languages including Arabic.
    Alternative: BAAI/bge-m3 (MIT). The first use downloads the weights
    (~640MB for Qwen3-0.6B) into HuggingFace's cache.

    Task prompts: several families need query/document discriminative
    prompts (config.HF_EMBEDDING_USE_PROMPTS). Auto-detected per family:
      - qwen  -> the model's built-in "query" prompt (docs stay raw);
      - bge-m3-> "Represent this sentence for searching relevant passages: "
      - e5    -> "query: " / "passage: "
    config.HF_EMBEDDING_QUERY_PROMPT / HF_EMBEDDING_DOC_PROMPT override.
    """

    provider_name = "huggingface"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.batch_size = getattr(cfg, "HF_EMBEDDING_BATCH_SIZE",
                                  self.batch_size)
        self.device = (getattr(cfg, "HF_EMBEDDING_DEVICE", "") or None)
        self.use_prompts = bool(getattr(cfg, "HF_EMBEDDING_USE_PROMPTS", True))
        self.query_prompt_override = getattr(cfg,
                                             "HF_EMBEDDING_QUERY_PROMPT", "")
        self.doc_prompt_override = getattr(cfg, "HF_EMBEDDING_DOC_PROMPT", "")

    def _make_client(self):
        if self._client is not None:
            return self._client
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # actionable, no raw traceback
            raise SystemExit(
                "[embedder] MISSING DEPENDENCY for EMBEDDING_PROVIDER="
                f"'huggingface': sentence-transformers is not importable.\n"
                f"           Fix:   {sys.executable} -m pip install "
                "sentence-transformers\n"
                f"           (installs torch + the model runner; no API key\n"
                f"            is needed — embeddings run 100% locally)\n"
                f"           Original error: {exc}"
            )
        print(f"[embedder] loading local model {self.model!r} "
              f"(device={self.device or 'auto'}) — first use downloads the "
              "weights, later runs load from the local cache")
        try:
            self._client = SentenceTransformer(self.model,
                                               device=self.device)
        except OSError as exc:
            raise SystemExit(
                "[embedder] could not load the local model from "
                f"huggingface.co: {exc}\n"
                "           First use needs internet to download the weights "
                "(later runs use the local cache).\n"
                "           Fixes: check your internet/proxy; or set "
                "HF_ENDPOINT=https://hf-mirror.com if HF is blocked;\n"
                "           or pre-download with `huggingface-cli download "
                f"{self.model}` and set HF_HUB_OFFLINE=1.\n"
                f"           Model: {self.model}"
            )
        # sentence-transformers >= 6 renamed get_sentence_embedding_dimension()
        # to get_embedding_dimension(); support both so the lab never breaks
        # on a versions bump.
        dim_getter = (getattr(self._client, "get_embedding_dimension", None)
                      or getattr(self._client,
                                 "get_sentence_embedding_dimension", None))
        if dim_getter is None:
            raise SystemExit(
                "[embedder] cannot detect the embedding dimension for "
                f"{self.model!r}: this sentence-transformers version exposes "
                "neither get_embedding_dimension() nor "
                "get_sentence_embedding_dimension().")
        self._dimension = int(dim_getter())
        return self._client

    def _family_prompts(self) -> tuple[str, str]:
        """(query_prompt, doc_prompt) for the loaded model family."""
        if not self.use_prompts:
            return "", ""
        lower = self.model.lower()
        if "qwen" in lower:
            query = (self.query_prompt_override or
                     self._client.prompts.get("query", "") or "")
            return query, (self.doc_prompt_override or "")
        if "bge-m3" in lower:
            return (self.query_prompt_override or
                    "Represent this sentence for searching relevant "
                    "passages: "), (self.doc_prompt_override or "")
        if "e5" in lower:
            return (self.query_prompt_override or "query: "),                 (self.doc_prompt_override or "passage: ")
        return (self.query_prompt_override or ""),             (self.doc_prompt_override or "")

    def _embed_batch(self, texts: list[str],
                     input_type: str = "search_document") -> list:
        model = self._make_client()
        is_query = input_type == "search_query"
        query_prompt, doc_prompt = ("", "") if not self.use_prompts else             self._family_prompts()
        lower = self.model.lower()
        # Qwen3 embeddings: use the model's registered prompt (documented
        # inference), not a hand-written prefix.
        if (is_query and query_prompt and "qwen" in lower
                and not self.query_prompt_override):
            kwargs = {"prompt_name": "query"}
        else:
            kwargs = {}
            parts = []
            for text in texts:
                if is_query and query_prompt:
                    parts.append(query_prompt + text)
                elif not is_query and doc_prompt:
                    parts.append(doc_prompt + text)
                else:
                    parts.append(text)
            texts = parts
        enc_kwargs = {
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": False,
            **kwargs,
        }
        try:
            vectors = model.encode(texts, convert_to_numpy=True, **enc_kwargs)
        except TypeError:
            # sentence-transformers >= 6 may drop convert_to_numpy; encode()
            # then returns a torch.Tensor — handled by _to_python_floats.
            vectors = model.encode(texts, **enc_kwargs)
        return self._to_python_floats(vectors)

    @staticmethod
    def _to_python_floats(vectors) -> list:
        """Coerce numpy float32 rows / torch tensors into plain Python floats.

        list(np.float32 array) yields numpy.float32 scalars, which are NOT
        JSON-serializable; the embed cache then crashes on save. .tolist()
        returns native Python numbers for both numpy and torch.
        """
        rows = []
        for v in vectors:
            row = v.tolist() if hasattr(v, "tolist") else list(v)
            rows.append([float(x) for x in row])
        return rows

    def _provider_notes(self) -> list[str]:
        return [
            f"local model: {self.model} "
            f"(free, offline, no quota; download on first use)",
            f"dimension: {self._dimension or 'auto'} | "
            f"device: {self.device or 'auto'}",
            "embedding space is model-specific — switch provider only with "
            "ingest --reset",
        ]




def build_embedder(cfg) -> BaseEmbedder:
    """Factory: instantiate the provider named in config.py."""
    providers = {
        "gemini": GeminiEmbedder,
        "openai": OpenAIEmbedder,
        "cohere": CohereEmbedder,
        "voyage": VoyageEmbedder,
        "jina": JinaEmbedder,
        "nvidia": NvidiaEmbedder,
        "huggingface": HuggingFaceEmbedder,
        "hf": HuggingFaceEmbedder,
        "sentence_transformers": HuggingFaceEmbedder,
        "sentence-transformers": HuggingFaceEmbedder,
    }
    name = (cfg.EMBEDDING_PROVIDER or "").strip().lower()
    if name not in providers:
        raise SystemExit(
            f"[embedder] unknown EMBEDDING_PROVIDER {cfg.EMBEDDING_PROVIDER!r}. "
            f"Supported: {', '.join(sorted(providers))}. Edit config.py."
        )
    print(f"[embedder] building provider '{name}' "
          f"(model: {cfg.active_embedding_model()})")
    return providers[name](cfg)
