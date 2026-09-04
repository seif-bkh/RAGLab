"""embedder.py — hosted embedding API behind a tiny interface + local cache.

One class per provider (OpenAI / Cohere / Voyage), all inheriting from
BaseEmbedder so the rest of the lab only sees:

    embedder = build_embedder(config)
    embedder.embed_texts(["some text", ...])   -> list[list[float]]
    embedder.embed_query("question")           -> list[float]
    embedder.startup_report()                  -> prints provider/model/dimension
                                                  + runs the 3-language sanity check

Behaviour (all of it explicit and printable):
- keys come from .env, never from code;
- batch embedding (EMBEDDING_BATCH_SIZE per API call);
- retry with exponential backoff on rate limits / connection errors;
- embeddings cached in embeddings_cache.json, keyed by
  sha256(model_name + "\n" + text), so metadata-only re-ingests cost nothing.
"""

import hashlib
import json
import os
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

    def save(self):
        payload = {
            "model": self.model,
            "entries": self.entries,
            "note": "key = sha256(model + '\\n' + text); preview shows the text tail",
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


def cache_key(model: str, text: str) -> str:
    """Deterministic key: sha256 over model name and the exact text."""
    payload = (model + "\n" + text).encode("utf-8")
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
    def _call_with_retry(self, texts: list[str], input_type: str) -> list:
        """Call _embed_batch, retrying on any provider error (bounded).

        Retries are bounded by max_retries; each failure is printed together
        with the exception type so rate limits are never silent.
        """
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._embed_batch(texts, input_type)
            except Exception as exc:  # noqa: BLE001 — providers raise many SDK-specific types
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
            key = cache_key(self.model, text)
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
        print("[embedder] multi-language sanity check follows (this may hit the API once):")
        self._sanity_check()
        print("=" * 72)

    def _sanity_check(self):
        """Embed three phrases, one per language, and print pairwise cosine
        similarity so you can see whether they live in one shared space."""
        phrases = [
            ("English", "savings account"),
            ("French", "compte épargne"),
            ("Arabic", "حساب التوفير"),
        ]
        vectors = self.embed_texts([p[1] for p in phrases])
        self._dimension = len(vectors[0]) if vectors else None

        print("[sanity]  dimension:", self._dimension)
        for i in range(len(phrases)):
            for j in range(i + 1, len(phrases)):
                sim = cosine(vectors[i], vectors[j])
                print(f"[sanity]  cosine({phrases[i][0]:<7}, {phrases[j][0]:<7}) = {sim:+.4f}")
        print("[sanity]  interpretation: if all three similarities are clearly positive, "
              "the model places the languages in a shared space; negative/near-zero "
              "values mean cross-lingual retrieval is likely to fail.")
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
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, timeout=60.0, max_retries=0)
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
        import cohere

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
        import voyageai

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
