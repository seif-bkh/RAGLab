"""tests_offline.py — no-API regression tests run by `run_tests.sh`.

Covers the pure-Python parts that need no provider key and no network:
query translation plumbing (graceful degradation), language detection,
best-variant fusion (including the three tie-break policies), RRF/blend
mechanics and the lambda=0/lambda=1 edge cases. Exits 0 when everything
passes; prints one PASS/FAIL line per check.
"""
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import tempfile

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:  # pragma: no cover — numpy ships with torch/ST anyway
    np = None
    HAVE_NUMPY = False

sys.path.insert(0, str(Path(__file__).resolve().parent))

from translate import QueryTranslator, detect_language  # noqa: E402
from store import (best_variant_merge, blend_hybrid,  # noqa: E402
                     keyword_search, rrf_merge)

ok = True


def check(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        ok = False


# --- translation plumbing ---------------------------------------------------
check("detect ar/fr/en",
      detect_language("ما هو المبلغ؟") == "ar"
      and detect_language("Quel est le montant ?") == "fr"
      and detect_language("What is the amount?") == "en")
check("parse numbered lines",
      QueryTranslator._parse("1. A\n2. B\n", 2) == ["A", "B"])
check("parse rejects wrong count", QueryTranslator._parse("1. A\n", 2) is None)


class FakeCfg:
    QUERY_TRANSLATION_MODEL = "gemini-3.5-flash-lite"
    QUERY_TRANSLATION_CACHE_PATH = Path(TemporaryDirectory().name) / "t.json"


tr = QueryTranslator(FakeCfg())
check("no key -> graceful degradation",
      tr.available is False
      and len(tr.build_variants("x", "ar", ["fr", "ar"])) == 1)


# --- fusion helpers ---------------------------------------------------------
def H(i, rank, sim, rrf=None, kw=None, lang="ar"):
    return {"id": f"chunk_{i}", "text": f"t{i}", "metadata": {"language": lang},
            "rank": rank, "similarity": sim, "rrf_score": rrf,
            "keyword_score": kw, "blend_score": None, "kw_norm": None}


va = [H(1, 1, 0.80), H(2, 2, 0.75), H(3, 3, 0.70)]
vf = [H(4, 1, 0.90, lang="fr"), H(3, 2, 0.71, lang="fr"),
      H(1, 3, 0.60, lang="fr")]
merged = best_variant_merge([va, vf], score_key="similarity",
                            labels=["ar(original)", "fr(translated)"])
check("normalized fusion order", [h["id"] for h in merged] ==
      ["chunk_4", "chunk_1", "chunk_2", "chunk_3"],
      str([(h["id"], h["from_variant"], round(h["relative_score"], 3))
           for h in merged]))
check("variant ranks + from_variant kept",
      merged[1]["from_variant"] == "ar(original)"
      and merged[1]["variant_ranks"] == {"ar(original)": 1,
                                         "fr(translated)": 3}
      and merged[0]["from_variant"] == "fr(translated)")

rrf = rrf_merge([H(1, 1, 0.9)], [H(2, 1, 0.8, kw=3.0)])
check("rrf_merge fills rrf_score", all(h["rrf_score"] > 0 for h in rrf))

bl = blend_hybrid([H(1, 1, 0.9)], [H(1, 1, 0.8, kw=3.0),
                                   H(2, 2, 0.8, kw=1.0)], lambd=0.7)
check("blend normalizes BM25 by max and ranks by blend",
      [h["id"] for h in bl] == ["chunk_1", "chunk_2"]
      and abs(bl[0]["blend_score"] - (0.7 * 0.9 + 0.3 * 1.0)) < 1e-9
      and abs(bl[1]["kw_norm"] - 1.0 / 3.0) < 1e-9,
      str([(h["id"], round(h["blend_score"], 4), round(h["kw_norm"], 4))
           for h in bl]))
check("blend lambda=1 is pure vector order",
      [h["id"] for h in blend_hybrid([H(1, 1, 0.9), H(2, 2, 0.8)],
                                     [H(2, 1, 0.8, kw=5.0)],
                                     lambd=1.0)] == ["chunk_1", "chunk_2"])
bl0 = blend_hybrid([H(1, 1, 0.9), H(2, 2, 0.5)],
                   [H(2, 1, 0.8, kw=5.0), H(1, 2, 0.8, kw=1.0)], lambd=0.0)
check("blend lambda=0 is pure keyword order (BM25 normalized)",
      [h["id"] for h in bl0] == ["chunk_2", "chunk_1"]
      and abs(bl0[0]["kw_norm"] - 1.0) < 1e-9)
check("blend keeps vector-only hits (kw part 0)",
      len(blend_hybrid([H(1, 1, 0.9, kw=None)], [], lambd=0.7)) == 1
      and abs(blend_hybrid([H(1, 1, 0.9, kw=None)], [],
                           lambd=0.7)[0]["blend_score"] - 0.63) < 1e-9)

# Cross-variant champion ties: two DIFFERENT chunks each top their own variant
# (both relative_score=1.0); the three policies must differ as documented.
va2 = [{"id": "x", "text": "tx", "metadata": {"language": "en"}, "rank": 1,
        "similarity": 0.30, "rrf_score": None, "keyword_score": None,
        "blend_score": None, "kw_norm": None},
       {"id": "y", "text": "ty", "metadata": {"language": "ar"}, "rank": 2,
        "similarity": 0.25, "rrf_score": None, "keyword_score": None,
        "blend_score": None, "kw_norm": None}]
vb2 = [{"id": "y", "text": "ty", "metadata": {"language": "ar"}, "rank": 1,
        "similarity": 0.60, "rrf_score": None, "keyword_score": None,
        "blend_score": None, "kw_norm": None},
       {"id": "x", "text": "tx", "metadata": {"language": "en"}, "rank": 2,
        "similarity": 0.20, "rrf_score": None, "keyword_score": None,
        "blend_score": None, "kw_norm": None}]
labels2 = ["en(original)", "ar(translated)"]
m_raw = best_variant_merge([va2, vb2], score_key="similarity",
                           labels=labels2, tie_break="raw")
m_vo = best_variant_merge([va2, vb2], score_key="similarity",
                          labels=labels2, tie_break="variant_order")
m_sm = best_variant_merge([va2, vb2], score_key="similarity",
                          labels=labels2, tie_break="same_lang_margin")
check("raw tie-break prefers absolute score",
      [h["id"] for h in m_raw] == ["y", "x"])
check("variant_order tie-break prefers original variant",
      [h["id"] for h in m_vo] == ["x", "y"])
check("same_lang_margin tie-break prefers clearer champion",
      [h["id"] for h in m_sm] == ["y", "x"]
      and m_sm[0]["from_variant"] == "ar(translated)")

# best_variant_merge must carry blend scores (used by evaluate mode=blend)
vbl = [{"id": "a", "text": "t", "metadata": {"language": "fr"}, "rank": 1,
        "similarity": 0.9, "blend_score": 0.93, "kw_norm": 1.0}]
m = best_variant_merge([vbl], score_key="blend_score",
                       labels=["fr(original)"])
check("fusion carries blend_score + kw_norm",
      m[0]["blend_score"] == 0.93 and m[0]["kw_norm"] == 1.0
      and m[0]["relative_score"] == 1.0)


# --- HuggingFace provider logic (stubbed model, no network) -----------------
# Validates prompt selection + batching + dimension detection without
# downloading any weights: sentence_transformers is replaced by a fake that
# records encode() kwargs and returns normalized vectors.
import math
import types

import config as cfg
from embedder import build_embedder

_hf_calls = []


def _fake_encode(self, texts, batch_size=None, normalize_embeddings=None,
                 show_progress_bar=None, convert_to_numpy=None, **kwargs):
    _hf_calls.append((list(texts), dict(kwargs)))
    rows = []
    for i, _ in enumerate(texts):
        row = [0.0] * 8
        row[i % 8] = 1.0
        rows.append(row)
    # Reproduce sentence-transformers >= 6 behaviour: numpy float32 rows
    # (json.dumps cannot serialize np.float32 scalars).
    if HAVE_NUMPY:
        return [np.array(r, dtype=np.float32) for r in rows]
    return rows


class _FakeSentenceTransformer:
    """ST >= 6 API: get_embedding_dimension (the renamed method)."""

    def __init__(self, model_name, device=None):
        self.prompts = {"query": "Instruct: retrieve relevant passages\nQuery: "}
        self._model_name = model_name

    def get_embedding_dimension(self):
        return 8

    encode = _fake_encode


class _FakeSentenceTransformerLegacy:
    """Old ST API: only get_sentence_embedding_dimension exists."""

    def __init__(self, model_name, device=None):
        self.prompts = {"query": ""}
        self._model_name = model_name

    def get_sentence_embedding_dimension(self):
        return 8

    encode = _fake_encode


def _patch_hf(model_name, fake_cls=None):
    global _hf_calls
    _hf_calls = []
    old_model = cfg.HF_EMBEDDING_MODEL
    old_provider = cfg.EMBEDDING_PROVIDER
    old_cache = cfg.EMBEDDING_CACHE_PATH
    cfg.HF_EMBEDDING_MODEL = model_name
    cfg.EMBEDDING_PROVIDER = "huggingface"
    cache_path = Path(tempfile.mkdtemp()) / "emb.json"
    cfg.EMBEDDING_CACHE_PATH = cache_path
    try:
        sys.modules["sentence_transformers"] = types.ModuleType(
            "sentence_transformers")
        sys.modules["sentence_transformers"].SentenceTransformer = \
            fake_cls or _FakeSentenceTransformer
        emb = build_embedder(cfg)
        vecs = emb.embed_texts(["doc one", "doc two"],
                               input_type="search_document")
        q = emb.embed_query("question?")
        return emb, vecs, q, cache_path
    finally:
        cfg.HF_EMBEDDING_MODEL = old_model
        cfg.EMBEDDING_PROVIDER = old_provider
        cfg.EMBEDDING_CACHE_PATH = old_cache
        sys.modules.pop("sentence_transformers", None)


# Qwen3: documents raw, queries via the model's built-in prompt.
emb, vecs, q, cache_path = _patch_hf("Qwen/Qwen3-Embedding-0.6B")
check("hf provider detects dimension", emb._dimension == 8
      and len(vecs[0]) == 8)
check("hf qwen: docs raw + queries use prompt_name='query'",
      _hf_calls[0][1] == {} and _hf_calls[1][1].get("prompt_name") == "query",
      str(_hf_calls))

# BGE-M3 family: hand-written retrieval instruction on queries only.
emb, _, _, _ = _patch_hf("BAAI/bge-m3")
check("hf bge-m3: query instruction prefixed, docs raw",
      _hf_calls[0][0] == ["doc one", "doc two"]
      and all(t.startswith("Represent this sentence for searching relevant "
                           "passages: ") for t in _hf_calls[1][0]),
      repr(_hf_calls[1][0][0][:60]))

# E5 family: query:/passage: prefixes on both sides.
emb, _, _, _ = _patch_hf("intfloat/multilingual-e5-large")
check("hf e5: query:/passage: prefixes",
      all(t.startswith("passage: ") for t in _hf_calls[0][0])
      and all(t.startswith("query: ") for t in _hf_calls[1][0]),
      repr(_hf_calls[1][0][0][:30]))

# Explicit overrides beat family defaults (raw prefix, no prompt_name).
cfg.HF_EMBEDDING_QUERY_PROMPT = "Q> "
cfg.HF_EMBEDDING_DOC_PROMPT = "D> "
emb, _, _, _ = _patch_hf("Qwen/Qwen3-Embedding-0.6B")
check("hf explicit prompt overrides",
      _hf_calls[0][0] == ["D> doc one", "D> doc two"]
      and _hf_calls[1][0] == ["Q> question?"]
      and "prompt_name" not in _hf_calls[1][1],
      repr(_hf_calls))
cfg.HF_EMBEDDING_QUERY_PROMPT = ""
cfg.HF_EMBEDDING_DOC_PROMPT = ""

# Prompts can be disabled entirely (raw text both sides).
cfg.HF_EMBEDDING_USE_PROMPTS = False
emb, _, _, _ = _patch_hf("Qwen/Qwen3-Embedding-0.6B")
check("hf prompts disabled -> raw text both sides",
      _hf_calls[0][0] == ["doc one", "doc two"]
      and _hf_calls[1][0] == ["question?"])
cfg.HF_EMBEDDING_USE_PROMPTS = True


# --- Jina provider logic (stubbed HTTP, no network) --------------------------
# Validates: task mapping (retrieval.passage/retrieval.query), normalized=true,
# provider-scoped cache, dimension auto-detect, retryable-429 vs fail-fast 400.
import urllib.error
import urllib.request
import embedder as embedder_mod

_jina_calls = []


def _jina_fake_open_factory(status: int | None = None):
    def _fake_open(req, timeout=None):
        _jina_calls.append({
            "url": req.full_url,
            "auth": req.get_header("Authorization"),
            "ctype": req.get_header("Content-type"),
            "payload": json.loads(req.data.decode("utf-8")),
        })
        if status is not None:
            raise urllib.error.HTTPError(
                req.full_url, status, "simulated", {}, None)
        n = len(_jina_calls[-1]["payload"]["input"])
        body = {"object": "list", "data": [
            {"object": "embedding", "index": i, "embedding": [0.25] * 6}
            for i in range(n)]}
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")
        return _Resp()
    return _fake_open


def _patch_jina(status=None):
    """Run one jina embed_texts+embed_query with urlopen stubbed."""
    global _jina_calls
    _jina_calls = []
    old_provider = cfg.EMBEDDING_PROVIDER
    old_cache = getattr(cfg, "JINA_EMBEDDING_CACHE_PATH", None)
    old_key = os.environ.get("JINA_API_KEY", "")
    cfg.EMBEDDING_PROVIDER = "jina"
    cfg.JINA_EMBEDDING_CACHE_PATH = Path(tempfile.mkdtemp()) / "jina_emb.json"
    os.environ["JINA_API_KEY"] = "test-secret"
    real_open = urllib.request.urlopen
    urllib.request.urlopen = _jina_fake_open_factory(status)
    try:
        emb = build_embedder(cfg)
        vecs = emb.embed_texts(["doc one"], input_type="search_document")
        q = emb.embed_query("question?")
        return emb, vecs, q
    finally:
        urllib.request.urlopen = real_open
        cfg.EMBEDDING_PROVIDER = old_provider
        cfg.JINA_EMBEDDING_CACHE_PATH = old_cache
        if old_key:
            os.environ["JINA_API_KEY"] = old_key
        else:
            os.environ.pop("JINA_API_KEY", None)


emb, vecs, q = _patch_jina()
check("jina: passage/query task mapping + normalized + bearer auth",
      _jina_calls[0]["payload"]["task"] == "retrieval.passage"
      and _jina_calls[1]["payload"]["task"] == "retrieval.query"
      and _jina_calls[0]["payload"]["normalized"] is True
      and _jina_calls[0]["payload"]["model"] == "jina-embeddings-v5-omni-small"
      and _jina_calls[0]["auth"] == "Bearer test-secret"
      and _jina_calls[0]["ctype"] == "application/json",
      str(_jina_calls[0]["payload"]))
check("jina: input is a list of {text} objects (sample format)",
      [_jina_calls[0]["payload"]["input"][0]["text"]] == ["doc one"]
      and _jina_calls[1]["payload"]["input"][0]["text"] == "question?",
      str(_jina_calls[0]["payload"]["input"]))
check("jina: dimension auto-detected + scoped cache file",
      emb._dimension == 6 and emb.cache.path.name == "jina_emb.json"
      and Path(emb.cache.path).exists(),
      f"dim={emb._dimension} cache={emb.cache.path}")

# Retryability classification: 429/503 retryable, 400 fails fast (the
# backoff sleep stays out of the unit test).
check("jina: rate-limit/5xx retryable, client errors fail fast",
      embedder_mod.BaseEmbedder._is_retryable(
          embedder_mod.JinaHTTPError(429, "quota"))
      and embedder_mod.BaseEmbedder._is_retryable(
          embedder_mod.JinaHTTPError(503, "unavailable"))
      and not embedder_mod.BaseEmbedder._is_retryable(
          embedder_mod.JinaHTTPError(400, "bad request"))
      and not embedder_mod.BaseEmbedder._is_retryable(
          embedder_mod.JinaHTTPError(401, "unauthorized")))
try:
    _patch_jina(status=400)
    check("jina: 400 propagates as JinaHTTPError", False, "no error raised")
except embedder_mod.JinaHTTPError as _exc:
    check("jina: 400 propagates as JinaHTTPError",
          _exc.status_code == 400 and "Jina API HTTP 400" in str(_exc),
          str(_exc)[:60])

# Missing key -> actionable SystemExit.
old_key = os.environ.pop("JINA_API_KEY", None)
old_provider = cfg.EMBEDDING_PROVIDER
cfg.EMBEDDING_PROVIDER = "jina"
try:
    try:
        build_embedder(cfg)._embed_batch(["x"], "search_document")
        check("jina missing key fails actionably", False, "no error raised")
    except SystemExit as _exc:
        check("jina missing key fails actionably",
              "JINA_API_KEY" in str(_exc) and ".env" in str(_exc),
              str(_exc)[:60])
finally:
    cfg.EMBEDDING_PROVIDER = old_provider
    if old_key:
        os.environ["JINA_API_KEY"] = old_key


# --- NVIDIA provider logic (stubbed HTTP, no network) ------------------------
# Validates the NeMo Retriever path: input_type passage|query (NOT Jina's
# retrieval.* task names), plain-string input list, optional Matryoshka
# dimensions, provider-scoped cache, retryable-429 vs fail-fast 400,
# NaN/malformed/order validation, missing-key SystemExit. Also stubs the
# NVIDIA chat-completions translator (Kimi/DeepSeek) payload + parsing.
_nv_calls = []


def _nvidia_fake_open_factory(status: int | None = None, bad: str | None = None):
    def _fake_open(req, timeout=None):
        _nv_calls.append({
            "url": req.full_url,
            "auth": req.get_header("Authorization"),
            "ctype": req.get_header("Content-type"),
            "payload": json.loads(req.data.decode("utf-8")),
        })
        if status is not None:
            raise urllib.error.HTTPError(
                req.full_url, status, "simulated", {}, None)
        n = len(_nv_calls[-1]["payload"]["input"])
        if bad == "nan":
            dim = len(_nv_calls[-1]["payload"].get("dimensions") or [0.25] * 6)
            embs = [[0.25] * (dim - 1) + [float("nan")] for _ in range(n)]
        elif bad == "missing":
            embs = [None] * n  # items without an "embedding" key
        elif bad == "short":
            embs = [[0.25] * 6]  # fewer rows than inputs
        else:
            embs = [[0.25] * 6 for _ in range(n)]
        body = {"object": "list", "data": [
            {"object": "embedding", "index": i, "embedding": e}
            for i, e in enumerate(embs)]}
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")
        return _Resp()
    return _fake_open


def _patch_nvidia(dim: int = 0, status: int | None = None,
                  bad: str | None = None):
    """Run one nvidia embed_texts+embed_query with urlopen stubbed."""
    global _nv_calls
    _nv_calls = []
    old_provider = cfg.EMBEDDING_PROVIDER
    old_cache = getattr(cfg, "NVIDIA_EMBEDDING_CACHE_PATH", None)
    old_dim = getattr(cfg, "NVIDIA_EMBEDDING_DIM", 0)
    old_key = os.environ.get("NVIDIA_API_KEY", "")
    cfg.EMBEDDING_PROVIDER = "nvidia"
    cfg.NVIDIA_EMBEDDING_CACHE_PATH = Path(tempfile.mkdtemp()) / "nvidia_emb.json"
    cfg.NVIDIA_EMBEDDING_DIM = dim
    os.environ["NVIDIA_API_KEY"] = "test-secret"
    real_open = urllib.request.urlopen
    urllib.request.urlopen = _nvidia_fake_open_factory(status, bad)
    try:
        emb = build_embedder(cfg)
        if bad == "short":  # row-count mismatch needs >= 2 inputs
            vecs = emb.embed_texts(["a", "b"], input_type="search_document")
            return emb, vecs, None
        vecs = emb.embed_texts(["doc one"], input_type="search_document")
        q = emb.embed_query("question?")
        return emb, vecs, q
    finally:
        urllib.request.urlopen = real_open
        cfg.EMBEDDING_PROVIDER = old_provider
        cfg.NVIDIA_EMBEDDING_CACHE_PATH = old_cache
        cfg.NVIDIA_EMBEDDING_DIM = old_dim
        if old_key:
            os.environ["NVIDIA_API_KEY"] = old_key
        else:
            os.environ.pop("NVIDIA_API_KEY", None)


emb, vecs, q = _patch_nvidia()
check("nvidia: passage/query input_type + bearer auth + plain-string input",
      _nv_calls[0]["payload"]["input_type"] == "passage"
      and _nv_calls[1]["payload"]["input_type"] == "query"
      and _nv_calls[0]["payload"]["input"] == ["doc one"]
      and _nv_calls[1]["payload"]["input"] == ["question?"]
      and "task" not in _nv_calls[0]["payload"]
      and _nv_calls[0]["payload"]["model"] == "nvidia/llama-3.2-nv-embedqa-1b-v2"
      and _nv_calls[0]["auth"] == "Bearer test-secret"
      and _nv_calls[0]["ctype"] == "application/json"
      and _nv_calls[0]["url"].endswith("/v1/embeddings"),
      str(_nv_calls[0]["payload"]))
check("nvidia: Matryoshka dimensions honored + auto-detect without dim",
      _patch_nvidia(dim=384) and _nv_calls[0]["payload"].get("dimensions") == 384,
      str(_nv_calls[0]["payload"].get("dimensions")))
emb, _, _ = _patch_nvidia()
check("nvidia: dimension auto-detected + scoped cache file",
      emb._dimension == 6 and emb.cache.path.name == "nvidia_emb.json"
      and Path(emb.cache.path).exists(),
      f"dim={emb._dimension} cache={emb.cache.path}")

# Retryability: NvidiaHTTPError subclasses JinaHTTPError; 429/503 retryable,
# 400/401 fail fast (backoff stays out of the unit test).
check("nvidia: rate-limit/5xx retryable, client errors fail fast",
      embedder_mod.BaseEmbedder._is_retryable(
          embedder_mod.NvidiaHTTPError(429, "quota"))
      and embedder_mod.BaseEmbedder._is_retryable(
          embedder_mod.NvidiaHTTPError(503, "unavailable"))
      and not embedder_mod.BaseEmbedder._is_retryable(
          embedder_mod.NvidiaHTTPError(400, "bad request"))
      and not embedder_mod.BaseEmbedder._is_retryable(
          embedder_mod.NvidiaHTTPError(401, "unauthorized")))
try:
    _patch_nvidia(status=400)
    check("nvidia: 400 propagates as NvidiaHTTPError", False, "no error raised")
except embedder_mod.NvidiaHTTPError as _exc:
    check("nvidia: 400 propagates as NvidiaHTTPError",
          _exc.status_code == 400 and "NVIDIA API HTTP 400" in str(_exc),
          str(_exc)[:60])

# Strict response validation: NaN / missing embedding / wrong count.
for bad, needle, label in (
        ("nan", "contains NaN", "NaN rejected"),
        ("missing", "no embedding list", "embedding-less item rejected"),
        ("short", "returned 1 embeddings for 2 inputs", "row-count mismatch"),
):
    try:
        _patch_nvidia(bad=bad)
        check(f"nvidia: {label}", False, "no error raised")
    except RuntimeError as _exc:
        check(f"nvidia: {label}", needle in str(_exc), str(_exc)[:70])

# Missing key -> actionable SystemExit.
old_key = os.environ.pop("NVIDIA_API_KEY", None)
old_provider = cfg.EMBEDDING_PROVIDER
cfg.EMBEDDING_PROVIDER = "nvidia"
try:
    try:
        build_embedder(cfg)._embed_batch(["x"], "search_document")
        check("nvidia missing key fails actionably", False, "no error raised")
    except SystemExit as _exc:
        check("nvidia missing key fails actionably",
              "NVIDIA_API_KEY" in str(_exc) and ".env" in str(_exc),
              str(_exc)[:60])
finally:
    cfg.EMBEDDING_PROVIDER = old_provider
    if old_key:
        os.environ["NVIDIA_API_KEY"] = old_key


# --- NVIDIA query-translation (stubbed chat-completions) ---------------------
_nv_chat_calls = []


def _nvidia_chat_open_factory(content: str | None = None):
    def _fake_open(req, timeout=None):
        _nv_chat_calls.append({
            "auth": req.get_header("Authorization"),
            "payload": json.loads(req.data.decode("utf-8")),
        })
        body = {"choices": [{"message": {"role": "assistant",
                                         "content": content or "1. ترجمة"}}]}
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")
        return _Resp()
    return _fake_open


class NvidiaCfg(FakeCfg):
    QUERY_TRANSLATION_PROVIDER = "nvidia"
    NVIDIA_TRANSLATION_MODEL = "moonshotai/kimi-k3"
    NVIDIA_TRANSLATION_FALLBACK_MODELS = "deepseek-ai/deepseek-v4-pro"
    NVIDIA_TRANSLATION_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


old_key = os.environ.get("NVIDIA_API_KEY")
real_open = urllib.request.urlopen
urllib.request.urlopen = _nvidia_chat_open_factory()
try:
    os.environ["NVIDIA_API_KEY"] = "test-secret"
    tr_nv = QueryTranslator(NvidiaCfg())
    out = tr_nv._chat("moonshotai/kimi-k3", "1. ما هو المبلغ؟")
finally:
    urllib.request.urlopen = real_open
    if old_key:
        os.environ["NVIDIA_API_KEY"] = old_key
    else:
        os.environ.pop("NVIDIA_API_KEY", None)
check("nvidia chat: thinking off + temp 0 + bearer + content parsed",
      tr_nv.available and out == "1. ترجمة"
      and _nv_chat_calls[0]["payload"]["model"] == "moonshotai/kimi-k3"
      and _nv_chat_calls[0]["payload"]["temperature"] == 0
      and _nv_chat_calls[0]["payload"]["chat_template_kwargs"] == \
          {"thinking": False}
      and _nv_chat_calls[0]["auth"] == "Bearer test-secret"
      and _nv_chat_calls[0]["payload"]["messages"][0]["role"] == "user",
      str(_nv_chat_calls[0]["payload"]["model"]))

# Reasoning fences are stripped if a model returns them anyway.
real_open = urllib.request.urlopen
urllib.request.urlopen = _nvidia_chat_open_factory(
    "```thinking\n1. ترجمة\n```")
try:
    os.environ["NVIDIA_API_KEY"] = "test-secret"
    tr_nv = QueryTranslator(NvidiaCfg())
    fenced = tr_nv._chat("moonshotai/kimi-k3", "1. ما هو المبلغ؟")
finally:
    urllib.request.urlopen = real_open
    if old_key:
        os.environ["NVIDIA_API_KEY"] = old_key
    else:
        os.environ.pop("NVIDIA_API_KEY", None)
check("nvidia chat: reasoning fence stripped", fenced == "1. ترجمة",
      repr(fenced))

# No NVIDIA key -> translator unavailable (graceful, never a crash).
old_key = os.environ.pop("NVIDIA_API_KEY", None)
try:
    tr_nv = QueryTranslator(NvidiaCfg())
finally:
    if old_key:
        os.environ["NVIDIA_API_KEY"] = old_key
check("nvidia translator: missing key degrades gracefully",
      tr_nv.available is False
      and tr_nv.build_variants("x", "ar", ["fr", "ar"])[0]["lang"] == "ar",
      f"available={tr_nv.available}")


# --- the two bugs the user hit on sentence-transformers 6.x -----------------
# (1) float32 embeddings must be JSON-serializable in the cache.
payload = json.loads(cache_path.read_text(encoding="utf-8"))
all_floats = all(
    isinstance(v, float)
    for entry in payload["entries"].values()
    for v in entry["embedding"])
check("hf float32 rows cached as JSON python floats",
      all_floats and len(payload["entries"]) >= 1,
      f"entries={len(payload['entries'])}")

# (2) dimension detection must work when only the NEW method exists.
emb, _, _, _ = _patch_hf("Qwen/Qwen3-Embedding-0.6B")
check("hf dimension detect via ST>=6 get_embedding_dimension",
      emb._dimension == 8 and emb._dimension == int(
          emb._client.get_embedding_dimension()))

# (3) ...and still work on ST<=5 (legacy method name only).
emb_leg, _, _, _ = _patch_hf("BAAI/bge-m3", _FakeSentenceTransformerLegacy)
check("hf dimension detect falls back to legacy method",
      emb_leg._dimension == 8)


# --- BM25 keyword search honors the document-metadata cue -------------------
class _FakeCollection:
    def __init__(self, ids, docs, metas):
        self._ids = ids; self._docs = docs; self._metas = metas
    def get(self, include=None, limit=None, **kw):
        return {"ids": list(self._ids), "documents": list(self._docs),
                "metadatas": list(self._metas)}
    def count(self):
        return len(self._ids)

# rq14-like: the cue "BCT 2019-08" only exists in the SOURCE name, never in
# the chunk text; without metadata both chunks score 0 (no token overlap),
# with metadata the Circulaire chunk wins.
fc = _FakeCollection(
    ["cir::0", "loi::1"],
    ["تكون عمليات الصيرفة الاسلامية اما في شكل عمليات تمويل تجاري او عمليات تمويل تشاركي او في شكل ودائع استثمارية.",
     "تكون عمليات الصيرفة الاسلامية اما في شكل عمليات تمويل تجاري او عمليات تمويل تشاركي."],
    [{"source": "Circulaire_BCT_2019-08.pdf", "heading": "الفصل 2"},
     {"source": "Loi_2016-48.pdf", "heading": "الفصل 1"}])
q = "bct circular 2019 08 islamic banking operations"
h_off = keyword_search(fc, q, k=5, include_metadata=False)
h_on = keyword_search(fc, q, k=5, include_metadata=True)
check("BM25 without metadata: no doc-cue match (both zero)",
      not any(h["keyword_score"] > 0 for h in h_off),
      str([round(h["keyword_score"], 3) for h in h_off]))
check("BM25 with metadata: source name lifts the right document",
      h_on and h_on[0]["id"] == "cir::0" and h_on[0]["keyword_score"] > 0,
      str([(h["id"], round(h["keyword_score"], 3)) for h in h_on[:2]]))

# Heading cue: "guide interne" matches the Guide source/heading too.
fc2 = _FakeCollection(
    ["guide::2", "madkhal::3"],
    ["تندرج ضمن عمليات الصيرفة الاسلامية في التمويل التجاري عمليات التمويل بصيغة المرابحة والاجارة.",
     "النقد في الاسلام ليس سلعة."],
    [{"source": "Guide_Interne_Operations_Bancaires_Islamiques.docx",
      "heading": "2- عمليات الصيرفة الاسلامية في التمويل التجاري"},
     {"source": "Madkhal_Sayrafa_Islamiya.docx", "heading": "المقدمة"}])
h2 = keyword_search(fc2, "guide interne operations commercial financing",
                    k=5, include_metadata=True)
check("BM25 heading cue also works for the Guide",
      h2 and h2[0]["id"] == "guide::2",
      str([(h["id"], round(h["keyword_score"], 3)) for h in h2[:2]]))


# --- loader/chunker: paragraph boundaries + sentence-aligned hard splits -----
# Regression for the rq13 failure: normalize_text() used to DROP every blank
# line, so each document became one giant paragraph and the chunker hard-split
# it at word boundaries — slicing expected-match phrases in half. These checks
# pin the paragraph-preserving behavior and the sentence-aligned split.
from zipfile import ZipFile  # noqa: E402
from loader import normalize_text, read_docx  # noqa: E402
from chunker import chunk_document  # noqa: E402

_nm = normalize_text("P1.\n\nP2.\n- a\n- b\n\n| x | y |\n| z | w |\n\nP3.")
check("normalize_text keeps paragraph + block boundaries",
      "\n\n" in _nm and _nm.count("\n\n") == 4
      and "- a\n- b" in _nm and "| x | y |\n| z | w |" in _nm,
      repr(_nm))
_nm2 = normalize_text("Wrapped line one\ncontinues here.\n\nNext paragraph.")
check("normalize_text reflows wrapped lines into ONE paragraph",
      "\n\n" in _nm2 and "\ncontinues" not in _nm2,
      repr(_nm2))

_DOCX_XML = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<w:document xmlns:w="http://schemas.openxmlformats.org/'
             'wordprocessingml/2006/main"><w:body>'
             '<w:p><w:r><w:t>First docx paragraph.</w:t></w:r></w:p>'
             '<w:p><w:r><w:t>Second docx paragraph.</w:t></w:r></w:p>'
             '<w:p><w:r><w:t>Third docx key phrase.</w:t></w:r></w:p>'
             '</w:body></w:document>')
with tempfile.TemporaryDirectory() as _td:
    _docx_path = Path(_td) / "sample.docx"
    with ZipFile(_docx_path, "w") as _zf:
        _zf.writestr("word/document.xml", _DOCX_XML)
    _docx_norm = normalize_text(read_docx(_docx_path))
check("read_docx keeps every paragraph separate",
      _docx_norm.count("\n\n") == 2
      and "First docx paragraph." in _docx_norm
      and "Third docx key phrase." in _docx_norm,
      repr(_docx_norm))

# Oversized paragraph: the cut must land BETWEEN sentences, never inside one.
_key_sent = "Beta sentence contains the KEYPHRASE here."
_big_para = ("Alpha sentence opens the paragraph. " + _key_sent +
             " Gamma sentence closes the paragraph. " * 12)
_doc = {"name": "t.txt", "text": _big_para, "language": "en",
        "source": "t.txt", "origin": "data/"}
_chunks = chunk_document(_doc, chunk_size=25, overlap=5)
_txts = [c.text for c in _chunks]
check("hard split keeps every sentence whole in one chunk",
      any(_key_sent in t for t in _txts)
      and all(t.strip().endswith(".") for t in _txts),
      f"chunks={len(_txts)}")
check("oversized paragraph is still flagged (transparency)",
      any(c.notes for c in _chunks),
      str([c.notes for c in _chunks if c.notes][:1]))

# --- collection staleness fingerprint (P0-1) --------------------------------
from chunker import chunk_fingerprint          # noqa: E402
from store import ensure_fresh_chunks, chunk_fp  # noqa: E402


class FpCfg:
    CHUNK_SIZE_TOKENS = 220
    CHUNK_OVERLAP_TOKENS = 40
    SPLIT_ON_HEADINGS_FIRST = True
    CHUNK_OVERLAP_SENTENCE_AWARE = True


class FpCol:
    def __init__(self, fp):
        self.fp = fp

    def get(self, include=None, limit=None, **kw):
        meta = {"chunk_fp": self.fp} if self.fp is not None else {}
        return {"metadatas": [meta] if self.fp is not None else [{}]}


try:
    ensure_fresh_chunks(FpCol(chunk_fp(FpCfg())), FpCfg())
    _fp_fresh = True
except Exception as _e:  # noqa: BLE001
    _fp_fresh = False
check("fresh fingerprint: retrieval allowed",
      _fp_fresh, "")

_fp_old = chunk_fingerprint(220, 40, True, True)
_fp_new = chunk_fingerprint(340, 60, True, True)
check("fingerprint changes with chunk size",
      _fp_old != _fp_new and chunk_fp(FpCfg()) == _fp_old, "")

try:
    ensure_fresh_chunks(FpCol("chunkv1:s500:o100:h1:sen0"), FpCfg())
    _fp_stale = False
except RuntimeError as _e:
    _fp_stale = True
    _fp_msg = str(_e)
check("stale collection: actionable refusal",
      _fp_stale and "ingest --reset" in _fp_msg,
      str(_fp_msg)[:120])

try:
    ensure_fresh_chunks(FpCol(None), FpCfg())
    _fp_legacy = True
except Exception as _e:  # noqa: BLE001
    _fp_legacy = False
check("legacy collection (no fingerprint): warn but do not fail",
      _fp_legacy, "")

class _EmptyCol:
    def get(self, include=None, limit=None, **kw):
        return {"metadatas": []}


try:
    ensure_fresh_chunks(_EmptyCol(), FpCfg())
    _fp_empty = True
except Exception as _e:  # noqa: BLE001
    _fp_empty = False
check("empty collection: no fingerprint required",
      _fp_empty, "")

sys.exit(0 if ok else 1)
