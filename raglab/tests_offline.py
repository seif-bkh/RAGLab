"""tests_offline.py — no-API regression tests run by `run_tests.sh`.

Covers the pure-Python parts that need no provider key and no network:
query translation plumbing (graceful degradation), language detection,
best-variant fusion (including the three tie-break policies), RRF/blend
mechanics and the lambda=0/lambda=1 edge cases. Exits 0 when everything
passes; prints one PASS/FAIL line per check.
"""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from translate import QueryTranslator, detect_language  # noqa: E402
from store import best_variant_merge, blend_hybrid, rrf_merge  # noqa: E402

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


class _FakeSentenceTransformer:
    def __init__(self, model_name, device=None):
        self.prompts = {"query": "Instruct: retrieve relevant passages\nQuery: "}
        self._model_name = model_name

    def get_sentence_embedding_dimension(self):
        return 8

    def encode(self, texts, batch_size=None, normalize_embeddings=None,
               show_progress_bar=None, convert_to_numpy=None, **kwargs):
        _hf_calls.append((list(texts), dict(kwargs)))
        vecs = []
        for i, _ in enumerate(texts):
            vec = [0.0] * 8
            vec[i % 8] = 1.0
            vecs.append(vec)
        return vecs


def _patch_hf(model_name):
    global _hf_calls
    _hf_calls = []
    old_model = cfg.HF_EMBEDDING_MODEL
    old_provider = cfg.EMBEDDING_PROVIDER
    old_cache = cfg.EMBEDDING_CACHE_PATH
    cfg.HF_EMBEDDING_MODEL = model_name
    cfg.EMBEDDING_PROVIDER = "huggingface"
    cfg.EMBEDDING_CACHE_PATH = Path(tempfile.mkdtemp()) / "emb.json"
    try:
        sys.modules["sentence_transformers"] = types.ModuleType(
            "sentence_transformers")
        sys.modules["sentence_transformers"].SentenceTransformer = \
            _FakeSentenceTransformer
        emb = build_embedder(cfg)
        vecs = emb.embed_texts(["doc one", "doc two"],
                               input_type="search_document")
        q = emb.embed_query("question?")
        return emb, vecs, q
    finally:
        cfg.HF_EMBEDDING_MODEL = old_model
        cfg.EMBEDDING_PROVIDER = old_provider
        cfg.EMBEDDING_CACHE_PATH = old_cache
        sys.modules.pop("sentence_transformers", None)


# Qwen3: documents raw, queries via the model's built-in prompt.
emb, vecs, q = _patch_hf("Qwen/Qwen3-Embedding-0.6B")
check("hf provider detects dimension", emb._dimension == 8
      and len(vecs[0]) == 8)
check("hf qwen: docs raw + queries use prompt_name='query'",
      _hf_calls[0][1] == {} and _hf_calls[1][1].get("prompt_name") == "query",
      str(_hf_calls))

# BGE-M3 family: hand-written retrieval instruction on queries only.
emb, _, _ = _patch_hf("BAAI/bge-m3")
check("hf bge-m3: query instruction prefixed, docs raw",
      _hf_calls[0][0] == ["doc one", "doc two"]
      and all(t.startswith("Represent this sentence for searching relevant "
                           "passages: ") for t in _hf_calls[1][0]),
      repr(_hf_calls[1][0][0][:60]))

# E5 family: query:/passage: prefixes on both sides.
emb, _, _ = _patch_hf("intfloat/multilingual-e5-large")
check("hf e5: query:/passage: prefixes",
      all(t.startswith("passage: ") for t in _hf_calls[0][0])
      and all(t.startswith("query: ") for t in _hf_calls[1][0]),
      repr(_hf_calls[1][0][0][:30]))

# Explicit overrides beat family defaults (raw prefix, no prompt_name).
cfg.HF_EMBEDDING_QUERY_PROMPT = "Q> "
cfg.HF_EMBEDDING_DOC_PROMPT = "D> "
emb, _, _ = _patch_hf("Qwen/Qwen3-Embedding-0.6B")
check("hf explicit prompt overrides",
      _hf_calls[0][0] == ["D> doc one", "D> doc two"]
      and _hf_calls[1][0] == ["Q> question?"]
      and "prompt_name" not in _hf_calls[1][1],
      repr(_hf_calls))
cfg.HF_EMBEDDING_QUERY_PROMPT = ""
cfg.HF_EMBEDDING_DOC_PROMPT = ""

# Prompts can be disabled entirely (raw text both sides).
cfg.HF_EMBEDDING_USE_PROMPTS = False
emb, _, _ = _patch_hf("Qwen/Qwen3-Embedding-0.6B")
check("hf prompts disabled -> raw text both sides",
      _hf_calls[0][0] == ["doc one", "doc two"]
      and _hf_calls[1][0] == ["question?"])
cfg.HF_EMBEDDING_USE_PROMPTS = True

sys.exit(0 if ok else 1)
