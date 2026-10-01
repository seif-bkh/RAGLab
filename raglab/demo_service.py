#!/usr/bin/env python3
"""Deterministic DEMO service — Experiment 7 (owner directive 2026-10-01
«أنجزها كلها»).

A fully working RAGLab service with ZERO API keys, for the owner to try the
1.3.0 behavior interactively:

- embeddings: an injected deterministic stub (same 8-dim one-hot the offline
  tests use) — retrieval runs in rrf mode so BM25 carries the ranking;
- the answer 'model' is a deterministic quote-bot: it answers with a verbatim
  slice of the top source, THROUGH THE REAL CITATION GATE (quote membership,
  number sourcing, allowed/in-force checks all execute for real);
- the REAL docs/ corpus (restructure arm, 339 chunks);
- Phase-5/6 gates ON so the full behavior is visible: sufficiency fields,
  the commitment (refusal before the model + partial downgrade), the audit
  trail (/audit), /numbers, unit ids on citations.

Honest limits (by design): the answers are verbatim quotes, not generated
prose — the deployed live arm (real embeddings + a real chat model) is what
CI measures. This demo exists to let the owner FEEL the deterministic layers.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DOCS = HERE.parent / "docs"

# --- 1) deterministic embedding stub (injected before the service builds it)
st_mod = types.ModuleType("sentence_transformers")


class _StubSentenceTransformer:
    def __init__(self, model_name, device=None):
        self.prompts = {"query": ""}

    @staticmethod
    def get_embedding_dimension():
        return 8

    def encode(self, texts, batch_size=None, normalize_embeddings=None,
               show_progress_bar=None, convert_to_numpy=None, **kwargs):
        rows = []
        for t in texts:
            row = [0.0] * 8
            row[hash(t) % 8] = 1.0
            rows.append(row)
        return rows


st_mod.SentenceTransformer = _StubSentenceTransformer
sys.modules["sentence_transformers"] = st_mod


# --- 2) the deterministic answer 'model' (through the REAL gate)
class DemoQuoteClient:
    """Answers with verbatim source slices — the citation gate validates
    them for real (membership + numbers + allowed + in-force)."""
    base_url = "https://demo.local/none"
    api_key = "none"

    @staticmethod
    def _slice(text, n=170):
        return " ".join(text.split())[:n]

    def chat(self, model, messages, *, max_tokens=4096):
        payload = json.loads(messages[1]["content"])
        sources = payload.get("sources") or []
        claims = []
        for s in sources[:2]:
            quote = self._slice(s["text"])
            if len(quote) >= 12:
                claims.append({"text": quote,
                               "evidence": [{"source_id": s["source_id"],
                                             "quote": quote}]})
        out = {"answerable": bool(claims), "claims": claims}
        return {"text": json.dumps(out, ensure_ascii=False),
                "served_model": "demo-deterministic-quotebot", "usage": {},
                "seconds": 0.0}


def build_corpus_index(local):
    """Pre-build the collection so the service starts ready to answer."""
    import chunker
    import loader
    from embedder import build_embedder
    from store import get_collection, store_chunks

    docs = loader.load_all([DOCS])
    chunks = chunker.chunk_all(docs, local)
    embedder = build_embedder(local)
    collection = get_collection(local, reset=True)
    store_chunks(collection, list(zip(chunks, embedder.embed_texts(
        [c.text for c in chunks]))), local)
    return collection.count()


def main() -> int:
    import uvicorn
    import profiles
    import service
    from answer import AnswerGenerator
    from profiles import build_lab_config

    state = profiles.default_state()
    state["embedding"] = {"provider": "huggingface",
                          "model": "Qwen/Qwen3-Embedding-0.6B"}
    state["answer"] = {"provider": "nvidia", "model": "demo-deterministic"}
    state["chunking"] = {"mode": "restructure", "size": 220, "overlap": 40}
    state["retrieval"] = {"mode": "rrf", "lang_filter": None}
    state["data_dirs"] = [str(DOCS)]

    demo_dir = HERE / "results" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    overrides = {
        "CHROMA_DIR": demo_dir / "chroma",
        "EMBEDDING_CACHE_PATH": demo_dir / "emb.json",
        "NVIDIA_EMBEDDING_CACHE_PATH": demo_dir / "emb.json",
        "ANSWER_CACHE_PATH": demo_dir / "answers.json",
        "RESULTS_DIR": demo_dir,
        "SUFFICIENCY_FIELDS_ENABLED": True,     # show the evidence state
        "ANSWER_SUFFICIENCY_COMMITMENT": True,  # refuse/partial for real
    }
    local = build_lab_config(state)
    for key, value in overrides.items():
        setattr(local, key, value)

    print("[demo] building the deterministic index over docs/ ...")
    n = build_corpus_index(local)
    print(f"[demo] index ready: {n} chunks")

    generator = AnswerGenerator(local, client=DemoQuoteClient(),
                                approved_models=("demo-deterministic",))
    app = service.create_app(state, generator=generator,
                             allow_profile_switch=False,
                             config_overrides=overrides)
    print("[demo] serving on 0.0.0.0:8000 — try POST /answer, GET /audit, "
          "GET /numbers, GET /docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
