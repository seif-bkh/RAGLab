"""One retrieval path shared by query, evaluation, and grounded answers."""
from store import (best_variant_merge, blend_hybrid, collection_languages,
                   keyword_search, query_vector, rrf_merge)
from translate import detect_language


def retrieve(cfg, embedder, collection, text, *, language=None, translator=None,
             mode="vector", top_k=5, candidate_k=None, lang_filter=None,
             variant_strategy=None, blend_lambda=None):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if mode not in {"vector", "rrf", "blend"}:
        raise ValueError(f"Unknown retrieval mode {mode!r}")
    strategy = variant_strategy or getattr(cfg, "QUERY_VARIANT_STRATEGY", "best")
    if strategy not in {"original", "best", "translated"}:
        raise ValueError(f"Unknown query variant strategy {strategy!r}")
    language = language or detect_language(text)
    corpus_langs = [lang_filter] if lang_filter else collection_languages(collection)
    if translator is not None and strategy != "original":
        variants = translator.build_variants(text, language, corpus_langs)
        if strategy == "translated":
            variants = [v for v in variants if v["lang"] in corpus_langs] or variants[:1]
    else:
        variants = [{"label": f"{language}(original)", "lang": language,
                     "translated": False, "text": text}]
    count = collection.count()
    if not count:
        return [], variants
    candidates = min(count, max(top_k, candidate_k or getattr(cfg, "RETRIEVAL_CANDIDATE_K", 20)))
    lambd = getattr(cfg, "HYBRID_BLEND_LAMBDA", 0.7) if blend_lambda is None else blend_lambda
    lists = []
    for variant in variants:
        vector = embedder.embed_query(variant["text"])
        hits = query_vector(collection, vector, k=candidates, lang=lang_filter, cfg=cfg)
        if mode != "vector":
            keywords = keyword_search(
                collection, variant["text"], k=candidates, cfg=cfg, lang=lang_filter,
                include_metadata=getattr(cfg, "KEYWORD_SEARCH_INCLUDE_METADATA", True))
            hits = (rrf_merge(hits, keywords, k=cfg.RRF_RANK_CONSTANT) if mode == "rrf" else
                    blend_hybrid(hits, keywords, lambd=lambd))[:candidates]
        lists.append(hits)
    score_key = {"vector": "similarity", "rrf": "rrf_score", "blend": "blend_score"}[mode]
    hits = best_variant_merge(lists, score_key=score_key, labels=[v["label"] for v in variants],
                              tie_break=getattr(cfg, "FUSION_TIE_BREAK", "same_lang_margin"))
    return hits[:top_k], variants


def expand_neighbors(collection, hits, radius=0):
    """Add adjacent chunks from the same source, not ground-truth-selected text.

    Each original hit is followed by its neighbors; de-duplicate by stable ID.
    The answer builder applies the separate overall context token budget.
    """
    if radius < 0 or radius > 2:
        raise ValueError("Neighbor radius must be 0, 1 or 2")
    if not radius:
        return hits
    output, seen = [], set()
    for hit in hits:
        if hit["id"] not in seen:
            output.append(hit)
            seen.add(hit["id"])
        meta = hit.get("metadata") or {}
        index, source = meta.get("chunk_index"), meta.get("source")
        if index is None or not source:
            continue
        ids = [f"{source}::chunk_{i:04d}" for i in range(max(0, index-radius), index+radius+1)
               if i != index and f"{source}::chunk_{i:04d}" not in seen]
        if not ids:
            continue
        records = collection.get(ids=ids, include=["documents", "metadatas"])
        by_id = {i: (t, m) for i, t, m in zip(records["ids"], records["documents"], records["metadatas"])}
        for identifier in ids:
            if identifier in by_id:
                text, metadata = by_id[identifier]
                output.append({"id": identifier, "text": text, "metadata": metadata,
                               "context_neighbor_of": hit["id"], "similarity": None})
                seen.add(identifier)
    return output
