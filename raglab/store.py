"""store.py — local persistent ChromaDB collection + vector/keyword retrieval.

ChromaDB runs locally (PersistentClient on chroma_db/), cosine distance.
Every record carries: chunk text, its embedding, and metadata with document
name, language, section heading, chunk index, source label, ingestion
timestamp and embedding model name.

Keyword fallback: a plain BM25 implementation over the stored chunk texts
(no extra dependency), ran alongside vector search with --hybrid and merged
by reciprocal rank fusion (RRF).
"""

import math
import re
from collections import Counter
from datetime import datetime, timezone

import chromadb

# ---------------------------------------------------------------------------
# Collection handling
# ---------------------------------------------------------------------------


def _client(cfg):
    """One PersistentClient per process, pointed at config.CHROMA_DIR."""
    try:
        settings = chromadb.config.Settings(anonymized_telemetry=False)
    except Exception:  # noqa: BLE001 — settings API varies between versions
        settings = None
    kwargs = {"path": str(cfg.CHROMA_DIR)}
    if settings is not None:
        kwargs["settings"] = settings
    return chromadb.PersistentClient(**kwargs)


def get_collection(cfg, reset: bool = False):
    """Open (or create) the local persistent collection with cosine distance.

    With reset=True the collection is deleted first, so ingestion starts from
    a clean slate.
    """
    client = _client(cfg)
    name = cfg.CHROMA_COLLECTION_NAME

    if reset:
        try:
            client.delete_collection(name)
            print(f"[store] deleted collection '{name}' (reset requested)")
        except Exception as exc:  # noqa: BLE001 — deleting a missing collection is fine
            print(f"[store] no existing collection to delete ({type(exc).__name__})")

    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"[store] collection '{name}' ready | count={collection.count()} | "
          f"distance=cosine | path={cfg.CHROMA_DIR}")
    return collection


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def store_chunks(collection, chunks_with_embeddings: list, cfg) -> int:
    """Add (chunk, embedding) pairs to the collection in small batches.

    chunks_with_embeddings: list of (Chunk, list[float]) tuples.
    When cfg.INDEX_EXCLUDE_BOILERPLATE is True, chunks tagged
    section_type != "content" are skipped (loudly) — they would otherwise
    act as retrieval magnets. Returns the number of records stored.
    """
    if not chunks_with_embeddings:
        print("[store] nothing to store")
        return 0

    exclude = bool(getattr(cfg, "INDEX_EXCLUDE_BOILERPLATE", False))
    kept, skipped = [], []
    for chunk, embedding in chunks_with_embeddings:
        if exclude and getattr(chunk, "section_type", "content") != "content":
            skipped.append(chunk)
            continue
        kept.append((chunk, embedding))

    if skipped:
        print(f"[store] excluding {len(skipped)} boilerplate chunk(s) "
              f"(INDEX_EXCLUDE_BOILERPLATE=True):")
        for chunk in skipped:
            print(f"[store]   skip {chunk.source}::chunk_{chunk.index:04d} "
                  f"[{chunk.section_type}] heading={chunk.heading[:50]!r}")
    elif exclude:
        print("[store] no boilerplate chunks to exclude")

    if not kept:
        print("[store] nothing kept after boilerplate filter")
        return 0

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = len(kept)

    for start in range(0, total, cfg.STORE_BATCH_SIZE):
        batch = kept[start : start + cfg.STORE_BATCH_SIZE]

        ids, embeddings, documents, metadatas = [], [], [], []
        for chunk, embedding in batch:
            ids.append(f"{chunk.source}::chunk_{chunk.index:04d}")
            embeddings.append(embedding)
            documents.append(chunk.text)
            metadatas.append({
                "document": chunk.source,
                "language": chunk.language,
                "heading": chunk.heading,
                "chunk_index": chunk.index,
                "source": chunk.source,
                "origin": chunk.origin,
                "section_type": chunk.section_type,
                "ingested_at": timestamp,
                "embedding_model": cfg.EMBEDDING_MODEL,
                "token_count": chunk.token_count,
            })

        collection.add(ids=ids, embeddings=embeddings,
                       documents=documents, metadatas=metadatas)
        print(f"[store] added {min(start + len(batch), total)}/{total} records")

    print(f"[store] done | stored {total} of {len(chunks_with_embeddings)} chunk(s) | "
          f"collection count={collection.count()}")
    return collection.count()


# ---------------------------------------------------------------------------
# Vector retrieval
# ---------------------------------------------------------------------------


def query_vector(collection, query_embedding: list, k: int, lang: str = None) -> list:
    """Vector search: top-k hits, optional 'where' filter on language.

    Chroma returns distances (0 = identical for cosine). We convert to
    similarity = 1 - distance for readability and keep both.
    Each hit dict: {id, text, metadata, distance, similarity, rank}.
    """
    params = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if lang:
        params["where"] = {"language": lang}

    results = collection.query(**params)
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    hits = []
    for rank, (doc_id, text, meta, distance) in enumerate(
        zip(ids, docs, metas, dists), start=1
    ):
        hits.append({
            "id": doc_id,
            "text": text,
            "metadata": meta,
            "distance": distance,
            "similarity": 1.0 - distance,
            "rank": rank,
        })
    return hits


# ---------------------------------------------------------------------------
# Keyword fallback (plain BM25) + RRF merge
# ---------------------------------------------------------------------------


def _tokenize_kw(text: str) -> list[str]:
    """Lowercase word tokens; unicode-aware so Arabic is tokenized too."""
    return re.findall(r"[\w]+", text.casefold())


class BM25Index:
    """Minimal textbook BM25 over stored chunk texts. Built per query call."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.texts = texts
        self.k1 = k1
        self.b = b
        self.tokenized = [_tokenize_kw(t) for t in texts]
        self.doc_lengths = [len(t) for t in self.tokenized]
        self.avgdl = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0
        # df[t] = number of documents containing term t (the chunk set is small).
        self.df: Counter = Counter()
        for tokens in self.tokenized:
            for term in set(tokens):
                self.df[term] += 1
        self.n = len(texts)

    def score(self, query: str) -> list[float]:
        q_tokens = _tokenize_kw(query)
        scores = [0.0] * self.n
        for doc_idx, tokens in enumerate(self.tokenized):
            tf_counts = Counter(tokens)
            dl = self.doc_lengths[doc_idx]
            for term in set(q_tokens):
                tf = tf_counts.get(term, 0)
                if tf == 0:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[doc_idx] += idf * tf * (self.k1 + 1) / denom
        return scores


def keyword_search(collection, query: str, k: int) -> list:
    """BM25 over every stored chunk text; returns top-k keyword hits."""
    all_docs = collection.get(include=["documents", "metadatas"])
    if not all_docs["ids"]:
        print("[store] keyword search: collection is empty")
        return []

    bm25 = BM25Index(all_docs["documents"])
    scores = bm25.score(query)
    ranked = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:k]

    hits = []
    for rank, idx in enumerate(ranked, start=1):
        score = scores[idx]
        if score <= 0.0:
            break  # no term overlap at all: stop after the non-zero tail
        hits.append({
            "id": all_docs["ids"][idx],
            "text": all_docs["documents"][idx],
            "metadata": all_docs["metadatas"][idx],
            "keyword_score": score,
            "rank": rank,
        })
    return hits


def collection_languages(collection) -> list[str]:
    """Distinct chunk languages in the collection (drives translation targets)."""
    metas = collection.get(include=["metadatas"])["metadatas"] or []
    return sorted({m.get("language") for m in metas if m.get("language")})


def _variant_lang(label: str) -> str:
    """'fr(translated)' -> 'fr', 'ar(original)' -> 'ar'."""
    return str(label).split("(")[0].strip()


def best_variant_merge(variant_hit_lists: list, score_key: str = "similarity",
                       labels: list = None) -> list:
    """Language-normalized best-score fusion across query-language variants.

    Raw scores are NOT comparable across variants: the original (same-language)
    query scores its chunks ~0.80 while a translated query scores its chunks
    ~0.76, so a raw maximum systematically favors the query's own language —
    exactly the language-routing problem translation is meant to fix.

    Therefore each variant's scores are first normalized by the variant's OWN
    best score (relative similarity: how close a chunk is to the best match in
    that query language), then every chunk keeps its best relative score plus
    the variant that produced it and all (variant, rank) pairs. Ties (e.g. the
    correct chunk is rank 1 in its own language AND another variant's best is
    also 1.0) break by variant order, so the original-language ranking is
    preserved and same-language queries are unaffected.

    score_key selects the comparable score within a variant: "similarity"
    (cosine) for vector-only retrieval, "rrf_score" for hybrid.
    """
    if not variant_hit_lists:
        return []
    labels = [str(l) for l in (labels or range(len(variant_hit_lists)))]
    fused: dict[str, dict] = {}
    for v_idx, (label, hits) in enumerate(zip(labels, variant_hit_lists)):
        # Normalization constant: best score inside THIS variant.
        top = max((h.get(score_key) for h in hits
                   if h.get(score_key) is not None), default=None)
        if top is None or top <= 0:
            continue
        for hit in hits:
            val = hit.get(score_key)
            if val is None:
                continue
            rel = val / top
            entry = fused.setdefault(hit["id"], {
                "id": hit["id"],
                "text": hit["text"],
                "metadata": hit["metadata"],
                "similarity": None,
                "rrf_score": None,
                "keyword_score": None,
                "best_score": None,
                "relative_score": None,
                "from_variant": None,
                "variant_ranks": {},
                "_variant_order": v_idx,
                "_best_same_lang": False,
            })
            entry["variant_ranks"][label] = hit["rank"]
            # Keep the best raw value of EACH score type so hybrid entries
            # stay recognizably hybrid; ranking uses the RELATIVE score.
            for key in ("similarity", "rrf_score", "keyword_score"):
                raw = hit.get(key)
                if raw is not None and (entry[key] is None or raw > entry[key]):
                    entry[key] = raw
            if entry["best_score"] is None or val > entry["best_score"]:
                entry["best_score"] = val
            # Relative selection: strictly better wins. On an exact tie the
            # variant whose language matches the chunk's language wins (the
            # translated query is the effective search language for that side
            # of the corpus); the original variant is the final fallback, so
            # same-language questions keep their ranking.
            chunk_lang = (entry["metadata"] or {}).get("language")
            variant_lang = _variant_lang(label)
            same_lang = bool(chunk_lang and chunk_lang == variant_lang)
            if (entry["relative_score"] is None
                    or rel > entry["relative_score"]
                    or (rel == entry["relative_score"]
                        and same_lang and not entry["_best_same_lang"])):
                entry["relative_score"] = rel
                entry["from_variant"] = label
                entry["_variant_order"] = v_idx
                entry["_best_same_lang"] = same_lang

    merged = [e for e in fused.values() if e["relative_score"] is not None]
    merged.sort(key=lambda e: (-e["relative_score"], e["_variant_order"],
                               -(e.get("best_score") or 0.0)))
    for rank, entry in enumerate(merged, start=1):
        entry["rank"] = rank
        entry.pop("_variant_order", None)
        entry.pop("_best_same_lang", None)
    return merged


def rrf_merge(vector_hits: list, keyword_hits: list, k: int = 60) -> list:
    """Reciprocal rank fusion: 1/(k + rank) per list, summed, then re-ranked."""
    fused: dict[str, dict] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        entry = fused.setdefault(hit["id"], {
            "id": hit["id"],
            "text": hit["text"],
            "metadata": hit["metadata"],
            "vector_rank": None,
            "keyword_rank": None,
            "similarity": None,
            "keyword_score": None,
            "rrf_score": 0.0,
        })
        entry["vector_rank"] = rank
        entry["similarity"] = hit["similarity"]
        entry["rrf_score"] += 1.0 / (k + rank)

    for rank, hit in enumerate(keyword_hits, start=1):
        entry = fused.setdefault(hit["id"], {
            "id": hit["id"],
            "text": hit["text"],
            "metadata": hit["metadata"],
            "vector_rank": None,
            "keyword_rank": None,
            "similarity": None,
            "keyword_score": None,
            "rrf_score": 0.0,
        })
        entry["keyword_rank"] = rank
        entry["keyword_score"] = hit["keyword_score"]
        entry["rrf_score"] += 1.0 / (k + rank)

    merged = sorted(fused.values(), key=lambda e: e["rrf_score"], reverse=True)
    for rank, entry in enumerate(merged, start=1):
        entry["rank"] = rank
    return merged
