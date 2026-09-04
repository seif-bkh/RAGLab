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
    Returns the number of records added.
    """
    if not chunks_with_embeddings:
        print("[store] nothing to store")
        return 0

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = len(chunks_with_embeddings)

    for start in range(0, total, cfg.STORE_BATCH_SIZE):
        batch = chunks_with_embeddings[start : start + cfg.STORE_BATCH_SIZE]

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
                "ingested_at": timestamp,
                "embedding_model": cfg.EMBEDDING_MODEL,
                "token_count": chunk.token_count,
            })

        collection.add(ids=ids, embeddings=embeddings,
                       documents=documents, metadatas=metadatas)
        print(f"[store] added {min(start + len(batch), total)}/{total} records")

    print(f"[store] done | collection count={collection.count()}")
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
