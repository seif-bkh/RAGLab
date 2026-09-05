"""store.py — local persistent ChromaDB collection + vector/keyword retrieval.

ChromaDB runs locally (PersistentClient on chroma_db/), cosine distance.
Every record carries: chunk text, its embedding, and metadata with document
name, language, section heading, chunk index, source label, ingestion
timestamp and embedding model name.

Keyword fallback: a plain BM25 implementation over the stored chunk texts
(no extra dependency), ran alongside vector search with --hybrid and merged
by reciprocal rank fusion (RRF).
"""

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone

import chromadb

from chunker import chunk_fingerprint
from embedder import _vector_issue

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


def chunk_fp(cfg) -> str:
    """Fingerprint of the chunking inputs currently configured (see
    chunker.chunk_fingerprint — it must match exactly what chunk_all uses)."""
    return chunk_fingerprint(
        chunk_size=cfg.CHUNK_SIZE_TOKENS,
        overlap=cfg.CHUNK_OVERLAP_TOKENS,
        split_on_headings=cfg.SPLIT_ON_HEADINGS_FIRST,
        sentence_aware_overlap=cfg.CHUNK_OVERLAP_SENTENCE_AWARE)


def ensure_fresh_chunks(collection, cfg) -> None:
    """Refuse retrieval/extension over a collection built with other settings.

    Chunks are immutable records: if the chunker version, size, overlap or
    heading/sentence policy changed, every stored chunk's text is outdated
    and any retrieval result is silently wrong — so raise an actionable
    error instead of returning misleading hits.

    Collections from before fingerprinting (no 'chunk_fp' metadata field)
    warn instead of failing: they are almost certainly stale, but the old
    ones had no way to know, so we do not hard-block first use.
    """
    metas = (collection.get(include=["metadatas"], limit=1)
             .get("metadatas") or [])
    if not metas:
        return  # empty collection: the next ingest will tag it
    stored = (metas[0] or {}).get("chunk_fp")
    current = chunk_fp(cfg)
    if not stored:
        print("[store] WARNING: collection carries no chunk fingerprint "
              "(built by an older RAGLab). Rebuild with "
              "`raglab ingest --reset` to enable staleness checks.")
        return
    if stored != current:
        raise RuntimeError(
            "[store] collection is STALE — chunks were built with "
            f"fingerprint '{stored}' but the current settings produce "
            f"'{current}'. Chunk texts changed, so retrieval results would "
            "be wrong. Rebuild with:  raglab ingest --reset")


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
    if collection.count() > 0:
        # Adding to an existing collection: never mix chunk formats.
        ensure_fresh_chunks(collection, cfg)

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
    dropped_total = 0

    for start in range(0, total, cfg.STORE_BATCH_SIZE):
        batch = kept[start : start + cfg.STORE_BATCH_SIZE]

        bad: list[dict] = []
        filtered = []
        for chunk, embedding in batch:
            issue = _vector_issue(embedding)
            if issue is None:
                filtered.append((chunk, embedding))
            else:
                # chroma rejects a degenerate vector with a cryptic numpy
                # error ("0-dimensional array nan"); drop it LOUDLY with its
                # chunk id instead of hiding it or crashing the whole ingest.
                bad.append({
                    "id": f"{chunk.source}::chunk_{chunk.index:04d}",
                    "issue": issue,
                    "source": chunk.source,
                    "chunk_index": chunk.index,
                    "text": chunk.text[:300],
                })
        if bad:
            dropped_total += len(bad)
            print(f"[store] WARNING: dropping {len(bad)} chunk(s) with "
                  f"invalid vectors ({', '.join(b['issue'] for b in bad)}):")
            for b in bad:
                print(f"[store]   drop {b['id']} [{b['issue']}] "
                      f"{b['text'][:100]!r}")
            drop_path = cfg.RESULTS_DIR / "dropped_vectors.json"
            drop_path.parent.mkdir(parents=True, exist_ok=True)
            drop_path.write_text(
                json.dumps(bad, ensure_ascii=False, indent=1),
                encoding="utf-8")
        batch = filtered
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
                "embedding_model": cfg.active_embedding_model(),
                "token_count": chunk.token_count,
                "chunk_fp": chunk_fp(cfg),
            })

        if ids:
            collection.add(ids=ids, embeddings=embeddings,
                           documents=documents, metadatas=metadatas)
            print(f"[store] added {min(start + len(batch), total)}/{total} records"
                  + (f" (dropped {len(bad)} invalid)" if bad else ""))

    print(f"[store] done | stored {total - dropped_total} of "
          f"{len(chunks_with_embeddings)} chunk(s)"
          + (f" ({dropped_total} invalid dropped)"
             if dropped_total else "")
          + f" | collection count={collection.count()}")
    return collection.count()


# ---------------------------------------------------------------------------
# Vector retrieval
# ---------------------------------------------------------------------------


def query_vector(collection, query_embedding: list, k: int, lang: str = None,
                 cfg=None) -> list:
    """Vector search: top-k hits, optional 'where' filter on language.

    Chroma returns distances (0 = identical for cosine). We convert to
    similarity = 1 - distance for readability and keep both.
    Each hit dict: {id, text, metadata, distance, similarity, rank}.
    cfg: when given, the collection's chunk fingerprint must match the
    current chunking settings (ensure_fresh_chunks) — stale indexes are
    refused with an actionable error instead of returning wrong hits.
    """
    if cfg is not None:
        ensure_fresh_chunks(collection, cfg)
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


def keyword_search(collection, query: str, k: int,
                    include_metadata: bool = True, cfg=None) -> list:
    """BM25 over every stored chunk text; returns top-k keyword hits.

    include_metadata (config KEYWORD_SEARCH_INCLUDE_METADATA) appends the
    source file name + heading to each text: questions that name the document
    ("BCT circular 2019-08", "the internal guide") then match the right
    corpus even if the cue never appears inside the chunk text itself.
    cfg: when given, the collection's chunk fingerprint must match the
    current chunking settings (ensure_fresh_chunks) — stale indexes are
    refused with an actionable error instead of returning wrong hits.
    """
    if cfg is not None:
        ensure_fresh_chunks(collection, cfg)
    all_docs = collection.get(include=["documents", "metadatas"])
    if not all_docs["ids"]:
        print("[store] keyword search: collection is empty")
        return []

    texts = all_docs["documents"]
    if include_metadata:
        metas = all_docs["metadatas"] or []
        # Filenames/headings use underscores ("Guide_Interne_..."); _tokenize_kw
        # treats '_' as a word char, so replace them with spaces or the whole
        # name becomes ONE unsearchable token.
        texts = [
            f"{t}\n{(m or {}).get('source', '').replace('_', ' ')}\n"
            f"{(m or {}).get('heading', '').replace('_', ' ')}"
            for t, m in zip(texts, metas)
        ]
        print("[store] keyword search: metadata (source + heading) appended "
              "to BM25 corpus (KEYWORD_SEARCH_INCLUDE_METADATA)")

    bm25 = BM25Index(texts)
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
                       labels: list = None,
                       tie_break: str = "same_lang_margin") -> list:
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

    tie_break decides rank 1 when two DIFFERENT chunks each top their own
    variant (both relative_score == 1.0) — see config.FUSION_TIE_BREAK.
    """
    if not variant_hit_lists:
        return []
    labels = [str(l) for l in (labels or range(len(variant_hit_lists)))]
    # Confidence margin of each variant's own top-1: how clearly its best
    # match tops its second-best (relative scores). Used by the
    # "same_lang_margin" tie-break; a sharp answer beats a broad match.
    margins: dict[str, float] = {}
    for label, hits in zip(labels, variant_hit_lists):
        rels = sorted(((h.get(score_key) or 0.0) for h in hits if h.get(score_key)
                       is not None and h.get(score_key) > 0), reverse=True)
        if len(rels) >= 2 and rels[0] > 0:
            margins[label] = 1.0 - rels[1] / rels[0]
        else:
            margins[label] = 0.0
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
                "blend_score": None,
                "kw_norm": None,
                "best_score": None,
                "relative_score": None,
                "from_variant": None,
                "variant_ranks": {},
                "_variant_order": v_idx,
                "_best_same_lang": False,
                "_winner_margin": 0.0,
            })
            entry["variant_ranks"][label] = hit["rank"]
            # Keep the best raw value of EACH score type so hybrid entries
            # stay recognizably hybrid; ranking uses the RELATIVE score.
            for key in ("similarity", "rrf_score", "keyword_score",
                        "blend_score", "kw_norm"):
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
                entry["_winner_margin"] = margins.get(label, 0.0)

    merged = [e for e in fused.values() if e["relative_score"] is not None]
    # Tie-break order matters: two DIFFERENT chunks can each be the top-1 of a
    # different query variant (both relative_score == 1.0). Measured in CI:
    #  - variant_order: translated champions demoted → cross-lingual h1 drops;
    #  - raw: cross-variant absolute scores are biased (same-language queries
    #    score higher) → verbatim/same-language h1 drops;
    #  - same_lang_margin: prefer the champion whose query language matches
    #    the chunk's language, then the clearest champion (largest relative
    #    margin over its variant's second-best). Default; CI keeps the table.
    def _tie_key(e):
        if tie_break == "raw":
            return (-e["relative_score"], -(e.get("best_score") or 0.0),
                    e["_variant_order"])
        if tie_break == "variant_order":
            return (-e["relative_score"], e["_variant_order"],
                    -(e.get("best_score") or 0.0))
        same = 0 if e["_best_same_lang"] else 1
        return (-e["relative_score"], same, -e["_winner_margin"],
                -(e.get("best_score") or 0.0), e["_variant_order"])

    merged.sort(key=_tie_key)
    for rank, entry in enumerate(merged, start=1):
        entry["rank"] = rank
        entry.pop("_variant_order", None)
        entry.pop("_best_same_lang", None)
        entry.pop("_winner_margin", None)
    return merged


def blend_hybrid(vector_hits: list, keyword_hits: list,
                 lambd: float = 0.7) -> list:
    """Score-blend fusion: score = lambda*cosine + (1-lambda)*normalized BM25.

    Why: RRF is rank-only and rank-less chunks can beat fact chunks; a score
    blend keeps the dense similarity as the primary signal and uses BM25 as a
    boost. BM25 magnitudes depend on the query (no upper bound), so keyword
    scores are normalized by their OWN max inside this variant (same
    normalization idea as best_variant_merge); cosine is already in [0,1].

    - vector-only chunks: keyword part 0.0;
    - keyword-only chunks: cosine part 0.0 (they can still rank on BM25);
    - every hit keeps its raw similarity / keyword_score for transparency,
      plus "blend_score" (the fused value used for ranking) and "kw_norm".
    Deterministic tie-break: higher similarity first, then id.
    """
    if not vector_hits:
        # Keyword-only: keep the same formula (kw_norm normalized by this
        # variant's own max) so the fused score is never None downstream.
        k_entries = [dict(h) for h in keyword_hits]
        k_max = max((h.get("keyword_score") or 0.0) for h in k_entries) or 1.0
        for entry in k_entries:
            norm = min(1.0, (entry.get("keyword_score") or 0.0) / k_max)
            entry["kw_norm"] = norm
            entry["blend_score"] = (1.0 - lambd) * norm
        return _rank_by_blend(k_entries)
    if not keyword_hits:
        # Pure vector: same formula, no BM25 context (kw part = 0).
        out = [dict(h, blend_score=(h.get("similarity") or 0.0) * lambd,
                    kw_norm=0.0, keyword_score=None) for h in vector_hits]
        return _rank_by_blend(out)

    kw_by_id = {h["id"]: h for h in keyword_hits}
    max_kw = max((h.get("keyword_score") or 0.0) for h in keyword_hits) or 1.0

    merged: dict[str, dict] = {}
    for h in vector_hits:
        entry = dict(h)
        entry["blend_score"] = 0.0
        entry["kw_norm"] = 0.0
        merged[h["id"]] = entry
    for h in keyword_hits:
        entry = merged.setdefault(h["id"], {
            "id": h["id"],
            "text": h["text"],
            "metadata": h["metadata"],
            "rank": None,
            "similarity": None,
            "distance": None,
            "blend_score": 0.0,
            "kw_norm": 0.0,
        })
        entry.setdefault("keyword_score", 0.0)
        if (h.get("keyword_score") or 0.0) > (entry.get("keyword_score") or 0.0):
            entry["keyword_score"] = h.get("keyword_score")

    for entry in merged.values():
        sim = entry.get("similarity") or 0.0
        kw = entry.get("keyword_score") or 0.0
        entry["kw_norm"] = min(1.0, kw / max_kw)
        entry["blend_score"] = lambd * sim + (1.0 - lambd) * entry["kw_norm"]
    return _rank_by_blend(list(merged.values()))


def _rank_by_blend(entries: list) -> list:
    """Sort by blend_score desc (tie: similarity desc, then id) and number ranks."""
    entries.sort(key=lambda e: (-(e.get("blend_score") or 0.0),
                                -(e.get("similarity") or 0.0),
                                str(e.get("id"))))
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
    return entries


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
