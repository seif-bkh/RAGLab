#!/usr/bin/env python3
"""Offline retrieval investigation (no model calls, no chromadb).

Reproduces the restructure arm's BM25 path (store.BM25Index, metadata
appended exactly like store.keyword_search) over the adopted docs/ corpus,
then ranks the owner's live casino/alcohol questions and locates every
target chunk (ميسر / خمر / محرّم evidence) in the ranking.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import chunker  # noqa: E402
import loader  # noqa: E402

# Verbatim copy of store.BM25Index (pure stdlib) so the probe needs no
# chromadb install — behavior must stay identical to store.keyword_search.
import math  # noqa: E402
import re  # noqa: E402
from collections import Counter  # noqa: E402


def _tokenize_kw(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold())


class BM25Index:  # noqa: D101 — copied from store.py
    def __init__(self, texts, k1: float = 1.5, b: float = 0.75):
        self.texts = texts
        self.k1, self.b = k1, b
        self.tokenized = [_tokenize_kw(t) for t in texts]
        self.doc_lengths = [len(t) for t in self.tokenized]
        self.avgdl = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0
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

DOCS = HERE.parent.parent / "docs"

CFG = SimpleNamespace(
    CHUNKING_MODE="restructure",
    CHUNK_SIZE_TOKENS=220,
    CHUNK_OVERLAP_TOKENS=40,
    SPLIT_ON_HEADINGS_FIRST=True,
    CHUNK_OVERLAP_SENTENCE_AWARE=True,
    RESTRUCTURE_RTL_REPAIR="1",
    CHROMA_COLLECTION_NAME="probe",
    CHROMA_DIR="/tmp/probe_chroma",
    STORE_BATCH_SIZE=64,
    INDEX_EXCLUDE_BOILERPLATE=False,
    KEYWORD_SEARCH_INCLUDE_METADATA=True,
)

# The four live questions (2 EN refused, 1 AR answered with generic chunks,
# plus the alcohol question in Arabic) and the guided-round rescue terms.
QUERIES = [
    ("EN-casino", "can i get financement for casino project?"),
    ("EN-alcohol", "can i engage in alcoholic activities?"),
    ("AR-casino", "هل يجوز للبنك تمويل مشروع كازينو/قمار؟"),
    ("AR-alcohol", "هل يجوز للبنك تمويل أنشطة الخمر؟"),
    ("GUIDED-ميسر", "ميسر"),
    ("GUIDED-خمر", "خمر"),
    ("GUIDED-محرمات", "محرمات"),
]

TARGET_TERMS = ["ميسر", "خمر", "محرم", "محظور", "قمار", "كازينو"]


def main() -> int:
    docs = loader.load_all([DOCS])
    print(f"[probe] loaded {len(docs)} docs: {[d['name'] for d in docs]}")
    chunks = chunker.chunk_all(docs, CFG)
    print(f"[probe] {len(chunks)} restructure chunks | "
          f"{sum(c.token_count for c in chunks)} tokens")

    # Same text assembly as store.keyword_search(include_metadata=True).
    texts = [
        f"{c.text}\n{c.source.replace('_', ' ')}\n{(c.heading or '').replace('_', ' ')}"
        for c in chunks
    ]
    bm25 = BM25Index(texts)

    # Where do the target-evidence chunks sit, globally?
    print("\n=== TARGET EVIDENCE CHUNKS (contain ميسر/خمر/محرّم) ===")
    target_ids = []
    for i, c in enumerate(chunks):
        marks = [t for t in TARGET_TERMS if t in c.text]
        if marks:
            target_ids.append(i)
            snippet = c.text.replace("\n", " ")[:110]
            print(f"  idx={i:4d} {c.source[:34]:34s} hdr="
                  f"{(c.heading or '')[:38]!r} marks={marks}")
            print(f"         {snippet}")

    for label, q in QUERIES:
        scores = bm25.score(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        print(f"\n=== {label} :: {q!r} ===")
        print("  TOP 10 (rank | score | target? | id | heading | snippet):")
        for rank, idx in enumerate(order[:10], start=1):
            c = chunks[idx]
            star = "★TARGET" if idx in target_ids else "       "
            snip = c.text.replace("\n", " ")[:80]
            print(f"  {rank:2d}. {scores[idx]:7.3f} {star} "
                  f"{c.source[:22]:22s} hdr={(c.heading or '')[:30]!r}")
            print(f"        {snip}")
        # ranks of every target chunk with a non-zero score
        hits = []
        for rank, idx in enumerate(order, start=1):
            if idx in target_ids and scores[idx] > 0:
                hits.append((rank, round(scores[idx], 3),
                             chunks[idx].source[:22],
                             [t for t in TARGET_TERMS if t in chunks[idx].text]))
        in5 = [h for h in hits if h[0] <= 5]
        in20 = [h for h in hits if h[0] <= 20]
        print(f"  TARGET chunks ranked: {hits if hits else 'NONE (all score 0)'}")
        print(f"  -> in top-5: {len(in5)} | in top-20: {len(in20)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
