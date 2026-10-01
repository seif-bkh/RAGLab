#!/usr/bin/env python3
"""Deterministic feature-based reranker — after retrieval, before the top-k cut.

Phase 3 intervention 3 (RAGLAB_ROADMAP.md): reorder the retrieved candidates
with DETERMINISTIC signals only (query-term coverage in text and heading, exact
phrase, a rank prior, a degenerate-length penalty) — no model calls, no network,
same output for the same input. The point is the documented hit@1 vs hit@3/5
gap (BM25 62/82/87, live vector 71/80/87): the right chunk is often IN the
candidate pool but not at the top.

Governance:
- Every weight below is review data (plan item 6,
  raglab/audits/PHASE3_INTERVENTIONS.md). Defaults are round and principled;
  the before/after measurement on the adopted 50-case set is recorded in the
  plan's execution record — if the numbers regress, the weights change or the
  intervention stays off.
- OFF by default (config.RERANK_ENABLED): the deployed path is unchanged until
  the owner activates it after reviewing the numbers.
- The reranker only REORDERS the candidate pool it is given; it can never
  introduce text that was not retrieved.
"""

from __future__ import annotations

from evaluate import normalize_for_match

# A minimal governed stop list for the COVERAGE FEATURES only (not for
# retrieval itself): function words that would dilute the term-coverage signal.
# Deterministic, lowercase, fixed order.
STOP_WORDS = frozenset({
    # Arabic function words
    "ما", "هو", "هي", "هذا", "هذه", "على", "عن", "في", "من", "الى", "إلى",
    "التي", "الذي", "او", "أو", "و", "ثم", "كل", "بين", "عند", "مع", "هل",
    "كيف", "ماذا", "لماذا", "متى", "اين", "أين", "هناك",
    # English function words
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "are", "was", "were", "what", "which", "how", "does", "do",
    # French function words
    "le", "la", "les", "de", "du", "des", "et", "ou", "un", "une", "dans",
    "sur", "pour", "au", "aux", "est", "que", "quoi", "comment",
})

# Governed weights (review data — see module docstring).
DEFAULT_WEIGHTS = {
    "prior_rank": 1.0,        # 1/base_position: the retriever's own ordering
    "term_coverage": 3.0,     # fraction of query terms present in the text
    "heading_coverage": 1.0,  # fraction of query terms present in the heading
    "phrase_bonus": 2.0,      # the whole normalized query occurs in the text
    "short_penalty": 0.5,     # degenerate chunks (fewer than 8 raw words)
}


def query_terms(text: str) -> list[str]:
    """Normalized, stop-filtered, de-duplicated query terms (order kept)."""
    out: list[str] = []
    seen: set = set()
    for raw in normalize_for_match(text).split():
        term = raw.strip("،؛.:!?()[]\"'«»-؟")
        if len(term) < 2 or term in STOP_WORDS or term in seen:
            continue
        seen.add(term)
        out.append(term)
    return out


def _heading_of(hit: dict) -> str:
    return (hit.get("heading")
            or (hit.get("metadata") or {}).get("heading")
            or "")


def _features(query_text: str, hit: dict, position: int,
              terms: list[str] | None = None) -> dict:
    terms = query_terms(query_text) if terms is None else terms
    text_norm = normalize_for_match(hit.get("text") or "")
    head_norm = normalize_for_match(_heading_of(hit))
    phrase = normalize_for_match(query_text) in text_norm
    if terms:
        in_text = sum(1 for t in terms if t in text_norm)
        in_head = sum(1 for t in terms if t in head_norm)
        term_coverage = in_text / len(terms)
        heading_coverage = in_head / len(terms)
    else:
        term_coverage = heading_coverage = 0.0
    n_words = len((hit.get("text") or "").split())
    return {
        "base_position": position,
        "prior_rank": 1.0 / position,
        "term_coverage": term_coverage,
        "heading_coverage": heading_coverage,
        "phrase_bonus": 1.0 if phrase else 0.0,
        "short_penalty": 1.0 if 0 < n_words < 8 else 0.0,
    }


def rerank_hits(query_text: str, hits: list,
                weights: dict | None = None) -> list:
    """Deterministic reorder of `hits` (best first).

    score = sum(weight * feature); ties break by the ORIGINAL position, so an
    equally-scoring candidate keeps the retriever's order. Every output hit is
    annotated with h["rerank"] = {"score", "features"} — the inputs are not
    otherwise modified.
    """
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    terms = query_terms(query_text)
    scored = []
    for position, hit in enumerate(hits, start=1):
        feats = _features(query_text, hit, position, terms)
        score = (w["prior_rank"] * feats["prior_rank"]
                 + w["term_coverage"] * feats["term_coverage"]
                 + w["heading_coverage"] * feats["heading_coverage"]
                 + w["phrase_bonus"] * feats["phrase_bonus"]
                 - w["short_penalty"] * feats["short_penalty"])
        scored.append((score, position, hit, feats))
    scored.sort(key=lambda t: (-t[0], t[1]))
    out = []
    for score, _position, hit, feats in scored:
        annotated = dict(hit)
        annotated["rerank"] = {"score": round(score, 6),
                               "features": feats}
        out.append(annotated)
    return out
