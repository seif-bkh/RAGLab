#!/usr/bin/env python3
"""Per-micro retrieval & fusion — Experiment 3 (owner directive 2026-10-01
«أنجزها كلها»; the independent wiring gate registered with the item-6
decomposition layer).

When enabled, retrieval runs PER MICRO-QUESTION (the decomposed, reformulated
questions that each carry one requirement) and fuses the pools — so evidence
for a compound/implicit request is searched for requirement by requirement
instead of once for the whole sentence.

- fusion is RECIPROCAL-RANK over the pools (score-agnostic: works in every
  mode), deduplicated by hit id, with the per-micro provenance attached
  (micro_ranks — additive metadata);
- the MAIN question's pool participates like any other (no artificial head
  bonus: the fusion itself decides);
- caps are DECLARED REVIEW DATA: MICROS_MAX (micros actually searched) and
  MICRO_POOL_K (per-micro retrieval depth);
- a single-micro decomposition short-circuits to the plain retrieval —
  identical behavior;
- gated by PER_MICRO_RETRIEVAL_ENABLED (default OFF — activation is the
  owner's gate; the measurement protocol is deterministic-then-live, same
  as every deployed intervention).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ---------------------------------------------------------------------------
# Declared caps (REVIEW DATA)
# ---------------------------------------------------------------------------

MICROS_MAX = 6        # micros actually searched per request
MICRO_POOL_K = 10     # retrieval depth per micro
RRF_K = 60            # fusion constant (retrieval.RRF_RANK_CONSTANT default)


def fuse_micro_pools(pools: list[list[dict]], k: int,
                     rrf_k: int = RRF_K) -> list[dict]:
    """Reciprocal-rank fusion of per-micro hit lists (pure, deterministic).

    Each pool keeps its own ranking; a hit's fused score is the sum of
    1/(rrf_k + rank) over the pools that returned it. Ties break by the best
    single-pool rank, then by first-seen order — fully deterministic.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first: dict[str, dict] = {}
    order: list[str] = []
    for pool in pools:
        for hit in pool:
            hid = hit["id"]
            if hid not in first:
                first[hid] = dict(hit)
                order.append(hid)
            scores[hid] = scores.get(hid, 0.0) + 1.0 / (rrf_k + hit["rank"])
            best_rank[hid] = min(best_rank.get(hid, 10 ** 9), hit["rank"])
    ranked = sorted(order, key=lambda hid: (-scores[hid], best_rank[hid],
                                            order.index(hid)))
    out = []
    for hid in ranked[:k]:
        row = dict(first[hid])
        row["rank"] = len(out) + 1
        row["micro_rrf_score"] = round(scores[hid], 6)
        out.append(row)
    return out


def micro_questions(question: str) -> list[dict]:
    """The micros worth searching: main + sub-question micros, capped."""
    import decompose
    d = decompose.decompose(question)
    micros = [m for m in d["micro_questions"] if m["source_kind"] == "main"]
    micros += [m for m in d["micro_questions"] if m["source_kind"] != "main"]
    return micros[:MICROS_MAX]


def retrieve_micro(cfg, embedder, collection, text, *, top_k=None, **kwargs):
    """Retrieve per micro-question and fuse. Returns (hits, variants) with
    the MAIN question's variants (the outermost caller's contract)."""
    import retrieval
    from evaluate import prepare_query_text

    micros = micro_questions(text)
    top_k = top_k or getattr(cfg, "ANSWER_TOP_K", 20)
    if len(micros) <= 1:
        return retrieval.retrieve(cfg, embedder, collection,
                                  prepare_query_text(text), top_k=top_k,
                                  _in_micro=True, **kwargs)

    pools: list[list[dict]] = []
    variants: list[dict] = []
    for i, micro in enumerate(micros):
        hits, var = retrieval.retrieve(
            cfg, embedder, collection, prepare_query_text(micro["text"]),
            top_k=min(MICRO_POOL_K, top_k), _in_micro=True, **kwargs)
        if i == 0:
            variants = var              # the main question's variants
        pools.append(hits[:MICRO_POOL_K])

    fused = fuse_micro_pools(pools, k=top_k)
    # attach the per-micro provenance (additive metadata)
    for hit in fused:
        ranks = {}
        for i, pool in enumerate(pools):
            for h in pool:
                if h["id"] == hit["id"]:
                    ranks[micros[i]["id"]] = h["rank"]
                    break
        if ranks:
            meta = dict(hit.get("metadata") or {})
            meta["micro_ranks"] = ranks
            hit["metadata"] = meta
    return fused, variants
