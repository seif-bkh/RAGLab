#!/usr/bin/env python3
"""Relational expansion under an intent policy — Phase 5, item 3
(raglab/audits/PHASE5_UNDERSTANDING.md).

Expands retrieval along the Phase-4 relations, UNDER A DECLARED INTENT
POLICY — the expansion is additive beside the vector results (no result is
ever removed; extras only compete for the tail of the top-k window):

- internal cross-references: allowed for إجرائي and استثناء requests only
  (the procedure references its base articles; the exception must arrive
  WITH its rule — both edge directions);
- the grounding edge (Circulaire ← law art 11): allowed when the request
  explicitly spans BOTH the Circulaire and the law (document-family
  patterns both match);
- depth ≤ 2 (edges of extras followed once), node cap per query.

Every extra hit carries `via_relation` metadata and the full VERBATIM unit
text (bigger than a chunk — containment-checked evidence). Gated by
RELATIONAL_EXPANSION_ENABLED (default OFF — activation is the owner's
data-driven gate, per the env-gate pattern of rerank/lexicon).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import intent       # noqa: E402
import restructure  # noqa: E402
import units        # noqa: E402

LAW_DOC = "Loi_2016-48.pdf"

# ---------------------------------------------------------------------------
# Declared policy (REVIEW DATA — veto entry by entry)
# ---------------------------------------------------------------------------

# intent type → allowed internal-edge directions ("" = no expansion)
EXPANSION_POLICY: dict[str, tuple[str, ...]] = {
    "إجرائي": ("from",),          # the procedure references its base articles
    "استثناء": ("from", "to"),    # the exception WITH its rule, both directions
}

DEPTH_MAX = 2          # edges of extras followed once more
NODE_CAP = 4           # max extra units per query

# Cross-document (grounding) trigger: the request names BOTH families
_CIRCULAIRE_RE = re.compile(r"منشور|circulaire", re.I)
_LOI_RE = re.compile(r"قانون\s*(?:عدد\s*)?48|قانون البنوك|law|loi", re.I)


def _law_context():
    """Units + internal edges, built once per call (deterministic)."""
    codex = restructure._adopted_codex_text(LAW_DOC)
    if codex is None:
        return [], []
    law_units = units.extract_law_units(codex)
    import relations
    edges = relations.extract_internal_references(law_units)
    return law_units, edges


def _source_unit_ids(hits: list[dict], law_units: list[dict]) -> set[str]:
    """Law chunks map to units by their heading marker («الفصل183»)."""
    by_heading = {u["heading"]: u["unit_id"] for u in law_units}
    ids = set()
    for h in hits:
        meta = h.get("metadata", {}) or {}
        if (meta.get("document") or meta.get("source")) != LAW_DOC:
            continue
        uid = by_heading.get(meta.get("heading") or "")
        if uid:
            ids.add(uid)
    return ids


def expand(cfg, query: str, hits: list[dict], top_k: int,
           score_key: str | None = None) -> list[dict]:
    """Policy-gated unit hits BESIDE the retrieved results.

    The retrieved list keeps its order at the head of the window; extras
    occupy the TAIL slots of top-k (they displace only the weakest,
    post-rerank tail hits — never a removal, and the strong vector results
    keep their ranks). This makes the expansion measurable at hit@k while
    keeping «التوسيع بجوار النتائج لا إحلالها».
    """
    law_units, edges = _law_context()
    if not law_units or not edges:
        return hits

    r = intent.classify(query)
    directions = EXPANSION_POLICY.get(r["intent_type"], ())
    grounding = bool(_CIRCULAIRE_RE.search(query) and _LOI_RE.search(query))
    if not directions and not grounding:
        return hits

    sources = _source_unit_ids(hits, law_units)
    has_circulaire_hits = any(
        (h.get("metadata", {}) or {}).get("document", "").startswith("Circulaire")
        for h in hits)
    if not sources and not (grounding and has_circulaire_hits):
        return hits

    by_id = {u["unit_id"]: u for u in law_units}
    covered = {h["id"] for h in hits}
    extras: list[dict] = []
    seen = set(sources)

    def add_unit(uid: str, via: str):
        if uid in seen or uid not in by_id or len(extras) >= NODE_CAP:
            return
        seen.add(uid)
        u = by_id[uid]
        # The extra carries the ACTIVE MODE's ranking key at 0.0 — it ranks
        # after every retrieval hit (tail-slot semantics) and downstream
        # score math (evaluate separation) never sees a None score.
        extra = {
            "id": uid,
            "text": u["text"],
            "score": 0.0,                      # ranks after every retrieval hit
            "metadata": {
                "document": LAW_DOC, "source": LAW_DOC, "language": "ar",
                "heading": u["heading"], "unit_id": uid,
                "via_relation": via,           # the expansion's provenance
            },
        }
        if score_key:
            extra[score_key] = 0.0
        extras.append(extra)

    # depth 1: edges from the retrieved units (and to them, per policy)
    frontier = set(sources)
    for e in edges:
        if "from" in directions and e["from_unit"] in frontier:
            add_unit(e["to_unit"], f"internal:{e['from_unit']}→{e['to_unit']}")
        if "to" in directions and e["to_unit"] in frontier:
            add_unit(e["from_unit"], f"internal-rev:{e['to_unit']}←{e['from_unit']}")
    # depth 2: one more hop from the extras
    if len(extras) < NODE_CAP and DEPTH_MAX >= 2:
        frontier2 = {e["id"] for e in extras}
        for e in edges:
            if len(extras) >= NODE_CAP:
                break
            if "from" in directions and e["from_unit"] in frontier2:
                add_unit(e["to_unit"], f"internal-d2:{e['from_unit']}→{e['to_unit']}")

    # the grounding edge: the Circulaire is BUILT ON law art 11 — when the
    # request spans both families and Circulaire evidence was retrieved, the
    # article it is grounded on rides along (deduped if already covered).
    if grounding and has_circulaire_hits:
        add_unit("loi-2016-48:art011", "grounding:Circulaire→art011")

    if not extras:
        return hits
    kept = [dict(h, rank=i + 1) for i, h in enumerate(hits[:max(0, top_k - len(extras))])]
    tail = [dict(h, rank=len(kept) + i + 1) for i, h in enumerate(extras)]
    return kept + tail
