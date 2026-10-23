#!/usr/bin/env python3
"""Minimal relations tables — Phase 4, item 3 (raglab/audits/PHASE4_KNOWLEDGE.md).

Exactly TWO relation tables, per the plan — no graph store:

1. GROUNDING (إرساء): institutional documents grounded on law articles.
   Declared data; every edge carries its verbatim codex evidence plus the
   audit reference that documents it.
2. INTERNAL REFERENCES (إحالات داخلية): cross-references between the law's
   own articles («الفصل X من هذا القانون»), extracted DETERMINISTICALLY
   from the item-1 units. The rule is deliberately narrow — the reference
   form must literally name «من هذا القانون», so references to OTHER laws
   («الفصل 254 من المجلة الجزائية») never become edges.

Every edge is documented with its source: the evidence span must appear
VERBATIM in the source unit's text (grounding: in the source document's
adopted codex), and the target unit must exist. validate_relations()
enforces all of this; the CLI exits non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import restructure  # noqa: E402
import units        # noqa: E402

LAW_DOC = "Loi_2016-48.pdf"

# ---------------------------------------------------------------------------
# Table 1 — grounding edges (declared data; REVIEW DATA like TYPE_RULES)
# ---------------------------------------------------------------------------

GROUNDING_EDGES: list[dict] = [
    {
        "relation": "مؤسس على",
        "from_document": "Circulaire_BCT_2019-08.pdf",
        "to_unit": "loi-2016-48:art011",
        "evidence": "وخاصة الفصل 11 منه وما بعده",
        "documented_by": "audits/MASTER_INDEX.md §6 (السلسلة القانونية: الفصل 11 "
                         "من القانون — تعريف الصيرفة الإسلامية ومراقبة مطابقتها — "
                         "أساس إحالة المنشور)",
    },
]

# ---------------------------------------------------------------------------
# Table 2 — internal cross-references (deterministic extraction)
# ---------------------------------------------------------------------------

# «الفصل24 من هذا القانون» / «بالفصل 34 من هذا القانون» (the glued ب- prefix
# leaves الفصل intact as a substring) / «الفصول 49 و50 و51 من هذا القانون».
# Plural lists yield one edge per named article number.
_INTERNAL_REF_RE = re.compile(
    r"(?:الفصل|الفصول)\s*((?:\d+)(?:\s*و\s*\d+){0,5})\s*من هذا القانون")


def extract_internal_references(law_units: list[dict]) -> list[dict]:
    """Cross-reference edges between the law's own articles.

    One edge per (source unit, referenced article number, occurrence). The
    evidence span is the matched text — verbatim in the source unit's body by
    construction.
    """
    edges: list[dict] = []
    for u in law_units:
        for m in _INTERNAL_REF_RE.finditer(u["text"]):
            for num_str in re.findall(r"\d+", m.group(1)):
                edges.append({
                    "relation": "إحالة داخلية",
                    "from_unit": u["unit_id"],
                    "to_unit": f"{units.UNIT_PREFIX}:art{int(num_str):03d}",
                    "evidence": m.group(0),
                })
    return edges


def _target_unit_ids(law_units: list[dict]) -> set[str]:
    return {u["unit_id"] for u in law_units}


def validate_relations(law_units: list[dict]) -> dict:
    """The plan's verification: every edge documented with its codex source.

    Checks: grounding evidence verbatim in the source document's adopted
    codex; grounding target exists; internal evidence verbatim in the source
    unit's body; internal targets exist (within the law's article range).
    """
    report: dict = {
        "grounding_edges": len(GROUNDING_EDGES),
        "internal_edges": 0,
        "violations": [],
        "self_references": 0,
    }
    unit_ids = _target_unit_ids(law_units)
    by_id = {u["unit_id"]: u for u in law_units}

    for e in GROUNDING_EDGES:
        codex = restructure._adopted_codex_text(e["from_document"])
        if codex is None or e["evidence"] not in codex:
            report["violations"].append(
                f"grounding evidence not verbatim in {e['from_document']}: {e['evidence']!r}")
        if e["to_unit"] not in unit_ids:
            report["violations"].append(f"grounding target missing: {e['to_unit']}")

    internal = extract_internal_references(law_units)
    report["internal_edges"] = len(internal)
    for e in internal:
        src = by_id.get(e["from_unit"])
        if src is None or e["evidence"] not in src["text"]:
            report["violations"].append(
                f"internal evidence not verbatim in {e['from_unit']}: {e['evidence']!r}")
        if e["to_unit"] not in unit_ids:
            report["violations"].append(f"internal target missing: {e['to_unit']}")
        if e["from_unit"] == e["to_unit"]:
            report["self_references"] += 1
    return report


def main() -> int:
    parser = argparse.ArgumentParser(prog="relations",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--emit", action="store_true",
                        help="write the relations data file (declared data)")
    parser.add_argument("--out", default=None,
                        help="output path (default: relations_loi_2016_48.json)")
    args = parser.parse_args()

    codex = restructure._adopted_codex_text(LAW_DOC)
    if codex is None:
        raise SystemExit(f"no adopted codex for {LAW_DOC}")
    law_units = units.extract_law_units(codex)
    report = validate_relations(law_units)
    print(json.dumps(report, ensure_ascii=False, indent=1))

    if args.emit:
        internal = extract_internal_references(law_units)
        out = Path(args.out) if args.out else HERE / "relations_loi_2016_48.json"
        out.write_text(json.dumps({
            "_comment": "Minimal relations tables — Phase 4 item 3. Grounding "
                        "(declared) + internal cross-references (deterministic "
                        "from the item-1 units). Regenerate: python relations.py "
                        "--emit. Every edge documented with verbatim codex "
                        "evidence; validated by validate_relations.",
            "grounding": GROUNDING_EDGES,
            "internal": internal,
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"[relations] wrote {out}")

    return 0 if not report["violations"] else 1


if __name__ == "__main__":
    sys.exit(main())
