#!/usr/bin/env python3
"""Knowledge units for Loi 2016-48 — Phase 4, item 1 (raglab/audits/PHASE4_KNOWLEDGE.md).

Extracts the law's articles from the ADOPTED corrected codex as knowledge
units with a STRICTLY UTILITARIAN functional typing:

  unit = {
    "unit_id":  "loi-2016-48:art001"          — stable across chunking changes
    "heading":  "الفصل 1"                      — the article marker, verbatim
    "path":     "العنوان الأول > الباب الأول > الفصل 1"
    "text":     verbatim codex body            — never reworded, never trimmed
    "description": first sentence (≤160 chars) — an INDEXABLE surface, not a summary
    "type":     definition | procedure | penalty | delegation | general
    "numeric":  bool                           — feeds the Phase-4 numbers path
  }

Determinism and sourcing:
- The section split REUSES restructure.py's own marker machinery
  (_extract_markers + _is_section_start) — one source of truth for what a
  "الفصل N" boundary is; cross-references ("بالفصل 24") correctly stay body
  text and never start a section.
- The functional types come from the GOVERNED RULES TABLE below: every rule is
  review data (PHASE4_KNOWLEDGE.md item 1.6 — the owner vetoes rules one by
  one). First match wins in the declared order; "general" is the fallback.
- The emitted data file (units_loi_2016_48.json) is declared data: regenerate
  with  python units.py --emit  and review the diff.

Coverage contract: every article marker in the codex becomes exactly one unit
(198/198 for this law; ordinal-spelled first article included).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import restructure  # noqa: E402  — the marker machinery is the shared source of truth

SOURCE_DOC = "Loi_2016-48.pdf"
UNIT_PREFIX = "loi-2016-48"

# ---------------------------------------------------------------------------
# Governed functional-type rules (REVIEW DATA — veto entry by entry)
# Order matters: first match wins. Patterns match the article BODY.
# ---------------------------------------------------------------------------

TYPE_RULES: list[tuple[str, list[str]]] = [
    ("penalty", [
        r"يعاقب",                # العقوبات الجزائية
        r"خطية",
        r"غرامة",
        r"عقوبات? جزائية",
        r"عقوبات? تاديبية",
    ]),
    ("definition", [
        r"^تعتبر",
        r"^يعتبر",
        r"^يعد",
        r"على معنى هذا القانون",
    ]),
    ("delegation", [
        r"^يصدر",
        r"بقرار من",
        r"^يمثل تونس",
        r"باتفاقية",
        r"يمكن بمقتضى اتفاقية",
    ]),
    ("procedure", [
        r"^يجب",
        r"^يمكن",
        r"^تحدد",
        r"^يخضع",
        r"^تخضع",
        r"^يتولى",
        r"^على ",
        r"^تحال",
        r"^يقع",
        r"^تقدم",
        r"^ترفع",
    ]),
    ("general", []),          # fallback — everything else
]

# Numeric-provision flag (a separate dimension, not a type — feeds item 4).
NUMERIC_PATTERNS = [
    r"\d+\s*%", r"في المائة", r"دينار", r"خلال\s+\d+", r"داخل\s+\d+",
    r"أقصى مدة", r"اجل", r"آجال", r"نسبة",
]

_ORDINAL_TO_NUM = {
    "الاول": 1, "الثاني": 2, "الثالث": 3, "الرابع": 4, "الخامس": 5,
    "السادس": 6, "السابع": 7, "الثامن": 8, "التاسع": 9, "العاشر": 10,
}


def _article_number(marker: str) -> int | None:
    m = re.search(r"(\d+)", marker)
    if m:
        return int(m.group(1))
    for ordinal, num in _ORDINAL_TO_NUM.items():
        if ordinal in marker:
            return num
    return None


def _classify(body: str) -> tuple[str, bool]:
    """(type, numeric) — first matching rule wins; numeric is a flag."""
    type_name = "general"
    for name, patterns in TYPE_RULES:
        if name == "general":
            break
        if any(re.search(p, body) for p in patterns):
            type_name = name
            break
    numeric = any(re.search(p, body) for p in NUMERIC_PATTERNS)
    return type_name, numeric


def _description(body: str, cap: int = 160) -> str:
    """First sentence (or first cap chars) — an indexable surface."""
    text = re.sub(r"\s+", " ", body).strip()
    m = re.search(r"[.؛؟]", text)
    head = text[:m.end()] if m and m.end() <= cap + 20 else text[:cap].rsplit(" ", 1)[0]
    return head.strip()


def extract_law_units(codex_text: str) -> list[dict]:
    """Split the adopted codex into article-level units.

    Walks the text line by line through restructure._extract_markers — the
    same call the deployed chunker uses — so a section boundary here is a
    section boundary there. العناوين and الأبواب accumulate into each
    article's heading path; every الفصل section becomes exactly one unit.
    """
    report = restructure.RestructureReport(name=SOURCE_DOC)
    units: list[dict] = []
    title = bab = ""
    current = None  # (marker, lines) for a الفصل in progress

    def flush():
        nonlocal current
        if current is None:
            return
        marker, lines = current
        body = "\n".join(lines).strip()
        num = _article_number(marker)
        current = None
        if num is None:      # a malformed marker is reported, never silently kept
            raise ValueError(f"unparseable article marker: {marker!r}")
        type_name, numeric = _classify(body)
        units.append({
            "unit_id": f"{UNIT_PREFIX}:art{num:03d}",
            "heading": marker,
            "path": " > ".join(x for x in (title, bab, marker) if x),
            "text": body,
            "description": _description(body),
            "type": type_name,
            "numeric": numeric,
        })

    for line in codex_text.split("\n"):
        pieces = restructure._extract_markers(line, report)
        if not pieces:
            # Blank separator lines are part of the VERBATIM contract: keep
            # them so a codex paragraph break (\n\n) survives into the unit
            # body instead of being flattened to a single newline.
            if current is not None and not line.strip():
                current[1].append("")
            continue
        for piece in pieces:
            if piece.startswith("!"):
                marker = piece[1:].strip()
                if marker.startswith("العنوان"):
                    flush()
                    title, bab = marker, ""
                elif marker.startswith("الباب"):
                    flush()
                    bab = marker
                elif marker.startswith("الفصل"):
                    flush()
                    current = (marker, [])
                else:          # القسم وغيرها — لا تُنشئ وحدات
                    flush()
            elif current is not None:
                current[1].append(piece)
    flush()
    return units


def coverage_report(units: list[dict], codex_text: str) -> dict:
    """The automated coverage contract (plan item 1.5)."""
    import evaluate
    norm = evaluate.normalize_for_match
    nums = sorted(int(u["unit_id"].rsplit("art", 1)[1]) for u in units)
    verbatim_ok = all(norm(u["text"]) in norm(codex_text) for u in units)
    return {
        "n_units": len(units),
        "first_article": nums[0] if nums else None,
        "last_article": nums[-1] if nums else None,
        "missing_numbers": sorted(set(range(nums[0], nums[-1] + 1)) - set(nums)) if nums else [],
        "duplicate_ids": len(units) - len({u["unit_id"] for u in units}),
        "empty_bodies": sum(1 for u in units if not u["text"].strip()),
        "empty_descriptions": sum(1 for u in units if not u["description"].strip()),
        "all_text_verbatim_in_codex": verbatim_ok,
        "by_type": {t: sum(1 for u in units if u["type"] == t)
                    for t, _ in TYPE_RULES},
        "numeric_units": sum(1 for u in units if u["numeric"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="units", description=__doc__.splitlines()[0])
    parser.add_argument("--emit", action="store_true",
                        help="write the units data file (declared data)")
    parser.add_argument("--out", default=None,
                        help="output path (default: units_loi_2016_48.json in raglab/)")
    args = parser.parse_args()

    codex = restructure._adopted_codex_text(SOURCE_DOC)
    if codex is None:
        raise SystemExit(f"no adopted codex for {SOURCE_DOC}")
    units = extract_law_units(codex)
    cov = coverage_report(units, codex)
    print(json.dumps(cov, ensure_ascii=False, indent=1))
    if args.emit:
        out = Path(args.out) if args.out else HERE / "units_loi_2016_48.json"
        out.write_text(json.dumps({
            "_comment": f"Knowledge units for {SOURCE_DOC} — Phase 4 item 1. "
                        f"Regenerate: python units.py --emit. Stable unit_ids; "
                        f"verbatim codex text; governed types (units.py TYPE_RULES).",
            "units": units,
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"[units] wrote {out}")
    # The coverage contract: full range, no duplicates, no empty bodies, verbatim.
    ok = (not cov["missing_numbers"] and cov["duplicate_ids"] == 0
          and cov["empty_bodies"] == 0 and cov["empty_descriptions"] == 0
          and cov["all_text_verbatim_in_codex"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
