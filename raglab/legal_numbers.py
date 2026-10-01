#!/usr/bin/env python3
"""Structured legal-numbers path — Phase 4, item 4 (raglab/audits/PHASE4_KNOWLEDGE.md).

Deterministic extraction of the law's legal numbers (نسب / حدود مالية / آجال /
عقوبات مالية) from the item-1 units into programmatically-queryable records,
plus the Circulaire corrections table (option ج, owner decision 2026-09-28) as
the path's first scheduled input.

Sourcing contract (the plan's verification):
- every record carries its unit_id and the VERBATIM span it was read from
  (validated: raw in unit text);
- every value is derived deterministically from that span via the GOVERNED
  NUM_WORDS vocabulary below (REVIEW DATA — same veto regime as TYPE_RULES);
- the Circulaire corrections table anchors every official value to the
  ADOPTED codex (verbatim evidence) with its الرائد الرسمي source; the raw
  corrupted forms are kept as historical documentation from the §4 audit.

New additive endpoint (respects the AGENTS §2.7 endpoint freeze):
GET /numbers  — serves these records (service.py; filters: unit_id, kind).
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
# Governed Arabic number-word vocabulary (REVIEW DATA — veto entry by entry)
# Covers exactly the forms attested in the law's adopted codex.
# ---------------------------------------------------------------------------

NUM_WORDS: dict[str, int] = {
    # units
    "واحد": 1, "واحدة": 1, "اثنين": 2, "اثنتين": 2,
    "ثلاثة": 3, "ثلاث": 3, "اربعة": 4, "اربع": 4,
    "خمسة": 5, "خمس": 5, "ستة": 6, "ست": 6,
    "سبعة": 7, "سبع": 7, "ثمانية": 8, "ثماني": 8, "تسعة": 9, "عشرة": 10,
    # tens
    "عشرين": 20, "ثلاثين": 30, "اربعين": 40, "خمسين": 50,
    "ستين": 60, "سبعين": 70, "ثمانين": 80, "تسعين": 90,
    # hundreds
    "مائة": 100, "مئة": 100, "مائتي": 200,
    # thousands / millions (scale words multiply what precedes them)
    "الف": 1_000, "الاف": 1_000, "الفين": 2_000,
    "مليون": 1_000_000, "ملايين": 1_000_000, "مليوني": 2_000_000,
}
_SCALE_WORDS = {"الف", "الاف", "الفين", "مليون", "ملايين", "مليوني"}

# Duration units (value kept in its own unit of measure; dual forms = 2).
_DURATION_UNITS: dict[str, tuple[int, str]] = {
    "يوم": (1, "يوم"), "ايام": (1, "يوم"),
    "شهر": (1, "شهر"), "اشهر": (1, "شهر"), "شهرين": (2, "شهر"), "شهران": (2, "شهر"),
    "سنة": (1, "سنة"), "سنوات": (1, "سنة"), "اعوام": (1, "سنة"),
    "عام": (1, "سنة"), "عامين": (2, "سنة"), "عامان": (2, "سنة"),
}


def parse_number_phrase(phrase: str) -> int | None:
    """Parse a governed number phrase («خمسة وعشرين», «مائة الف», «مليوني»).

    Tokens are whitespace-separated; a standalone or glued-leading «و» is a
    plain connector. Semantics: units/tens/hundreds accumulate in `small`;
    a scale word multiplies the pending `small` (or 1) into `total`; the dual
    scale words («الفين», «مليوني») carry their own value. Returns None when
    any token is outside the governed vocabulary — the caller then skips the
    span (never guesses).
    """
    tokens: list[str] = []
    for t in phrase.split():
        if t == "و":
            continue
        if t.startswith("و") and t[1:] in NUM_WORDS:
            t = t[1:]
        tokens.append(t)
    if not tokens:
        return None
    total = 0    # committed scaled values
    small = 0    # pending units/tens/hundreds
    for tok in tokens:
        val = NUM_WORDS.get(tok)
        if val is None:
            return None
        if tok in ("الفين", "مليوني"):          # dual scales stand alone
            total += (small + val) if small else val
            small = 0
        elif tok in _SCALE_WORDS:               # multiply what precedes
            total += (small if small else 1) * val
            small = 0
        else:
            small += val
    return total + small


# ---------------------------------------------------------------------------
# Deterministic extraction from the law's units
# ---------------------------------------------------------------------------

# Tight alternation of the GOVERNED vocabulary — only number words can sit
# directly before دينار / في المائة, so a match is parseable BY CONSTRUCTION
# (no guessing which preceding word is the number).
_NUM_ALT = "|".join(sorted(NUM_WORDS, key=len, reverse=True))
_NUM_PHRASE = rf"(?:{_NUM_ALT})(?:\s+(?:و\s*)?(?:{_NUM_ALT}))*"
_PERCENT_RE = re.compile(r"(\d{1,3})\s*%")
_PERCENT_WORDS_RE = re.compile(rf"({_NUM_PHRASE})\s+في المائة")
_DINAR_RE = re.compile(rf"({_NUM_PHRASE})\s+دينار")
# number phrase REQUIRED before the unit word (dual forms handled separately)
_DEADLINE_RE = re.compile(
    r"(خلال|داخل|اقصاه|لمدة|غضون)\s+((?:[\u0621-\u064a]+\s+){0,2}[\u0621-\u064a]+?)\s+"
    r"(يوم|ايام|شهر|اشهر|سنة|سنوات|اعوام|عام)")
# dual duration forms carry the number 2 themselves («اقصاه شهران»)
_DEADLINE_DUAL_RE = re.compile(
    r"(?:خلال|داخل|اقصاه|لمدة|غضون)\s+(شهرين|شهران|عامين|عامان)")
_PENALTY_CONTEXT_RE = re.compile(r"خطية|غرامة")


def extract_legal_numbers(law_units: list[dict]) -> list[dict]:
    """One record per matched span: {unit_id, kind, value, unit, raw}."""
    records: list[dict] = []
    for u in law_units:
        text = u["text"]
        is_penalty_unit = u.get("type") == "penalty"

        for m in _PERCENT_RE.finditer(text):
            records.append({"unit_id": u["unit_id"], "kind": "نسبة",
                            "value": int(m.group(1)), "unit": "%",
                            "raw": m.group(0)})

        for m in _PERCENT_WORDS_RE.finditer(text):
            value = parse_number_phrase(m.group(1))
            if value is None:
                continue          # belt: the tight regex already guarantees parse
            records.append({"unit_id": u["unit_id"], "kind": "نسبة",
                            "value": value, "unit": "%",
                            "raw": m.group(0)})

        for m in _DINAR_RE.finditer(text):
            value = parse_number_phrase(m.group(1))
            if value is None:
                continue
            kind = ("عقوبة مالية" if (is_penalty_unit or
                    _PENALTY_CONTEXT_RE.search(text[max(0, m.start()-60):m.end()+60]))
                    else "حد مالي")
            records.append({"unit_id": u["unit_id"], "kind": kind,
                            "value": value, "unit": "دينار",
                            "raw": m.group(0).strip()})

        for m in _DEADLINE_RE.finditer(text):
            _, dur_unit = _DURATION_UNITS[m.group(3)]
            value = parse_number_phrase(m.group(2))
            if value is None:
                continue
            records.append({"unit_id": u["unit_id"], "kind": "أجل",
                            "value": value, "unit": dur_unit,
                            "raw": m.group(0)})

        for m in _DEADLINE_DUAL_RE.finditer(text):
            _, dur_unit = _DURATION_UNITS[m.group(1)]
            records.append({"unit_id": u["unit_id"], "kind": "أجل",
                            "value": 2, "unit": dur_unit,
                            "raw": m.group(0)})
    return records


def validate_numbers(law_units: list[dict]) -> dict:
    """The verification contract: raw spans verbatim, values sane, countable."""
    by_id = {u["unit_id"]: u for u in law_units}
    records = extract_legal_numbers(law_units)
    violations: list[str] = []
    for r in records:
        src = by_id.get(r["unit_id"])
        if src is None or r["raw"] not in src["text"]:
            violations.append(f"raw not verbatim in {r['unit_id']}: {r['raw']!r}")
        if r["value"] is None or r["value"] <= 0:
            violations.append(f"non-positive value: {r!r}")
        if r["kind"] == "نسبة" and not (0 < r["value"] <= 100):
            violations.append(f"percentage out of range: {r!r}")
    by_kind: dict[str, int] = {}
    for r in records:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    return {"records": len(records), "by_kind": by_kind,
            "violations": violations}


# ---------------------------------------------------------------------------
# The Circulaire corrections table (option ج, 2026-09-28) — declared data.
# Every official value is anchored VERBATIM in the adopted Circulaire codex;
# the raw corrupted form is historical documentation (audit §4).
# ---------------------------------------------------------------------------

CIRCULAIRE_CORRECTIONS: list[dict] = [
    {"section": "الترويسة", "topic": "تاريخ المنشور",
     "raw_form": "اكتوبر9112", "official_evidence": "14 اكتوبر 2019",
     "note": "القيمة الرسمية من الرائد الرسمي (تدقيق §4-أ)"},
    {"section": "الترويسة", "topic": "عدد المنشور وسنته",
     "raw_form": "عدد 80 لسنة2019", "official_evidence": "عدد 80 لسنة 2019",
     "note": "الهوية المعتمدة تبقى 80/2019 في المدونة؛ الرسمي «8» موثق للمرجع فقط (إغلاق 1.1)"},
    {"section": "الإحالات التشريعية", "topic": "قانون الإيجار المالي",
     "raw_form": "القانون عدد 98 لسنة4881 المؤرخ في62 جويلية 4881",
     "official_evidence": "القانون عدد 89 لسنة 1994 المؤرخ في 26 جويلية 1994",
     "note": None},
    {"section": "الإحالات التشريعية", "topic": "القانون 48-2016 (البنوك)",
     "raw_form": "القانون عدد 19 لسنة6142 المؤرخ في 44 جويلية 6142",
     "official_evidence": "القانون عدد 48 لسنة 2016 المؤرخ في 11 جويلية 2016",
     "note": "ورد فاسدًا في موضعين"},
    {"section": "الإحالات التشريعية", "topic": "أساس الإحالة على القانون",
     "raw_form": "خاصة الفصل 44 منه وما بعده",
     "official_evidence": "خاصة الفصل 11 منه وما بعده",
     "note": "أساس حافة الإرساء (البند 3)"},
    {"section": "الإحالات التشريعية", "topic": "رأي لجنة مراقبة المطابقة",
     "raw_form": "راي لجنة مراقبة المطابقة عدد 9 المؤرخ في6 اكتوبر 2019",
     "official_evidence": "راي لجنة مراقبة المطابقة عدد 8 المؤرخ في 2 اكتوبر 2019",
     "note": None},
    {"section": "الإحالات التشريعية", "topic": "الإحالة على القانون 35-2016",
     "raw_form": "الفصل 16 من القانون عدد35 لسنة6142",
     "official_evidence": "الفصل 42\nمن القانون عدد 35 لسنة 2016",
     "note": "الدليل يعبر سطرًا في المدونة"},
    {"section": "الإحالات التشريعية", "topic": "مقتضيات الترخيص",
     "raw_form": "الفصل 51 من القانون عدد19 لسنة 6142",
     "official_evidence": "الفصل 54 من القانون عدد 48 لسنة 2016",
     "note": None},
    {"section": "ترقيم الفصول 14–20", "topic": "الإفصاح قبل كل عملية",
     "raw_form": "الفصل 11: يتعين على البنك الافصاح", "official_evidence": "الفصل 14:",
     "note": "الترقيم 14–20 فاسد كله في الاستخراج الخام؛ صُحح في المدونة المعتمدة"},
    {"section": "ترقيم الفصول 14–20", "topic": "الإعلام نصف السنوي",
     "raw_form": "الفصل 11 : على البنك ان يعلم كل ستة اشهر", "official_evidence": "الفصل 15:",
     "note": None},
    {"section": "ترقيم الفصول 14–20", "topic": "الضمانات",
     "raw_form": "الفصل 11: يمكن ان تقترن عمليات التمويل", "official_evidence": "الفصل 16:",
     "note": None},
    {"section": "ترقيم الفصول 14–20", "topic": "هامش الجدية",
     "raw_form": "الفصل 11 : يمكن للبنك او المؤسسة المالية", "official_evidence": "الفصل 17:",
     "note": None},
    {"section": "ترقيم الفصول 14–20", "topic": "إعلام البنك المركزي بالمنتجات",
     "raw_form": "الفصل81 : على البنوك والمؤسسات المالية اعلام البنك المركزي", "official_evidence": "الفصل 18:",
     "note": None},
    {"section": "ترقيم الفصول 14–20", "topic": "المعايير الدولية وAAOIFI",
     "raw_form": "الفصل 11: تلتزم ا لبنوك", "official_evidence": "الفصل 19:",
     "note": None},
    {"section": "ترقيم الفصول 14–20", "topic": "المراقبة والتقرير السنوي",
     "raw_form": "الفصل 02 : يراقب البنك المركزي التونس ي", "official_evidence": "الفصل 20:",
     "note": None},
]


def validate_corrections() -> list[str]:
    """Every official value must be anchored verbatim in the adopted codex."""
    codex = restructure._adopted_codex_text("Circulaire_BCT_2019-08.pdf")
    if codex is None:
        return ["no adopted Circulaire codex"]
    violations = []
    for e in CIRCULAIRE_CORRECTIONS:
        if e["official_evidence"] not in codex:
            violations.append(
                f"official evidence not verbatim in the adopted codex: "
                f"{e['topic']}: {e['official_evidence']!r}")
    return violations


def manual_sample(records: list[dict], n: int = 8) -> list[dict]:
    """The plan's «عينة يدوية معروضة» — a deterministic spread sample."""
    if not records:
        return []
    step = max(1, len(records) // n)
    return records[::step][:n]


def main() -> int:
    parser = argparse.ArgumentParser(prog="numbers",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--emit", action="store_true",
                        help="write the numbers data file (declared data)")
    parser.add_argument("--out", default=None,
                        help="output path (default: legal_numbers_loi_2016_48.json)")
    args = parser.parse_args()

    codex = restructure._adopted_codex_text(LAW_DOC)
    if codex is None:
        raise SystemExit(f"no adopted codex for {LAW_DOC}")
    law_units = units.extract_law_units(codex)
    report = validate_numbers(law_units)
    corr_violations = validate_corrections()
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print("corrections:", len(CIRCULAIRE_CORRECTIONS),
          "| correction violations:", len(corr_violations))
    print("--- manual sample ---")
    for r in manual_sample(extract_legal_numbers(law_units)):
        print(f"  {r['unit_id']} [{r['kind']}] {r['value']} {r['unit']} "
              f"← {r['raw']!r}")

    if args.emit:
        out = Path(args.out) if args.out else HERE / "legal_numbers_loi_2016_48.json"
        out.write_text(json.dumps({
            "_comment": "Structured legal numbers — Phase 4 item 4. Deterministic "
                        "extraction from the item-1 units via the governed "
                        "NUM_WORDS vocabulary; every raw span is verbatim in its "
                        "unit. Regenerate: python numbers.py --emit.",
            "law_numbers": extract_legal_numbers(law_units),
            "circulaire_corrections": CIRCULAIRE_CORRECTIONS,
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"[numbers] wrote {out}")

    return 0 if (not report["violations"] and not corr_violations) else 1


if __name__ == "__main__":
    sys.exit(main())
