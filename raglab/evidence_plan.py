#!/usr/bin/env python3
"""Evidence plan — Phase 5, item 2 (raglab/audits/PHASE5_UNDERSTANDING.md).

Derives evidence REQUIREMENTS from the classified intent — WITHOUT assuming
the answer. A requirement says what must be PRESENT in the evidence (a
definition-type unit, a structured number, a penalty unit together with its
governing rule, both sides of a comparison, an exception with its base rule,
procedural text) — never what the answer is.

Governance:
- DERIVATION_RULES is DECLARED REVIEW DATA (veto rule by rule): intent type →
  requirement specs, each with an Arabic description for the owner's review.
- «غير مصنف» derives the wide requirement (any valid evidence, no type
  assumption) — consistent with intent.py's declared default path.
- Compound requests derive one sub-plan per sub-question.
- Read-only layer: nothing in the deployed path imports it.

The measurable contract (plan item 2.7): on both adopted sets every case
derives a plan deterministically; where the evidence lands in the LAW (typed
units exist), each requirement is checked against the actual evidence unit —
the consistency ratio is reported per set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate    # noqa: E402  — normalize_for_match (shared matching rule)
import intent      # noqa: E402
import restructure  # noqa: E402
import units       # noqa: E402

# ---------------------------------------------------------------------------
# Declared derivation table (REVIEW DATA — veto requirement by requirement)
# ---------------------------------------------------------------------------

DERIVATION_RULES: dict[str, list[dict]] = {
    "تعريفي": [
        {"req": "definition_or_purpose_unit",
         "desc": "نص تعريفي للمفهوم المطلوب (وحدة تعريف أو نص: تعتبر/يعتبر/يعد/على معنى)، "
                 "أو نص تأسيسي/غرضي عند سؤال الكيان أو الهدف (صندوق/إطار/لجنة/هيئة/آلية/يهدف)"},
    ],
    "رقمي": [
        {"req": "numeric_evidence",
         "desc": "سجل من المسار المهيكل للأرقام أو وحدة معلمة رقمية تحمل الرقم المطلوب"},
    ],
    "عقوبي": [
        {"req": "penalty_unit", "desc": "وحدة نوعها عقوبة"},
        {"req": "with_governing_rule",
         "desc": "مع النص العام المرتبط (إحالة داخلية من وحدة العقوبة أو نص الفعل المعاقَب عليه)"},
    ],
    "مقارن": [
        {"req": "both_sides_evidence", "desc": "دليل لكل طرف من طرفي المقارنة"},
    ],
    "استثناء": [
        {"req": "exception_with_rule", "desc": "نص الاستثناء مع قاعدته (لا الاستثناء وحده)"},
    ],
    "إجرائي": [
        {"req": "procedural_evidence",
         "desc": "وحدة إجراء أو تفويض أو حكم عام تحمل الفعل/الشرط المطلوب"},
    ],
    "غير مصنف": [
        {"req": "wide_evidence", "desc": "أي دليل لغوي صالح — لا افتراض نوع (المسار الافتراضي المعلن)"},
    ],
}

_DEFINITION_TEXT_RE = re.compile(r"تعتبر|يعتبر|^يعد|على معنى")
# Entity/purpose questions (تعريفي عن كيان أو غرض) legitimately land in
# institutional provisions, not concept definitions.
_ENTITY_OR_PURPOSE_RE = re.compile(
    r"صندوق|إطار|لجنة|هيئة|بنك|آلية|جهاز|منشأة|يهدف|هدف |What.*(?:committee|body|framework)"
    r"|Which .*(?:committee|fund|structure)|Quelle (?:structure|institution)")
# glued prefixes (للفصل/بالفصل) drop the alef — match the bare «فصلN» core
_INTERNAL_REF_RE = re.compile(r"\u0641\u0635\u0644\s*\d+|\u0641\u0635\u0648\u0644\s*\d+")
_EXCEPTION_RE = re.compile(r"استثناء|باستثناء|عدا|ما عدا|لا يشمل")


def derive_plan(question: str) -> dict:
    """Intent → requirement list (compound: one sub-plan per sub-question)."""
    r = intent.classify(question)
    reqs = [dict(spec) for spec in DERIVATION_RULES.get(r["intent_type"], [])]
    plan = {
        "question": question,
        "intent_type": r["intent_type"],
        "ambiguous_concepts": r["ambiguous_concepts"],
        "requirements": reqs,
        "sub_plans": ([derive_plan(sq) for sq in r["sub_questions"]]
                      if r["sub_questions"] else []),
    }
    return plan


# ---------------------------------------------------------------------------
# Requirement satisfaction — checked ONLY where typed units exist (the law)
# ---------------------------------------------------------------------------

def _law_evidence_units(case: dict, law_units: list[dict]) -> list[dict]:
    """The law units whose verbatim text contains the case's expected evidence."""
    subs = (case.get("expected_substrings")
            or ([case["expected_substring"]] if case.get("expected_substring") else []))
    hits = []
    for sub in subs:
        for u in law_units:
            if evaluate.normalize_for_match(sub) in evaluate.normalize_for_match(u["text"]):
                hits.append(u)
                break        # first containing unit per requirement
    return hits


def requirement_satisfied(req: dict, case: dict, evidence_units: list[dict],
                          numeric_unit_ids: set[str]) -> bool | None:
    """True/False where typed law evidence exists; None when the check cannot
    apply (evidence outside the typed catalog — reported separately, never
    guessed)."""
    if not evidence_units:
        return None
    kind = req["req"]
    if kind == "definition_or_purpose_unit":
        definitional = any(u.get("type") == "definition"
                           or _DEFINITION_TEXT_RE.search(u["text"])
                           for u in evidence_units)
        if definitional:
            return True
        question = case.get("question", "")
        if _ENTITY_OR_PURPOSE_RE.search(question):
            return any(u.get("type") in ("general", "procedure", "delegation")
                       for u in evidence_units)
        return False
    if kind == "numeric_evidence":
        return any(u.get("numeric") or u["unit_id"] in numeric_unit_ids
                   for u in evidence_units)
    if kind == "penalty_unit":
        return any(u.get("type") == "penalty" for u in evidence_units)
    if kind == "with_governing_rule":
        # the exception-with-rule analog for penalties: the penalty text must
        # reference its governing provision (internal cross-reference)
        return any(_INTERNAL_REF_RE.search(u["text"]) for u in evidence_units)
    if kind == "both_sides_evidence":
        subs = case.get("expected_substrings") or []
        return len(subs) >= 2 or len(evidence_units) >= 2
    if kind == "exception_with_rule":
        return bool(_EXCEPTION_RE.search(" ".join(u["text"] for u in evidence_units)))
    if kind == "procedural_evidence":
        return any(u.get("type") in ("procedure", "delegation", "general", "definition")
                   for u in evidence_units)
    if kind == "wide_evidence":
        return True
    return None


def measure_coverage(cases: list[dict], law_units: list[dict],
                     numeric_unit_ids: set[str]) -> dict:
    """Per set: every case derives a plan (deterministic); where the evidence
    lands in typed law units, each requirement is checked — consistency ratio.
    """
    rows = []
    for case in cases:
        plan = derive_plan(case["question"])
        ev_units = _law_evidence_units(case, law_units)
        checks = []
        for req in plan["requirements"]:
            checks.append({
                "req": req["req"],
                "status": requirement_satisfied(req, case, ev_units, numeric_unit_ids),
            })
        rows.append({"id": case["id"], "intent_type": plan["intent_type"],
                     "evidence_in_law": bool(ev_units), "checks": checks})
    typed = [r for r in rows if r["evidence_in_law"]]
    consistent = [r for r in typed
                  if all(c["status"] for c in r["checks"])]
    return {
        "n": len(rows),
        "derived_plans": len(rows),                     # determinism: always
        "typed_evidence_cases": len(typed),
        "typed_consistent": len(consistent),
        "consistency_ratio": (len(consistent) / len(typed)) if typed else None,
        "per_case": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="evidence_plan",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--sets", nargs="*", default=["questions_50.json",
                                                      "questions_targets.json"])
    args = parser.parse_args()

    codex = restructure._adopted_codex_text("Loi_2016-48.pdf")
    if codex is None:
        raise SystemExit("no adopted law codex")
    law_units = units.extract_law_units(codex)
    import legal_numbers
    records = legal_numbers.extract_legal_numbers(law_units)
    numeric_unit_ids = {r["unit_id"] for r in records}

    report = {}
    for name in args.sets:
        cases = json.loads((HERE / name).read_text(encoding="utf-8"))["cases"]
        m = measure_coverage(cases, law_units, numeric_unit_ids)
        report[name] = {k: v for k, v in m.items() if k != "per_case"}
        viol = [r for r in m["per_case"] if r["evidence_in_law"]
                and not all(c["status"] for c in r["checks"])]
        report[name]["inconsistent_cases"] = [(r["id"], r["intent_type"],
                                               [c for c in r["checks"]
                                                if c["status"] is False])
                                               for r in viol]
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
