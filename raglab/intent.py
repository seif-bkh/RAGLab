#!/usr/bin/env python3
"""Request intent — Phase 5, item 1 (raglab/audits/PHASE5_UNDERSTANDING.md).

Deterministic-FIRST intent classification before retrieval:
  - request type (تعريفي/رقمي/إجرائي/عقوبي/مقارن/استثناء) — or the DECLARED
    default path when unclassifiable (wide hybrid retrieval, no assumptions);
  - عام/شخصي (general/personal framing);
  - scope: the document a query explicitly names (governance-axes backed) —
    never assumed;
  - sub-questions for compound requests only;
  - influential-ambiguity detection: concept terms whose evidence anchors
    span MULTIPLE documents (MASTER_INDEX §6's shared-concepts table) —
    ambiguity that changes WHICH evidence is required, not mere wording.

Governance:
- INTENT_RULES is DECLARED REVIEW DATA (same veto regime as TYPE_RULES):
  ordered, first match wins, every decision explained by the rules that
  fired (fired_rules).
- No guessing: an unclassifiable request is reported as such and routed to
  the declared default path; the unclassified ratio is MEASURED on both
  adopted question sets (unclassified_ratio).
- The layer is INERT until a declared activation: nothing in the deployed
  retrieval path calls it (item 3 wires the expansion behind an env gate).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import governance  # noqa: E402

# ---------------------------------------------------------------------------
# Declared rule tables (REVIEW DATA — veto rule by rule; order governs)
# ---------------------------------------------------------------------------

INTENT_TYPES = ("عقوبي", "رقمي", "مقارن", "استثناء", "تعريفي", "إجرائي")

INTENT_RULES: list[tuple[str, list[str]]] = [
    ("عقوبي", [r"عقاب", r"عقوب", r"سجن", r"خطية", r"غرامة", r"يعاقب", r"جزاء"]),
    ("رقمي", [r"كم\s", r"كمية", r"نسبة", r"مبلغ", r"أقصى", r"اقصى", r"راس المال",
              r"رأس المال", r"دينار", r"مدة", r"آجل", r"اجل ", r"خلال ",
              r"في أي سنة", r"How many", r"how many", r"Combien", r"combien",
              r"maximum", r"période"]),
    ("مقارن", [r"الفرق", r"مقارنة", r"أيهما", r"ايهما", r"بالمقارنة", r"الفارق",
               r"قارن"]),
    ("استثناء", [r"استثناء", r"باستثناء", r"لا يشمل", r"عدا", r"ما عدا"]),
    ("تعريفي", [r"ما هو", r"ما هي", r"عرّف", r"عرف ", r"تعريف", r"ماذا يعني",
                r"ما معنى", r"المقصود", r"ماذا يهدف", r"إلى ماذا", r"^ما\s+\S+",
                r"متى", r"أين ", r"تاريخ", r"What is", r"what is", r"What does",
                r"what does", r"Which ", r"which ", r"Quelle\b", r"quelle\b",
                r"Quel\b", r"quel\b", r"Quels\b", r"quels\b", r"Qu'est",
                r"Quand\b", r"quand\b", r"When ", r"when ", r"Where ",
                r"where ", r"Who ", r"who "]),
    ("إجرائي", [r"كيف", r"إجراءات", r"اجراءات", r"شروط", r"الشرط", r"خطوات",
                r"من يملك", r"من يعين", r"من يتولى", r"من يتحمل", r"على عاتق من",
                r"على من ", r"ماذا يحدث", r"هل يمكن", r"هل يجب", r"هل يحق",
                r"هل يجوز", r"يجوز", r"جائز", r"تنطبق", r"أحكام", r"أي أساس",
                r"ما الإجراء", r"ما الاجراء", r"Who must", r"who must",
                r"Who gives", r"who gives", r"Qui doit", r"qui doit",
                r"Que doivent", r"que doivent", r"charge-t-elle",
                r"must approve",
                # French procedural markers — measured 2026-10-05: without
                # these, «Quelles sont les étapes et modalités…» had no
                # procedural rule to fire and the definitional rule took it,
                # so the plan demanded a definition the guide's 5.3 procedure
                # section can never supply.
                r"\bétapes?\b", r"\bmodalités?\b", r"\bprocédures?\b",
                r"\bcomment\b", r"\bComment\b"]),
]

# Personal framing (advisory to a person — OOS-prone; the default is عام)
PERSONAL_PATTERNS = [r"هل يمكنني", r"هل لي", r"هل لي أن", r"أنا ", r"انا ",
                     r"أريد", r"اريد", r"حسابي", r"أموالي", r"اموالي", r"نفسي"]

# Compound detection: a second requirement marker inside one request
_COMPOUND_MARKERS = [r"وما\s", r"وكم\s", r"وكيف\s", r"ومن\s", r"وماذا\s",
                     r"وعلى ماذا", r"وعلى أي", r"ومتى", r"وأين", r"؟.*؟",
                     r" ثم "]

# Shared concepts whose evidence anchors span multiple documents
# (MASTER_INDEX §6 — the cross-document map; ambiguity here changes WHICH
# evidence a correct answer needs, so it is INFLUENTIAL).
AMBIGUOUS_CONCEPTS: dict[str, tuple[str, ...]] = {
    "المرابحة": ("Loi_2016-48.pdf", "Circulaire_BCT_2019-08.pdf",
                 "Guide_Interne_Operations_Bancaires_Islamiques.docx",
                 "Madkhal_Sayrafa_Islamiya.docx"),
    "الاجارة": ("Loi_2016-48.pdf", "Circulaire_BCT_2019-08.pdf",
                "Guide_Interne_Operations_Bancaires_Islamiques.docx"),
    "الاستصناع": ("Loi_2016-48.pdf", "Circulaire_BCT_2019-08.pdf",
                  "Guide_Interne_Operations_Bancaires_Islamiques.docx"),
    "السلم": ("Loi_2016-48.pdf", "Circulaire_BCT_2019-08.pdf",
              "Guide_Interne_Operations_Bancaires_Islamiques.docx"),
    "المضاربة": ("Loi_2016-48.pdf", "Circulaire_BCT_2019-08.pdf",
                 "Guide_Interne_Operations_Bancaires_Islamiques.docx",
                 "Madkhal_Sayrafa_Islamiya.docx"),
    "المشاركة": ("Loi_2016-48.pdf", "Circulaire_BCT_2019-08.pdf",
                 "Guide_Interne_Operations_Bancaires_Islamiques.docx",
                 "Madkhal_Sayrafa_Islamiya.docx"),
    "الودائع الاستثمارية": ("Loi_2016-48.pdf", "Circulaire_BCT_2019-08.pdf",
                            "Guide_Interne_Operations_Bancaires_Islamiques.docx",
                            "Madkhal_Sayrafa_Islamiya.docx"),
}

# Explicit document naming (scope is only ever what the query itself says)
SCOPE_PATTERNS: list[tuple[str, str]] = [
    (r"قانون\s*(?:عدد\s*)?48|قانون البنوك", "Loi_2016-48.pdf"),
    (r"منشور|circulaire", "Circulaire_BCT_2019-08.pdf"),
    (r"الدليل|دليل", "Guide_Interne_Operations_Bancaires_Islamiques.docx"),
    (r"المدخل|مدخل الى", "Madkhal_Sayrafa_Islamiya.docx"),
]


def classify(question: str) -> dict:
    """Deterministic intent for one request — explained, never guessed."""
    q = question.strip()
    fired: list[tuple[str, str]] = []

    intent_type = "غير مصنف"
    for type_name, patterns in INTENT_RULES:
        for pat in patterns:
            m = re.search(pat, q)
            if m:
                intent_type = type_name
                fired.append((f"intent:{type_name}", m.group(0)))
                break
        if intent_type != "غير مصنف":
            break

    personal = False
    for pat in PERSONAL_PATTERNS:
        m = re.search(pat, q)
        if m:
            personal = True
            fired.append(("personal", m.group(0)))
            break

    scope = None
    for pat, doc in SCOPE_PATTERNS:
        m = re.search(pat, q)
        if m:
            scope = doc
            fired.append((f"scope:{doc}", m.group(0)))
            break

    sub_questions = None
    for pat in _COMPOUND_MARKERS:
        if re.search(pat, q):
            sub_questions = _split_compound(q)
            fired.append(("compound", pat))
            break

    ambiguous = [c for c in AMBIGUOUS_CONCEPTS if c in q]
    if ambiguous:
        fired.append(("ambiguous", "+".join(ambiguous)))
        # an explicit scope RESOLVES the influential ambiguity
        if scope:
            fired.append(("scope-resolves-ambiguity", scope))

    return {
        "question": q,
        "intent_type": intent_type,
        "default_path": intent_type == "غير مصنف",
        "personal": personal,
        "scope": scope,
        "sub_questions": sub_questions,
        "ambiguous_concepts": ambiguous if not scope else [],
        "fired_rules": fired,
    }


def _split_compound(question: str) -> list[str]:
    """Split a compound request at its second-requirement markers."""
    parts = re.split(r"وما\s|وكم\s|وكيف\s|ومن\s|وماذا\s|وعلى ماذا|وعلى أي|ومتى|وأين| ثم |؟", question)
    return [p.strip(" ،,.؟?") for p in parts if p.strip(" ،,.؟?")]


def unclassified_ratio(cases: list[dict]) -> dict:
    """Measured on an adopted question set (the plan's per-phase metric)."""
    rows = [classify(c["question"]) for c in cases]
    n = len(rows)
    unclassified = sum(1 for r in rows if r["default_path"])
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["intent_type"]] = by_type.get(r["intent_type"], 0) + 1
    return {"n": n, "unclassified": unclassified,
            "ratio": (unclassified / n) if n else None,
            "by_type": by_type}


def main() -> int:
    parser = argparse.ArgumentParser(prog="intent",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--sets", nargs="*", default=["questions_50.json",
                                                      "questions_targets.json"],
                        help="adopted question sets to measure")
    args = parser.parse_args()
    report = {}
    for name in args.sets:
        cases = json.loads((HERE / name).read_text(encoding="utf-8"))["cases"]
        report[name] = unclassified_ratio(cases)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
