#!/usr/bin/env python3
"""Question decomposition & reformulation — Phase 5, item 6
(raglab/audits/PHASE5_UNDERSTANDING.md; added by the owner's directive
2026-10-01: «السؤال مفككًا إلى أسئلة صغرى تحدد مطالبه بتحديد النية»).

The intermediate understanding layer BETWEEN the raw question and the
evidence plan:

    سؤال المستخدم
      → تفكيك (المركب إلى أسئلة فرعية — جدول النية المعتمد)
      → لكل (سؤال/سؤال فرعي): النية + الموضوع المستخرج
      → كل مطلب من خطة الدليل (البند 2) يُعاد صياغته سؤالًا صغيرًا مستقلًا
      → كل مطلب مرتبط بالحقول التي يستطيع المساعد الإجابة منها
        (أسطح المرحلة 4: أنواع الوحدات، المسار المهيكل للأرقام، حواف
        العلاقات) بأعداد حية من المعطيات المعلنة
      + أسئلة توضيح مولدة للغموض المؤثر (استشارية لا مانعة)

Governance:
- MICRO_TEMPLATES, ANSWERABLE_SURFACES, and the subject-extraction patterns
  are DECLARED REVIEW DATA (veto line by line).
- INTENT_RULES (adopted) and DERIVATION_RULES are CONSUMED as-is — this
  layer never modifies them; it composes them into the decomposition.
- Read-only understanding layer: nothing in the deployed path imports it;
  any wiring (e.g. per-micro retrieval) is a separate owner gate.

The measurable contract: on both adopted sets every question decomposes
deterministically into >= 1 micro-question; every micro-question carries
exactly one requirement of the plan (bijection: every requirement is asked
by >= 1 micro); every requirement's answerable_from is non-empty; every
influentially-ambiguous question carries >= 1 clarification.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evidence_plan  # noqa: E402
import intent         # noqa: E402

# ---------------------------------------------------------------------------
# Declared subject extraction (REVIEW DATA)
# ---------------------------------------------------------------------------

# leading interrogative framing — stripped to expose the SUBJECT
_AR_INTERROGATIVE = re.compile(
    r"^(?:ماذا يعني|ما معنى|ما المقصود بـ|المقصود بـ|ما هي|ما هو|ماذا يهدف|"
    r"إلى ماذا|ما الإجراء|ما الاجراء|ما الفرق بين|الفرق بين|قارن بين|"
    r"قارن|عرّف|عرف|ما|كيف|هل|من|كم|متى|أين|لماذا|على من|في ماذا|ماذا)\s+")
# scope clauses — naming a document/source, not part of the subject
_AR_SCOPE = re.compile(
    r"\s*(?:وفق|حسب|طبقا ل|طبقاً ل|على معنى|خلافا (?:لأحكام )?|وفقا ل|"
    r"بمقتضى|بواسطة|من طرف|في إطار)\s*"
    r"(?:منشور|المنشور|قانون|القانون|الدليل|دليل|المادة|المدخل|معايير|"
    r"نص|أحكام|اتفاقية).*$")
_LAT_INTERROGATIVE = re.compile(
    r"^(?:what is|what does|what do|which|who|whom|how many|how much|how|"
    r"when|where|according to|under|in what|quelle est|quel est|quels|"
    r"quelles|quel|quelle|combien|que doivent|qu'est|quand|en quelle anne)"
    r"[\s:]+", re.I)
_LAT_SCOPE = re.compile(
    r"\s*(?:according to|under|dans le|in the)\s+.*$", re.I)
_TRAIL_QM = "؟?!."

# مقارن framing: «... من حيث X وY» — aspects are compared, the head is the subject
_MIN_HAWA = re.compile(r"\s*من حيث\s+(.*)$")


def extract_subject(text: str) -> str:
    """The question's SUBJECT (declared strip rules; never empty — the
    fallback is the text minus its question mark)."""
    s = text.strip().rstrip(_TRAIL_QM).strip()
    s = _AR_INTERROGATIVE.sub("", s, count=1)
    s = _LAT_INTERROGATIVE.sub("", s, count=1)
    s = _AR_SCOPE.sub("", s).strip()
    s = _LAT_SCOPE.sub("", s).strip()
    s = s.strip(" ،,.")
    return s or text.strip().rstrip(_TRAIL_QM).strip()


def _aspects(text: str) -> list[str]:
    """Comparison aspects after «من حيث» (split at و) — empty when absent."""
    m = _MIN_HAWA.search(text)
    if not m:
        return []
    return [a.strip(" ،,.") for a in re.split(r"\s+و", m.group(1)) if a.strip(" ،,.")]


# ---------------------------------------------------------------------------
# Declared micro-question templates (REVIEW DATA — one per requirement kind)
# ---------------------------------------------------------------------------

MICRO_TEMPLATES: dict[str, str] = {
    "definition_or_purpose_unit": "ما نص التعريف أو الغرض لـ«{subject}»؟",
    "numeric_evidence": "ما القيمة أو الرقم المنصوص لـ«{subject}»؟",
    "penalty_unit": "ما نص العقوبة المتعلقة بـ«{subject}»؟",
    "with_governing_rule": "ما النص العام المرتبط بـ«{subject}» (القاعدة التي تحيل إليها)؟",
    "exception_with_rule": "ما نص الاستثناء وقاعدته لـ«{subject}»؟",
    "procedural_evidence": "ما الإجراءات والأحكام المتعلقة بـ«{subject}»؟",
    "both_sides_evidence": "ما دليل كل طرف من طرفي المقارنة حول «{subject}»؟",
    "wide_evidence": "ما الدليل اللغوي المتعلق بـ«{subject}»؟",
}

# ---------------------------------------------------------------------------
# Declared answerable surfaces (REVIEW DATA) — requirement → the Phase-4
# knowledge surfaces the assistant can answer from, with LIVE counts.
# ---------------------------------------------------------------------------

_SURFACES_CACHE: dict = {}


def _catalog_counts() -> dict:
    """Live counts from the Phase-4 declared catalogs (cached)."""
    if "counts" not in _SURFACES_CACHE:
        import restructure
        import units
        codex = restructure._adopted_codex_text("Loi_2016-48.pdf")
        if codex is None:
            _SURFACES_CACHE["counts"] = {}
        else:
            law_units = units.extract_law_units(codex)
            by_type: dict[str, int] = {}
            for u in law_units:
                by_type[u["type"]] = by_type.get(u["type"], 0) + 1
            import legal_numbers
            import relations
            _SURFACES_CACHE["counts"] = {
                "units_total": len(law_units),
                "by_type": by_type,
                "numeric_records": len(legal_numbers.extract_legal_numbers(law_units)),
                "internal_edges": len(relations.extract_internal_references(law_units)),
            }
    return _SURFACES_CACHE["counts"]


def _n(kind: str) -> int:
    return _catalog_counts().get("by_type", {}).get(kind, 0)


ANSWERABLE_SURFACES: dict[str, list[str]] = {
    "definition_or_purpose_unit": [
        "وحدات القانون نوع definition ({definition})",
        "نصوص «تعتبر/يعد/على معنى» في المنشور والدليل",
    ],
    "numeric_evidence": [
        "المسار المهيكل للأرقام ({numeric_records} سجلًا: نسبة/أجل/حد مالي/عقوبة مالية)",
        "نقطة النهاية GET /numbers (تصفية unit_id/kind)",
    ],
    "penalty_unit": [
        "وحدات القانون نوع penalty ({penalty})",
    ],
    "with_governing_rule": [
        "حواف الإحالات الداخلية ({internal_edges} حافة)",
    ],
    "exception_with_rule": [
        "نصوص الاستثناء في الوثائق + وحدة القاعدة العامة",
    ],
    "procedural_evidence": [
        "وحدات القانون نوع procedure ({procedure})",
        "وحدات القانون نوع delegation ({delegation})",
        "أحكام عامة في المنشور والدليل (نصوص يجب/يتولى/شرط)",
    ],
    "both_sides_evidence": [
        "دليل كل طرف من طرفي المقارنة (مصدران مرسَّان على الأقل)",
    ],
    "wide_evidence": [
        "أي دليل لغوي في المدونة ({units_total} وحدة قانون + 141 مقطعًا للوثائق الثلاث الأخرى)",
    ],
}


def answerable_from(req_kind: str) -> list[str]:
    """The declared surfaces, with live counts filled in."""
    counts = dict(_catalog_counts())
    counts.update(counts.get("by_type", {}))
    return [tpl.format(**counts) for tpl in ANSWERABLE_SURFACES.get(req_kind, [])]


# ---------------------------------------------------------------------------
# Declared clarification templates (REVIEW DATA) — influential ambiguity
# ---------------------------------------------------------------------------

def _clarifications(classified: dict) -> list[str]:
    out = []
    for concept in classified.get("ambiguous_concepts") or []:
        docs = ", ".join(intent.AMBIGUOUS_CONCEPTS.get(concept, ()))
        out.append(f"أي وثيقة أو سياق تقصد لـ«{concept}»؟ المفهوم مشترك بين: {docs}")
    return out


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------

def decompose(question: str) -> dict:
    """The intermediate layer: question → micro-questions (each with its
    intent, its ONE requirement, and the fields the assistant can answer
    from), plus advisory clarifications for influential ambiguity."""
    plan = evidence_plan.derive_plan(question)
    classified = intent.classify(question)
    clarifications = _clarifications(classified)

    micros: list[dict] = []

    def add(source_text: str, source_kind: str, sub_index: int | None,
            sub_intent: str, req: dict, subject: str):
        micros.append({
            "id": f"m{len(micros) + 1}",
            "text": MICRO_TEMPLATES[req["req"]].format(subject=subject),
            "source": source_text,
            "source_kind": source_kind,          # main | sub-question i
            "sub_question_index": sub_index,
            "intent_type": sub_intent,
            "requirement": req["req"],
            "requirement_desc": req["desc"],
            "answerable_from": answerable_from(req["req"]),
        })

    # the main question's requirements
    subject = extract_subject(question)
    aspects = _aspects(question) if plan["intent_type"] == "مقارن" else []
    for req in plan["requirements"]:
        if req["req"] == "both_sides_evidence" and aspects:
            # one micro per compared aspect — all anchored to the same
            # requirement (each aspect needs its own both-sides evidence)
            head = _MIN_HAWA.sub("", subject).strip(" ،,.") or subject
            for a in aspects:
                micros.append({
                    "id": f"m{len(micros) + 1}",
                    "text": f"ما أحكام «{head}» من حيث «{a}»؟",
                    "source": question,
                    "source_kind": "main",
                    "sub_question_index": None,
                    "intent_type": plan["intent_type"],
                    "requirement": req["req"],
                    "requirement_desc": req["desc"],
                    "answerable_from": answerable_from(req["req"]),
                    "aspect": a,
                })
        else:
            add(question, "main", None, plan["intent_type"], req, subject)

    # each sub-question of a compound request: its own intent + requirements
    for i, sp in enumerate(plan.get("sub_plans") or [], start=1):
        sub_text = sp["question"]        # derive_plan built it from the sub-question
        sub_subject = extract_subject(sub_text)
        for req in sp["requirements"]:
            add(sub_text, f"sub-question {i}", i, sp["intent_type"], req,
                sub_subject)

    return {
        "question": question,
        "intent_type": plan["intent_type"],
        "ambiguous_concepts": classified.get("ambiguous_concepts") or [],
        "clarifications": clarifications,
        "subject": subject,
        "micro_questions": micros,
    }


# ---------------------------------------------------------------------------
# Measurement (the contract above, on the adopted sets)
# ---------------------------------------------------------------------------

def measure(cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        d = decompose(case["question"])
        asked = [m["requirement"] for m in d["micro_questions"]]
        # bijection: every requirement of the plan is asked by >= 1 micro
        plan_reqs = [r["req"] for r in evidence_plan.derive_plan(
            case["question"])["requirements"]]
        for sp in evidence_plan.derive_plan(case["question"]).get("sub_plans") or []:
            plan_reqs += [r["req"] for r in sp["requirements"]]
        unasked = [r for r in dict.fromkeys(plan_reqs) if r not in asked]
        empty_surfaces = sum(1 for m in d["micro_questions"]
                             if not m["answerable_from"])
        rows.append({
            "id": case["id"], "intent": d["intent_type"],
            "n_micros": len(d["micro_questions"]), "unasked_reqs": unasked,
            "empty_surfaces": empty_surfaces,
            "clarifications": len(d["clarifications"]),
            "ambiguous": bool(d["ambiguous_concepts"]),
        })
    n = len(rows)
    return {
        "n": n,
        "decomposed": sum(1 for r in rows if r["n_micros"] >= 1),
        "micros_total": sum(r["n_micros"] for r in rows),
        "bijection_violations": sum(1 for r in rows if r["unasked_reqs"]),
        "empty_surface_micros": sum(r["empty_surfaces"] for r in rows),
        "ambiguous_without_clarification": sum(
            1 for r in rows if r["ambiguous"] and not r["clarifications"]),
        "per_case": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="decompose",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--sets", nargs="*", default=["questions_50.json",
                                                      "questions_targets.json"])
    parser.add_argument("--question", help="print the full decomposition of one question")
    args = parser.parse_args()

    if args.question:
        print(json.dumps(decompose(args.question), ensure_ascii=False, indent=1))
        return 0

    report = {}
    for name in args.sets:
        cases = json.loads((HERE / name).read_text(encoding="utf-8"))["cases"]
        m = measure(cases)
        report[name] = {k: v for k, v in m.items() if k != "per_case"}
        report[name]["violations"] = [
            {kk: r[kk] for kk in ("id", "intent", "n_micros", "unasked_reqs",
                                  "empty_surfaces", "clarifications")}
            for r in m["per_case"]
            if r["unasked_reqs"] or r["empty_surfaces"]
            or (r["ambiguous"] and not r["clarifications"])]
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
