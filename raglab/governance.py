#!/usr/bin/env python3
"""Governance-axes registry — Phase 4, item 2 (raglab/audits/PHASE4_KNOWLEDGE.md).

The central registry gives every corpus document DECLARED governance axes that
ride on its chunks' metadata at ingest time (store.store_chunks). The deployed
retrieval path already returns chunk metadata with every hit, so the axes
become filterable/inspectable provenance without any retrieval change.

Scope of THIS item (the plan is the contract):
- The law (Loi 2016-48) gets the full axis set.
- The three other corpus documents stay DEFERRED: their minimal corresponding
  axes are a later registry extension, explicitly reviewed — never silently
  invented here.

The axis values are REVIEW DATA (the plan: «القرار عند صاحب الملف عند تعارض»).
Conflict resolution — when two sources disagree about a document's axes — is
the file owner's gate; with a single source now, no conflict path exists yet.

Fingerprint checks (the plan's «فحوص بصمات»), enforced by tests:
1. every registry key is a real corpus document (no phantom entries);
2. corpus documents are exactly registry + deferred (nothing silently
   unregistered);
3. every entry carries the full declared axis set, values non-empty;
4. after ingest, every chunk of a registered document carries all axes.
"""

from __future__ import annotations

# The declared axis schema — keys are the metadata field names on each chunk.
AXES: tuple[str, ...] = (
    "gov_type",        # نوع الوثيقة
    "gov_authority",   # سلطة المرجعية
    "gov_issue",       # الإصدار (الرائد الرسمي)
    "gov_status",      # السريان
    "gov_audience",    # الجمهور المعني
    "gov_scope",       # النطاق
)

# Full registry — the law only, per the item-2 plan.
GOVERNANCE_AXES: dict[str, dict[str, str]] = {
    "Loi_2016-48.pdf": {
        "gov_type": "قانون",
        "gov_authority": "سلطة تشريعية عليا",
        "gov_issue": "الرائد الرسمي للجمهورية التونسية عدد 58 لسنة 2016",
        "gov_status": "نافذ",
        "gov_audience": "العموم",
        "gov_scope": "البنوك والمؤسسات المالية",
    },
}

# Corpus documents deliberately not yet in the registry (minimal axes later,
# as a reviewed registry extension — Phase-4 follow-up, not this item).
DEFERRED_DOCUMENTS: tuple[str, ...] = (
    "Circulaire_BCT_2019-08.pdf",
    "Guide_Interne_Operations_Bancaires_Islamiques.docx",
    "Madkhal_Sayrafa_Islamiya.docx",
)


def axes_for(document: str) -> dict[str, str]:
    """The declared governance axes for a document ({} when unregistered)."""
    entry = GOVERNANCE_AXES.get(document)
    return dict(entry) if entry else {}


def registry_report(corpus_documents) -> dict:
    """Fingerprint-check data: registry vs corpus coverage."""
    corpus = set(corpus_documents)
    registered = set(GOVERNANCE_AXES)
    deferred = set(DEFERRED_DOCUMENTS)
    return {
        "corpus_documents": sorted(corpus),
        "registered": sorted(registered),
        "deferred": sorted(deferred),
        "phantom_entries": sorted(registered - corpus),
        "unregistered_undeclared": sorted(corpus - registered - deferred),
        "empty_axis_values": sorted(
            doc for doc, axes in GOVERNANCE_AXES.items()
            if any(not (axes.get(a) or "").strip() for a in AXES)),
        "schema_violations": sorted(
            doc for doc, axes in GOVERNANCE_AXES.items()
            if set(axes) != set(AXES)),
    }


def registry_ok(corpus_documents) -> bool:
    r = registry_report(corpus_documents)
    return not (r["phantom_entries"] or r["unregistered_undeclared"]
                or r["empty_axis_values"] or r["schema_violations"])
