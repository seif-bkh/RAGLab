#!/usr/bin/env python3
"""Governed institutional lexicon — deterministic query expansion before search.

Phase 3 intervention 2 (RAGLAB_ROADMAP.md): a MANAGED table (colloquial->formal,
abbreviations, cross-lingual equivalents) applied as a deterministic,
NON-GENERATIVE query expansion BEFORE embedding and BM25, so both retrievers see
the governed equivalents. This is deliberately separate from the retired
translation path: no model calls, no language detection, no variant strategies —
just the table below.

Governance:
- The seed table is REVIEW DATA: the owner vetoes or approves entries one by one
  (plan item 6, raglab/audits/PHASE3_INTERVENTIONS.md).
- Every 'to' form is verified present in the ADOPTED corrected codex
  (2026-10-01). Terms absent from the codex (e.g. التورق، الصكوك) are
  deliberately excluded — the lexicon maps INTO the corpus vocabulary, never
  beyond it.
- An empty table (or LEXICON_ENABLED off) reproduces the current behaviour
  exactly: expand_query returns [text] and retrieval is untouched.
- Entries match EXACT surface forms (whole-word, case-insensitive). Prefixed
  Arabic forms need their own entries (التسليف is a separate row from تسليف);
  growing that coverage is a governed table decision, not a code behaviour.

Usage (wired in retrieval.retrieve behind cfg.LEXICON_ENABLED, default OFF):
    expand_query("What is murabaha?")
    -> ["What is murabaha?", "What is المرابحة?"]
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# The governed seed table (alphabetical by 'from'; one row = one vetoable entry)
# ---------------------------------------------------------------------------

SEED_ENTRIES: list[dict] = [
    # abbreviations -----------------------------------------------------------
    {"from": "BCT", "to": "البنك المركزي التونسي", "kind": "abbreviation",
     "note": "التوقيع الرسمي للبنك المركزي التونسي"},
    # cross-lingual equivalents (en -> ar) -------------------------------------
    {"from": "murabaha", "to": "المرابحة", "kind": "equivalent-en",
     "note": "صيغة التمويل بصيغة المرابحة"},
    {"from": "mudaraba", "to": "المضاربة", "kind": "equivalent-en",
     "note": ""},
    {"from": "musharaka", "to": "المشاركة", "kind": "equivalent-en",
     "note": ""},
    {"from": "ijara", "to": "الإجارة", "kind": "equivalent-en",
     "note": ""},
    {"from": "salam", "to": "السلم", "kind": "equivalent-en",
     "note": "بيع السلم (الشراء الآجل بثمن عاجل)"},
    {"from": "islamic banking", "to": "الصيرفة الإسلامية", "kind": "equivalent-en",
     "note": ""},
    {"from": "riba", "to": "الربا", "kind": "equivalent-en",
     "note": ""},
    {"from": "deposits", "to": "الودائع", "kind": "equivalent-en",
     "note": ""},
    # cross-lingual equivalents (fr -> ar) -------------------------------------
    {"from": "mourabaha", "to": "المرابحة", "kind": "equivalent-fr",
     "note": ""},
    {"from": "moudaraba", "to": "المضاربة", "kind": "equivalent-fr",
     "note": ""},
    {"from": "moucharaka", "to": "المشاركة", "kind": "equivalent-fr",
     "note": ""},
    {"from": "banque centrale", "to": "البنك المركزي التونسي", "kind": "equivalent-fr",
     "note": "المقابلة المؤسسية التونسية"},
    {"from": "dépôts", "to": "الودائع", "kind": "equivalent-fr",
     "note": ""},
    # colloquial -> formal (ar -> ar) ------------------------------------------
    {"from": "تسليف", "to": "القرض", "kind": "colloquial",
     "note": "الاستعمال التونسي الدارج للقرض"},
    {"from": "التسليف", "to": "القرض", "kind": "colloquial",
     "note": "صيغة التعريف للمدخل السابق"},
]

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

# Arabic LETTERS only (hamza..yeh). The full \u0600-\u06FF block also holds
# Arabic punctuation (؟ ، ؛) and digits, which must count as word borders.
_AR = r"\u0621-\u064A"
_LATIN = r"A-Za-zÀ-ÿ"


def _pattern(form: str) -> re.Pattern:
    """Whole-word, case-insensitive pattern.

    Arabic word boundaries cannot rely on \\b (unreliable with Arabic script),
    so boundaries are explicit: the form must not touch same-script letters on
    either side (Arabic letters for Arabic forms, Latin letters for Latin
    forms). Digits, punctuation, spaces and string edges all count as borders.
    """
    edge = _AR if re.search(rf"[{_AR}]", form) else _LATIN
    return re.compile(rf"(?<![{edge}]){re.escape(form)}(?![{edge}])",
                      re.IGNORECASE)


def expansion_pairs(text: str, entries: list[dict] | None = None) -> list[tuple]:
    """Governed expansions of `text` as (label, form) pairs, excluding the
    original: one pair per matched entry (the matched span replaced by the
    governed equivalent) plus — when two or more entries matched — a combined
    pair with all replacements. Fixed order (table order); duplicates dropped
    preserving first occurrence. No match -> [].
    """
    entries = SEED_ENTRIES if entries is None else entries
    matched = [e for e in entries if _pattern(e["from"]).search(text)]
    pairs: list[tuple] = []
    seen: set = set()
    for entry in matched:
        variant = _pattern(entry["from"]).sub(entry["to"], text, count=1)
        if variant not in seen:
            pairs.append((f"lexicon:{entry['from']}", variant))
            seen.add(variant)
    if len(matched) >= 2:
        combined = text
        for entry in matched:
            combined = _pattern(entry["from"]).sub(entry["to"], combined, count=1)
        if combined not in seen:
            pairs.append(("lexicon:combined", combined))
    return pairs


def expand_query(text: str, entries: list[dict] | None = None) -> list[str]:
    """Deterministic expansion: the original text first, then every governed
    form from expansion_pairs. No match -> [text] (exact current behaviour).
    """
    return [text] + [form for _, form in expansion_pairs(text, entries)]


def matched_entries(text: str, entries: list[dict] | None = None) -> list[dict]:
    """The governed entries a query matched (transparency/diagnostics)."""
    entries = SEED_ENTRIES if entries is None else entries
    return [e for e in entries if _pattern(e["from"]).search(text)]
