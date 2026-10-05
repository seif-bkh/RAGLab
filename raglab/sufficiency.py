#!/usr/bin/env python3
"""Sufficiency & conflict — Phase 5, item 4
(raglab/audits/PHASE5_UNDERSTANDING.md).

Deterministic checks of whether the RETRIEVED evidence suffices for the
evidence plan (item 2): requirement-by-requirement coverage, conflict
detection under a DECLARED adjustable precedence, escalation when the
decision is impossible, and bounded guided search rounds.

States (explicit, per query):
- «كافٍ»            every plan requirement covered by an anchored hit;
- «غير كافٍ»        ≥1 requirement uncovered — refusal tied to EVIDENCE
                    ABSENCE (the missing requirements are named; never a
                    classification failure);
- «متعارض»          covered, but anchored texts from ≥2 precedence levels
                    disagree (detected via the adopted Circulaire
                    corrections table) — resolved BY the declared
                    precedence, the winning text is recorded;
- «غير محسوم»       cannot decide deterministically (cross-script question
                    with no distinctive shared term, or a conflict the
                    precedence cannot resolve) — escalated, never guessed.

Governance:
- PRECEDENCE, the anchor thresholds, the stopword tables and the untyped
  shape patterns are DECLARED REVIEW DATA (veto line by line).
- The anchor uses the hit TEXT only — never document names or metadata
  (a question naming a document must still find its subject IN the text).
- Read-only evaluation layer: nothing in the deployed path imports it.

The measurable contract (plan items 4.5/4.7): on both adopted sets the
sufficiency decision must agree with actual evidence presence in the
retrieved pool (zero false sufficiency, zero false refusal), every OOS
case must be «غير كافٍ», and multi-requirement cases must be
differentiated (which requirement is missing), never blanket-rejected.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate    # noqa: E402  — normalize_for_match (shared matching rule)
import evidence_plan  # noqa: E402
import legal_numbers  # noqa: E402
import restructure  # noqa: E402
import units       # noqa: E402

# ---------------------------------------------------------------------------
# Declared precedence table (REVIEW DATA — veto / reorder line by line)
# ---------------------------------------------------------------------------

PRECEDENCE: dict[str, int] = {
    "Loi_2016-48.pdf": 1,             # الرائد الرسمي — القانون
    "Circulaire_BCT_2019-08.pdf": 2,  # منشور البنك المركزي
    "Guide_Interne_Operations_Bancaires_Islamiques.docx": 3,  # الدليل الداخلي
    "Madkhal_Sayrafa_Islamiya.docx": 4,                       # المدخل التعليمي
}

# ---------------------------------------------------------------------------
# Declared anchor rules (REVIEW DATA)
# ---------------------------------------------------------------------------

# Same-script questions: a 4-term exact overlap with a hit is topical
# evidence; SHORT questions cannot reach 4, so the bar is relative —
# min(AR_ANCHOR_MIN, ceil(AR_ANCHOR_SHARE * n_terms)) (measured on the
# adopted sets: every ar hit-case clears its bar, the OOS best false hit
# never does).
AR_ANCHOR_MIN = 4
AR_ANCHOR_SHARE = 0.6
# Cross-script questions share only embedded terms/digits — TWO shared terms
# distinctive in the corpus (df over chunk texts <= CROSS_DF_MAX) are
# required (a single generic Latin word appearing once is noise, not signal).
CROSS_ANCHOR_MIN = 2
CROSS_DF_MAX = 30

# The df cap is CALIBRATED on the adopted 339-chunk corpus (fallback token
# estimator) but must SCALE with the live corpus size: df counts grow with
# (a) finer token estimators — tiktoken counts ~2.5x more tokens for Arabic,
# so the same corpus chunks into ~2.5x more pieces (the owner's server:
# 858 baked-in chunks, df(المرابحة) ≈ 40 even WITHOUT duplicates), and
# (b) legitimate corpus growth. Rarity is a SHARE, not an absolute count:
# duplication and finer chunking scale every term equally (بنك sits in ~70%
# of chunks however many copies exist; مرابحة ~5%). cap = max(CROSS_DF_MAX,
# ceil(CROSS_DF_MAX * N / 339)) — on the calibrated corpus this is EXACTLY
# CROSS_DF_MAX (measured: both adopted sets byte-identical).
CROSS_DF_CALIBRATION_N = 339


def _df_cap(df: dict[str, int] | None) -> int:
    """The anchor df cap for THIS corpus (see CROSS_DF_CALIBRATION_N).
    Hand-built dicts (tests) carry no size -> the calibrated corpus ->
    the historical absolute cap."""
    n = (df or {}).get("__corpus_size__") if df else None
    if not isinstance(n, int) or n <= CROSS_DF_CALIBRATION_N:
        return CROSS_DF_MAX
    return max(CROSS_DF_MAX,
               -(-CROSS_DF_MAX * n // CROSS_DF_CALIBRATION_N))

# Declared cross-script bridges (REVIEW DATA — 2026-10-02, the owner's field
# case: 'what is murabaha?' was refused although the corpus defines المرابحة
# in 17 chunks; df('murabaha')=0 because the Latin transliteration never
# appears in the corpus texts). A question term listed here is equivalent to
# its Arabic counterpart for ANCHORING (coverage) purposes only — retrieval,
# matching and the citation gate are untouched. A bridged match anchors
# SINGLY (a declared, human-curated equivalence is high-precision by
# construction — unlike incidental lexical overlap, which still needs
# CROSS_ANCHOR_MIN terms) but remains df-checked (df <= CROSS_DF_MAX) so a
# bridge onto boilerplate (بنك، تونس) can never anchor. Measured corpus df:
# مرابحة=17 مضاربة=28 مشاركة=22 صكوك=2 اجارة=17 تكافل=3 ربا=13 استصناع=11.
CROSS_SCRIPT_BRIDGES: dict[str, str] = {
    "murabaha": "مرابحة", "morabaha": "مرابحة", "mourabaha": "مرابحة",
    "murabah": "مرابحة",
    "mudaraba": "مضاربة", "moudaraba": "مضاربة", "modaraba": "مضاربة",
    "musharaka": "مشاركة", "mousharaka": "مشاركة",
    "sukuk": "صكوك",
    "ijara": "اجارة", "ijarah": "اجارة",
    "takaful": "تكافل",
    "riba": "ربا",
    "salaf": "سلف",
    "istisna": "استصناع",
}
# Owner activation 2026-10-02 (the failing question was re-sent after the
# build offer — read as the directive). "0" restores the pre-bridge exact-
# lexical behavior exactly.
CROSS_BRIDGE_ENABLED = os.getenv("CROSS_SCRIPT_BRIDGES_ENABLED", "1") == "1"

# ---------------------------------------------------------------------------
# Field-vocabulary bridges (REVIEW DATA — 2026-10-05, the Phase-9 answer
# probe: TM03/TM04 retrieved the RIGHT guide sections and were still refused).
#
# Measured cause (raglab/audits/PHASE9_STRUCTURE/13_CROSS_SCRIPT_COVERAGE.md):
# the corpus is Arabic-only, so a French/English question shares ZERO terms
# with its own evidence chunk —
#   Guide…docx::chunk_0043 («5.1- عمليات فتح وقبول الاعتمادات المستندية», which
#   literally reads «الاعتماد المستندي … هو تعهد مكتوب صادر من بنك … ويسلم
#   للبائع … بناء على طلب المشتري») -> shared_terms=[], _anchors()=False.
# The seed table above only carries Islamic-finance TRANSLITERATIONS
# (murabaha, mudaraba, …); it has no row for the operational vocabulary the
# guide is actually written in, so nothing can anchor and every requirement
# stays uncovered -> «غير كافٍ» -> refusal on an answerable question.
#
# Same contract as the seed table: ANCHORING only (retrieval, matching and the
# citation gate are untouched), single bridged term anchors, still df-checked
# (df <= cap), so a bridge onto boilerplate can never anchor. Measured corpus
# df (339 chunks): اعتماد=12 مستندي=4 صرف=14 — all under CROSS_DF_MAX=30.
#
# Default ON (2026-10-05, owner directive after the live measurement: 2/2
# answered with the arm vs 2/2 refused without it, on the owner's 1713-chunk
# index, understood_as absent -> the ORIGINAL question passed the first pass).
# "0" restores the pre-fix behavior exactly.
FIELD_BRIDGE_TERMS: dict[str, str] = {
    # fr — «opération de change» is the guide's own heading 5.3 subject
    "change": "صرف",
}

# Multi-word concepts whose single tokens are NOT safely equivalent on their
# own («credit» alone is not «مستندي»), so the equivalence is declared on the
# PHRASE. Keys are matched on the question's raw normalized token sequence
# (stopwords kept, so «opération DE change» is one key) with Latin accents
# folded — the corpus spells its own Latin glosses with accents («Lettre de
# Crédit Documentaire») while questions arrive both ways.
FIELD_BRIDGE_PHRASES: dict[str, str] = {
    "documentary credit": "الاعتماد المستندي",
    "documentary credits": "الاعتماد المستندي",
    "credit documentaire": "الاعتماد المستندي",
    "credits documentaires": "الاعتماد المستندي",
    "lettre de credit": "الاعتماد المستندي",
    "operation de change": "عمليات الصرف",
    "operations de change": "عمليات الصرف",
    "change a terme": "الصرف الاجل",
}

# Longest phrase key looked up (in tokens). Declared, not inferred.
FIELD_BRIDGE_MAX_PHRASE = 4

FIELD_BRIDGES_ENABLED = (
    os.getenv("SUFFICIENCY_FIELD_BRIDGES_ENABLED", "1") == "1")

# A rare term (df <= RARE_DF) carries subject signal; common-term overlap
# (بنك، تونس، شركة) is boilerplate noise. Used by the escalation rule.
RARE_DF = 5

_AR_STOP_RAW = """
ما هي هو كيف هل في على من إلى عن مع وفق حسب بين ماذا لماذا متى أين الذي التي هذا هذه
ذلك كل بعض غير بعد قبل لدى منها فيه بها أي قد لقد ثم أو أم لا نعم عند عندما حيث كما
الى او وهو وهي ما وفي ومن فيها عليها إليها لها منها عنه عليه به فيه
""".split()

# post-core residue (فيها -> strip ف -> «يها») matches everything — junk
_CORE_JUNK = {"يها", "ها", "يا", "لا", "ما"}
LAT_STOP = set("""
what which how when where who whom whose is are was were does do did according
to of in for under must the a an and or quelle quel quelles quels comment
quand qui est sont selon dans pour sous doit doivent la le les un une des du
de et au aux en par sur avec ce cette ces que quoi dont must many much
""".split())

# agglutinated Arabic prefixes — matching uses the bare core (relations.py
# lesson: «بالفصل34» must match «فصل34»)
_PREFIXES = ("وال", "بال", "فال", "كال", "لل", "ال", "و", "ف", "ب", "ل", "ك")

# Arabic punctuation INSIDE the \u0600-\u06FF range (؟ ؛ ، …) would glue
# itself to tokens («مشاركة؟») and break exact matching — strip it.
_AR_PUNCT = "؟!؛،…·"


def _stop_sets():
    ar = {evaluate.normalize_for_match(w) for w in _AR_STOP_RAW}
    lat = {w.lower() for w in LAT_STOP}
    return ar, lat


_AR_STOP_N, _LAT_STOP_N = _stop_sets()

# ---------------------------------------------------------------------------
# Declared bounded guided search (REVIEW DATA)
# ---------------------------------------------------------------------------

MAX_GUIDED_ROUNDS = 2        # hard cap
GUIDED_TERMS_PER_ROUND = 5   # rarest uncovered terms per round
ROUND_K = 20                 # retrieval depth of a guided round

# A cross-script question's own terms have df=0 in an Arabic corpus, so the
# round below selected NOTHING and broke immediately (measured on TM03/TM04:
# guided_rounds=[] — the rescue was dead exactly where it was needed). When
# this is on, the round searches the DECLARED Arabic equivalents instead
# (_bridged_terms — governed table, no model call, no rephrasing).
# Default ON (measured 2026-10-05 over the 8-arm matrix): with the bridges on,
# the guided round turns TM04 from غير كافٍ to كافٍ on the offline arm even
# though its evidence sits at BM25 ranks 27/29/36 — the round now reaches what
# the first window cannot. Both frozen sets are byte-identical in all eight
# arms (FS=0 FR=0 agreement=1.0). "0" restores the dead-round behavior.
# COST: up to MAX_GUIDED_ROUNDS extra retrievals, and only on the path that
# was already failing.
GUIDED_BRIDGED_ENABLED = (
    os.getenv("SUFFICIENCY_GUIDED_BRIDGED_ENABLED", "1") == "1")

# ---------------------------------------------------------------------------
# Declared shape patterns for UNTYPED documents (REVIEW DATA) — the typed
# law path reuses evidence_plan's adopted checks via unit types.
# ---------------------------------------------------------------------------

SHAPE_PATTERNS: dict[str, list[str]] = {
    "definition_or_purpose_unit": [
        r"تعتبر|يعتبر|^يعد|على معنى|تعريف|تعني|يقصد",
        # تعريف سردي/كياني (تاريخ التأسيس والظهور) — أسئلة «متى وأين تأسس»
        r"تأسس|تاسس|نشأ|ظهر|تجربة|سنة\s*\d{4}|\d{4}",
    ],
    "numeric_evidence": [r"\d|دينار|في المائة|نسبة|مبلغ|أجل|عقوبة"],
    "penalty_unit": [r"عقوب|خطية|غرامة|يعاقب"],
    "with_governing_rule": [r"فصل\s*\d+|الفصول\s*\d+|الفصل\s*\d+"],
    "exception_with_rule": [r"استثناء|باستثناء|عدا|ما عدا|لا يشمل"],
    "procedural_evidence": [r"يجب|يتم|تتولى|يتولى|يقدم|تقدم|يمنح|تمنح|شرط|"
                            r"يستوجب|يمكن للبنك|على البنك",
                            # شرح/مواقف معيارية في الوثائق غير المنمّطة
                            r"ينظر|يرى|حكم |مبدأ|يتحمل|تتحمل|يجوز|لا يجوز"],
    "both_sides_evidence": [],   # coverage = >= 2 anchored hits (rule, not text)
    "wide_evidence": [],         # any anchored hit
}

# ---------------------------------------------------------------------------
# Declarative definition/purpose shapes for UNTYPED documents (REVIEW DATA —
# 2026-10-05, same probe as the field bridges above).
#
# The guide defines documentary credit as «الاعتماد المستندي … هو تعهد مكتوب
# صادر من بنك …» — an equative nominal sentence with NO explicit definition
# verb, so none of the patterns above fire and the requirement stayed
# uncovered even though the evidence was in the pool.
#
# Both additions were measured for looseness over the 339-chunk corpus before
# being declared (a pattern that matches most chunks would make the
# definitional requirement vacuous — a false-sufficiency risk):
#   existing definition pattern   91/339 chunks
#   «هو + <اسم جامد>» (below)      1/339 chunk  — exactly chunk_0043
#   bare «هو + …»                14/339 chunks  — rejected as too loose
#   «يستعمل|يستخدم|يهدف|الغرض»    55/339 chunks  — rejected as too loose
#   tight purpose forms (below)    3/339 chunks
# Default ON with the field bridges above (same measurement); "0" restores
# the previous shapes exactly.
DEFINITION_SHAPE_PATTERNS: list[str] = [
    # equative definition: «X هو <what X IS>» — the noun list is the declared
    # set of heads the corpus actually uses for «what the thing is».
    r"هو\s+(تعهد|عقد|اتفاق|التزام|عملية|صيغة|بيع|شراء|تفويض|مستند|أمر|خطاب|ضمان)",
    # declared purpose/use, in the corpus's own tight forms only
    r"ويستعمل|ويستخدم|يهدف\s+الى|الغرض\s+منه|الغاية\s+منه",
]

DEFINITION_SHAPE_ENABLED = (
    os.getenv("SUFFICIENCY_DEFINITION_SHAPE_ENABLED", "1") == "1")

_CONFLICT_UNRESOLVABLE = "unresolvable"

_CONFLICT_RESOLVED = "resolved-by-precedence"


# ---------------------------------------------------------------------------
# Terms & anchoring
# ---------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    raw = re.split(r"[^\w\u0600-\u06FF]+", text)
    return [t.strip(_AR_PUNCT) for t in raw if t.strip(_AR_PUNCT)]


def _core(token: str) -> str:
    s = evaluate.normalize_for_match(token)
    for p in _PREFIXES:
        if len(s) > len(p) + 1 and s.startswith(p):
            return s[len(p):]
    return s


def question_terms(question: str) -> list[str]:
    """Distinctive question terms (digits kept, stopwords dropped, in order)."""
    out: list[str] = []
    for t in _tokens(question):
        n = evaluate.normalize_for_match(t)
        if n.isdigit():
            term = n
        else:
            if len(n) < 3 or n in _AR_STOP_N or n.lower() in _LAT_STOP_N:
                continue
            term = _core(n)
            if term in _CORE_JUNK:
                continue
        if term and term not in out:
            out.append(term)
    return out


def _hit_terms(text: str) -> set[str]:
    toks = _tokens(text)
    return {_core(t) for t in toks if len(_core(t)) >= 2} | {
        evaluate.normalize_for_match(t) for t in toks if t.isdigit()
    }


def shared_terms(question: str, hit_text: str) -> set[str]:
    """Exact-term overlap between the question and a hit TEXT (never metadata)."""
    return set(question_terms(question)) & _hit_terms(hit_text)


# Latin words that NAME THE DOMAIN OR THE DOCUMENTS (bank/finance/guide/law)
# — they are not subject terms; a single shared one is noise, never a signal.
GENERIC_LATIN = {"finance", "financial", "bank", "banking", "banks", "islamic",
                 "guide", "internal", "interne", "circular", "circulaire",
                 "law", "loi", "bct"}

# A mostly-Latin question with a few embedded Arabic terms is treated on the
# cross path: the embedded terms are signals, not a script.
MIXED_AR_MAX = 2


def _is_cross_script(question: str) -> bool:
    """<= MIXED_AR_MAX Arabic tokens -> the question cannot lexically meet
    the corpus except through embedded terms/digits (cross rules apply)."""
    n_ar = len(re.findall(r"[\u0600-\u06FF]+", question))
    return n_ar <= MIXED_AR_MAX


# The /answer response fields this module feeds (Phase-5 item 5, additive-
# only §2.7). The service adds exactly these keys when
# SUFFICIENCY_FIELDS_ENABLED=1 and none of them when it is off.
RESPONSE_FIELDS = ("evidence_status", "requirements_covered",
                   "requirements_missing", "conflicts", "refusal_reason")

_DF_CACHE: dict = {}


def df_for_collection(collection) -> dict[str, int]:
    """build_df over a live collection's texts, cached per (name, count) —
    the corpus is small and re-tokenizing it per request would be waste."""
    key = (getattr(collection, "name", None) or id(collection),
           collection.count())
    if key not in _DF_CACHE:
        docs = collection.get(include=["documents"])["documents"] or []
        _DF_CACHE.clear()          # keep a single entry: corpora are small
        _DF_CACHE[key] = build_df(docs)
    return _DF_CACHE[key]


def build_df(texts: list[str]) -> dict[str, int]:
    """Document frequency of every term over the corpus texts (declared
    distinctiveness signal for cross-script anchoring). Carries the corpus
    size under __corpus_size__ so the df cap can scale (never a term:
    tokenization cannot yield that key)."""
    df: dict[str, int] = {}
    for text in texts:
        for term in _hit_terms(text):
            df[term] = df.get(term, 0) + 1
    df["__corpus_size__"] = len(texts)
    return df


def _anchor_bar(question: str) -> int:
    """Declared same-script anchor bar: min(AR_ANCHOR_MIN,
    ceil(AR_ANCHOR_SHARE * n_terms)) — short questions get a short bar."""
    n = len(question_terms(question))
    return min(AR_ANCHOR_MIN, max(1, -(-int(AR_ANCHOR_SHARE * 100 * n) // 100)))


def _bridge_key(token: str) -> str:
    """Latin accent-folded key for the declared bridge tables: the corpus
    spells its own Latin glosses with accents («Lettre de Crédit
    Documentaire») while questions arrive both ways, and NFKC (used by
    normalize_for_match) does NOT fold them. Arabic is untouched by NFD."""
    folded = unicodedata.normalize("NFD", token)
    return "".join(c for c in folded
                   if not unicodedata.combining(c)).casefold()


def _bridge_phrase_keys(question: str) -> list[str]:
    """Declared phrase keys carried by the question, longest first, over the
    RAW token sequence (stopwords KEPT — «opération DE change» is one key, and
    question_terms() would have dropped «de»)."""
    toks = [_bridge_key(_core(t)) for t in _tokens(question)]
    keys: list[str] = []
    for n in range(min(FIELD_BRIDGE_MAX_PHRASE, len(toks)), 1, -1):
        for i in range(len(toks) - n + 1):
            keys.append(" ".join(toks[i:i + n]))
    return keys


def _bridged_terms(question: str) -> set[str]:
    """Normalized Arabic equivalents of the question's DECLARED bridge terms
    (empty when the gate is off or the question carries none)."""
    if not CROSS_BRIDGE_ENABLED:
        return set()
    out: set[str] = set()
    tables = [CROSS_SCRIPT_BRIDGES]
    if FIELD_BRIDGES_ENABLED:
        tables.append(FIELD_BRIDGE_TERMS)
        for key in _bridge_phrase_keys(question):
            arabic = FIELD_BRIDGE_PHRASES.get(key)
            if arabic:
                out |= _hit_terms(arabic)
    for term in question_terms(question):
        for table in tables:
            arabic = table.get(term)
            if arabic:
                out |= _hit_terms(arabic)   # same normalization as the hit side
    return out


def _anchors(question: str, hit_text: str, df: dict[str, int] | None) -> bool:
    """Does this hit topically anchor the question? (declared rules)"""
    shared = shared_terms(question, hit_text)
    if _is_cross_script(question):
        if df is None:
            return False
        # digits are df-checked like words — «48»/«2016» sit in 240 chunk
        # headers and carry no subject signal
        cap = _df_cap(df)
        distinctive = {t for t in shared if df.get(t, 0) <= cap}
        if len(distinctive) >= CROSS_ANCHOR_MIN:
            return True
        # declared cross-script bridge: a bridged DISTINCTIVE term anchors
        # singly (curated equivalence — but still df-checked, so a bridge
        # onto boilerplate can never anchor)
        bridged = _bridged_terms(question) & _hit_terms(hit_text)
        return any(df.get(t, 0) <= cap for t in bridged)
    return bool(shared) and len(shared) >= _anchor_bar(question)


# ---------------------------------------------------------------------------
# Requirement shape checks
# ---------------------------------------------------------------------------

def _shape_patterns(req_kind: str) -> list[str]:
    """Declared shape patterns for a requirement kind. The 2026-10-05
    definition/purpose additions are appended ONLY when their gate is on, so
    the flag has no import-time side effect and unset reproduces the previous
    patterns exactly."""
    patterns = SHAPE_PATTERNS.get(req_kind, [])
    if DEFINITION_SHAPE_ENABLED and req_kind == "definition_or_purpose_unit":
        return patterns + DEFINITION_SHAPE_PATTERNS
    return patterns


_LAW_CACHE: dict = {}


def _law_context() -> tuple[dict[str, dict], set[str]]:
    """(heading -> unit, numeric unit ids) — built once, deterministic."""
    if "ctx" not in _LAW_CACHE:
        codex = restructure._adopted_codex_text("Loi_2016-48.pdf")
        if codex is None:
            _LAW_CACHE["ctx"] = ({}, set())
        else:
            law_units = units.extract_law_units(codex)
            law_map = {u["heading"]: u for u in law_units}
            numeric_ids = {r["unit_id"]
                           for r in legal_numbers.extract_legal_numbers(law_units)}
            _LAW_CACHE["ctx"] = (law_map, numeric_ids)
    return _LAW_CACHE["ctx"]


def shape_ok(req_kind: str, hit: dict, law_map: dict[str, dict],
             numeric_ids: set[str], question: str) -> bool | None:
    """Is this requirement's evidence SHAPE present in this hit?

    Typed path (law chunks via their heading) reuses the adopted evidence_plan
    logic; untyped texts use the declared SHAPE_PATTERNS. None = the check
    cannot apply (never guessed as True or False)."""
    meta = hit.get("metadata") or {}
    text = hit.get("text", "")
    unit = law_map.get(meta.get("heading") or "") \
        if (meta.get("document") or meta.get("source")) == "Loi_2016-48.pdf" else None

    if unit is not None:  # typed path — adopted item-2 semantics
        if req_kind == "definition_or_purpose_unit":
            if unit.get("type") == "definition" or re.search(
                    r"تعتبر|يعتبر|^يعد|على معنى", text):
                return True
            if evidence_plan._ENTITY_OR_PURPOSE_RE.search(question):
                return unit.get("type") in ("general", "procedure", "delegation")
            return False
        if req_kind == "numeric_evidence":
            return bool(unit.get("numeric") or unit["unit_id"] in numeric_ids)
        if req_kind == "penalty_unit":
            return unit.get("type") == "penalty"
        if req_kind == "with_governing_rule":
            return bool(evidence_plan._INTERNAL_REF_RE.search(text))
        if req_kind == "procedural_evidence":
            return unit.get("type") in ("procedure", "delegation", "general",
                                        "definition")
        # exception/both_sides/wide on typed units fall through to patterns
    if req_kind == "both_sides_evidence":
        return None   # rule-based (>=2 anchored hits), not text-shaped
    if req_kind == "wide_evidence":
        return True   # any anchored hit is valid wide evidence (declared)
    patterns = _shape_patterns(req_kind)
    return any(re.search(p, text) for p in patterns)


# ---------------------------------------------------------------------------
# Conflict detection (declared, via the adopted corrections table)
# ---------------------------------------------------------------------------

def detect_conflicts(anchored_hits: list[dict]) -> list[dict]:
    """Two anchored texts from DIFFERENT precedence levels that disagree —
    detected where the adopted Circulaire codex carries a corrupted number
    (raw_form) whose official value (official_evidence) is carried by another
    anchored document. The declared precedence picks the winning text."""
    conflicts = []
    for row in legal_numbers.CIRCULAIRE_CORRECTIONS:
        raw = evaluate.normalize_for_match(row["raw_form"])
        official = evaluate.normalize_for_match(row["official_evidence"])
        raw_hits = [h for h in anchored_hits
                    if raw in evaluate.normalize_for_match(h.get("text", ""))]
        off_hits = [h for h in anchored_hits
                    if official in evaluate.normalize_for_match(h.get("text", ""))]
        if not raw_hits or not off_hits:
            continue
        r_doc = (raw_hits[0].get("metadata") or {}).get("document") or ""
        o_doc = (off_hits[0].get("metadata") or {}).get("document") or ""
        base = {"topic": row["topic"], "raw": row["raw_form"],
                "official": row["official_evidence"]}
        if r_doc not in PRECEDENCE or o_doc not in PRECEDENCE:
            # an unranked document cannot be compared — escalate, never guess
            conflicts.append(dict(base, resolution=_CONFLICT_UNRESOLVABLE,
                                  winning_text=None))
            break
        r_lvl, o_lvl = PRECEDENCE[r_doc], PRECEDENCE[o_doc]
        if r_lvl == o_lvl:
            # equal precedence (possible under an owner-adjusted table) —
            # the conflict cannot be resolved by this table
            conflicts.append(dict(base, resolution=_CONFLICT_UNRESOLVABLE,
                                  winning_text=None))
            break
        winner = "official" if o_lvl < r_lvl else "raw"
        conflicts.append(dict(
            base, resolution=_CONFLICT_RESOLVED,
            winning_text=("official_evidence" if o_lvl < r_lvl else "raw_form")))
        break   # one recorded conflict is enough to flag the query
    return conflicts


# ---------------------------------------------------------------------------
# The sufficiency check
# ---------------------------------------------------------------------------

def _pool_anchor_shape(question: str, req_kind: str, pool: list[dict],
                       law_map: dict, numeric_ids: set[str], df) -> list[dict]:
    """Anchored hits whose shape satisfies the requirement (the coverers)."""
    coverers = []
    for h in pool:
        if not _anchors(question, h.get("text", ""), df):
            continue
        ok = shape_ok(req_kind, h, law_map, numeric_ids, question)
        if ok:
            coverers.append(h)
    return coverers


def check(question: str, hits: list[dict], df: dict[str, int] | None = None,
          search_fn=None, top_k: int = 20) -> dict:
    """Sufficiency state for one question against its retrieved pool.

    search_fn(query, k) -> [hit, ...] enables the bounded guided rounds
    (declared cap); without it the check is one-shot."""
    plan = evidence_plan.derive_plan(question)
    law_map, numeric_ids = _law_context()
    pool = list(hits[:top_k])
    guided_rounds: list[dict] = []

    def requirements_of(p: dict) -> list[dict]:
        reqs = [dict(r, _sub=None) for r in p["requirements"]]
        for i, sp in enumerate(p.get("sub_plans") or []):
            reqs += [dict(r, _sub=i + 1) for r in sp["requirements"]]
        return reqs

    def evaluate_pool(p):
        rows, anchored_all = [], []
        for req in requirements_of(p):
            coverers = _pool_anchor_shape(question, req["req"], pool,
                                          law_map, numeric_ids, df)
            if req["req"] == "both_sides_evidence":
                anchored = [h for h in pool
                            if _anchors(question, h.get("text", ""), df)]
                comparison_q = bool(re.search(r"فرق|بين|قارن|مقارنة", question))
                # both sides: two anchored hits, or a comparison question
                # anchored once (the corpus may treat both sides together)
                covered = len(anchored) >= 2 or (comparison_q and len(anchored) >= 1)
                coverers = anchored[:2]
            else:
                covered = bool(coverers)
            anchored_all += coverers
            rows.append({"req": req["req"], "sub_question": req["_sub"],
                         "covered": covered,
                         "by": [c.get("id") for c in coverers[:2]]})
        return rows, anchored_all

    rows, anchored_all = evaluate_pool(plan)

    # bounded guided rounds: only while evidence is missing
    searched_terms: set[str] = set()
    r = 0
    while (search_fn is not None and r < MAX_GUIDED_ROUNDS
           and any(not x["covered"] for x in rows)):
        r += 1
        terms = question_terms(question)
        anchorable = set()
        for h in pool:
            anchorable |= shared_terms(question, h.get("text", ""))
        # rarest not-yet-anchorable RETRIEVABLE terms target the gap
        # (df == 0 terms exist in no chunk — searching them cannot help)
        rest = [t for t in terms if t not in anchorable
                and (df is None or df.get(t, 0) > 0)]
        if not rest and GUIDED_BRIDGED_ENABLED and _is_cross_script(question):
            # the question's own terms exist in no chunk (df=0); searching
            # them cannot help. Fall back to the DECLARED Arabic equivalents,
            # rarest first — the same governed table the anchor uses.
            rest = [t for t in _bridged_terms(question)
                    if (df is None or df.get(t, 0) > 0)]
        # drop what an earlier round already searched BEFORE truncating, so
        # round 2 takes the NEXT rarest terms instead of repeating round 1
        # (measured 2026-10-05 on the owner's index: two identical rounds,
        # «صرف عمليات» twice, the second retrieving nothing new).
        rest = [t for t in rest if t not in searched_terms]
        rest.sort(key=lambda t: (df.get(t, 0) if df else 0))
        batch = rest[:GUIDED_TERMS_PER_ROUND]
        if not batch:
            break
        searched_terms.update(batch)
        new_hits = search_fn(" ".join(batch), ROUND_K) or []
        seen = {h["id"] for h in pool}
        added = [h for h in new_hits if h["id"] not in seen]
        guided_rounds.append({"round": r, "query": " ".join(batch),
                              "new_hits": len(added)})
        if not added:
            break
        pool = pool + added
        rows, anchored_all = evaluate_pool(plan)

    missing = [x for x in rows if not x["covered"]]
    result = {
        "question": question,
        "intent_type": plan["intent_type"],
        "requirements": rows,
        "missing": [x["req"] for x in missing],
        "guided_rounds": guided_rounds,
        "final_pool": pool,
    }
    if missing:
        # Nothing that covers the requirements — is the subject itself absent
        # (refusal: evidence absence) or is there unverifiable contact
        # (escalation, never a guess)?
        any_anchored = any(_anchors(question, h.get("text", ""), df)
                           for h in pool)
        if not any_anchored:
            shared_any: set[str] = set()
            for h in pool:
                shared_any |= shared_terms(question, h.get("text", ""))
            if _is_cross_script(question):
                cap = _df_cap(df)
                distinctive = {t for t in shared_any
                               if (df or {}).get(t, 0) <= cap}
                # one tantalizing DELIBERATE signal — an embedded Arabic term,
                # a specific number, or a non-generic technical word — is an
                # honest escalation; a generic domain word or nothing at all
                # means the subject is absent (refusal).
                if len(distinctive) == 1:
                    term = next(iter(distinctive))
                    if (not term.isascii() or term.isdigit()
                            or term not in GENERIC_LATIN):
                        result["state"] = "غير محسوم"
                        result["reason"] = ("cross-script: single distinctive "
                                            "shared term " + term)
                        return result
            else:
                bar = _anchor_bar(question)
                best = max((len(shared_terms(question, h.get("text", "")))
                            for h in pool), default=0)
                if best == bar - 1:
                    # one term below the bar — escalation only when the whole
                    # overlap is RARE terms (subject signal, not boilerplate)
                    best_hit = max(pool, key=lambda h: len(
                        shared_terms(question, h.get("text", ""))))
                    sh = shared_terms(question, best_hit.get("text", ""))
                    if sh and all((df or {}).get(t, 0) <= RARE_DF for t in sh):
                        result["state"] = "غير محسوم"
                        result["reason"] = ("one shared term below the anchor "
                                            "bar, all rare (subject signal)")
                        return result
        result["state"] = "غير كافٍ"
        result["reason"] = "evidence absent for: " + ", ".join(result["missing"])
        return result

    # covered — conflict check on the anchored evidence
    conflicts = detect_conflicts(anchored_all)
    if conflicts and any(c["resolution"] == _CONFLICT_UNRESOLVABLE for c in conflicts):
        result["state"] = "غير محسوم"
        result["reason"] = "conflict the precedence cannot resolve"
        result["conflicts"] = conflicts
        return result
    if conflicts:
        result["state"] = "متعارض"
        result["reason"] = "texts differ; resolved by declared precedence"
        result["conflicts"] = conflicts
        return result
    result["state"] = "كافٍ"
    result["reason"] = "all requirements covered"
    return result


# ---------------------------------------------------------------------------
# Measurement (verification contract — uses the adopted sets' expectations)
# ---------------------------------------------------------------------------

def _gold_present(case: dict, pool: list[dict]) -> list[bool]:
    """Per expected substring: is it inside some pool hit's text? (oracle —
    measurement only, never used by the sufficiency decision itself)."""
    subs = (case.get("expected_substrings")
            or ([case["expected_substring"]] if case.get("expected_substring") else []))
    out = []
    for sub in subs:
        ns = evaluate.normalize_for_match(sub)
        out.append(any(ns in evaluate.normalize_for_match(h.get("text", ""))
                       for h in pool))
    return out


def measure(cases: list[dict], search_fn, k: int = 20,
            df: dict[str, int] | None = None) -> dict:
    """Per set: sufficiency state vs actual evidence presence in the pool
    (the final pool = one-shot top-k + whatever the bounded guided rounds
    added). Zero false sufficiency + zero false refusal + all OOS
    «غير كافٍ» is the plan's transition criterion."""
    rows = []
    for case in cases:
        hits = search_fn(case["question"], k) or []
        res = check(case["question"], hits, df=df, search_fn=search_fn,
                    top_k=k)
        final_pool = res["final_pool"]
        one_shot = _gold_present(case, hits)
        final = _gold_present(case, final_pool)
        is_oos = case["category"] == "out-of-scope"
        if is_oos:
            expected = "غير كافٍ"
        else:
            expected = "كافٍ" if all(final) else "غير كافٍ"
        state = res["state"]
        effective = ("كافٍ" if state == "متعارض" else
                     "غير محسوم" if state == "غير محسوم" else state)
        false_sufficient = effective == "كافٍ" and expected == "غير كافٍ"
        false_refusal = effective == "غير كافٍ" and expected == "كافٍ"
        rows.append({
            "id": case["id"], "category": case["category"],
            "intent": res["intent_type"], "state": state,
            "expected": expected, "effective": effective,
            "n_covered": sum(1 for x in res["requirements"] if x["covered"]),
            "n_requirements": len(res["requirements"]),
            "one_shot_gold": one_shot, "final_gold": final,
            "guided_rounds": res["guided_rounds"],
            "missing": res["missing"],
            "false_sufficient": false_sufficient, "false_refusal": false_refusal,
            "rescued": (not all(one_shot) and all(final)
                        and effective == "كافٍ"),
            "ok": (not false_sufficient and not false_refusal
                   and (not is_oos or state == "غير كافٍ")),
        })
    oos = [r for r in rows if r["category"] == "out-of-scope"]
    return {
        "n": len(rows),
        "false_sufficient": sum(r["false_sufficient"] for r in rows),
        "false_refusal": sum(r["false_refusal"] for r in rows),
        "escalated": sum(1 for r in rows if r["effective"] == "غير محسوم"),
        "guided_rescues": sum(1 for r in rows if r["rescued"]),
        "oos_all_insufficient": (all(r["state"] == "غير كافٍ" for r in oos)
                                 if oos else None),
        "agreement": (sum(r["ok"] for r in rows) / len(rows)) if rows else None,
        "per_case": rows,
    }


# ---------------------------------------------------------------------------
# CLI — deterministic local measurement on the BM25 keyword arm
# ---------------------------------------------------------------------------

class _Corpus:
    """Keyword-only collection adapter (no embeddings — deterministic BM25,
    the same keyword_search the deployed hybrid mode uses)."""

    def __init__(self, chunks: list[dict]):
        self._chunks = chunks

    def count(self):
        return len(self._chunks)

    def get(self, include=None, **_):
        return {"ids": [c["id"] for c in self._chunks],
                "documents": [c["text"] for c in self._chunks],
                "metadatas": [c["metadata"] for c in self._chunks]}


def bm25_store(docs_dir: Path | None = None):
    """(search_fn, df) over the docs corpus — deterministic, no API calls."""
    import chunker
    import loader
    import store
    from types import SimpleNamespace
    import config as config_mod

    cfg = SimpleNamespace(**{k: getattr(config_mod, k)
                             for k in dir(config_mod) if k.isupper()})
    cfg.CHUNKING_MODE = "restructure"
    docs = loader.load_all([docs_dir or (HERE / ".." / "docs")])
    chunks = chunker.chunk_all(docs, cfg)
    corpus = [{"id": f"{c.source}::chunk_{c.index:04d}", "text": c.text,
               "metadata": {"document": c.source, "source": c.source,
                            "language": c.language, "heading": c.heading}}
              for c in chunks]
    collection = _Corpus(corpus)

    def search(query: str, k: int) -> list[dict]:
        return store.keyword_search(collection, query, k=k,
                                    include_metadata=True)

    return search, build_df([c["text"] for c in corpus])


def live_store():
    """(search_fn, df) on the DEPLOYED live arm — the real embedder + the
    stored collection + the deployed retrieval (vector + rerank per config).
    Same contract as bm25_store; --live (CI) measures the sufficiency states
    on the live vector arm — read-only, no /answer call."""
    import config as cfg
    from embedder import build_embedder
    from evaluate import prepare_query_text
    from retrieval import retrieve
    from store import get_collection

    embedder = build_embedder(cfg)
    collection = get_collection(cfg, reset=False)
    if not collection.count():
        raise SystemExit("live collection is empty — ingest first")

    def search(query, k):
        hits, _ = retrieve(cfg, embedder, collection, prepare_query_text(query),
                           mode="vector", top_k=k, candidate_k=k)
        return hits

    return search, df_for_collection(collection)


def main() -> int:
    parser = argparse.ArgumentParser(prog="sufficiency",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--sets", nargs="*", default=["questions_50.json",
                                                      "questions_targets.json"])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--live", action="store_true",
                        help="measure on the deployed live arm (real embedder "
                             "+ stored collection) instead of local BM25")
    args = parser.parse_args()

    if getattr(args, 'live', False):
        search, df = live_store()
    else:
        search, df = bm25_store()
    report = {}
    for name in args.sets:
        cases = json.loads((HERE / name).read_text(encoding="utf-8"))["cases"]
        m = measure(cases, search, k=args.k, df=df)
        report[name] = {k: v for k, v in m.items() if k != "per_case"}
        report[name]["disagreements"] = [
            {kk: r[kk] for kk in ("id", "category", "intent", "state",
                                  "expected", "effective", "one_shot_gold",
                                  "final_gold", "missing", "guided_rounds")}
            for r in m["per_case"]
            if r["false_sufficient"] or r["false_refusal"]
            or r["effective"] == "غير محسوم"]
        report[name]["states"] = {}
        for r in m["per_case"]:
            report[name]["states"][r["state"]] = \
                report[name]["states"].get(r["state"], 0) + 1
        report[name]["per_case_compact"] = [
            {"id": r["id"], "state": r["state"], "expected": r["expected"],
             "n_covered": r["n_covered"], "n_requirements": r["n_requirements"],
             "guided_rounds": len(r["guided_rounds"]), "rescued": r["rescued"]}
            for r in m["per_case"]]
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
