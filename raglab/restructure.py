"""restructure.py — the restructured chunking strategy (CHUNKING_MODE="restructure").

The strategy is three explicit stages, each visible in the printed report:

Stage 1  normalize_structure(doc)
    Semantic normalization: messy raw extraction (PDF pages, visual-order
    Arabic, repeated page headers, glued legal markers like "الفصل2",
    inconsistent bullets) becomes clean, hierarchy-explicit Markdown:
    section titles -> # / ## / ### headings, tables -> Markdown tables with
    separator rows, implicit lists -> standard "- " items, page garbage gone.

Stage 2  enrich_context(markdown, title)
    Context enrichment: a breadcrumb line
        > **Context:** <doc title> > <H2> > <H3>
    is injected directly above every H2/H3 heading and above any table that
    is not already under its own heading, so every subsection (and every
    table) stays self-contained even after it is separated from its title.

Stage 3  recursive_structural_chunk(markdown, ...)
    Recursive structural chunking: a recursive text splitter that walks the
    hierarchy-aware separator list
        ["\\n# ", "\\n## ", "\\n### ", "\\n\\n", "\\n", " "]
    splitting only as finely as the token budget forces, so a section that
    fits the budget stays whole (one logical topic = one chunk), and only
    oversized sections fall through to paragraphs, lines and finally words.
    Every chunk is re-anchored with its context line, so a continuation
    chunk never opens without saying which document section it belongs to.

Nothing here calls an API. Every decision is printed (lines repaired,
headings invented, markers split) so the strategy is inspectable the same
way the rest of this lab is.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from chunker import (CHUNK_FINGERPRINT_VERSION, Chunk, classify_section,
                     count_tokens)
from loader import normalize_arabic, normalize_text

# ---------------------------------------------------------------------------
# Tunables for stage 1 (all printed in the report, none hidden)
# ---------------------------------------------------------------------------

# A line is "Arabic enough" for RTL repair when at least this share of its
# letters are Arabic script.
RTL_ARABIC_SHARE = 0.5
# Minimum line length (chars) considered for RTL repair: short lines
# (headings, labels) are left alone.
RTL_MIN_LEN = 40
# The flipped score must beat the original by at least this many bigram
# hits, otherwise the line is kept as-is (a near-tie is not a decision).
RTL_MARGIN = 2
# Document-level gate for the visual-order (RTL) zone reconstruction: when
# at least RTL_DOMINANT_MIN rendered lines score clearly better AFTER zone
# reconstruction than before (by RTL_MARGIN), and they outnumber the
# opposite direction by RTL_DOMINANT_RATIO to 1, the document is treated as
# extracted in visual order and EVERY rendered line gets zone-reconstructed
# (the reconstruction is the inverse of the extraction transform, not a
# per-line guess). Documents that fail the gate are left byte-identical.
RTL_DOMINANT_MIN = 20
RTL_DOMINANT_RATIO = 4
# Repeated-line garbage (page headers/footers): a line counted this many
# times and shorter than this length is a running header, not content.
REPEAT_HEADER_MIN_COUNT = 3
REPEAT_HEADER_MAX_LEN = 120

# Lines that are religious preambles, never document titles.
_TITLE_SKIP_RE = re.compile(r"^(بسم|الحمد|والصلاة|المدثر)|صلى الله عليه وسلم")
# Official-gazette running headers (e.g. "الرائد الرسمي للجمهورية التونسية
# 15 جويلية 2016 عدد 58", sometimes in visual order: "لجم الرسمي الرائد
# التونسية هورية"). They reappear on every page, sometimes fused into a
# content line, so they are dropped wherever they stand.
_GAZETTE_HEADER_RE = re.compile(
    r"(الرائد|لجم|هورية).{0,60}(جويلية|عدد)|(جويلية|عدد).{0,60}(الرائد|لجم|هورية)")
_GAZETTE_HEADER_MAX_LEN = 160
# The header SPAN as it opens a page's first fused line, once "[page N]" is
# stripped. Two orderings occur in the corpus (odd/even gazette pages):
#   "صفحة2516 للجم الرسمي الرائد التونسية هورية–– 15 جويلية 2016 عدد58 <content>"
#   "عدد58 التونسية للجمهورية الرسمي الرائد–– 15 جويلية 2016 صفحة2519 <content>"
# The prefix logic below stops at the FIRST صفحة/عدد token, which leaves the
# body of the header fused to the content line; this span consumes the whole
# header. It only matches at line start, so the publication formula that
# cites the gazette inside body text ("وينشر بالرائد الرسمي للجمهورية
# التونسية ...") is never touched.
_GAZETTE_MONTHS = (r"جانفي|فيفري|مارس|أفريل|افريل|ماي|جوان|جويلية|أوت|اوت|"
                   r"سبتمبر|أكتوبر|اكتوبر|نوفمبر|ديسمبر")
_GAZETTE_SPAN_RE = re.compile(
    r"^(?:صفحة\s*\d{1,4}|عدد\s*\d{1,3})?\s*"
    r"(?:(?:التونسية|للجمهورية|للجم|الرسمي|الرائد|هورية)[\s\u2013\u2014\-]*){2,8}"
    r"\d{1,2}\s+(?:" + _GAZETTE_MONTHS + r")\s+\d{4}\s*"
    r"(?:عدد\s*\d{1,3}|صفحة\s*\d{1,4})?\s*")

# Common Arabic words that appear fused after a legal marker in visual-order
# extractions ("الاولفصل" == "الفصل الاول" stored right-to-left).
_LEGAL_ORDINALS = ("الاول", "الثاني", "الثالث", "الرابع", "الخامس", "السادس",
                   "السابع", "الثامن", "التاسع", "العاشر", "الحادي عشر",
                   "الثاني عشر")
_MARKER_WORD_RE = re.compile(r"^(الفصل|العنوان|الباب|القسم)")

# Curated logical-order BIGRAMS of formal Tunisian banking/legal Arabic.
# Used ONLY to score original-vs-flipped line order (never to edit text).
# True pairs: word order is the signal, so each hit counts double.
_LOGICAL_BIGRAMS = (
    "القانون عدد", "المؤرخ في", "الجمهورية التونسية", "الرائد الرسمي",
    "باسم الشعب", "نواب الشعب", "يهدف الى", "الحفاظ على", "المحافظة على",
    "على اساس", "على سبيل", "على وجه", "على غرار", "على ان", "على اثر",
    "على معنى", "في مجال", "في شكل", "في اطار", "في اطار المشاريع",
    "في اطار عقد", "في اطار التمويل", "في اطار ممارسة", "في اطار عمليات",
    "في جميع", "في حدود", "في حالة", "في نهاية", "في بداية", "في الغرض",
    "في هذه", "في تلك", "في صورة", "في احوال", "من اجل", "من بين",
    "من طرف", "من طرف اشخاص", "من قبل", "من غير", "منه", "منها",
    "لغرض", "لرهن", "لبيع", "لشراء", "لاقتناء", "لانجاز", "لتقديم",
    "لتحقيق", "لتوفير", "لغير", "لذمة", "عند تأخره", "عند الاقتضاء",
    "عند الحاجة", "عند التوقيع", "عند الطلب", "عند الاثر", "عند العقد",
    "طبقا لاحكام", "طبقا ل", "وفقا لاحكام", "وفقا ل",
    "الشريعة الاسلامية", "المعاملات المالية", "البنوك والمؤسسات",
    "المؤسسات المالية", "المؤسسات المالية الاسلامية", "البنك المركزي",
    "البنك المركزي التونسي", "عمليات الصيرفة", "الصيرفة الاسلامية",
    "المالية الاسلامية", "الامر بالشراء", "ثم بيعها", "بثمن يعادل",
    "تكلفة شرائها", "هامش ربح", "اقساط معلومة", "اجال معلومة",
    "بمقتضى", "بمقتضاها", "بموجبها", "بموجب عقد", "كل عملية",
    "يتولى بمقتضاها", "يتولى بموجبها", "يجب ان", "لا يجوز", "يمكن ان",
    "يمكن للبنك", "يجب على", "يتعين على", "يتمثل في", "يخضع على",
    "الضوابط الشرعية", "المعايير الدولية", "معايير الصيرفة",
    "هيئة مراقبة", "هيئة المحاسبة", "المراجعة للمؤسسات",
    "قبل استلام", "قبل ابرام", "قبل التوقيع", "بعد استلام", "بعد ابرام",
    "بين البنك", "بين المتعاقدين", "بين الحريف", "او عقارات", "او خدمات",
    "منقولات او", "او سلع", "او عقار", "الاوراق المالية", "القروض المضمونة",
    "الغرض من", "المدة المحددة", "مصادر الشريعة", "المرجع الشرعية",
)
# High-frequency content words (containment match, affix-tolerant): a word
# hits when it CONTAINS the lemma (والمؤسسات ~= المؤسسات). Weak signal —
# they appear in both orders — so they count once against the double-
# weighted bigrams.
_LOGICAL_UNIGRAMS = (
    "القانون", "الفصل", "العنوان", "الباب", "البنوك", "البنك", "المؤسسات",
    "المؤسسة", "المالية", "الصيرفة", "الاسلامية", "العمليات", "التمويل",
    "الودائع", "الحرفاء", "الحريف", "الامر", "بالشراء", "المشروع", "المشاريع",
    "النشاط", "المقترض", "المستثمرين", "المودعين", "المستفيدة", "الشركة",
    "السلعة", "السلع", "المبيع", "المشتري", "البائع", "الضمان", "الضمانات",
    "الفائدة", "الربا", "الهدف", "المبلغ", "المبالغ", "النسبة", "الاجل",
    "الاجال", "الوثائق", "القرار", "القرارات", "المنشور", "المنشورات",
    "التشريع", "التشريعات", "الترتيبات", "المعايير", "المستندات",
    "المستند", "الشروط", "العقود", "الحساب", "الحسابات", "الرصيد",
    "الاموال", "النقود", "القروض", "التمويلات", "الخدمات", "البطاقات",
    "الشيكات", "الحوالات", "التحويلات", "السحب", "الايداع", "السيولة",
    "الخزينة", "المحاسبة", "المراجعة", "الرقابة", "المراقبة", "الامتثال",
    "الضوابط", "القواعد", "المبادئ", "شراء", "بيع", "تمويل", "عملية",
    "اعمار", "المصارف", "المصرفية",
)

# The loader joins "[page N]" with the PDF's own page-number line into one
# text line ("[page 1] 1"), so match the marker as a line PREFIX: strip it,
# keep the remainder (and drop a bare page-number remainder).
_PAGE_MARKER_RE = re.compile(r"^\[page \d+\]\s*")
_MARKER_RE = re.compile(
    r"(?:^|(?<=\s))((?:الفصل|العنوان|الباب|القسم)"
    r"(?:\s*\d+|\s+(?:" + "|".join(_LEGAL_ORDINALS) + r")))")
# Fused visual-order form: "<ordinal><marker>" glued (e.g. "الاولفصل").
_FUSED_MARKER_RE = re.compile(
    r"(?:^|(?<=\s))((" + "|".join(_LEGAL_ORDINALS) + r")(?:الفصل|العنوان|الباب|القسم))")
# Same visual-order form with a space between ordinal and marker
# ("الثاني العنوان"). In logical order the marker always precedes its
# ordinal ("العنوان الثاني"), so ordinal-first can only be visual order.
_SPACED_FUSED_MARKER_RE = re.compile(
    r"(?:^|(?<=\s))((" + "|".join(_LEGAL_ORDINALS) + r")\s+(?:الفصل|العنوان|الباب|القسم))")
# PDF text blocks often lose their line breaks entirely: one page becomes a
# single 2000+ char "line". Such a line is not a paragraph — it is a run of
# sentences. Resplit it at sentence boundaries so that per-line processing
# (RTL repair, marker extraction) sees sentence-sized units.
LONG_LINE_MAX = 400
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.؟؛۔])\s+")


@dataclass
class RestructureReport:
    """What stage 1/2 did to one document — printed and stored per run."""
    name: str
    pages_removed: int = 0
    repeated_lines_dropped: int = 0
    gazette_headers_removed: int = 0
    long_lines_resplit: int = 0
    rtl_doc_dominant: bool = False
    rtl_lines_reconstructed: int = 0
    rtl_marker_prefixes: int = 0
    rtl_digit_islands: int = 0
    codex_adopted: bool = False
    markers_extracted: int = 0
    headings_invented: int = 0
    tables_normalized: int = 0
    list_items_normalized: int = 0
    doc_title: str = ""
    headings: list = None  # [(level, text)]

    def __post_init__(self):
        if self.headings is None:
            self.headings = []

    def print(self):
        print(f"[restructure] {self.name}: title={self.doc_title[:60]!r}")
        print(f"[restructure]   pages_removed={self.pages_removed} "
              f"repeated_lines_dropped={self.repeated_lines_dropped} "
              f"gazette_headers_removed={self.gazette_headers_removed} "
              f"long_lines_resplit={self.long_lines_resplit} "
              f"codex_adopted={self.codex_adopted}")
        print(f"[restructure]   rtl: visual-order doc={self.rtl_doc_dominant} "
              f"rendered-lines reconstructed={self.rtl_lines_reconstructed} "
              f"(marker prefixes {self.rtl_marker_prefixes}, "
              f"digit islands {self.rtl_digit_islands}) | "
              f"markers_extracted={self.markers_extracted} "
              f"headings_invented={self.headings_invented}")
        print(f"[restructure]   tables_normalized={self.tables_normalized} "
              f"list_items_normalized={self.list_items_normalized} | "
              f"headings={len(self.headings)}")


# ---------------------------------------------------------------------------
# Stage 1 — semantic normalization
# ---------------------------------------------------------------------------

def _arabic_share(line: str) -> float:
    arabic = len(re.findall(r"[\u0600-\u06FF]", line))
    letters = len(re.findall(r"[A-Za-z\u0600-\u06FF\u00C0-\u00FF]", line))
    return (arabic / letters) if letters else 0.0


def _order_score(words: list[str]) -> float:
    """Plausibility of a word order: double-weighted curated bigram hits
    plus single-weighted, affix-tolerant unigram hits."""
    bigrams = 0
    for a, b in zip(words, words[1:]):
        if f"{a} {b}" in _LOGICAL_BIGRAMS:
            bigrams += 1
    unigrams = 0
    for w in words:
        for lem in _LOGICAL_UNIGRAMS:
            if len(lem) >= 5 and lem in w:
                unigrams += 1
                break
    return bigrams * 2 + unigrams


def _has_digit(tok: str) -> bool:
    return any(c.isdigit() for c in tok)


# Sentence-terminal punctuation re-emitted by the extractor at the raw
# line's END regardless of where it belongs in the logical line (verified
# on Loi 2016-48). Commas are NOT in this set: they ride their own word.
_TERMINAL_SUFFIX_RE = re.compile(r"[.!\u061F\u061B:]+$")


def _is_digit_island(tok: str) -> bool:
    """True for tokens that belong to a keep-in-place digit run.

    The extractor keeps digit runs in logical order while reversing the
    Arabic around them, and fuses boundary material into single tokens.
    Discriminator = the Arabic residue left after stripping digits,
    punctuation and dashes: at most a clinging short prefix/suffix (و، ن،
    ولا — "47و", "ن48", "ولا78") rides the digit run (island); a real word
    fused with its number ("الصادرة64", "يتعلق)1(", "المبرم30") belongs to
    its Arabic run and is un-reversed with it (verified line-by-line on the
    Loi body: "عدد بالقانون الصادرة64" -> "الصادرة 64 بالقانون عدد").
    """
    if not _has_digit(tok):
        return False
    residue = re.sub(r"[\d\s\(\)\[\]\{\}.,،؛:!\u061F\u061B\u061D?\-\u2010-\u2015\u0640]", "", tok)
    return len(residue) <= 3


def _reconstruct_line_visual_order(line: str,
                                   report: "RestructureReport | None" = None) -> str:
    """Un-reverse ONE physical rendered line of a visual-order Arabic PDF.

    Empirical extraction model (verified line-by-line on Loi 2016-48, the
    JORT gazette PDF): each rendered line reaches us as a sequence of runs —
    pure-Arabic runs are word-REVERSED, runs containing digits keep their
    logical order, and the runs themselves appear in logical order. The
    sentence-terminal period is re-emitted at the raw line's end. The
    faithful inverse is therefore NOT a whole-line flip (which would also
    reverse the digit runs and the run order, scrambling enumerations and
    sentence halves): keep each maximal digit-token run ("island") in place,
    reverse each pure-Arabic run back, and re-attach the trailing period at
    the end. A leading structural marker (الفصلNN ـ ...) stays in place,
    with its content reconstructed after it. Letters fused to digits inside
    a token are split at the boundary (الفصول46و -> الفصول 46 و).

    Only ever called for documents that passed the document-dominance gate
    (see normalize_structure); logical-order documents never reach this.
    """
    s = line.strip()
    if not s or s.startswith("|"):
        return line
    prefix = ""
    m = (_MARKER_RE.match(s) or _FUSED_MARKER_RE.match(s)
         or _SPACED_FUSED_MARKER_RE.match(s))
    if m:
        prefix = m.group(0).rstrip()
        s = s[m.end():].strip()
        # a bare dash/tatweel separator token belongs to the marker, not content
        head = s.split()
        if head and head[0] in ("ـ", "—", "–", "-"):
            prefix += " " + head[0]
            s = " ".join(head[1:])
    if not s:
        return prefix

    # Split letters fused to digits BEFORE zoning (with an optional
    # parenthesis cluster in between, as in يتعلق)1( ): the Arabic part of a
    # fused token belongs to its (reversed) Arabic run and the digit part
    # to the keep-in-place island — verified on the Loi body: stored
    # "عدد بالقانون الصادرة64 لسنة 2009" must give "الصادرة بالقانون عدد 64
    # لسنة 2009", which only this order produces.
    s = re.sub(r"(?<=[\u0600-\u06FF])(?=[()\[\]]*\d)", " ", s)
    s = re.sub(r"(?<=\d)(?=[\u0600-\u06FF])", " ", s)
    s = re.sub(r"(?<=\d\))(?=[\u0600-\u06FF])", " ", s)
    s = re.sub(r"(?<=\d\()(?=[\u0600-\u06FF])", " ", s)

    toks = s.split()
    # Hold the raw line's trailing sentence punctuation aside: it belongs at
    # the TRUE end of the line, i.e. the end of the reconstructed line.
    trail = ""
    if toks:
        last = toks[-1]
        if _TERMINAL_SUFFIX_RE.fullmatch(last):
            trail = last
            toks.pop()
        else:
            mm = _TERMINAL_SUFFIX_RE.search(last)
            if mm and mm.start() > 0 and not last[:mm.start()].isdigit():
                trail = last[mm.start():]
                toks[-1] = last[:mm.start()]

    out: list[str] = []
    i = 0
    while i < len(toks):
        if _is_digit_island(toks[i]):
            j = i
            while j < len(toks) and _is_digit_island(toks[j]):
                j += 1
            out.extend(toks[i:j])
            if report is not None:
                report.rtl_digit_islands += j - i
            i = j
        else:
            j = i
            while j < len(toks) and not _is_digit_island(toks[j]):
                j += 1
            out.extend(reversed(toks[i:j]))
            i = j
    body = re.sub(r"\s{2,}", " ", " ".join(out)).strip()
    if trail:
        body = (body + " " + trail).strip()
    if report is not None:
        report.rtl_lines_reconstructed += 1
        if prefix:
            report.rtl_marker_prefixes += 1
    return (prefix + " " + body).strip() if prefix else body


def _drop_repeated_headers(lines: list[str], report: RestructureReport) -> list[str]:
    """Drop page-running headers/footers (a line repeated on many pages).

    The first occurrence is kept (it may be meaningful front matter, e.g. the
    gazette reference line); later duplicates are dropped and counted.
    """
    from collections import Counter

    def key(l: str) -> str:
        return re.sub(r"\d+", "#", l.strip())

    counts = Counter(key(l) for l in lines if l.strip())
    out: list[str] = []
    seen: dict[str, int] = {}
    repeated_keys: set[str] = set()
    for l in lines:
        k = key(l)
        stripped = l.strip()
        # Table rows are exempt: TOC tables legitimately repeat row shapes
        # (different section numbers collapse to the same digit-stripped
        # key), and flush_table already dedupes true header repeats.
        # A line ending in sentence-terminal punctuation is a complete
        # sentence, not a running header: legal texts legitimately repeat
        # standard closing formulas after several chapters (Loi 2016-48 has
        # "ويضبط البنك المركزي التونسي شروط تطبيق هذا الفصل" after 3 of
        # them) — those are content and must survive.
        if (stripped and not stripped.startswith("|")
                and counts[k] >= REPEAT_HEADER_MIN_COUNT
                and len(stripped) <= REPEAT_HEADER_MAX_LEN
                and not stripped.endswith((".", "؟", "؛", "!"))):
            repeated_keys.add(k)
            seen[k] = seen.get(k, 0) + 1
            if seen[k] > 1:
                report.repeated_lines_dropped += 1
                continue
        out.append(l)
    return out, repeated_keys


def _strip_gazette_header(line: str, report: RestructureReport) -> str:
    """Drop official-gazette running headers, whole or as a line prefix.

    A short line that looks like a gazette header is dropped outright. A
    longer line that OPENS with a gazette header (PDF extraction fuses the
    header into the page's first content line) keeps its remainder.
    """
    m = _GAZETTE_SPAN_RE.match(line)
    if m:
        report.gazette_headers_removed += 1
        return line[m.end():].strip()
    if len(line) <= _GAZETTE_HEADER_MAX_LEN:
        # Whole-line drop only with a corroborating token: a month-year date,
        # or the gazette-specific words هورية / للجم as standalone tokens.
        # Otherwise a short body line that merely cites a عدد next to الرائد
        # (law citations, the publication formula) would be dropped whole.
        if (_GAZETTE_HEADER_RE.search(line)
                and (re.search(_GAZETTE_MONTHS + r"\s+\d{4}", line)
                     or re.search(r"(?:^|\s)هورية(?:^|\s)", line)
                     or re.search(r"(?:^|\s)للجم(?:^|\s)", line))):
            report.gazette_headers_removed += 1
            return ""
        return line
    head = line[:120]
    if not _GAZETTE_HEADER_RE.search(head):
        return line
    m = re.match(r"^.{0,120}?(?:صفحة\d+|عدد\s*\d+)\s*", line)
    if m and _GAZETTE_HEADER_RE.search(m.group(0)):
        report.gazette_headers_removed += 1
        return line[m.end():].strip()
    return line


def _is_structural_line(line: str) -> bool:
    """Lines that must never be reflow-joined (mirrors loader._is_block_line):
    headings, table rows, list items, horizontal rules."""
    s = line.strip()
    if not s:
        return False
    if s.startswith("#") or s.startswith("|"):
        return True
    if re.match(r"^[-*+]\s+\S", s):
        return True
    if re.match(r"^\d+[.)]\s+\S", s):
        return True
    if re.match(r"^(---+|\*\*\*+)\s*$", s):
        return True
    return False


def _resplit_long(line: str, report: RestructureReport) -> list[str]:
    """A text block that lost its line breaks: resplit into sentence-sized
    lines so the later per-line passes see units they can judge."""
    if len(line) > LONG_LINE_MAX and not line.startswith("|"):
        parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(line) if p.strip()]
        if len(parts) > 1:
            report.long_lines_resplit += len(parts) - 1
            return parts
    return [line] if line else []


def _reflow_lines(lines: list[str], report: RestructureReport) -> list[str]:
    """Re-form paragraphs from physical extracted lines.

    PDFs now arrive with every rendered line on its own line break (loader
    reflow=False — the RTL zone reconstruction needs them separate). After
    the per-line reconstruction, consecutive paragraph lines are joined back
    into paragraphs (blank lines and structural lines are boundaries), then
    over-long joined lines are resplit. For sources that already arrive
    reflowed (DOCX: blank-separated paragraphs), this is a no-op.
    """
    out: list[str] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            joined = re.sub(r"[ \t]+", " ", " ".join(para)).strip()
            out.extend(_resplit_long(joined, report))
            para.clear()

    for line in lines:
        s = line.strip()
        if not s:
            flush()
            continue
        if _is_structural_line(s):
            flush()
            out.extend(_resplit_long(s, report))
            continue
        para.append(s)
    flush()
    return out


# End-of-document signature anchor (Loi 2016-48): the president's signature
# closes law 48. Everything AFTER it in the file is the start of ANOTHER law
# printed on the same gazette pages (49-2016, the Raiffeisen loan — foreign
# contamination documented in the 1.2 audit). That tail is extracted in a
# different, quasi-logical order and is left in stored form.
_RECONSTRUCT_TAIL_STOP = ("السبسي",)


# --- Adopted corrected codices (owner decisions 2026-09-30) ----------------
# The stored extractions carry documented corruption. The algorithmic zone
# reconstruction below is comparison material only (owner 2026-09-29: it stays
# too rigid); the operative repair is the language-model track — one repair
# log per document in audits/ (Loi: 215 entries / 29 batches; Circulaire: 35;
# Guide: 23; Madkhal: 21), each machine-checked by audits/llm_repair_check.py
# (verbatim source quotes, documented conservation, full line coverage) —
# whose output codex audits/*_corrected.md the owner ADOPTED on 2026-09-30
# (Loi: «لا اعتراض، واصل»; the three others: «طيب واصل» after the packages
# were presented at the adoption gate). With RTL repair enabled, a document's
# codex law text replaces its stored extraction for stage 1: the dominance
# gate below then sees a logical-order document and the zone reconstruction
# stays a no-op. The source files in docs/ are never modified; the audited
# codex files are the single source of truth for the adopted text.
_ADOPTED_CODEX = {
    "Loi_2016-48.pdf": "audits/Loi_2016-48_corrected.md",
    "Circulaire_BCT_2019-08.pdf": "audits/Circulaire_BCT_2019-08_corrected.md",
    "Guide_Interne_Operations_Bancaires_Islamiques.docx":
        "audits/Guide_Interne_Operations_Bancaires_Islamiques_corrected.md",
    "Madkhal_Sayrafa_Islamiya.docx": "audits/Madkhal_Sayrafa_Islamiya_corrected.md",
}


def _adopted_codex_text(name: str) -> str | None:
    """Law text of the adopted corrected codex, loader-canonical, or None.

    Form guarantees, so the adopted text behaves exactly like loader output:
    - the codex's metadata header (everything above its single '---' rule)
      is stripped;
    - the law text runs through the loader's own normalization for that file
      type (normalize_text(reflow=False) for PDFs — physical rendered lines;
      reflow=True for DOCX — paragraph reflow) plus normalize_arabic and a
      per-line strip (tatweel removal can expose new edge spaces). The
      Circulaire/Loi codices carry the PDF's line runs; the Guide/Madkhal
      codices are line-for-line mirrors of the stored text INCLUDING its
      blank-line block structure;
    - block boundaries are restored around structural lines. The PDF codices
      keep the stored extraction's line layout, which has no blank lines
      inside articles, so paragraph reflow glues a chapter subtitle onto the
      next article's head line; the marker then sits mid-sentence, reads as
      a cross-reference, and the article loses its heading (31 of 198 article
      heads lost that way in the stored Loi arm, still 20 with the raw
      codex). Every marker-initial line in these codices is a true heading
      (Loi: 198 فصل + 33 عنوان/باب/قسم; Circulaire: 20 فصل + 4 عنوان —
      scan-verified), and a marker at line start was already a section start
      for _extract_markers, so restoring the boundary promotes no
      cross-reference. The first line (the document's own title block) gets
      its own boundary for the same reason: it is a block in the printed
      document, and _pick_title only sees lines.
    """
    rel = _ADOPTED_CODEX.get(name)
    if not rel:
        return None
    path = Path(__file__).resolve().parent / rel
    if not path.is_file():
        print(f"[restructure] WARNING: adopted codex not found: {path} "
              "(falling back to the stored extraction)")
        return None
    raw = path.read_text(encoding="utf-8")
    if "\n---\n" not in raw:
        print("[restructure] WARNING: adopted codex has no header rule: "
              f"{path}")
        return None
    law = raw.split("\n---\n", 1)[1].lstrip("\n")
    reflow = not name.lower().endswith(".pdf")
    text = normalize_arabic(normalize_text(law, reflow=reflow))
    out: list[str] = []
    for i, line in enumerate(text.split("\n")):
        s = line.strip()
        if s and out and out[-1].strip() and (i == 1 or _MARKER_RE.match(s)):
            out.append("")          # restore the block boundary
        out.append(s)
    return "\n".join(out)


def _prepare_lines(text: str, report: RestructureReport,
                   reconstruct: bool = False) -> tuple[list[str], set[str]]:
    """Pre-pass over the raw normalized extraction, in dependency order:

    1. strip "[page N]" markers (and bare page-number remainders),
    2. drop official-gazette running headers (whole lines or fused prefixes),
    3. un-reverse each physical rendered line (zone reconstruction) when the
       document passed the visual-order dominance gate,
    4. re-form paragraphs from the rendered lines, resplitting long ones,
    5. drop other repeated short lines (running headers/footers).

    Returns (lines, repeated_keys) — the keys of dropped repeated groups,
    used later to keep garbage out of the title pick.
    """
    lines: list[str] = []
    for raw in text.split("\n"):
        m = _PAGE_MARKER_RE.match(raw.strip())
        line = raw.strip()
        if m:
            report.pages_removed += 1
            rest = line[m.end():].strip()
            line = rest if rest and not rest.isdigit() else ""
            if not line:
                lines.append("")      # page break = paragraph boundary
                continue
        line = _strip_gazette_header(line, report)
        if not line:
            lines.append("")
            continue
        if reconstruct:
            line = _reconstruct_line_visual_order(line, report)
            if any(marker in line for marker in _RECONSTRUCT_TAIL_STOP):
                # signature reached: the rest of the file is another law's
                # text (different extraction order) — keep it as stored
                reconstruct = False
        lines.append(line)
    lines = _reflow_lines(lines, report)
    lines, repeated_keys = _drop_repeated_headers(lines, report)
    return lines, repeated_keys


def _normalize_bullets(line: str, report: RestructureReport) -> str:
    """Standardize list markers: en-dash bullets and "N/" rules become "- N/ "."""
    m = re.match(r"^\u2013\s+(\S.*)$", line)          # "– item"
    if m:
        report.list_items_normalized += 1
        return f"- {m.group(1)}"
    m = re.match(r"^(\d{1,2})/\s*(\S.*)$", line)       # "1/ rule" numbered rules
    if m:
        report.list_items_normalized += 1
        return f"- {m.group(1)}/ {m.group(2)}"
    return line


def _is_section_start(rest: str, start: int) -> bool:
    """A marker starts a new section only at the head of the text or right
    after a sentence end. Mid-sentence markers are cross-references
    ("طبقا لاحكام الفصل 43 من هذا القانون") and must stay in the body —
    over-splitting corrupts context worse than under-splitting does."""
    if start == 0:
        return True
    prev = rest[:start].rstrip()
    if not prev:
        return True
    # ":" counts: "…قرر ما يلي : الفصل الاول : …" — the colon introduces the
    # section, it does not close the previous sentence.
    return prev[-1] in ".؟؛۔:"


def _extract_markers(line: str, report: RestructureReport) -> list[str]:
    """Split one body line at legal section markers ("الفصل 4", "العنوان
    الثاني", fused visual-order "الثانيباب").

    Returns the list of (segment, is_new_section_start) pieces as plain
    strings: a leading "!" prefix marks a new section. Markers found mid-line
    are exactly the implicit hierarchy the raw extraction buried in the text.
    """
    pieces: list[str] = []
    content = ""   # body text seen so far (may embed cross-references)
    rest = line
    while True:
        m = _MARKER_RE.search(rest)
        fm = _FUSED_MARKER_RE.search(rest)
        sff = _SPACED_FUSED_MARKER_RE.search(rest)
        candidates = [c for c in (m, fm, sff) if c]
        if not candidates:
            seg = (content + rest).strip()
            if seg:
                pieces.append(seg)
            break
        hit = min(candidates, key=lambda c: c.start())
        if not _is_section_start(rest, hit.start()):
            # Cross-reference: the marker stays body text; keep scanning.
            content += rest[:hit.end()]
            rest = rest[hit.end():]
            continue
        seg = (content + rest[:hit.start()]).strip()
        content = ""
        if seg:
            pieces.append(seg)
        if hit is fm or hit is sff:
            # Visual order (glued or spaced): normalize to logical "marker ordinal".
            parts = hit.group(1).split()
            ordinal = parts[0]
            marker = parts[-1]
            marker_text = f"{marker} {ordinal}"
        else:
            marker_text = hit.group(1)
        # The "!" piece carries ONLY the marker: the following text is the
        # new section's body and must be emitted exactly once (as the next
        # content piece), not glued here as well.
        pieces.append(f"!{marker_text}")
        after = rest[hit.end():]
        after = after.lstrip(" \u00a0")
        if after.startswith(":"):
            after = after[1:].lstrip()
        rest = after  # keep scanning: one line may carry several markers
    return pieces


def _pick_title(lines: list[str], repeated_keys: set[str]) -> str:
    """The document's own title, best effort.

    An explicit "# " line wins. Otherwise the first title-like line: 12-70
    characters, not a religious preamble, not a table row, not a legal
    marker line, not a bullet, and not part of a repeated (running header)
    group. When nothing qualifies, the first meaningful line is truncated
    to 70 characters.
    """
    for l in lines:
        s = l.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()
    fallback = ""
    for l in lines:
        s = l.strip()
        if not s or s.startswith(("#", "|", "- ")):
            continue
        if re.match(r"^\d{1,2}[.)]\s", s):  # numbered list item
            continue
        if _TITLE_SKIP_RE.search(s):
            continue
        if (_MARKER_RE.search(s) or _FUSED_MARKER_RE.search(s)
                or _SPACED_FUSED_MARKER_RE.search(s)):
            continue
        if re.sub(r"\d+", "#", s) in repeated_keys:
            continue
        if not fallback:
            fallback = s
        if 12 <= len(s) <= 85:
            return s
    return fallback[:85]


def _short_title(remainder: str) -> str | None:
    """If the text after a marker is a short title (not a full sentence),
    keep it with the heading. "احكام عامة" -> title; "تنطبق احكام ..." -> no."""
    r = remainder.strip()
    if not r or len(r) > 50:
        return None
    if re.search(r"[.؟؟؛]", r[:15]):
        return None
    # A sentence of 8+ words is content, not a title.
    if len(r.split()) > 8:
        return None
    return r


def normalize_structure(doc: dict, repair_rtl: bool = True) -> tuple[str, RestructureReport]:
    """Stage 1: messy extraction -> clean hierarchical Markdown.

    Returns (markdown_text, report). The markdown starts with an H1 document
    title (invented from the first meaningful line when the source had none)
    and uses ## for books/chapters/numbered sections and ### for articles.
    """
    report = RestructureReport(name=doc.get("name", "?"))
    text = doc.get("text", "")

    # Owner-adopted corrected codex (2026-09-30): with RTL repair enabled and
    # a vetted codex for this document, the codex's law text REPLACES the
    # stored extraction for stage 1. The dominance gate below then measures a
    # logical-order document and leaves every line unreversed.
    if repair_rtl:
        adopted = _adopted_codex_text(doc.get("name", ""))
        if adopted is not None:
            text = adopted
            report.codex_adopted = True

    # Document-level gate for the RTL zone reconstruction: is THIS document
    # extracted in visual order? Sample every scorable rendered line and
    # compare its curated-bigram score before vs after zone reconstruction.
    # Only a document that wins this comparison overwhelmingly gets its
    # lines un-reversed; everything else is left byte-identical.
    dominant = False
    if repair_rtl:
        wins_recon = wins_raw = 0
        for raw in text.split("\n"):
            s = raw.strip()
            if (len(s) < RTL_MIN_LEN or _arabic_share(s) < RTL_ARABIC_SHARE
                    or len(s.split()) < 4 or s.startswith("|")):
                continue
            recon = _reconstruct_line_visual_order(s)
            o = _order_score(s.split())
            r = _order_score(recon.split())
            if r >= o + RTL_MARGIN:
                wins_recon += 1
            elif o >= r + RTL_MARGIN:
                wins_raw += 1
        dominant = (wins_recon >= RTL_DOMINANT_MIN
                    and wins_recon >= RTL_DOMINANT_RATIO * max(wins_raw, 1))
    report.rtl_doc_dominant = dominant

    lines, repeated_keys = _prepare_lines(text, report, reconstruct=dominant)

    out: list[str] = []
    doc_title = _pick_title(lines, repeated_keys)
    h1_seen = False
    h1_emitted = False
    in_table = False
    table_rows: list[str] = []

    def flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            in_table = False
            return
        # Ensure a Markdown separator row under the header.
        rows = [r.strip() for r in table_rows]
        if len(rows) >= 2 and not re.fullmatch(r"[\s|:-]+", rows[1] or "|"):
            n_cols = len(rows[0].strip("|").split("|"))
            rows.insert(1, "| " + " | ".join(["---"] * n_cols) + " |")
        # A table whose header row is repeated verbatim inside it (DOCX
        # extraction repeats the header on long tables): dedupe the repeats.
        deduped = []
        for r in rows:
            if r == rows[0] and len(deduped) > 1 and not r == rows[1]:
                continue
            deduped.append(r)
        out.extend(deduped)
        out.append("")
        report.tables_normalized += 1
        in_table = False
        table_rows = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # -- tables: keep rows contiguous, fix the separator row. -------------
        if line.startswith("|"):
            if not in_table:
                flush_table()
                in_table = True
            table_rows.append(line)
            continue
        flush_table()

        if not line:
            out.append("")
            continue

        # -- H1 once: the document's own title (also feeds the context
        #    breadcrumb in stage 2 and the fingerprint in stage 3). ----------
        if not h1_emitted:
            if doc_title:
                if line.startswith("# ") and line.lstrip("#").strip() == doc_title:
                    # The line below IS the H1: let the pass-through emit it
                    # exactly once.
                    h1_emitted = True
                else:
                    out.append(f"# {doc_title}")
                    h1_emitted = True
                    # The picked title line itself is now the H1: skip it in
                    # the body so the title text is not duplicated.
                    if doc_title and line == doc_title and not h1_seen:
                        h1_seen = True
                        continue

        # -- existing markdown headings pass through (levels preserved). ------
        if line.startswith("#"):
            if line.startswith("# "):
                if h1_emitted and line.lstrip("#").strip() != doc_title:
                    # A document has one H1: demote any later "# " line.
                    out.append("## " + line.lstrip("#").strip())
                    out.append("")
                    continue
                doc_title = doc_title or line.lstrip("#").strip()
                if not h1_seen:
                    h1_seen = True
                    h1_emitted = True  # the document carries its own H1
            h1_seen = True
            out.append(line)
            out.append("")
            continue

        # -- RTL repair already happened per physical rendered line in
        # _prepare_lines (zone reconstruction under the document-dominance
        # gate: digit islands kept, Arabic runs un-reversed, markers kept at
        # line start). Here we only split the — now logical-order — line on
        # its structural markers.
        pieces = _extract_markers(line, report)
        line = _normalize_bullets(line, report)
        for piece in pieces:
            if piece.startswith("!"):
                # The piece carries only the marker ("الفصل 2", "الفصل4",
                # "العنوان الاول", ...); the section's body follows as its
                # own piece.
                marker_full = piece[1:].strip()
                mw = _MARKER_WORD_RE.match(marker_full)
                marker_word = mw.group(1) if mw else "الفصل"
                if marker_word == "الفصل":
                    heading_line, level = f"### {marker_full}", 3
                else:  # الباب / العنوان / القسم
                    heading_line, level = f"## {marker_full}", 2
                report.markers_extracted += 1
                report.headings.append((level, marker_full))
                out.append(heading_line)
                out.append("")
                continue
            else:
                piece = piece.strip()
                if not piece:
                    continue
                # Numbered top-level sections ("2- title") and decimal
                # subsections ("1.2- title") of the DOCX guides.
                m = re.match(r"^(\d+)\.(\d+)-\s*(.+)$", piece)
                if m:
                    out.append(f"### {m.group(1)}.{m.group(2)}- {m.group(3)}")
                    out.append("")
                    report.headings_invented += 1
                    report.headings.append((3, f"{m.group(1)}.{m.group(2)}- {m.group(3)}"))
                    continue
                m = re.match(r"^(\d+)-\s*(.+)$", piece)
                if m:
                    out.append(f"## {m.group(1)}- {m.group(2)}")
                    out.append("")
                    report.headings_invented += 1
                    report.headings.append((2, f"{m.group(1)}- {m.group(2)}"))
                    continue
                m = re.match(r"^([أ-ي])-\s*(.+)$", piece)
                if m:
                    out.append(f"### {m.group(1)}- {m.group(2)}")
                    out.append("")
                    report.headings_invented += 1
                    report.headings.append((3, f"{m.group(1)}- {m.group(2)}"))
                    continue
                if piece in ("توطئة", "خاتمة", "الفهرس", "المقدمة"):
                    out.append(f"## {piece}")
                    out.append("")
                    report.headings_invented += 1
                    report.headings.append((2, piece))
                    continue
                out.append(piece)
                out.append("")

    flush_table()

    # Collapse blank runs; no leading/trailing blanks.
    clean: list[str] = []
    blank = False
    for l in out:
        if l == "":
            if not blank:
                clean.append("")
            blank = True
        else:
            clean.append(l)
            blank = False
    while clean and clean[0] == "":
        clean.pop(0)
    while clean and clean[-1] == "":
        clean.pop()

    report.doc_title = doc_title
    report.print()
    return "\n".join(clean), report


# ---------------------------------------------------------------------------
# Stage 2 — context enrichment
# ---------------------------------------------------------------------------

CTX_PREFIX = "> **Context:** "


def enrich_context(markdown: str, doc_title: str) -> str:
    """Inject breadcrumb context lines above H2/H3 headings and standalone
    tables, so every section and table carries its own location marker."""
    lines = markdown.split("\n")
    out: list[str] = []
    h1, h2 = doc_title, ""
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("### "):
            h3 = stripped[4:].strip()
            ctx = CTX_PREFIX + " > ".join(p for p in (h1, h2, h3) if p)
            out.append(ctx)
            out.append(line)
            i += 1
            continue
        if stripped.startswith("## "):
            h2 = stripped[3:].strip()
            ctx = CTX_PREFIX + " > ".join(p for p in (h1, h2) if p)
            out.append(ctx)
            out.append(line)
            i += 1
            continue
        if stripped.startswith("# "):
            h1 = stripped[2:].strip()
            h2 = ""
            out.append(line)
            i += 1
            continue
        # Table not already under its own heading/context: add the breadcrumb.
        if stripped.startswith("|"):
            prev = out[-2] if len(out) >= 2 else (out[-1] if out else "")
            if (not prev.strip().startswith("##")
                    and not prev.strip().startswith(CTX_PREFIX)
                    and (h1 or h2)):
                ctx = CTX_PREFIX + " > ".join(p for p in (h1, h2) if p)
                out.append(ctx)
            out.append(line)
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Stage 3 — recursive structural chunking
# ---------------------------------------------------------------------------

# The hierarchy-aware separator list (from most to least structural).
STRUCTURAL_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "]


def _split_on(text: str, sep: str) -> list[str]:
    """Split keeping the separator attached to the part that follows it
    (so a re-join is lossless and chunks keep their heading markers)."""
    if sep == " ":
        parts = text.split(" ")
        if len(parts) <= 1:
            return [text]
        return [parts[0]] + [" " + p for p in parts[1:]]
    parts = text.split(sep)
    if len(parts) <= 1:
        return [text]
    return [parts[0]] + [sep + p for p in parts[1:]]


def _word_split(text: str, budget: int) -> list[str]:
    """Last resort: word-boundary pieces, each within the budget."""
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word]) if current else word
        if current and count_tokens(candidate) > budget:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _glue_context_lines(pieces: list[str]) -> list[str]:
    """A context line ("> **Context:** …") belongs to the heading that
    follows it: when a split leaves a context line at the head of a piece,
    re-attach it to the piece of the heading it describes, so the breadcrumb
    can never be chunked away from its section."""
    out: list[str] = []
    for p in pieces:
        if p.lstrip().startswith(CTX_PREFIX) and out:
            out[-1] = out[-1].rstrip() + "\n" + p.strip()
        else:
            out.append(p)
    return out


def _recursive_split(text: str, seps: list[str], sep_index: int,
                     budget: int) -> list[str]:
    """LangChain-style recursive splitting over a token budget.

    A piece that fits the budget is kept WHOLE (structure preserved); a
    piece that does not is re-split with the next, finer separator. Adjacent
    small pieces are merged up to the budget so the result is a list of
    maximal structural units, not a flat bag of fragments. Context lines are
    kept glued to the heading they describe.
    """
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= budget:
        return _glue_context_lines([text])
    if sep_index >= len(seps):
        return _word_split(text, budget)

    sep = seps[sep_index]
    parts = _split_on(text, sep)
    if len(parts) <= 1:
        return _recursive_split(text, seps, sep_index + 1, budget)

    result: list[str] = []
    merged: list[str] = []
    merged_tokens = 0

    def flush_merged():
        nonlocal merged, merged_tokens
        if merged:
            result.append("".join(merged))
            merged = []
            merged_tokens = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue
        part_tokens = count_tokens(part)
        if part_tokens > budget:
            flush_merged()
            result.extend(_recursive_split(part, seps, sep_index + 1, budget))
        else:
            if merged and merged_tokens + part_tokens > budget:
                flush_merged()
            merged.append(part if not merged else "\n\n" + part)
            merged_tokens += part_tokens
    flush_merged()
    return _glue_context_lines(result)


def _context_stack_for(line: str) -> tuple[int, str]:
    """(level, text) if line is a heading, else (0, '')."""
    m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
    if m:
        return len(m.group(1)), m.group(2).strip()
    return 0, ""


def _nearest_heading_and_context(chunk_text: str, h1: str, h2: str, h3: str,
                                 doc_title: str) -> tuple[str, str]:
    """Walk a chunk's own lines; the last real heading inside it is its
    anchor; if the chunk has none, it is a continuation of the section the
    chunker is currently in (h1/h2/h3 passed in)."""
    heading = ""
    for line in chunk_text.split("\n"):
        level, text = _context_stack_for(line)
        if level:
            heading = text
    if not heading:
        heading = h3 or h2 or h1 or doc_title
    crumbs = [p for p in (doc_title, h1, h2, h3) if p and p != heading]
    deduped = [c for i, c in enumerate(crumbs) if not (i and c == crumbs[i - 1])]
    context = " > ".join(deduped)
    return heading, context


def recursive_structural_chunk(enriched: str, doc: dict, budget: int,
                               overlap: int = 0) -> list[Chunk]:
    """Stage 3: split enriched markdown into self-contained chunks.

    budget is the per-chunk token budget (heading+context included, counted
    with the same tokenizer as the rest of the lab); overlap re-anchors
    continuation chunks with the trailing words of the previous chunk.
    """
    pieces = _recursive_split(enriched, STRUCTURAL_SEPARATORS, 0, budget)

    chunks: list[Chunk] = []
    doc_title = ""
    h1 = h2 = h3 = ""
    prev_text: str | None = None

    for piece in pieces:
        # Track the heading stack from the piece's own headings so a
        # continuation piece inherits the section it sits in.
        for line in piece.split("\n"):
            level, text = _context_stack_for(line)
            if level == 1:
                doc_title = doc_title or text
                h1, h2, h3 = text, "", ""
            elif level == 2:
                h2, h3 = text, ""
            elif level == 3:
                h3 = text

        heading, context = _nearest_heading_and_context(
            piece, h1, h2, h3, doc_title or doc.get("name", ""))

        body = piece.strip()
        # Drop an embedded context line, then re-add exactly one of our own
        # so every chunk carries its breadcrumb exactly once.
        body_lines = [l for l in body.split("\n")
                      if not l.strip().startswith(CTX_PREFIX)]
        body = "\n".join(body_lines).strip()

        overlap_note = []
        if overlap > 0 and prev_text and context:
            # Take the trailing `overlap` WORDS of the previous chunk but
            # keep their original whitespace/newlines, so a heading at the
            # start of this chunk's body is not glued mid-line.
            segs = re.split(r"(\s+)", prev_text)
            word_idx = target_pos = 0
            n_words = sum(1 for s in segs if s.strip())
            k = max(0, n_words - overlap)
            for i, s in enumerate(segs):
                if s.strip():
                    if word_idx == k:
                        target_pos = sum(len(x) for x in segs[:i])
                        break
                    word_idx += 1
            tail = prev_text[target_pos:].strip()
            if tail:
                body = tail + "\n" + body
                overlap_note = [f"continuation: {len(tail.split())} trailing "
                                f"words of previous chunk prepended"]

        if context:
            final_text = f"{CTX_PREFIX}{context}\n{body}"
        else:
            final_text = body

        section_index = len([c for c in chunks if c.source == doc.get("name")])
        chunks.append(Chunk(
            index=len(chunks),
            text=final_text,
            heading=heading,
            language=doc.get("language", "unknown"),
            source=doc.get("source", doc.get("name", "?")),
            token_count=count_tokens(final_text),
            origin=doc.get("origin", "data/"),
            section_type=classify_section(f"{'# ' if section_index == 0 else ''}{heading}",
                                          section_index),
            notes=overlap_note,
        ))
        prev_text = body
    return chunks


# ---------------------------------------------------------------------------
# Entry point used by chunker.chunk_all
# ---------------------------------------------------------------------------

def chunk_documents(docs: list[dict], cfg) -> list[Chunk]:
    """Restructure-chunk every document with config.py parameters.

    Stage 1 normalizes each document, stage 2 enriches context, stage 3
    applies the recursive structural splitter with the configured token
    budget (CHUNK_SIZE_TOKENS) and overlap (CHUNK_OVERLAP_TOKENS).
    """
    budget = cfg.CHUNK_SIZE_TOKENS
    overlap = cfg.CHUNK_OVERLAP_TOKENS
    repair_rtl = str(getattr(cfg, "RESTRUCTURE_RTL_REPAIR", "1")).strip() \
        not in {"0", "false", "no", "off"}
    print(f"[restructure] strategy=restructure | budget={budget} tokens "
          f"overlap={overlap} | rtl_repair={'on' if repair_rtl else 'off'}")
    all_chunks: list[Chunk] = []
    for doc in docs:
        markdown, _report = normalize_structure(doc, repair_rtl=repair_rtl)
        title = _report.doc_title or doc.get("name", "")
        enriched = enrich_context(markdown, title)
        doc_chunks = recursive_structural_chunk(enriched, doc, budget, overlap)
        print(f"[restructure]   {doc.get('name')}: {len(doc_chunks)} chunk(s), "
              f"{sum(c.token_count for c in doc_chunks)} tokens total")
        all_chunks.extend(doc_chunks)
    print(f"[restructure] total: {len(all_chunks)} chunk(s) across "
          f"{len(docs)} document(s)")
    return all_chunks
