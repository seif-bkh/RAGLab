"""chunker.py — one plain function that slices a document into chunks.

Design goals:
- split on headings first, then on paragraph boundaries, and only fall back
  to hard token-size cuts (never cutting mid-sentence if avoidable);
- prepend the nearest heading to every chunk so no chunk loses its context;
- turn Markdown table rows into full sentences (no flattened cells).

Token counting uses tiktoken ("cl100k_base"), the same family used by
text-embedding-3-large. The token budget applies to each chunk's body plus
its heading. Because overlap text is *reused* (appended to the next chunk,
not re-counted against its budget), a chunk may exceed CHUNK_SIZE_TOKENS by
up to CHUNK_OVERLAP_TOKENS — that is by design, so no context is lost between
chunks. Single paragraphs longer than the budget are hard-split at word
boundaries, with a note attached to the chunks.
"""

import re
from dataclasses import dataclass, field

import tiktoken

# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

_TOKENIZER = None
_TOKENIZER_WARNED = False


def count_tokens(text: str) -> int:
    """Token count for one text.

    Primary: tiktoken "cl100k_base" (matches text-embedding-3-large). The BPE
    file is downloaded by tiktoken on first use (cached under TIKTOKEN_CACHE_DIR);
    if that download is impossible (offline / blocked host), we fall back to a
    transparent estimator: max(word_count, ceil(chars/4)). The estimator is
    marked everywhere it is used (warning printed once).
    """
    global _TOKENIZER, _TOKENIZER_WARNED
    if _TOKENIZER is None and not _TOKENIZER_WARNED:
        try:
            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # noqa: BLE001 — any failure to fetch BPE data
            _TOKENIZER_WARNED = True
            print(f"[chunker] WARNING: tiktoken BPE file unavailable "
                  f"({type(exc).__name__}); using fallback token estimator "
                  f"max(words, chars/4). See README for TIKTOKEN_CACHE_DIR.")
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text, disallowed_special=()))
    words = len(text.split())
    approx = -(-len(text) // 4)  # ceil(chars / 4)
    return max(words, approx)


# Sentence boundaries for the sentence-aware overlap (covers Latin + Arabic).
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…؟;؛۔])\s+")

# Headings whose content is boilerplate (legal notice / terms / disclaimer),
# not retrievable facts. Language-agnostic keyword list; "content" sections
# are the only ones indexed when INDEX_EXCLUDE_BOILERPLATE is on.
# NOTE: "conditions/شروط" alone is NOT a boilerplate marker: "Conditions
# d'éligibilité" / "شروط الأهلية" are real product facts and must stay
# content. Only general legal terms/disclaimers are matched.
BOILERPLATE_HEADING_KEYWORDS = (
    "conditions générales", "conditions generales", "mentions", "avertissement",
    "avis légal", "avis legal", "terms", "notice", "disclaimer", "legal",
    # Arabic-normalized forms (أ/إ/آ -> ا); "شروط" alone is excluded on purpose.
    "الشروط العامة", "ملاحظات", "قانوني", "إخلاء", "مسؤولية", "اعلان", "عامة",
)


def split_sentences(text: str) -> list[str]:
    """Split text into whole sentences, whitespace-reflowed, fragments dropped."""
    parts = [re.sub(r"\s+", " ", s).strip()
             for s in SENTENCE_BOUNDARY_RE.split(text)]
    return [p for p in parts if p]


# Table column keywords, per purpose, language-agnostic on purpose.
FEE_COLUMN_KEYWORDS = ("frais mensuel", "frais de tenue", "tenue de compte",
                       "monthly fee", "الرسوم الشهرية", "رسوم", "شهرية")
# NOTE: chunk text is Arabic-normalized (أ/إ/آ/ٱ -> ا), so the Arabic keywords
# below use normalized forms: "اعفاء" (not "إعفاء"), "الرسوم الشهرية", "معفى".
WAIVER_COLUMN_KEYWORDS = ("exonération", "exoneration", "exonéré", "exonere",
                          "waived", "waiver", "اعفاء", "معفى")


@dataclass
class Chunk:
    """One chunk of one document. Plain data, everything printable."""
    index: int
    text: str
    heading: str
    language: str
    source: str
    token_count: int = 0
    origin: str = "data/"
    section_type: str = "content"  # "content" | "front-matter" | "legal"
    notes: list = field(default_factory=list)  # e.g. "long sentence hard-cut"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "heading": self.heading,
            "language": self.language,
            "source": self.source,
            "token_count": self.token_count,
            "origin": self.origin,
            "section_type": self.section_type,
            "notes": self.notes,
        }


def _heading_level(line: str) -> int:
    return len(line) - len(line.lstrip("#"))


def classify_section(heading: str, section_index: int) -> str:
    """Classify a section as "content", "front-matter" or "legal".

    - front-matter: the first H1 section (document title + preamble boilerplate).
    - legal: headings mentioning terms, conditions, notices, disclaimers
      (matched per-language via BOILERPLATE_HEADING_KEYWORDS).
    - content: everything else — the sections that actually answer questions.
    """
    if heading.startswith("# ") and section_index == 0:
        return "front-matter"
    low = heading.casefold()
    if any(k in low for k in BOILERPLATE_HEADING_KEYWORDS):
        return "legal"
    return "content"


def _split_sections(text: str):
    """Split the normalized text into (heading, body) sections.

    A section starts at a heading line (any level) and runs to the next
    heading. Text before the first heading belongs to heading "".
    """
    sections = []
    current_heading = ""
    current_body = []

    def flush():
        nonlocal current_body
        body = "\n".join(current_body).strip()
        if body:
            sections.append((current_heading, body))
        current_body = []

    for line in text.split("\n"):
        if line.lstrip().startswith("#"):
            flush()
            current_heading = line.lstrip().strip()
        else:
            current_body.append(line)
    flush()
    return sections


def _parse_table(lines: list) -> list[list[str]]:
    """Parse contiguous '|' lines into rows of stripped cells.

    Drops the Markdown separator row (--- | ---) if present.
    """
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) >= 2 and all(re.fullmatch(r":?-{2,}:?", c) for c in rows[1]):
        del rows[1]
    return rows


def _find_column(header: list, keywords) -> int:
    """Index of the first header cell containing any keyword, else -1."""
    for i, cell in enumerate(header):
        low = cell.lower()
        if any(k in low for k in keywords):
            return i
    return -1


# ---------------------------------------------------------------------------
# Table rows -> full sentences
# ---------------------------------------------------------------------------

def convert_table_rows(rows: list[list[str]], language: str) -> list[str]:
    """Turn a parsed table into one full sentence per data row.

    The generic fee-table shape is: Product | monthly fee | waiver condition.
    The sentence follows the requested pattern:
        "For the X account, the monthly fee is Y; it is waived if Z"
    in French, English or Arabic, with a graceful fallback for other tables.
    """
    if not rows:
        return []
    header, data_rows = rows[0], rows[1:] if len(rows) > 1 else []

    fee_col = _find_column(header, FEE_COLUMN_KEYWORDS)
    waiver_col = _find_column(header, WAIVER_COLUMN_KEYWORDS)

    sentences = []
    for row in data_rows:
        if fee_col != -1 and row:
            product = row[0] if row else "?"
            fee = row[fee_col] if fee_col < len(row) else "?"
            waiver = row[waiver_col] if waiver_col != -1 and waiver_col < len(row) else ""

            # "0,00 €" -> no fee at all.
            fee_numeric = fee.replace(" ", "").replace(",", ".").replace("€", "")
            free = fee_numeric in {"0", "0.0", "0.00", ""}

            if language == "ar":
                if free:
                    sentences.append(
                        f"بالنسبة لمنتج {product}، لا توجد رسوم شهرية (المبلغ المذكور: {fee})."
                    )
                elif waiver:
                    sentences.append(
                        f"بالنسبة لمنتج {product}، تبلغ الرسوم الشهرية {fee}؛ وتُعفى منها إذا {waiver}."
                    )
                else:
                    sentences.append(
                        f"بالنسبة لمنتج {product}، تبلغ الرسوم الشهرية {fee} دون أي شرط إعفاء."
                    )
            elif language == "en":
                if free:
                    sentences.append(
                        f"For the {product} account, there is no monthly fee "
                        f"(amount shown: {fee})."
                    )
                elif waiver:
                    sentences.append(
                        f"For the {product} account, the monthly fee is {fee}; "
                        f"it is waived if {waiver}."
                    )
                else:
                    sentences.append(
                        f"For the {product} account, the monthly fee is {fee}; "
                        f"there is no waiver condition."
                    )
            else:  # default French
                if free:
                    sentences.append(
                        f"Pour le produit {product}, il n'y a pas de frais mensuels "
                        f"(montant indiqué : {fee})."
                    )
                elif waiver:
                    sentences.append(
                        f"Pour le produit {product}, les frais mensuels s'élèvent à {fee} ; "
                        f"ils sont exonérés si {waiver}."
                    )
                else:
                    sentences.append(
                        f"Pour le produit {product}, les frais mensuels s'élèvent à {fee} ; "
                        f"aucune condition d'exonération n'est prévue."
                    )
            continue

        # Not the fee-table shape: fall back to a spelled-out row.
        pairs = " ; ".join(f"{h}: {v}" for h, v in zip(header, row) if h and v)
        if language == "ar":
            sentences.append(f"صف جدول ({' | '.join(header)}) : {pairs}.")
        elif language == "en":
            sentences.append(f"Table row ({' | '.join(header)}): {pairs}.")
        else:
            sentences.append(f"Ligne du tableau ({' | '.join(header)}) : {pairs}.")

    return sentences


def _paragraphs_and_tables(section_body: str) -> list[str]:
    """Split a section body into paragraphs, converting table blocks to sentences.

    Consecutive lines starting with '|' become one table block; everything else
    is split on blank lines into plain paragraphs.
    """
    pieces: list[str] = []
    paragraph: list[str] = []
    table_lines: list[str] = []

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            pieces.append(" ".join(paragraph).strip())
            paragraph = []

    def flush_table():
        nonlocal table_lines
        if table_lines:
            rows = _parse_table(table_lines)
            # The language is attached later; rows are converted with a generic
            # marker here and re-rendered language-aware in chunk_document.
            pieces.append(("__TABLE__", rows))
            table_lines = []

    for line in section_body.split("\n"):
        if line.strip().startswith("|"):
            flush_paragraph()
            table_lines.append(line)
        elif line.strip():
            flush_table()
            paragraph.append(line)
        else:
            flush_paragraph()
            flush_table()
    flush_paragraph()
    flush_table()
    return pieces


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def chunk_document(doc: dict,
                   chunk_size: int = 220,
                   overlap: int = 40,
                   split_on_headings: bool = True,
                   sentence_aware_overlap: bool = True) -> list[Chunk]:
    """Chunk one document dict (from loader.py) into a list of Chunk objects.

    Parameters mirror config.py: chunk_size (token budget), overlap (tokens
    shared between consecutive chunks), split_on_headings (True = sections
    first), sentence_aware_overlap (True = overlap only at sentence
    boundaries). Returns an empty list for an empty document.
    """
    chunks: list[Chunk] = []
    text = doc.get("text", "").strip()
    if not text:
        print("[chunker] WARNING: empty document, nothing to chunk:", doc.get("name"))
        return chunks

    print(f"[chunker] chunking {doc['name']} | size={chunk_size} overlap={overlap} "
          f"split_headings={split_on_headings} "
          f"sentence_overlap={sentence_aware_overlap} | tokens={count_tokens(text)}")

    sections = _split_sections(text) if split_on_headings else [("", text)]

    for section_index, (heading, body) in enumerate(sections):
        heading_text = heading if heading else ""
        section_type = classify_section(heading_text, section_index)
        if section_type != "content":
            print(f"[chunker]   section_type={section_type} "
                  f"(heading={heading_text[:60]!r})")
        pieces = _paragraphs_and_tables(body)
        # Convert tables to sentences per document language (marker tuple).
        paragraphs: list[str] = []
        for piece in pieces:
            if isinstance(piece, tuple) and piece[0] == "__TABLE__":
                paragraphs.extend(convert_table_rows(piece[1], doc["language"]))
            else:
                paragraphs.append(piece)

        # A heading costs tokens too; keep the final chunk within the budget.
        heading_tokens = count_tokens(heading_text) if heading_text else 0
        budget = chunk_size - heading_tokens
        if budget < 1:
            print(f"[chunker] WARNING: heading alone exceeds chunk_size "
                  f"({heading_text[:80]!r}); chunk may exceed the budget.")
            budget = max(1, chunk_size // 2)

        body_chunks = _pack_paragraphs(paragraphs, budget, overlap,
                                       sentence_aware=sentence_aware_overlap)
        for body_tokens, note in body_chunks:
            if heading_text:
                chunk_text = f"{heading_text}\n\n{body_tokens['text']}"
            else:
                chunk_text = body_tokens["text"]
            chunks.append(Chunk(
                index=len(chunks),
                text=chunk_text,
                heading=heading_text,
                language=doc["language"],
                source=doc["source"],
                token_count=count_tokens(chunk_text),
                origin=doc.get("origin", "data/"),
                section_type=section_type,
                notes=note,
            ))

    total_tokens = sum(c.token_count for c in chunks)
    print(f"[chunker]   produced {len(chunks)} chunk(s), {total_tokens} tokens total")
    return chunks


def _pack_paragraphs(paragraphs: list[str], budget: int, overlap: int,
                     sentence_aware: bool = True):
    """Group paragraphs into token-bounded bodies, with sentence-aware overlap.

    Returns a list of ({"text": ..., "tokens": int}, notes:list[str]).
    Falls back to word-boundary hard splitting only when a single paragraph
    exceeds the budget (marked with a note). Overlap text is whole sentences
    when sentence_aware=True, so chunks never open with a fragment.
    """
    if not paragraphs:
        return []

    result = []
    current_parts: list[str] = []
    current_tokens = 0
    current_note = []

    def flush(extra_parts=None, extra_tokens=0, extra_notes=None):
        nonlocal current_parts, current_tokens, current_note
        if current_parts:
            text = "\n\n".join(current_parts)
            result.append(({"text": text, "tokens": count_tokens(text)}, list(current_note)))
        if extra_parts:
            current_parts = list(extra_parts)
            current_tokens = extra_tokens
            current_note = list(extra_notes or [])
        else:
            current_parts = []
            current_tokens = 0
            current_note = []

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if para_tokens <= budget:
            if current_tokens + para_tokens > budget and current_parts:
                # Overlap: reuse whole trailing sentences of the previous chunk.
                tail_parts, tail_tokens, tail_notes = _overlap_parts(
                    current_parts, overlap, sentence_aware)
                flush(tail_parts, tail_tokens, tail_notes)
            current_parts.append(para)
            current_tokens += para_tokens
        else:
            # Single paragraph longer than the budget: hard split (word-level).
            # Every piece is flagged so `inspect` shows the cut explicitly.
            flush()
            for piece in _hard_split(para, budget):
                note = ["paragraph longer than budget: hard-split at word boundary"]
                result.append(({"text": piece, "tokens": count_tokens(piece)}, note))

    flush()
    return result


def _long_sentence(text: str) -> bool:
    return len(text) > 600  # heuristic: almost certainly one very long sentence


def _sentence_tail(text: str, max_tokens: int) -> tuple[list[str], bool]:
    """Return the trailing WHOLE sentences of `text` totalling ~max_tokens.

    Never cuts a sentence: if even the last sentence exceeds max_tokens it is
    still kept whole (returned with exceeded=True) — a complete clause beats
    a fragment of context. Empty result only when there is nothing to reuse.
    """
    if max_tokens <= 0:
        return [], False
    sentences = split_sentences(text)
    if not sentences:
        return [], False
    tail: list[str] = []
    tokens = 0
    for sentence in reversed(sentences):
        sentence_tokens = count_tokens(sentence)
        if tail and tokens + sentence_tokens > max_tokens:
            break
        tail.append(sentence)
        tokens += sentence_tokens
        if tokens >= max_tokens:
            break
    if not tail:
        # Single very long sentence: keep it whole anyway (context > budget).
        tail = [sentences[-1]]
        return tail, True
    return list(reversed(tail)), False


def _token_tail(parts: list[str], overlap: int) -> tuple[list[str], int]:
    """Fallback: last `overlap` tokens as whole words (only if
    CHUNK_OVERLAP_SENTENCE_AWARE is turned off)."""
    combined = " ".join(parts)
    tokens = combined.split()
    if not tokens or overlap <= 0:
        return [], 0
    tail = tokens[-overlap:]
    return [" ".join(tail)], count_tokens(" ".join(tail))


def _overlap_parts(parts: list[str], overlap: int, sentence_aware: bool):
    """Build the overlap tail for the next chunk: (parts, tokens, notes).

    sentence_aware=True reuses whole trailing sentences (any language);
    False falls back to word-level overlap for quick A/B comparison.
    """
    if overlap <= 0 or not parts:
        return [], 0, []
    combined = "\n\n".join(parts)
    if sentence_aware:
        sentences, exceeded = _sentence_tail(combined, overlap)
        if sentences:
            text = " ".join(sentences)
            notes = ["overlap sentence exceeds budget"] if exceeded else []
            return [text], count_tokens(text), notes
        return [], 0, []
    text_tokens, text_count = _token_tail(parts, overlap)
    return text_tokens, text_count, []


def _word_split(text: str, budget: int) -> list[str]:
    """Word-boundary split of one text, each piece within `budget`."""
    words = text.split()
    pieces, current = [], []
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


def _hard_split(text: str, budget: int) -> list[str]:
    """Split a single oversized paragraph to fit `budget`.

    Prefers SENTENCE boundaries (Latin + Arabic): a cut between sentences can
    never split an expected-match phrase. Only a sentence that alone exceeds
    the budget falls back to word-boundary cuts (marked by the caller).
    """
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            pieces.append(" ".join(current))
            current = []
            current_tokens = 0

    for sentence in split_sentences(text):
        sent_tokens = count_tokens(sentence)
        if sent_tokens <= budget:
            if current and current_tokens + sent_tokens > budget:
                flush()
            current.append(sentence)
            current_tokens += sent_tokens
            continue
        # One sentence longer than the whole budget: word-level fallback.
        flush()
        pieces.extend(_word_split(sentence, budget))
    flush()
    return pieces


def chunk_all(docs: list[dict], cfg) -> list[Chunk]:
    """Convenience wrapper: chunk every document with config.py parameters."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        doc_chunks = chunk_document(
            doc,
            chunk_size=cfg.CHUNK_SIZE_TOKENS,
            overlap=cfg.CHUNK_OVERLAP_TOKENS,
            split_on_headings=cfg.SPLIT_ON_HEADINGS_FIRST,
            sentence_aware_overlap=cfg.CHUNK_OVERLAP_SENTENCE_AWARE,
        )
        all_chunks.extend(doc_chunks)
    return all_chunks
