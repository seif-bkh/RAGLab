"""loader.py — turn files in data/ into normalized, printable document dicts.

A "document" is a plain dict, always with the same keys:

    {
        "name":      file name, e.g. "fiche_produit_banque_atlas_fr.md",
        "path":      absolute path of the source file,
        "text":      normalized full text (whitespace reflowed, Arabic normalized),
        "language":  "fr" | "ar" | "en" | "unknown",
        "source":    same as name (source label stored in Chroma metadata),
        "origin":    "data/",
    }

Nothing here calls an API and nothing here translates. Only text cleaning
and a language guess. Every function prints what it did so you can inspect
each intermediate step.
"""

import re
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

# --- Arabic normalization constants ---------------------------------------
ALEF_VARIANTS = "أإآٱ"  # all unify to bare alef "ا"
TATWEEL = "\u0640"       # "ـ" (kashida)
DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

# Arabic letter range used by the language heuristic.
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]")

FRENCH_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "est", "sont",
    "pour", "sur", "avec", "dans", "au", "aux", "par", "qui", "que", "ce",
    "cette", "ces", "ses", "son", "il", "elle", "nous", "vous", "ils", "elles",
    "pas", "plus", "si", "comme", "à", "en", "ou", "tous", "toutes", "moins",
    "chaque", "leur", "leurs", "dont", "aussi", "sous", "vers", "entre", "après",
}
ENGLISH_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "and", "to", "for", "with", "on",
    "in", "at", "by", "from", "that", "this", "it", "its", "as", "or", "if",
    "not", "no", "what", "how", "when", "which", "who", "does", "do", "can",
    "be", "your", "my", "you", "we", "they", "there", "more", "than", "any",
    "each", "every", "must", "will", "would",
}


def normalize_arabic(text: str) -> str:
    """Light Arabic normalization: unify alef variants, remove tatweel and diacritics.

    Does not translate, does not stem, does not shift the text's meaning.
    """
    out = text
    for variant in ALEF_VARIANTS:
        out = out.replace(variant, "ا")
    out = out.replace(TATWEEL, "")
    out = DIACRITICS_RE.sub("", out)
    return out


def _is_block_line(line: str) -> bool:
    """True for lines that must keep their own line break: headings, tables,
    list items, horizontal rules. Paragraph text gets reflowed instead."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#") or stripped.startswith("|"):
        return True
    if re.match(r"^[-*+]\s+\S", stripped):          # unordered list item
        return True
    if re.match(r"^\d+[.)]\s+\S", stripped):        # ordered list item
        return True
    if re.match(r"^(---+|\*\*\*+)\s*$", stripped):  # horizontal rule
        return True
    return False


def normalize_text(text: str) -> str:
    """Normalize whitespace and line breaks, preserving paragraph boundaries,
    headings and table rows.

    - Universal newlines -> \n
    - Non-breaking spaces -> plain spaces
    - One blank line between paragraphs, none inside a paragraph
    - Heading/table/list lines stay on their own line
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u2028", "\n").replace("\u2029", "\n")

    lines = text.split("\n")
    out_lines: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph():
        if paragraph:
            # Reflow: collapse runs of internal whitespace, keep one space.
            joined = " ".join(paragraph)
            joined = re.sub(r"[ \t]+", " ", joined).strip()
            if joined:
                out_lines.append(joined)
            paragraph.clear()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        if _is_block_line(line):
            flush_paragraph()
            out_lines.append(line)
        else:
            paragraph.append(line)

    flush_paragraph()

    # Exactly one blank line between blocks; no leading/trailing blank lines.
    clean = []
    blank = False
    for line in out_lines:
        if line == "":
            if not blank:
                clean.append("")
            blank = True
        else:
            clean.append(line)
            blank = False
    while clean and clean[0] == "":
        clean.pop(0)
    while clean and clean[-1] == "":
        clean.pop()
    return "\n".join(clean)


def detect_language(text: str) -> str:
    """Very small, fully transparent language heuristic.

    1. Arabic-script character count wins if it clearly dominates.
    2. Otherwise compare French vs English stopword frequency.
    Returns "fr", "ar", "en" or "unknown".
    """
    arabic_chars = len(ARABIC_RE.findall(text))
    latin_chars = len(LATIN_RE.findall(text))

    if arabic_chars > 20 and arabic_chars > latin_chars:
        return "ar"
    if latin_chars < 20:
        return "unknown"

    tokens = re.findall(r"[A-Za-zÀ-ÿ]+", text.lower())
    fr_hits = sum(1 for t in tokens if t in FRENCH_STOPWORDS)
    en_hits = sum(1 for t in tokens if t in ENGLISH_STOPWORDS)

    if fr_hits > en_hits:
        return "fr"
    if en_hits > 0:
        return "en"
    return "unknown"


def read_pdf(path: Path) -> str:
    """Extract text from a PDF, page by page, marking each page for inspection."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # Keep the marker so extracted-pages are visible in chunk output.
        pages.append(f"[page {i}]\n{text}")
    return "\n\n".join(pages)


def load_document(path: Path, origin: str = "data/") -> dict:
    """Load one file, normalize it, guess its language, return a document dict."""
    suffix = path.suffix.lower()
    print(f"[loader] reading {path.name} ({suffix})")

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported extension {suffix!r} for {path.name}")

    if suffix == ".pdf":
        raw = read_pdf(path)
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")

    text = normalize_text(raw)
    text = normalize_arabic(text)
    language = detect_language(text)

    print(f"[loader]   raw chars={len(raw):>6}  normalized chars={len(text):>6}  "
          f"language={language}")

    return {
        "name": path.name,
        "path": str(path.resolve()),
        "text": text,
        "language": language,
        "source": path.name,
        "origin": origin,
    }


def load_all(data_dir: Path) -> list[dict]:
    """Load every supported file in data_dir (sorted by name), skip nothing silently."""
    docs: list[dict] = []
    if not data_dir.is_dir():
        print(f"[loader] WARNING: data directory does not exist: {data_dir}")
        return docs

    for path in sorted(data_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                docs.append(load_document(path))
            except Exception as exc:  # noqa: BLE001 — lab tool: never hide a bad file
                print(f"[loader] ERROR loading {path.name}: {exc}")

    print(f"[loader] {len(docs)} document(s) loaded from {data_dir}")
    if not docs:
        print("[loader] WARNING: no supported files found "
              f"({', '.join(sorted(SUPPORTED_EXTENSIONS))}).")
    return docs
