#!/usr/bin/env python3
"""topic_map.py — a deterministic, source-grounded corpus topic catalog.

The Phase-8 interrogation asks the model to choose a technical topic only
when the original request has no evidence coverage. Bare structural labels
(e.g. "الفصل 12") are not useful topics by themselves, so each selectable
entry carries:

- a request-local prompt ID (short, exact-match selection handle);
- a stable topic_id (a law unit ID or a document-scoped section ID);
- the exact heading/path from the adopted source structure; and
- a short source excerpt (never an LLM-generated summary).

Law article excerpts come from units.extract_law_units; all other paths and
excerpts come from restructure.normalize_structure over the documents actually
loaded from the configured data directories. Parent headings remain in paths,
not as empty selectable topics. No embedding or model calls are made here.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

TOPIC_PROMPT_MAX = 400
TOPIC_PROMPT_CHAR_MAX = 55_000
TOPIC_EXCERPT_MAX = 160
_PROMPT_EXCERPT_MAX = 88
_PROMPT_PATH_MAX = 96
_CACHE: dict[tuple, list[dict]] = {}

_STRUCTURAL_HEADING_RE = re.compile(
    r"^(?:العنوان|الباب|الفصل|القسم)\s*(?:"
    r"[0-9٠-٩]+|الاول|الأول|الثاني|الثالث|الرابع|الخامس|السادس|"
    r"السابع|الثامن|التاسع|العاشر|الحادي عشر|الثاني عشر)\s*$"
)
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^[|\s:-]+$")

_SOURCE_ALIASES = {
    "Loi_2016-48.pdf": "LAW",
    "Circulaire_BCT_2019-08.pdf": "CIRCULAR",
    "Guide_Interne_Operations_Bancaires_Islamiques.docx": "GUIDE",
    "Madkhal_Sayrafa_Islamiya.docx": "MANUAL",
}
_NON_TOPIC_HEADINGS = {"الفهرس", "قائمة المحتويات", "جدول المحتويات",
                       "محتويات", "contents", "table of contents", "index"}


def _is_structural_heading(heading: str) -> bool:
    return bool(_STRUCTURAL_HEADING_RE.fullmatch(re.sub(r"\s+", " ", heading).strip()))


def _slug(source: str) -> str:
    stem = Path(source).stem.casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug or "document"


def _clip_source_text(text: str, limit: int) -> str:
    """Whitespace-normalize and cap an exact source excerpt at a word edge."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].rstrip(" ,،;؛:-")
    return clipped or text[:limit]


def _first_source_excerpt(body: str, limit: int = TOPIC_EXCERPT_MAX) -> str:
    """Take the first prose from a section without inventing a summary."""
    lines = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("> **Context:") or line.startswith("[page "):
            continue
        if _MARKDOWN_HEADING_RE.match(line) or line.startswith("<!--"):
            continue
        if line.startswith("|") and _TABLE_SEPARATOR_RE.fullmatch(line):
            continue
        if line.startswith("|"):
            # A table-only section can still expose a source-grounded cue.
            line = " ".join(cell.strip() for cell in line.strip("|").split("|"))
        line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line).strip()
        if line:
            lines.append(line)
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if not text:
        return ""

    # Prefer a complete first sentence when it is reasonably short.
    end = re.search(r"[.؟!؛](?:\s|$)", text)
    if end and 24 <= end.end() <= limit:
        text = text[:end.end()].strip()
    return _clip_source_text(text, limit)


def _section_nodes(markdown: str) -> list[dict]:
    """Parse normalized Markdown into heading nodes with their own body text."""
    root = {"level": 0, "heading": "", "body": [], "children": []}
    stack = [root]
    current = root

    for line in markdown.splitlines():
        match = _MARKDOWN_HEADING_RE.match(line.strip())
        if not match:
            current["body"].append(line)
            continue
        level = len(match.group(1))
        heading = match.group(2).strip()
        while len(stack) > 1 and stack[-1]["level"] >= level:
            stack.pop()
        parent = stack[-1]
        node = {"level": level, "heading": heading, "body": [],
                "children": [], "parent": parent}
        parent["children"].append(node)
        stack.append(node)
        current = node

    nodes: list[dict] = []

    def walk(parent: dict) -> None:
        for node in parent["children"]:
            nodes.append(node)
            walk(node)

    walk(root)
    return nodes


def _path_label(node: dict) -> str:
    """Preserve structural ancestors while attaching any source-provided title."""
    ancestors = []
    parent = node.get("parent")
    while parent and parent.get("level", 0) > 1:
        ancestors.append(parent)
        parent = parent.get("parent")
    ancestors.reverse()
    pieces = []
    for item in [*ancestors, node]:
        heading = item["heading"]
        if _is_structural_heading(heading):
            title = _first_source_excerpt("\n".join(item.get("body", [])), 72)
            if title and title != heading:
                heading = f"{heading} ({title})"
        pieces.append(heading)
    return " > ".join(pieces)


def _raw_path_label(node: dict) -> str:
    """Raw heading path for stable IDs (without any derived parent cue)."""
    ancestors = []
    parent = node.get("parent")
    while parent and parent.get("level", 0) > 1:
        ancestors.append(parent["heading"])
        parent = parent.get("parent")
    ancestors.reverse()
    return " > ".join([*ancestors, node["heading"]])


def _display_label(heading: str, description: str) -> str:
    heading = _clip_source_text(heading, 120)
    if _is_structural_heading(heading) and description:
        return f"{heading} — {_clip_source_text(description, 96)}"
    return heading


def _make_section_entries(doc: dict, markdown: str) -> list[dict]:
    source = (doc.get("source") or doc.get("name") or "").strip()
    nodes = _section_nodes(markdown)
    entries = []
    path_occurrences = {}
    for node in nodes:
        if node["level"] < 2:
            continue                         # the document title is not a topic
        heading = node["heading"].strip()
        if not heading:
            continue
        unnumbered_heading = re.sub(r"^\d+(?:\.\d+)?[-.)]\s*", "", heading)
        if unnumbered_heading.casefold() in _NON_TOPIC_HEADINGS:
            continue
        description = _first_source_excerpt("\n".join(node["body"]))
        description_source = "section_body"
        if not description and not _is_structural_heading(heading):
            # A meaningful parent heading is itself an exact source label,
            # even when its content is entirely in child sections.
            description = heading
            description_source = "heading"
        if not description:
            continue                         # never expose a bare chapter number
        raw_path = re.sub(r"\s+", " ", _raw_path_label(node)).strip()
        occurrence = path_occurrences.get(raw_path, 0) + 1
        path_occurrences[raw_path] = occurrence
        digest = hashlib.sha256(raw_path.casefold().encode("utf-8")).hexdigest()[:10]
        suffix = f"-{occurrence}" if occurrence > 1 else ""
        topic_id = f"{_slug(source)}:section-{digest}{suffix}"
        entries.append({
            "document": source,
            "heading": heading,
            "display": _display_label(heading, description),
            "path": _path_label(node),
            "description": description,
            "description_source": description_source,
            "topic_id": topic_id,
            "unit_id": None,
            "kind": "section",
        })
    return entries


def _make_law_entries(doc: dict, restructure, units) -> list[dict]:
    source = (doc.get("source") or doc.get("name") or "").strip()
    codex = restructure._adopted_codex_text(doc.get("name", ""))
    if codex is None:
        codex = doc.get("text", "")
    entries = []
    for unit in units.extract_law_units(codex):
        heading = (unit.get("heading") or "").strip()
        description = (unit.get("description") or "").strip()
        path = (unit.get("path") or heading).strip()
        unit_id = unit.get("unit_id")
        if not heading or not description or not unit_id:
            continue
        entries.append({
            "document": source,
            "heading": heading,
            "display": _display_label(heading, description),
            "path": path,
            "description": description,
            "description_source": "law_unit",
            "topic_id": unit_id,
            "unit_id": unit_id,
            "kind": "law_article",
        })
    return entries


def _source_alias(source: str) -> str:
    return _SOURCE_ALIASES.get(source, Path(source).stem[:24])


def _cache_key(data_dirs, supported_extensions) -> tuple:
    directories = [Path(d) for d in data_dirs]
    paths = tuple(str(d.resolve()) for d in directories)
    signatures = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in supported_extensions:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            signatures.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
    return paths, tuple(signatures)


def build_topic_map(data_dirs) -> list[dict]:
    """Build source-grounded topics from documents actually present in data_dirs.

    Entries are deterministic and include ``topic_id``, ``prompt_id``,
    ``document``, ``heading``, ``path``, ``description`` and ``display``.
    ``prompt_id`` is request-local; ``topic_id`` is the stable source anchor.
    """
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]
    directories = [Path(d) for d in data_dirs]

    import loader
    import restructure
    import units

    key = _cache_key(directories, loader.SUPPORTED_EXTENSIONS)
    if key in _CACHE:
        return _CACHE[key]

    docs = loader.load_all(directories)
    # The active corpus may be supplied through more than one configured
    # directory. Keep the first copy of a basename (directory order wins),
    # matching the repository's existing source-name identity contract.
    by_name = {}
    for doc in docs:
        by_name.setdefault(doc.get("name") or doc.get("source"), doc)

    entries: list[dict] = []
    for name in sorted(by_name, key=lambda n: (n != "Loi_2016-48.pdf", n.casefold())):
        doc = by_name[name]
        if name == "Loi_2016-48.pdf":
            entries.extend(_make_law_entries(doc, restructure, units))
            continue
        markdown, _report = restructure.normalize_structure(doc, repair_rtl=True)
        entries.extend(_make_section_entries(doc, markdown))

    # Stable de-duplication. The stable topic_id is the source anchor; repeated
    # copies of the same adopted document do not create duplicate choices.
    out, seen_ids = [], set()
    for entry in entries:
        topic_id = entry["topic_id"]
        if topic_id in seen_ids:
            continue
        seen_ids.add(topic_id)
        out.append(entry)
    for index, entry in enumerate(out, start=1):
        entry["prompt_id"] = f"t{index:03d}"
    _CACHE[key] = out
    return out


def prompt_entries(data_dirs) -> list[dict]:
    """The exact catalog visible to the model, respecting both prompt caps."""
    return build_topic_map(data_dirs)[:TOPIC_PROMPT_MAX]


def _truncate_for_prompt(text: str, limit: int) -> str:
    return _clip_source_text(text, limit)


def render_topic_map(entries: list[dict]) -> str:
    """Render compact path + verbatim cue lines; IDs are the only new output."""
    visible = entries[:TOPIC_PROMPT_MAX]
    for excerpt_limit in (_PROMPT_EXCERPT_MAX, 72, 56, 40, 24):
        lines = []
        for entry in visible:
            prompt_id = entry["prompt_id"]
            alias = _source_alias(entry["document"])
            path = _truncate_for_prompt(entry.get("path", ""), _PROMPT_PATH_MAX)
            description = _truncate_for_prompt(
                entry.get("description", ""), excerpt_limit)
            lines.append(f"- [{prompt_id}] {alias} | {path} | {description}")
        rendered = "\n".join(lines)
        if len(rendered) <= TOPIC_PROMPT_CHAR_MAX:
            return rendered
    raise ValueError("source-grounded topic map exceeds the bounded prompt size")


def for_prompt(data_dirs) -> str:
    """Bounded plain-text rendering for the interrogation prompt."""
    return render_topic_map(prompt_entries(data_dirs))
