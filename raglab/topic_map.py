#!/usr/bin/env python3
"""topic_map.py — the deterministic CORPUS TOPIC MAP (Phase 8, item 1).

The interrogation layer (interrogate.py) needs the corpus's own topics so
the language model CHOOSES from what the corpus actually covers instead of
inventing. This module derives that map deterministically from the corpus
itself — no model calls, no manual curation:

- the law's typed units (via the adopted codex + units.extract_law_units):
  each الفصل with its unit id — the finest, most citable topic list;
- every other document's section headings (via the chunker's heading
  extraction) — the guide, the circular, the manual.

The map is cached per (dirs, mode) — building it re-chunks the corpus once
per process. The prompt rendering is BOUNDED (TOPIC_PROMPT_MAX entries)
so the interrogation call stays small.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

TOPIC_PROMPT_MAX = 400      # bounded prompt rendering (the corpus has ~300)
_CACHE: dict[tuple, list[dict]] = {}


def build_topic_map(data_dirs) -> list[dict]:
    """Deterministic topic entries: {document, heading, unit_id?, kind}."""
    key = tuple(str(d) for d in data_dirs)
    if key in _CACHE:
        return _CACHE[key]

    entries: list[dict] = []

    # 1) the law's typed units — the finest topic list (with unit ids)
    try:
        import restructure
        import units
        codex = restructure._adopted_codex_text("Loi_2016-48.pdf")
        if codex:
            for u in units.extract_law_units(codex):
                heading = (u.get("heading") or "").strip()
                if heading:
                    entries.append({"document": "Loi_2016-48.pdf",
                                    "heading": heading,
                                    "unit_id": u.get("unit_id"),
                                    "kind": "law_article"})
    except Exception:
        pass  # no law in this corpus — the map still serves the other docs

    # 2) every document's section headings (chunker metadata)
    try:
        import chunker
        import loader
        from types import SimpleNamespace
        import config as config_mod
        cfg = SimpleNamespace(**{k: getattr(config_mod, k)
                                 for k in dir(config_mod) if k.isupper()})
        cfg.CHUNKING_MODE = "restructure"
        docs = loader.load_all([Path(d) for d in data_dirs])
        seen: set[tuple[str, str]] = set()
        for chunk in chunker.chunk_all(docs, cfg):
            heading = (getattr(chunk, "heading", None) or "").strip()
            source = (getattr(chunk, "source", None) or "").strip()
            if not heading or (source, heading) in seen:
                continue
            if getattr(chunk, "section_type", None) == "front-matter":
                continue          # document titles are not topics
            seen.add((source, heading))
            entries.append({"document": source, "heading": heading,
                            "unit_id": None, "kind": "section"})
    except Exception:
        pass

    # de-duplicate (law entries first — they carry unit ids)
    out, seen_ids = [], set()
    for e in entries:
        eid = (e["document"], e["heading"])
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        out.append(e)
    _CACHE[key] = out
    return out


def for_prompt(data_dirs) -> str:
    """Bounded plain-text rendering for the interrogation prompt."""
    lines = []
    for e in build_topic_map(data_dirs)[:TOPIC_PROMPT_MAX]:
        unit = f" [{e['unit_id']}]" if e.get("unit_id") else ""
        lines.append(f"- ({e['document']}{unit}) {e['heading']}")
    return "\n".join(lines)
