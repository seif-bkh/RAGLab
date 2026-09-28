#!/usr/bin/env python3
"""Verify the LLM-repair log (Loi_2016-48_llm_repair.md) against the corpus.

Three checks per entry, per the owner-approved methodology (2026-09-28:
language-model repair first, algorithm derivation afterwards):

1. SOURCE  — every «ق» line must appear verbatim (whitespace-collapsed) in
   the stored document text, in order: the log quotes the real corpus.
2. CONSERVATION — the word inventory of «ب» must equal that of «ق» except
   for diffs explicitly explained in the entry's «م» note (joins, splits,
   documented letter insertions/drops). The tool prints every diff; a diff
   with no matching note is a finding to fix.
3. ALGORITHM — for each «ق» line, compare the v2 zone-reconstruction
   (restructure._reconstruct_line_visual_order) with the «ب» text: how much
   of the human/LM repair a strict rule already reproduces. This feeds the
   final derive-the-algorithm step.

Usage: python audits/llm_repair_check.py [report.md] [doc-name]
Exit 0 = sources all verified (conservation diffs are printed, not fatal).
"""
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loader import load_document  # noqa: E402

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE.parent.parent / "docs"
DEFAULT_REPORT = HERE / "Loi_2016-48_llm_repair.md"
DEFAULT_DOC = "Loi_2016-48.pdf"

ENTRY_RE = re.compile(r"^### (E\d+\s*ب?)\s*\|\s*صفحة\s*([^|]+?)\s*\|\s*(.+)$")
PUNCT = "،.؛:()«»–—؟!…\"'٪%-|\u0640"


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def toks(s: str) -> list[str]:
    return [t.strip(PUNCT) for t in norm_ws(s).split() if t.strip(PUNCT)]


def parse_entries(md: str):
    entries, cur = [], None
    for line in md.split("\n"):
        m = ENTRY_RE.match(line)
        if m:
            cur = {"id": m.group(1).strip(), "page": m.group(2), "title": m.group(3),
                   "q": [], "b": [], "m": [], "where": None}
            entries.append(cur)
            continue
        if cur is None:
            continue
        if line.strip() == "ق:":
            cur["where"] = "q"
            continue
        if line.strip() == "ب:":
            cur["where"] = "b"
            continue
        if line.startswith("م:"):
            cur["where"] = "m"
            cur["m"].append(line[2:].strip())
            continue
        if line.strip().startswith("```"):
            continue
        if line.startswith("## ") or line.startswith("---"):
            cur = None
            continue
        if cur["where"] == "q" and line.strip():
            cur["q"].append(line)
        elif cur["where"] == "b" and line.strip():
            cur["b"].append(line)
        elif cur["where"] == "m" and line.strip():
            cur["m"].append(line.strip())
    return entries


def canon(s: str) -> list:
    """Presentation artifacts out: standalone tatweel dashes are separator
    decoration, not content — both sides drop them before comparing."""
    return [t for t in norm_ws(s).split() if t != "ـ"]


def agreement(alg: str, lm: str) -> float:
    """Positional agreement via token alignment (a single inserted token
    must not zero the score)."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, canon(alg), canon(lm)).ratio()


def is_exact(alg: str, lm: str) -> bool:
    return canon(alg) == canon(lm)


def main(argv):
    report_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_REPORT
    doc_name = argv[2] if len(argv) > 2 else DEFAULT_DOC
    doc = load_document(DOCS_DIR / doc_name, origin="docs/")
    stored_lines = [norm_ws(l) for l in doc["text"].split("\n")]
    stored_flat = norm_ws(doc["text"])

    md = report_path.read_text(encoding="utf-8")
    entries = parse_entries(md)
    print(f"[llm-check] report={report_path.name} entries={len(entries)}")

    src_fail = 0
    total_ratio, n_cmp = 0.0, 0
    exact_alg = 0
    for e in entries:
        # 1. source fidelity (in order, whitespace-collapsed)
        pos = 0
        ok_src = True
        for ql in e["q"]:
            n = norm_ws(ql)
            i = stored_flat.find(n, pos)
            if i < 0:
                ok_src = False
                print(f"[llm-check] {e['id']}: SOURCE NOT FOUND: {n[:70]!r}")
            else:
                pos = i + len(n)
        if not ok_src:
            src_fail += 1

        # 2. conservation
        cq, cb = Counter(toks("\n".join(e["q"]))), Counter(toks("\n".join(e["b"])))
        added = cb - cq
        removed = cq - cb
        if added or removed:
            print(f"[llm-check] {e['id']} ({e['title'][:30]}): inventory diff")
            for w, n in sorted(added.items()):
                print(f"    + {'×' + str(n) if n > 1 else ''} {w}")
            for w, n in sorted(removed.items()):
                print(f"    - {'×' + str(n) if n > 1 else ''} {w}")

        # 3. algorithm agreement (line-level zone reconstruction vs LM)
        try:
            import restructure as rst
            alg = norm_ws(" ".join(
                rst._reconstruct_line_visual_order(l) for l in e["q"]))
            lm = norm_ws("\n".join(e["b"]))
            if is_exact(alg, lm):
                exact_alg += 1
                ratio = 1.0
            else:
                ratio = agreement(alg, lm)
            total_ratio += ratio
            n_cmp += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[llm-check] {e['id']}: algorithm compare failed: {exc}")

    print(f"[llm-check] source fidelity: {'OK' if src_fail == 0 else str(src_fail) + ' ENTRIES FAILED'}")
    if n_cmp:
        print(f"[llm-check] algorithm-vs-LM (v2 zones baseline): "
              f"exact {exact_alg}/{n_cmp} entries, "
              f"mean agreement {total_ratio / n_cmp:.1%}")
    return 1 if src_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
