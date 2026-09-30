#!/usr/bin/env python3
"""Verify the LLM-repair log (Loi_2016-48_llm_repair.md) against the corpus.

Two checks per entry, per the owner-approved methodology (2026-09-28:
language-model repair first; the strict-algorithm derivation was ABANDONED
by owner decision — طرح أي التخلي عنها):

1. SOURCE  — every «ق» line must appear verbatim (whitespace-collapsed) in
   the stored document text, in order: the log quotes the real corpus.
2. CONSERVATION — the word inventory of «ب» must equal that of «ق» except
   for diffs explicitly explained in the entry's «م» note (joins, splits,
   documented letter insertions/drops). The tool prints every diff; a diff
   with no matching note is a finding to fix.
3. COVERAGE — walking the stored lines in order against the full «ق» stream,
   every skipped stored line must be classifiable (blank / loader page
   marker / gazette running header). An unaccounted substantive line is a
   coverage gap in the repair log (added with batch 27, which closed the
   three gaps the check first found: E42ب, E44ب, E54ب).

Usage: python audits/llm_repair_check.py [report.md] [doc-name]
Exit 0 = sources verified AND full line coverage; conservation diffs are
printed, not fatal.
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

    print(f"[llm-check] source fidelity: "
          f"{'OK' if src_fail == 0 else str(src_fail) + ' ENTRIES FAILED'}")

    # 3. coverage — every stored line is quoted in some «ق» (in entry order)
    #    or legitimately skipped (blank / loader page marker / gazette
    #    running header). Anything else is an unaccounted substantive line:
    #    a coverage gap in the repair log.
    def classify_skip(line: str) -> str:
        if not line.strip():
            return "blank"
        if re.fullmatch(r"\[page \d+\]", line.strip()):
            return "page-marker"
        if "الرائد" in line and "الرسمي" in line:
            return "gazette-header"
        return "UNACCOUNTED"

    pos = 0
    matched = 0
    order_fail = 0
    unaccounted = 0
    skipped: Counter = Counter()
    for e in entries:
        for ql in e["q"]:
            n = norm_ws(ql)
            j = pos
            while j < len(stored_lines) and stored_lines[j] != n:
                j += 1
            if j >= len(stored_lines):
                order_fail += 1
                print(f"[llm-check] {e['id']}: COVERAGE ORDER MISS: {n[:60]!r}")
                continue
            matched += 1
            for k in range(pos, j):
                c = classify_skip(stored_lines[k])
                skipped[c] += 1
                if c == "UNACCOUNTED":
                    unaccounted += 1
                    print(f"[llm-check] UNACCOUNTED stored line {k}: "
                          f"{stored_lines[k][:70]!r}")
            pos = j + 1
    for k in range(pos, len(stored_lines)):
        c = classify_skip(stored_lines[k])
        skipped[c] += 1
        if c == "UNACCOUNTED":
            unaccounted += 1
            print(f"[llm-check] UNACCOUNTED stored line {k}: "
                  f"{stored_lines[k][:70]!r}")

    cov = "OK" if unaccounted == 0 and order_fail == 0 else "FAILED"
    skipped_txt = ", ".join(f"{c} {n}" for c, n in sorted(skipped.items()))
    print(f"[llm-check] line coverage: {cov} (quoted {matched}"
          f" of {len(stored_lines)} stored lines; skipped: {skipped_txt}"
          f"; unaccounted {unaccounted})")
    return 1 if src_fail or unaccounted or order_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
