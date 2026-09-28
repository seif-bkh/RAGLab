#!/usr/bin/env python3
"""Verify the verbatim quotes in an audit report against the corpus document.

Audit reports (raglab/audits/*.md) use ONE quoting convention: every span between
«...» is a VERBATIM span of the stored corpus text, defects included. Everything
official/true from external sources is written in bold, never in «».

Usage (from raglab/audits/):
    python check_audit_quotes.py <report.md> <document-name-in-docs>

Exits 0 if every «...» span is found verbatim in the loaded document text,
1 otherwise. Skips degenerate spans (pure punctuation, < 3 chars).
"""
import re
import sys
from pathlib import Path

RAGLAB_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
sys.path.insert(0, str(RAGLAB_DIR))

from loader import load_document  # noqa: E402


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    report_path, doc_name = Path(argv[1]), argv[2]
    report = report_path.read_text(encoding="utf-8")

    doc = load_document(DOCS_DIR / doc_name, origin="docs/")
    text = doc["text"]

    spans = re.findall(r"«([^»]+)»", report)
    checked, skipped, failures = 0, 0, []
    for span in spans:
        if len(span.strip()) < 3 or not re.search(r"[^\W\d_]", span, re.UNICODE):
            skipped += 1  # ellipsis placeholders like «...», not real quotes
            continue
        checked += 1
        if span not in text:
            failures.append(span)

    print(f"[audit-check] report={report_path.name} document={doc_name}")
    print(f"[audit-check] «» spans found: {len(spans)} | checked: {checked} | "
          f"skipped (degenerate): {skipped}")
    for span in failures:
        print(f"[audit-check] NOT IN CORPUS: {span!r}")
    if failures:
        print(f"[audit-check] FAIL — {len(failures)} quote(s) not verbatim in the corpus")
        return 1
    print("[audit-check] OK — every «» quote is verbatim corpus text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
