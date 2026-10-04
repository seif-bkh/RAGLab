#!/usr/bin/env python3
"""Project the provisional Stage 1 evidence cases into evaluate.py's schema.

Only cases with declared source evidence are included. The source behavior
manifest remains untouched, and its 14 no-evidence cases are deliberately not
misrepresented as retrieval-answerable or out-of-scope retrieval controls.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAGLAB = HERE.parents[1]
REPO = RAGLAB.parent
sys.path.insert(0, str(RAGLAB))

import evaluate  # noqa: E402

MANIFEST = HERE / "stage1_cases_draft.json"
OUTPUT = RAGLAB / "results" / "phase9_stage1" / "stage1_vector_questions.json"
CORRECTED_SUFFIX = "_corrected.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_document(evidence_path: str) -> str:
    filename = Path(evidence_path).name
    if not filename.endswith(CORRECTED_SUFFIX):
        raise ValueError(f"Not an adopted corrected-codex path: {evidence_path!r}")
    stem = filename[:-len(CORRECTED_SUFFIX)]
    matches = sorted(p.name for p in (REPO / "docs").iterdir()
                     if p.is_file() and p.stem == stem)
    if len(matches) != 1:
        raise ValueError(f"Expected one docs/ source for {evidence_path!r}; found {matches}")
    return matches[0]


def main() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = []
    for candidate in payload.get("cases", []):
        refs = candidate.get("expected_evidence") or []
        if not refs:
            continue
        if candidate.get("expected_behavior") not in {
                "answer_direct", "answer_estimate", "answer_partial"}:
            raise ValueError(
                f"{candidate.get('id')}: evidence-bearing case has unexpected behavior")
        sources = {source_document(ref["file"]) for ref in refs}
        if len(sources) != 1:
            raise ValueError(
                f"{candidate.get('id')}: evaluator schema expects one target document")
        target_doc = next(iter(sources))
        target_path = REPO / "docs" / target_doc
        codex_path = REPO / refs[0]["file"]
        if not codex_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"Missing source for {candidate.get('id')}")

        quotes = [ref.get("quote", "").strip() for ref in refs]
        if not all(quotes):
            raise ValueError(f"{candidate.get('id')}: empty evidence quote")
        codex_text = evaluate.normalize_for_match(
            codex_path.read_text(encoding="utf-8"))
        if any(evaluate.normalize_for_match(quote) not in codex_text
               for quote in quotes):
            raise ValueError(
                f"{candidate.get('id')}: evidence quote absent from cited codex")

        case = {
            "id": candidate["id"],
            "question": candidate["question"],
            "language": candidate.get("language", "unknown"),
            "category": "cross-lingual" if candidate.get("language") != "ar"
                        else "paraphrase",
            "expected_document": target_doc,
        }
        if len(quotes) == 1:
            case["expected_substring"] = quotes[0]
        else:
            case["expected_substrings"] = quotes
        cases.append(case)

    if len(cases) != 7:
        raise ValueError(f"Expected 7 evidence-bearing candidates, found {len(cases)}")

    output = {
        "source_manifest": str(MANIFEST.relative_to(REPO)),
        "source_manifest_sha256": sha256(MANIFEST),
        "manifest_status": payload.get("status"),
        "case_labels_final": False,
        "note": "Retrieval-only projection; excludes cases without a gold evidence span.",
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    # The project evaluator consumes the same top-level {cases:[...]} format.
    loaded = evaluate.load_question_set(OUTPUT)
    if len(loaded) != 7:
        raise RuntimeError(f"Evaluator loaded {len(loaded)} cases, expected 7")
    print(f"[phase9-stage1] wrote {OUTPUT.relative_to(REPO)} "
          f"({len(loaded)} evidence cases; labels remain provisional)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
