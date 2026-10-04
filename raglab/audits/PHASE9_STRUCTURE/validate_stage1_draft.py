#!/usr/bin/env python3
"""Validate the draft Phase-9/Stage-1 behavior manifest.

This is an authoring guard only. It does not call a model, run retrieval,
approve the labels, or alter the production question sets.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = HERE / "stage1_cases_draft.json"
ALLOWED = {
    "answer_direct",
    "answer_estimate",
    "clarify",
    "answer_partial",
    "abstain",
    "personal_data_unavailable",
    "action_unavailable",
}
ANSWER_MODES = {"answer_direct", "answer_estimate", "answer_partial"}
NO_EVIDENCE_MODES = {
    "clarify", "abstain", "personal_data_unavailable", "action_unavailable"
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("FAIL: cases must be a non-empty list")

    errors: list[str] = []
    seen: set[str] = set()
    modes: dict[str, int] = {}
    owner_count = 0

    for i, case in enumerate(cases):
        cid = case.get("id", f"index-{i}")
        if cid in seen:
            errors.append(f"{cid}: duplicate id")
        seen.add(cid)
        mode = case.get("expected_behavior")
        modes[mode] = modes.get(mode, 0) + 1
        if mode not in ALLOWED:
            errors.append(f"{cid}: unsupported expected_behavior {mode!r}")
        if not str(case.get("question", "")).strip():
            errors.append(f"{cid}: empty question")
        if case.get("review_status") != "provisional_owner_review":
            errors.append(f"{cid}: draft label is not marked provisional")
        if case.get("origin") == "owner_seed":
            owner_count += 1
        if not case.get("reason_code"):
            errors.append(f"{cid}: missing reason_code")

        evidence = case.get("expected_evidence")
        if not isinstance(evidence, list):
            errors.append(f"{cid}: expected_evidence must be a list")
            evidence = []
        if mode in ANSWER_MODES and not evidence:
            errors.append(f"{cid}: answer mode requires at least one source span")
        if mode in NO_EVIDENCE_MODES and evidence:
            errors.append(f"{cid}: non-answer mode must not declare an answer span")
        if mode == "clarify" and not case.get("clarification_target"):
            errors.append(f"{cid}: clarify mode needs a clarification_target")
        if mode == "answer_estimate" and not case.get("derivation"):
            errors.append(f"{cid}: estimate mode needs a derivation")
        if mode == "answer_partial" and not case.get("expected_requirements"):
            errors.append(f"{cid}: partial mode needs explicit requirements")

        for j, ref in enumerate(evidence):
            rel = ref.get("file", "")
            quote = ref.get("quote", "")
            source = REPO / rel
            if not rel or not source.is_file():
                errors.append(f"{cid}: evidence[{j}] source missing: {rel!r}")
                continue
            if not quote:
                errors.append(f"{cid}: evidence[{j}] quote is empty")
                continue
            source_text = norm(source.read_text(encoding="utf-8"))
            if norm(quote) not in source_text:
                errors.append(f"{cid}: evidence[{j}] quote not found verbatim in {rel}")

    if owner_count != 12:
        errors.append(f"expected 12 verbatim owner seeds, found {owner_count}")

    if errors:
        print("Stage 1 draft validation: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Stage 1 draft validation: OK — {len(cases)} cases; "
          f"{owner_count} owner seeds; {len(seen)} unique ids")
    print("Modes: " + ", ".join(f"{key}={modes[key]}" for key in sorted(modes)))
    print("All expected evidence spans exist in the adopted corrected codices.")
    print("Labels remain provisional; this validator checks structure and sources only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
