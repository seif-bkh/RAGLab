#!/usr/bin/env python3
"""Summarize vector-only and deployed-default retrieval runs for GitHub Actions."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAGLAB = HERE.parents[1]
RESULTS = RAGLAB / "results" / "phase9_stage1"
RUNS = [
    ("vector-only", RESULTS / "stage1_vector_only.json"),
    ("deployed-default", RESULTS / "stage1_vector_default.json"),
]


def pct(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


def load_runs():
    return [(name, json.loads(path.read_text(encoding="utf-8")))
            for name, path in RUNS if path.is_file()]


def render(runs) -> str:
    lines = [
        "## Phase 9 — Stage 1 vector retrieval (retrieval-only)",
        "",
        "The case labels remain provisional. This run calls NVIDIA embeddings only; it does not generate answers.",
        "",
        "| arm | n | hit@1 | hit@3 | hit@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, run in runs:
        overall = run["metrics"]["overall"]
        lines.append(
            f"| {label} | {overall['n']} | {pct(overall['hit@1'])} | "
            f"{pct(overall['hit@3'])} | {pct(overall['hit@5'])} |")
    lines += ["", "### Per-case evidence rank", "",
              "| arm | case | rank completing declared evidence | requirements in top 20 | top-1 source / heading |",
              "|---|---|---:|---:|---|"]
    for label, run in runs:
        for question in run.get("questions", []):
            rank = question.get("correct_rank")
            rank_text = str(rank) if rank is not None else "miss"
            total = question.get("requirements_total")
            found = question.get("requirements_found")
            coverage = f"{found}/{total}" if total is not None else "1/1" if rank is not None else "0/1"
            hits = question.get("hits") or []
            if hits:
                top = hits[0]
                meta = top.get("metadata") or {}
                doc = meta.get("document") or meta.get("source") or "?"
                heading = str(top.get("heading") or meta.get("heading") or "?")
                top_text = f"{doc} — {heading}"
            else:
                top_text = "no hits"
            lines.append(f"| {label} | {question['id']} | {rank_text} | "
                         f"{coverage} | {top_text} |")
    return "\n".join(lines) + "\n"


def annotation(runs) -> str:
    parts = []
    for label, run in runs:
        overall = run["metrics"]["overall"]
        ranks = ",".join(
            f"{q['id']}={q.get('correct_rank') or 'miss'}"
            for q in run.get("questions", []))
        parts.append(
            f"{label} n={overall['n']} hit@1/3/5="
            f"{pct(overall['hit@1'])}/{pct(overall['hit@3'])}/{pct(overall['hit@5'])} "
            f"ranks[{ranks}]")
    return " | ".join(parts)


def main() -> int:
    runs = load_runs()
    if not runs:
        print("[phase9-stage1] no evaluation JSON found")
        return 1
    summary = render(runs)
    out = RESULTS / "stage1_vector_summary.md"
    out.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print("ANNO| phase9-stage1-vector | " + annotation(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
