#!/usr/bin/env python3
"""diag_sufficiency.py — why did /answer refuse THIS question? (2026-10-05)

WHY IT EXISTS: the owner's live run refused two cross-script questions while
the same code answered them in the offline harness. The console hides the
reason: local_front prints one line («غير كافٍ · ناقص: …») but the decision
inside service._commitment_gate is per-requirement, and one input to it —
`search_fn` — decides whether the guided rescue round can run AT ALL.

THE STRUCTURAL FACT THIS EXPOSES: service.py calls sufficiency.check() at
lines 915 and 960 WITHOUT search_fn, and the guided round is guarded by
`search_fn is not None` (sufficiency.py:646). So on the deployed path the
guided round never executes — measured here, not inferred: this script runs
check() twice over the SAME pool, once exactly as /answer does and once with
a search_fn, and prints both verdicts side by side.

Read-only: GET /chunks (paginated) + POST /search per question. No model call.

Usage (from raglab/, against a running service):
    python diag_sufficiency.py --ask "question" [--ask "another"] \
        [--base-url http://localhost:8000] [--token ...] [--k 5]
"""

from __future__ import annotations

import argparse
import os
import sys

from diag_bridge import df_cap
from local_front import Api, DEFAULT_BASE_URL

# The questions the 2026-10-05 cross-script arm was built for. Declared here,
# not imported from the frozen sets: those two files must never be read by a
# deployed diagnostic (governance: questions_50.json / questions_targets.json
# are frozen and no shipped code may depend on them).
DEFAULT_QUESTIONS = [
    "Quelles sont les étapes et modalités de réalisation d'une opération de "
    "change (vente et achat de devises) pour un client ?",
    "What is a documentary letter of credit (L/C) in the manual, and what "
    "kind of commitment does it represent?",
]


def fetch_texts(api: Api) -> tuple[int, list[str]]:
    """Every chunk text in the live index, paginated like diag_bridge."""
    texts: list[str] = []
    total = offset = 0
    while True:
        status, page = api.get("/chunks", params={"limit": 100, "offset": offset})
        if status != 200:
            raise RuntimeError(f"GET /chunks answered HTTP {status}: {page}")
        total = page.get("total", 0)
        items = page.get("items") or []
        texts.extend(item.get("text") or "" for item in items)
        offset += len(items)
        if offset >= total or not items:
            break
    return total, texts


def analyse(question: str, texts: list[str], hits: list[dict],
            search_fn=None) -> dict:
    """Sufficiency state for one question — BOTH ways.

    `as_served` reproduces service._commitment_gate byte for byte: check() with
    no search_fn, so the guided round is skipped. `with_rounds` passes a
    search_fn, which is what the measurement harness does and what /answer
    would have to do to reach the rescue. The two verdicts are the evidence.
    """
    import sufficiency as _suf
    df = _suf.build_df(texts)
    as_served = _suf.check(question, hits, df=df)
    with_rounds = _suf.check(question, hits, df=df, search_fn=search_fn)
    return {
        "question": question,
        "as_served": {
            "state": as_served["state"],
            "reason": as_served.get("reason"),
            "missing": as_served["missing"],
            "requirements": [{"req": r["req"], "covered": r["covered"]}
                             for r in as_served["requirements"]],
            "guided_rounds": as_served["guided_rounds"],
        },
        "with_rounds": {
            "state": with_rounds["state"],
            "missing": with_rounds["missing"],
            "guided_rounds": with_rounds["guided_rounds"],
        },
        "df_cap": df_cap(len(texts)),
    }


def _verdict(rep: dict) -> list[str]:
    """Plain-language notes. Pure — unit-tested."""
    notes: list[str] = []
    a, w = rep["as_served"], rep["with_rounds"]
    if a["state"] != "غير كافٍ":
        notes.append(f"served verdict is {a['state']} — /answer should NOT have "
                     "refused on this pool; the refusal came from elsewhere "
                     "(interrogation path or a different pool).")
        return notes
    notes.append("served verdict is غير كافٍ — /answer refuses before any "
                 "model call (service.py:915).")
    covered = [r["req"] for r in a["requirements"] if r["covered"]]
    notes.append(f"  covered {len(covered)}/{len(a['requirements'])} "
                 f"requirement(s); missing: {', '.join(a['missing']) or '—'}")
    if not covered:
        notes.append("  NOTHING covers the plan, so service.py takes the "
                     "interrogation branch — if the response has no "
                     "understood_as, interrogate() returned None (fail-closed) "
                     "or REPHRASE_INTERROGATION_ENABLED=0.")
    else:
        notes.append("  partial coverage, so the interrogation branch is "
                     "SKIPPED and the refusal is immediate.")
    if a["guided_rounds"] == [] and w["guided_rounds"]:
        notes.append("  GUIDED ROUND UNREACHABLE ON /answer: with a search_fn "
                     f"it runs {len(w['guided_rounds'])} round(s) "
                     f"({', '.join(r['query'] for r in w['guided_rounds'])}) "
                     f"and the verdict becomes {w['state']}. service.py passes "
                     "no search_fn, so the deployed path can never reach it.")
    elif w["state"] == a["state"]:
        notes.append("  the guided round changes nothing here — the rescue is "
                     "not what this question needs.")
    return notes


def run(api: Api, questions: list[str], k: int) -> int:
    import sufficiency as _suf
    total, texts = fetch_texts(api)
    print(f"[diag] index: {total} chunks | df cap = {df_cap(total)}")
    print("[diag] gates as loaded in THIS container:")
    for name in ("FIELD_BRIDGES_ENABLED", "GUIDED_BRIDGED_ENABLED",
                 "DEFINITION_SHAPE_ENABLED", "REPHRASE_INTERROGATION_ENABLED",
                 "ANSWER_SUFFICIENCY_COMMITMENT"):
        print(f"  {name} = {getattr(_suf, name, getattr(__import__('config'), name, '?'))}")
    print(f"  ANSWER_TOP_K = {getattr(__import__('config'), 'ANSWER_TOP_K', '?')}"
          f"  (probe k = {k})")
    print("[diag] NOTE: /search returns PII-scrubbed text, so a pool built "
          "here can differ slightly from the one /answer scored.")
    for question in questions:
        status, body = api.post("/search", payload={"question": question, "k": k})
        if status != 200:
            print(f"\n[diag] POST /search answered HTTP {status}: {body}")
            return 1
        hits = body.get("hits") or []

        def search_fn(q: str, kk: int, _api=api) -> list[dict]:
            st, bd = _api.post("/search", payload={"question": q, "k": kk})
            return (bd.get("hits") or []) if st == 200 else []

        rep = analyse(question, texts, hits, search_fn=search_fn)
        print(f"\n[diag] {question[:78]}")
        print(f"  retrieved={len(hits)}")
        for note in _verdict(rep):
            print(f"  {note}" if note.startswith("  ") else f"  {note}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="diag_sufficiency", description=__doc__.splitlines()[0])
    parser.add_argument("--ask", action="append", dest="ask", default=None,
                        help="question to probe (repeatable); defaults to the "
                             "two declared cross-script cases")
    parser.add_argument("--k", type=int, default=None,
                        help="retrieval depth (default: the service's ANSWER_TOP_K)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token", default=None)
    args = parser.parse_args(argv)
    questions = args.ask or list(DEFAULT_QUESTIONS)
    api = Api(args.base_url, token=args.token or os.getenv("RAGLAB_SERVICE_TOKEN"))
    k = args.k or int(getattr(__import__("config"), "ANSWER_TOP_K", 5))
    return run(api, questions, k)


if __name__ == "__main__":
    raise SystemExit(main())
