#!/usr/bin/env python3
"""Phase 9 — deterministic measurement of the cross-script coverage gate.

WHY THIS EXISTS
---------------
The live answer probe (`12_PHASE9_ANSWER_PROBE_RESULTS.md`) showed TM03 (fr)
and TM04 (en) retrieving the right guide sections and still being refused, and
increasing `top_k` 5 -> 12 -> 20 changed nothing. This harness reproduces that
refusal OFFLINE — no embeddings, no provider calls — on the project's own
deterministic keyword arm (`sufficiency.bm25_store`), and separates the three
gates a hit must pass before it can cover a requirement:

    retrieved  ->  anchored (`sufficiency._anchors`)  ->  shape
                   (`sufficiency.shape_ok`)  ->  requirement covered

ARMS
----
The two declared changes are gated OFF by default:

    SUFFICIENCY_FIELD_BRIDGES_ENABLED=1   field-vocabulary bridges
    SUFFICIENCY_DEFINITION_SHAPE_ENABLED=1 equative definition/purpose shapes

`--arms` re-executes this script once per combination (real import-time
gating, not an in-process patch) and prints the 2x2, so each half of the
change can be attributed separately.

    python3 measure_cross_script_coverage.py                # current config
    python3 measure_cross_script_coverage.py --arms         # the 2x2
    python3 measure_cross_script_coverage.py --regression   # + frozen sets

WHAT IT DOES NOT PROVE
----------------------
This is the BM25 arm. The deployed retrieval arm is the vector one, so
`retrieved` here is not the deployed ranking; the anchoring/shape/coverage
gates measured here ARE the deployed code. No golden answers exist for the six
probe cases, so nothing here is an answer-quality benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAGLAB = HERE.parent.parent
DOCS = RAGLAB.parent / "docs"
PROBES = HERE / "topic_map_live_probes.json"
sys.path.insert(0, str(RAGLAB))

GATES = ("SUFFICIENCY_FIELD_BRIDGES_ENABLED",
         "SUFFICIENCY_DEFINITION_SHAPE_ENABLED")

ARM_NAMES = {(0, 0): "off/off (current behavior)",
             (1, 0): "field-bridges ON",
             (0, 1): "definition-shape ON",
             (1, 1): "BOTH ON (proposed arm)"}


def _banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def _corpus_and_df():
    """The project's own deterministic keyword arm (no API)."""
    import sufficiency
    search, df = sufficiency.bm25_store(DOCS)
    return sufficiency, search, df


def _chunks_by_heading():
    """(document, heading) -> [chunk, ...] over the same corpus the arm uses."""
    import chunker
    import config as config_mod
    import loader
    from types import SimpleNamespace

    cfg = SimpleNamespace(**{k: getattr(config_mod, k)
                             for k in dir(config_mod) if k.isupper()})
    cfg.CHUNKING_MODE = "restructure"
    docs = loader.load_all([DOCS])
    chunks = chunker.chunk_all(docs, cfg)
    out: dict[tuple[str, str], list[dict]] = {}
    for c in chunks:
        key = (c.source, (c.heading or "").strip())
        out.setdefault(key, []).append(
            {"id": f"{c.source}::chunk_{c.index:04d}", "text": c.text,
             "metadata": {"document": c.source, "source": c.source,
                          "language": c.language, "heading": c.heading}})
    return out


def _accepted_sections(case: dict) -> list[tuple[str, str, str]]:
    """Resolve the case's accepted topic ids through topic_map (exact id)."""
    import topic_map
    if not case.get("accepted_topic_ids"):
        return []
    entries = topic_map.build_topic_map([DOCS])
    by_id = {e["topic_id"]: e for e in entries}
    out = []
    for tid in case["accepted_topic_ids"]:
        e = by_id.get(tid)
        out.append((tid, e["document"], (e.get("heading") or "").strip())
                   if e else (tid, None, None))
    return out


def measure_case(sufficiency, search, df, case, by_heading) -> dict:
    import evidence_plan
    question = case["question"]
    plan = evidence_plan.derive_plan(question)
    plan_requirements = [r["req"] for r in plan["requirements"]]
    for sub in (plan.get("sub_plans") or []):
        plan_requirements += [r["req"] for r in sub["requirements"]]
    plan_requirements = list(dict.fromkeys(plan_requirements))
    law_map, numeric_ids = sufficiency._law_context()

    hits = search(question, 20)
    pool_ids = [h["id"] for h in hits]

    evidence_rows = []
    evidence_chunks: dict[str, dict] = {}
    for tid, document, heading in _accepted_sections(case):
        if document is None:
            evidence_rows.append({"topic_id": tid, "resolved": False})
            continue
        for chunk in by_heading.get((document, heading), []):
            evidence_chunks[chunk["id"]] = chunk
            evidence_rows.append({
                "topic_id": tid, "resolved": True,
                "heading": heading, "chunk_id": chunk["id"],
                "in_top20_pool": chunk["id"] in pool_ids,
                "pool_position": (pool_ids.index(chunk["id"]) + 1
                                  if chunk["id"] in pool_ids else None),
                "shared_terms": sorted(
                    sufficiency.shared_terms(question, chunk["text"]))[:8],
                "anchored": sufficiency._anchors(question, chunk["text"], df),
                "shape": {req: sufficiency.shape_ok(req, chunk, law_map,
                                                    numeric_ids, question)
                          for req in plan_requirements},
            })

    one_shot = sufficiency.check(question, hits, df=df, top_k=20)
    guided = sufficiency.check(question, hits, df=df, search_fn=search,
                               top_k=20)

    # GATE PROBE — isolates the coverage gate from retrieval depth: the same
    # check, but with the case's own resolved evidence chunks forced into the
    # pool. «كافٍ» here means the gate accepts the evidence when retrieval
    # delivers it; it says nothing about whether retrieval delivers it.
    forced_ids = {h["id"] for h in hits}
    forced_pool = list(hits)
    for chunk_id, chunk in evidence_chunks.items():
        if chunk_id not in forced_ids:
            forced_pool.append(chunk)
            forced_ids.add(chunk_id)
    forced = (sufficiency.check(question, forced_pool, df=df,
                                top_k=len(forced_pool))
              if forced_pool else one_shot)

    # where the evidence actually sits on THIS arm (BM25), beyond the top-20
    deep = search(question, 60)
    deep_ids = [h["id"] for h in deep]
    for ev in evidence_rows:
        if ev.get("resolved"):
            ev["position_k60"] = (deep_ids.index(ev["chunk_id"]) + 1
                                  if ev["chunk_id"] in deep_ids else None)

    return {
        "id": case["id"], "language": case.get("language"),
        "question": question,
        "cross_script": sufficiency._is_cross_script(question),
        "question_terms": sufficiency.question_terms(question),
        "bridged_terms": sorted(sufficiency._bridged_terms(question)),
        "requirements": plan_requirements,
        "evidence_chunks": evidence_rows,
        "one_shot": {"state": one_shot["state"], "reason": one_shot["reason"],
                     "missing": one_shot["missing"],
                     "covered_by": {r["req"]: r["by"]
                                    for r in one_shot["requirements"]}},
        "guided": {"state": guided["state"], "missing": guided["missing"],
                   "guided_rounds": guided["guided_rounds"]},
        "gate_probe_forced_pool": {
            "state": forced["state"], "missing": forced["missing"],
            "covered_by": {r["req"]: r["by"] for r in forced["requirements"]
                           if r["by"]},
            "pool_size": len(forced_pool)},
    }


def run(regression: bool) -> dict:
    import sufficiency
    sufficiency, search, df = _corpus_and_df()
    by_heading = _chunks_by_heading()
    cases = json.loads(PROBES.read_text(encoding="utf-8"))["cases"]

    _banner("arm: field_bridges=%s definition_shape=%s | corpus=%d chunks "
            "| df_cap=%d" % (
                int(sufficiency.FIELD_BRIDGES_ENABLED),
                int(sufficiency.DEFINITION_SHAPE_ENABLED),
                df.get("__corpus_size__", -1), sufficiency._df_cap(df)))

    report = {"arm": {g: int(bool(os.getenv(g) == "1")) for g in GATES},
              "corpus_size": df.get("__corpus_size__"),
              "cases": [measure_case(sufficiency, search, df, c, by_heading)
                        for c in cases]}

    for row in report["cases"]:
        print("\n--- %s [%s] cross_script=%s" % (
            row["id"], row["language"], row["cross_script"]))
        print("    question     : %s" % row["question"])
        print("    requirement(s): %s" % row["requirements"])
        print("    bridged terms: %s" % (row["bridged_terms"] or "[]"))
        for ev in row["evidence_chunks"]:
            if not ev.get("resolved"):
                print("    topic %s -> NOT RESOLVED in topic_map"
                      % ev["topic_id"])
                continue
            print("    evidence %s" % ev["chunk_id"])
            print("       heading      : %s" % ev["heading"])
            print("       in top-20    : %s (position %s) | BM25 rank @60: %s"
                  % (ev["in_top20_pool"], ev["pool_position"],
                     ev.get("position_k60")))
            print("       shared_terms : %s" % (ev["shared_terms"] or "[]"))
            print("       anchored     : %s" % ev["anchored"])
            print("       shape        : %s" % ev["shape"])
        print("    one-shot state : %s | missing=%s | covered_by=%s" % (
            row["one_shot"]["state"], row["one_shot"]["missing"],
            {k: v for k, v in row["one_shot"]["covered_by"].items() if v}))
        print("    guided state   : %s | missing=%s | rounds=%s" % (
            row["guided"]["state"], row["guided"]["missing"],
            row["guided"]["guided_rounds"]))
        gp = row["gate_probe_forced_pool"]
        print("    GATE PROBE     : %s | missing=%s | covered_by=%s "
              "(pool forced to %d)" % (
                  gp["state"], gp["missing"], gp["covered_by"],
                  gp["pool_size"]))

    if regression:
        _banner("regression — the two FROZEN adopted sets on the same arm")
        report["regression"] = {}
        for name in ("questions_50.json", "questions_targets.json"):
            cases_set = json.loads(
                (RAGLAB / name).read_text(encoding="utf-8"))["cases"]
            m = sufficiency.measure(cases_set, search, k=20, df=df)
            report["regression"][name] = {
                k: m[k] for k in ("n", "false_sufficient", "false_refusal",
                                  "escalated", "guided_rescues",
                                  "oos_all_insufficient", "agreement")}
            report["regression"][name]["states"] = {}
            for r in m["per_case"]:
                s = report["regression"][name]["states"]
                s[r["state"]] = s.get(r["state"], 0) + 1
            report["regression"][name]["disagreements"] = [
                {k: r[k] for k in ("id", "category", "state", "expected",
                                   "effective", "missing")}
                for r in m["per_case"]
                if r["false_sufficient"] or r["false_refusal"]]
            print("\n%s" % name)
            print(json.dumps(report["regression"][name], ensure_ascii=False,
                             indent=1))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", action="store_true",
                    help="re-run once per gate combination and print the 2x2")
    ap.add_argument("--regression", action="store_true",
                    help="also measure the two frozen adopted sets")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the machine-readable report here")
    args = ap.parse_args()

    if args.arms:
        summary = []
        for fb in (0, 1):
            for ds in (0, 1):
                env = {**os.environ,
                       GATES[0]: str(fb), GATES[1]: str(ds)}
                cmd = [sys.executable, str(Path(__file__).resolve()),
                       "--json", str((args.json or Path("/tmp/xcov.json"))
                                     .with_suffix(".%d%d.json" % (fb, ds)))]
                if args.regression:
                    cmd.append("--regression")
                print("\n\n########## ARM field_bridges=%d "
                      "definition_shape=%d ##########" % (fb, ds))
                subprocess.run(cmd, env=env, check=False)
                payload = json.loads(
                    Path(cmd[cmd.index("--json") + 1]).read_text(
                        encoding="utf-8"))
                summary.append({
                    "arm": ARM_NAMES[(fb, ds)],
                    "states": {c["id"]: c["one_shot"]["state"]
                               for c in payload["cases"]},
                    "gate_probe": {
                        c["id"]: c["gate_probe_forced_pool"]["state"]
                        for c in payload["cases"]},
                    "regression": payload.get("regression"),
                })
        _banner("2x2 SUMMARY (one-shot state / gate-probe state per case)")
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0

    report = run(args.regression)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
