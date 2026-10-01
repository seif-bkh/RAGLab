#!/usr/bin/env python3
"""Unit-description indexing — Phase 4, item 5 (raglab/audits/PHASE4_KNOWLEDGE.md).

Indexes the 198 unit DESCRIPTION lines (item 1) on the EXISTING retrieval
machinery in an ISOLATED experimental collection, then measures the law
questions (the 12 Loi cases of the adopted 50-set) against the adopted
baseline. The gate: correct-evidence appearance ≥ baseline.

What "correct" means here: the retrieved unit's VERBATIM article text
contains the case's expected_substring (same normalized-containment rule as
the deployed evaluation — is_correct_hit). A unit is bigger than a chunk, so
containment is the honest comparable: the question is whether the small
description surface still surfaces the RIGHT article.

Modes:
- local/fake (tests): any embedder-like object works — machinery check only.
- --live: builds the real provider from the lab config, embeds the 198
  descriptions + the 12 queries, optionally reads the same-run restructure
  evaluation JSON for the per-question baseline, prints an ANNO line, and
  writes the measurement JSON (declared-data friendly).

The collection is isolated (its own chroma dir under results/) — the deployed
collections are never touched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate   # noqa: E402  — normalize_for_match (the shared matching rule)
import restructure  # noqa: E402
import units      # noqa: E402

LAW_DOC = "Loi_2016-48.pdf"
UNIT_COLLECTION = "phase4_unit_index"
TOP_K = 20


def law_units() -> list[dict]:
    codex = restructure._adopted_codex_text(LAW_DOC)
    if codex is None:
        raise SystemExit(f"no adopted codex for {LAW_DOC}")
    return units.extract_law_units(codex)


def law_cases(cases: list[dict]) -> list[dict]:
    """The 12 Loi cases of the adopted 50-set."""
    return [c for c in cases if c.get("expected_document") == LAW_DOC]


def unit_contains_evidence(unit_text: str, case: dict) -> bool:
    """The shared correctness rule: expected substring inside the unit text."""
    sub = case.get("expected_substring") or ""
    if not sub:
        return False
    return evaluate.normalize_for_match(sub) in evaluate.normalize_for_match(unit_text)


def build_unit_collection(store, units_list: list[dict], embedder, chroma_dir: Path,
                          reset: bool = True):
    """An ISOLATED collection over the description lines (existing machinery)."""
    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_dir))
    if reset:
        try:
            client.delete_collection(UNIT_COLLECTION)
        except Exception:
            pass
    col = client.get_or_create_collection(name=UNIT_COLLECTION,
                                          metadata={"hnsw:space": "cosine"})
    vectors = embedder.embed_texts([u["description"] for u in units_list])
    for s in range(0, len(units_list), 64):
        part = units_list[s:s + 64]
        vs = vectors[s:s + 64]
        col.add(ids=[u["unit_id"] for u in part],
                embeddings=vs,
                documents=[u["description"] for u in part],
                metadatas=[{"unit_id": u["unit_id"], "heading": u["heading"],
                            "path": u["path"], "type": u["type"],
                            "language": "ar"} for u in part])
    return col


def evaluate_units_index(units_list: list[dict], cases: list[dict], embedder,
                         chroma_dir: Path, top_k: int = TOP_K) -> dict:
    col = build_unit_collection(None, units_list, embedder, chroma_dir)
    by_id = {u["unit_id"]: u for u in units_list}
    per_question = []
    for case in cases:
        qv = embedder.embed_query(case["question"])
        got = col.query(query_embeddings=[qv], n_results=min(top_k, len(units_list)),
                        include=["metadatas", "documents", "distances"])
        hits = [{"unit_id": m.get("unit_id") or d_id,
                 "rank": i + 1,
                 "distance": got["distances"][0][i]}
                for i, (m, d_id) in enumerate(zip(got["metadatas"][0], got["ids"][0]))]
        correct_rank = None
        for h in hits:
            if unit_contains_evidence(by_id[h["unit_id"]]["text"], case):
                correct_rank = h["rank"]
                break
        per_question.append({
            "id": case["id"], "category": case.get("category"),
            "question": case["question"],
            "correct_unit_rank": correct_rank,
            "hit_at_1": bool(correct_rank is not None and correct_rank <= 1),
            "hit_at_3": bool(correct_rank is not None and correct_rank <= 3),
            "hit_at_5": bool(correct_rank is not None and correct_rank <= 5),
            "top_unit": hits[0]["unit_id"] if hits else None,
        })
    n = len(per_question)
    metrics = {
        "n": n,
        "hit@1": sum(q["hit_at_1"] for q in per_question) / n if n else None,
        "hit@3": sum(q["hit_at_3"] for q in per_question) / n if n else None,
        "hit@5": sum(q["hit_at_5"] for q in per_question) / n if n else None,
    }
    return {"metrics": metrics, "questions": per_question}


def baseline_for_law_questions(eval_json: Path, case_ids: list[str]) -> dict:
    """Same-run baseline: the 12 law questions' hit@k in the restructure arm."""
    run = json.loads(Path(eval_json).read_text(encoding="utf-8"))
    wanted = set(case_ids)
    rows = [q for q in run.get("questions", []) if q.get("id") in wanted
            and not q.get("is_out_of_scope")]
    n = len(rows)
    return {
        "n": n,
        "hit@1": sum(q["hit_at_1"] for q in rows) / n if n else None,
        "hit@3": sum(q["hit_at_3"] for q in rows) / n if n else None,
        "hit@5": sum(q["hit_at_5"] for q in rows) / n if n else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="unit_index",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true",
                        help="use the real provider from the lab config")
    parser.add_argument("--baseline", default=None,
                        help="same-run restructure eval JSON for the baseline")
    parser.add_argument("--questions", default="questions_50.json")
    parser.add_argument("--out", default="results/harness50/unit_index_eval.json")
    args = parser.parse_args()

    units_list = law_units()
    cases = law_cases(json.loads(
        (HERE / args.questions).read_text(encoding="utf-8"))["cases"])
    print(f"[unit_index] {len(units_list)} units | {len(cases)} law cases")

    if args.live:
        import config as cfg_mod
        from embedder import build_embedder
        cfg = cfg_mod  # module-level config carries env-derived settings
        embedder = build_embedder(cfg)
        chroma_dir = HERE / "results" / "unit_index_chroma"
    else:
        raise SystemExit("local mode is test-only (fake embedders); use --live")

    result = evaluate_units_index(units_list, cases, embedder, chroma_dir)
    m = result["metrics"]
    line = (f"units-index n={m['n']} "
            f"hit@1/3/5={m['hit@1']*100:.0f}/{m['hit@3']*100:.0f}/{m['hit@5']*100:.0f}")
    parts = [line]
    gate = None
    if args.baseline:
        base = baseline_for_law_questions(HERE / args.baseline,
                                          [c["id"] for c in cases])
        parts.append(f"baseline n={base['n']} "
                     f"hit@1/3/5={base['hit@1']*100:.0f}/{base['hit@3']*100:.0f}/"
                     f"{base['hit@5']*100:.0f}")
        gate = (m["hit@5"] >= base["hit@5"])   # the plan's gate: evidence ≥ baseline
        parts.append(f"gate(hit@5 >= baseline): {'PASS' if gate else 'FAIL'}")
        result["baseline"] = base
        result["gate"] = gate
    out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("ANNO| units-index | " + " | ".join(parts))
    print(f"[unit_index] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
