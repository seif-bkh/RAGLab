#!/usr/bin/env python3
"""Model-independence A/B: fix the retrieval, swap the answer model, measure divergence.

The target-state acceptance test (RAGLAB_GAP_ANALYSIS.md §8; target report §4/§9):
send the SAME retrieved context to different answer models. If the answers
diverge IN SUBSTANCE (not just wording), the context still admits more than one
reading — the work then returns to evidence completeness and clarity, not to
swapping models.

Retrieval is model-independent BY CONSTRUCTION in this pipeline (the answer
model never rewrites the query; pipeline_policy refuses query translation), so
this tool retrieves ONCE per question with the active embedding profile and
feeds the identical hits to every requested answer model. The citation gate is
the automated judge: a model that cannot produce verbatim-supported, numerically
grounded claims is refused — those refusals are counted per model.

Usage (CI or a keyed environment; spends answer-model calls):
  .venv/bin/python answer_ab.py --questions questions_v2.json \
      --model xkiro/qwen/qwen3.8-max:free \
      --model nvidia/nvidia/nemotron-3.5-lightning-30b-a3b

Each --model is provider/model-id wired exactly like the console's answer slot
(profiles.build_generator). Output: results/answer_ab/<set>/report.json plus a
compact printed table. The first --model is the reference arm; every other arm
is compared against it pairwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import profiles                                                # noqa: E402
from evaluate import load_question_set, prepare_query_text     # noqa: E402
from retrieval import retrieve                                 # noqa: E402
from translate import detect_language                          # noqa: E402
from artifacts import write_json                               # noqa: E402


# ---------------------------------------------------------------------------
# Core comparison (testable offline with injected factories)
# ---------------------------------------------------------------------------

def _arm_row(result: dict) -> dict:
    """Compact per-model view of one answer result."""
    sources = sorted({s.get("chunk_id") for s in result.get("sources") or []})
    return {
        "status": result.get("status"),
        "reason": result.get("reason"),
        "model": result.get("model"),
        "claims": len(result.get("claims") or []),
        "source_ids": sources,
        "validation_ok": result.get("validation_ok", True),
    }


def _jaccard(a: list, b: list) -> float | None:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return None            # both refused with no sources: not a divergence
    return len(sa & sb) / len(sa | sb)


def run_comparison(local, embedder, collection, cases: list, generators: list,
                   *, top_k: int | None = None, mode: str = "vector") -> dict:
    """One retrieval per question, every generator answers the same hits.

    generators: list of (label, generator) where generator.answer(question,
    hits, language, use_cache=False) returns the AnswerGenerator result dict.
    Returns the full report dict (metrics + per-question rows).
    """
    per_question = []
    for case in cases:
        question = case["question"]
        language = case.get("language") or detect_language(question)
        hits, _variants = retrieve(
            local, embedder, collection, prepare_query_text(question),
            language=language, mode=mode,
            top_k=top_k or local.ANSWER_TOP_K, variant_strategy="original")
        arms = {}
        for label, generator in generators:
            result = generator.answer(question, hits, language, use_cache=False)
            arms[label] = _arm_row(result)
        per_question.append({
            "id": case["id"],
            "question": question,
            "language": language,
            "category": case["category"],
            "retrieved": len(hits),
            "retrieved_ids": [h["id"] for h in hits],
            "arms": arms,
        })

    labels = [label for label, _ in generators]
    reference = labels[0]
    per_model = {
        label: {
            "answered": sum(1 for q in per_question
                            if q["arms"][label]["status"] == "answered"),
            "refused": sum(1 for q in per_question
                           if q["arms"][label]["status"] == "refused"),
            "gate_rejected": sum(
                1 for q in per_question
                if q["arms"][label]["reason"] in ("invalid_output",
                                                  "unsourced_number")),
            "mean_claims": (
                sum(q["arms"][label]["claims"] for q in per_question)
                / len(per_question) if per_question else None),
        }
        for label in labels
    }
    pairwise = {}
    for label in labels[1:]:
        pairs = [(q["arms"][reference], q["arms"][label]) for q in per_question]
        n = len(pairs) or 1
        status_agree = sum(1 for a, b in pairs if a["status"] == b["status"])
        jaccards = [j for j in (_jaccard(a["source_ids"], b["source_ids"])
                                for a, b in pairs) if j is not None]
        both_answered = [ (a, b) for a, b in pairs
                          if a["status"] == "answered" and b["status"] == "answered"]
        pairwise[f"{reference} vs {label}"] = {
            "n": len(pairs),
            "status_agreement": status_agree / n,
            "mean_source_jaccard": (sum(jaccards) / len(jaccards)
                                    if jaccards else None),
            "mean_source_jaccard_when_both_answered": (
                sum(j for j in (_jaccard(a["source_ids"], b["source_ids"])
                                for a, b in both_answered) if j is not None)
                / max(1, len(both_answered)) if both_answered else None),
            "substance_divergent_questions": [
                q["id"] for q in per_question
                if q["arms"][reference]["status"] != q["arms"][label]["status"]
                or (_jaccard(q["arms"][reference]["source_ids"],
                            q["arms"][label]["source_ids"]) or 0.0) < 0.5
            ],
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions": len(cases),
        "reference_arm": reference,
        "retrieval": {"mode": mode, "top_k": top_k or local.ANSWER_TOP_K,
                      "embedding_model": local.active_embedding_model(),
                      "model_independent": True},
        "per_model": per_model,
        "pairwise": pairwise,
        "per_question": per_question,
    }


def print_report(report: dict) -> None:
    print("=" * 78)
    print(f"MODEL-INDEPENDENCE A/B | {report['questions']} question(s) | "
          f"retrieval fixed ({report['retrieval']['mode']}, "
          f"top {report['retrieval']['top_k']}, "
          f"{report['retrieval']['embedding_model']})")
    print("=" * 78)
    print(f"{'arm':<44}{'answered':>9}{'refused':>9}{'gate':>6}{'claims':>8}")
    for label, d in report["per_model"].items():
        print(f"{label:<44}{d['answered']:>9}{d['refused']:>9}"
              f"{d['gate_rejected']:>6}"
              f"{(d['mean_claims'] or 0):>8.1f}")
    for pair, d in report["pairwise"].items():
        jac = d["mean_source_jaccard"]
        jac_txt = f"{jac:.3f}" if jac is not None else "n/a"
        print(f"\n{pair}: status agreement={d['status_agreement']:.3f} | "
              f"mean source Jaccard={jac_txt}")
        divergent = d["substance_divergent_questions"]
        if divergent:
            print(f"  substance-divergent questions ({len(divergent)}): "
                  f"{', '.join(divergent)}")
            print("  -> the context still admits more than one reading: the "
                  "work returns to evidence completeness, not the model.")
        else:
            print("  -> no substance divergence: differences are wording-only.")
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI (live; needs keys — the sandbox cannot reach providers)
# ---------------------------------------------------------------------------

def parse_model(spec: str) -> tuple[str, str]:
    provider, _, model = spec.partition("/")
    if not provider or not model:
        raise SystemExit(f"--model must be provider/model-id, got {spec!r}")
    return provider.strip().lower(), model.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="answer_ab", description=__doc__.splitlines()[0])
    parser.add_argument("--questions", default="questions_v2.json",
                        help="question-set file name in raglab/ or absolute path")
    parser.add_argument("--model", action="append", required=True,
                        metavar="PROVIDER/MODEL",
                        help="answer arm; repeat for each model to compare "
                             "(first is the reference)")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--mode", default="vector",
                        choices=["vector", "rrf", "blend"])
    parser.add_argument("--results", default=None,
                        help="output dir under raglab/results/ "
                             "(default: answer_ab/<question set stem>)")
    args = parser.parse_args()

    import embedder as embedder_mod
    from store import _client

    q_path = Path(args.questions)
    questions_path = q_path if q_path.is_absolute() else HERE / q_path
    cases = load_question_set(questions_path)

    state = profiles.default_state()
    local = profiles.build_lab_config(state)
    embedder_obj = embedder_mod.build_embedder(local)
    collection = _client(local).get_or_create_collection(
        name=local.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"})
    if not collection.count():
        print("[answer_ab] index empty — ingesting the corpus first")
        profiles.ingest(local, embedder_obj, state, reset=True)

    generators = []
    for spec in args.model:
        provider, model = parse_model(spec)
        arm_state = json.loads(json.dumps(state))     # deep copy of the profile
        arm_state["answer"] = {"provider": provider, "model": model}
        arm_local = profiles.build_lab_config(arm_state)
        label = f"{provider}/{model}"
        generators.append((label, profiles.build_generator(arm_local)))
        print(f"[answer_ab] arm ready: {label}")

    report = run_comparison(local, embedder_obj, collection, cases, generators,
                            top_k=args.top_k, mode=args.mode)
    print_report(report)

    out_dir = (HERE / "results" / args.results if args.results
               else HERE / "results" / "answer_ab" / questions_path.stem)
    write_json(out_dir / "report.json", report)
    print(f"[answer_ab] report saved to {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
