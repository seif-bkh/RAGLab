#!/usr/bin/env python3
"""Build the real-model A/B comparison from two `main.py evaluate` run JSONs.

Usage (from raglab/):
    python real_report.py results/harness50/real_eval_size.json \
                          results/harness50/real_eval_restructure.json \
                          --out results/harness50/real_models.md \
                          [--answer results/harness50/real_answer_murabaha.json]

Input schema: the run JSON that `evaluate.save_run` writes — `generated_at`,
`config`, `metrics` {overall, by_category, by_language, separation,
out_of_scope}, `questions` [per-question records with correct_rank, etc.].

Output:
  - the markdown comparison to `--out` (default results/harness50/real_models.md)
  - the same markdown on stdout (so it lands in the CI log)
  - a final line `ANNO|<compact one-line summary>` that the CI workflow pipes
    into a `::warning::` annotation. The annotation is the channel the sandbox
    can read through the check-run annotations API (job logs are not
    downloadable from there), so it must be self-sufficient: overall / by
    category / by language / separation / OOS / flips / misses / answer smoke.

Standard library only; no API calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ARMS = ("size", "restructure")
ARM_LABEL = {"size": "size (legacy recursive 220/40)",
             "restructure": "restructure (normalization + enrichment + structural split)"}
CATS = ("verbatim", "paraphrase", "cross-lingual")
LANGS = ("ar", "en", "fr")


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pct(value, digits=1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def fnum(value, digits=4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def rank_str(rank, top_k) -> str:
    if rank is None:
        return f">={top_k}" if top_k else "miss"
    return str(rank)


def pp(d1, d2) -> str:
    """Percentage-point delta between two metric dicts."""
    if d1.get("hit@1") is None or d2.get("hit@1") is None:
        return "n/a"
    delta = (d2["hit@1"] - d1["hit@1"]) * 100
    return f"{delta:+.1f} pp"


def metric_row(arm: str, d: dict) -> list:
    return [d.get("n", 0), pct(d.get("hit@1")), pct(d.get("hit@3")), pct(d.get("hit@5"))]


def build_md(runs: dict, answer: dict | None, top_k: int) -> str:
    size, restr = runs["size"], runs["restructure"]
    conf_s, conf_r = size["config"], restr["config"]
    out: list[str] = []
    w = out.append

    sha = os.environ.get("GITHUB_SHA", "local")
    w("# Real-model A/B — new restructure chunking vs legacy size chunking")
    w("")
    w(f"- Generated: {max(r['generated_at'] for r in runs.values())} by real-test.yml "
      f"(commit `{sha[:12]}`)")
    w(f"- Retrieval: provider={conf_s.get('provider')} model={conf_s.get('embedding_model')} "
      f"mode={conf_s.get('retrieval_mode')} top_k={conf_s.get('retrieval_top_k')}")
    w(f"- Chunking arms: size={conf_s.get('chunk_size_tokens')}/{conf_s.get('chunk_overlap_tokens')} "
      f"vs restructure={conf_r.get('chunk_size_tokens')}/{conf_r.get('chunk_overlap_tokens')} "
      "(same budget, structural split)")
    w(f"- Questions: questions_50.json — {len(size['questions'])} cases "
      f"({len([q for q in size['questions'] if not q['is_out_of_scope']])} evaluable + "
      f"{len([q for q in size['questions'] if q['is_out_of_scope']])} out-of-scope); "
      "translation disabled, one arm per collection")
    w("")
    w("## Overall hit rate (evaluable questions)")
    w("")
    w("| arm | n | hit@1 | hit@3 | hit@5 |")
    w("|---|---|---|---|---|")
    for arm in ARMS:
        row = metric_row(arm, runs[arm]["metrics"]["overall"])
        w(f"| {ARM_LABEL[arm]} | " + " | ".join(str(x) for x in row) + " |")
    o_s, o_r = size["metrics"]["overall"], restr["metrics"]["overall"]
    w("")
    w(f"**Δ (restructure − size):** hit@1 {pp(o_s, o_r)}, hit@3 "
      f"{(o_r['hit@3'] - o_s['hit@3']) * 100:+.1f} pp, hit@5 "
      f"{(o_r['hit@5'] - o_s['hit@5']) * 100:+.1f} pp"
      if None not in (o_s.get("hit@1"), o_r.get("hit@1"), o_s.get("hit@3"),
                      o_r.get("hit@3"), o_s.get("hit@5"), o_r.get("hit@5"))
      else "**Δ:** n/a")
    w("")

    for title, key, keys in (("By category", "by_category", CATS),
                             ("By query language", "by_language", LANGS)):
        w(f"## {title}")
        w("")
        w("| key | n | size hit@1/3/5 | restructure hit@1/3/5 |")
        w("|---|---|---|---|")
        m_s, m_r = size["metrics"][key], restr["metrics"][key]
        seen = []
        for k in keys:
            if k in m_s or k in m_r:
                seen.append(k)
        for k in [k for k in sorted(set(m_s) | set(m_r)) if k not in seen]:
            seen.append(k)
        for k in seen:
            d_s, d_r = m_s.get(k) or {}, m_r.get(k) or {}
            n = d_s.get("n", d_r.get("n", 0))
            s3 = " / ".join(pct(d_s.get(f"hit@{c}")) for c in (1, 3, 5))
            r3 = " / ".join(pct(d_r.get(f"hit@{c}")) for c in (1, 3, 5))
            w(f"| {k} | {n} | {s3} | {r3} |")
        w("")

    w("## Separation (mean ranking score)")
    w("")
    w("| arm | n correct retrieved | mean correct | mean best-incorrect | gap |")
    w("|---|---|---|---|---|")
    for arm in ARMS:
        s = runs[arm]["metrics"]["separation"]
        w(f"| {ARM_LABEL[arm]} | {s['n_correct_retrieved']} | {fnum(s['mean_correct_score'])} "
          f"| {fnum(s['mean_best_incorrect_score'])} | {fnum(s['gap_mean_correct_minus_best_incorrect'])} |")
    w("")

    w("## Out-of-scope (no expected match; what a refusal threshold must beat)")
    w("")
    w("| arm | n | max top-1 | mean top-1 |")
    w("|---|---|---|---|")
    for arm in ARMS:
        o = runs[arm]["metrics"]["out_of_scope"]
        w(f"| {ARM_LABEL[arm]} | {o['n']} | {fnum(o['max_top1_score'])} | {fnum(o['mean_top1_score'])} |")
    w("")
    w("| id | question | size top-1 | restructure top-1 |")
    w("|---|---|---|---|")
    oos = {q["id"]: q for q in size["questions"] if q["is_out_of_scope"]}
    for qid in sorted(oos):
        q = oos[qid]
        r = {qq["id"]: qq for qq in restr["questions"] if qq["id"] == qid}
        s_score = q["hits"][0]["score"] if q["hits"] else None
        r_score = (r[qid]["hits"][0]["score"] if r.get(qid, {}).get("hits") else None)
        w(f"| {qid} | {q['question'][:60]} | {fnum(s_score)} | {fnum(r_score)} |")
    w("")

    # Per-question side-by-side -------------------------------------------------
    w("## Per-question comparison (evaluable)")
    w("")
    w("| id | lang | category | question (trunc.) | size rank | restructure rank | note |")
    w("|---|---|---|---|---|---|---|")
    r_by_id = {q["id"]: q for q in restr["questions"]}
    flips_up, flips_down, note_rank = [], [], []
    for q in size["questions"]:
        if q["is_out_of_scope"]:
            continue
        rq = r_by_id.get(q["id"])
        if rq is None:
            continue
        rs, rr = q["correct_rank"], rq["correct_rank"]
        note = ""
        if rs != rr:
            if rs is None and rr is not None:
                note = "FLIP miss→hit"
                flips_up.append((q["id"], rs, rr))
            elif rs is not None and rr is None:
                note = "FLIP hit→miss"
                flips_down.append((q["id"], rs, rr))
            else:
                note = f"rank {rs}→{rr}"
                note_rank.append((q["id"], rs, rr))
        if rq.get("correct_any_lang_rank") is not None and rr is None:
            note = (note + "; " if note else "") + \
                f"answer present any-lang at {rq['correct_any_lang_rank']} (language routing)"
        w(f"| {q['id']} | {q.get('language')} | {q['category']} | {q['question'][:50]} "
          f"| {rank_str(rs, top_k)} | {rank_str(rr, top_k)} | {note} |")
    w("")
    w(f"Rank shifts: {len(note_rank)} questions moved ≥1 place; "
      f"{len(flips_up)} flipped miss→hit; {len(flips_down)} flipped hit→miss.")
    w("")

    w("## Misses per arm (correct chunk not in top-k)")
    w("")
    for arm in ARMS:
        misses = [q for q in runs[arm]["questions"]
                  if not q["is_out_of_scope"] and q["correct_rank"] is None]
        w(f"**{ARM_LABEL[arm]}** ({len(misses)}): " +
          (", ".join(f"{q['id']} ({q.get('language')}/{q['category']})" for q in misses)
           or "none"))
        w("")

    if answer is not None:
        w("## Answer smoke test (xKiro/Qwen, restructure collection)")
        w("")
        w(f"- question: {answer.get('question', '?')}")
        w(f"- status={answer.get('status')} reason={answer.get('reason')} "
          f"validation_ok={answer.get('validation_ok')} inference_performed={answer.get('inference_performed')}")
        claims = answer.get("claims") or []
        sources = answer.get("sources") or []
        w(f"- claims={len(claims)} sources={len(sources)}")
        if claims:
            w(f"- first claim: {claims[0]['text'][:160]}")
        w(f"- answer text: {str(answer.get('answer', ''))[:300]}")
        w("")

    w("## Reading notes")
    w("")
    w("- BM25-only reference (harness50, no embeddings): size 40/69/80 vs restructure "
      "47/78/89 overall hit@1/3/5; the five restructure misses there were q26–q30 "
      "(fr→ar cross-lingual) — the embedding arm is expected to close them.")
    w("- hit@1 with real embeddings measures whether the correct chunk is the single "
      "nearest neighbour; a healthy gap in the separation table is what makes a "
      "refusal threshold possible on the OOS score.")
    w("")
    return "\n".join(out) + "\n"


def build_anno(runs: dict, answer: dict | None) -> str:
    """The one self-sufficient line the workflow echoes as ::warning::."""
    size, restr = runs["size"], runs["restructure"]
    parts = []

    def triple(d):
        return (f"{(d['hit@1'] or 0) * 100:.0f}/{(d['hit@3'] or 0) * 100:.0f}"
                f"/{(d['hit@5'] or 0) * 100:.0f}" if d.get("hit@1") is not None else "n/a")

    o_s, o_r = size["metrics"]["overall"], restr["metrics"]["overall"]
    parts.append(f"overall n={o_s['n']} size={triple(o_s)} restructure={triple(o_r)}")
    for key, keys in (("by_cat", CATS), ("by_lang", LANGS)):
        seg = []
        m_s, m_r = size["metrics"][key.replace("_cat", "_category").replace("_lang", "_language")], \
            restr["metrics"][key.replace("_cat", "_category").replace("_lang", "_language")]
        for k in keys:
            d_s, d_r = m_s.get(k) or {}, m_r.get(k) or {}
            if not d_s and not d_r:
                continue
            seg.append(f"{k} {d_s.get('n', d_r.get('n', '?'))}: {triple(d_s)}->{triple(d_r)}")
        if seg:
            parts.append(key + " " + " ".join(seg))
    s_s, s_r = size["metrics"]["separation"], restr["metrics"]["separation"]
    parts.append(f"separation gap size={fnum(s_s['gap_mean_correct_minus_best_incorrect'], 3)} "
                 f"restructure={fnum(s_r['gap_mean_correct_minus_best_incorrect'], 3)}")
    o_s2, o_r2 = size["metrics"]["out_of_scope"], restr["metrics"]["out_of_scope"]
    parts.append(f"oos max/mean size={fnum(o_s2['max_top1_score'], 3)}/{fnum(o_s2['mean_top1_score'], 3)} "
                 f"restructure={fnum(o_r2['max_top1_score'], 3)}/{fnum(o_r2['mean_top1_score'], 3)}")
    r_by_id = {q["id"]: q for q in restr["questions"]}
    up, down = [], []
    for q in size["questions"]:
        if q["is_out_of_scope"]:
            continue
        rq = r_by_id.get(q["id"])
        if rq is None:
            continue
        if q["correct_rank"] is None and rq["correct_rank"] is not None:
            up.append(f"{q['id']}({q.get('language')})->{rq['correct_rank']}")
        elif q["correct_rank"] is not None and rq["correct_rank"] is None:
            down.append(f"{q['id']}({q.get('language')}) rank{q['correct_rank']}->miss")
    parts.append("flips +[" + ",".join(up) + "] -[" + ",".join(down) + "]")
    for arm in ARMS:
        misses = [q["id"] for q in runs[arm]["questions"]
                  if not q["is_out_of_scope"] and q["correct_rank"] is None]
        parts.append(f"misses {arm}=[" + ",".join(misses) + "]")
    # Top-1 detail for restructure misses so the annotation is self-sufficient:
    # what the retriever actually returned instead of the expected chunk.
    rmiss = [q for q in restr["questions"]
             if not q["is_out_of_scope"] and q["correct_rank"] is None]
    if rmiss:
        seg = []
        for q in rmiss:
            if q["hits"]:
                h = q["hits"][0]
                seg.append(f"{q['id']} top1={h['score']:.3f} "
                           f"{h.get('document', '?')} | {str(h.get('heading') or '?')[:44]}")
            else:
                seg.append(f"{q['id']} no-hits")
        parts.append("restr_miss_detail: " + "; ".join(seg))
    if answer is not None:
        parts.append(f"answer status={answer.get('status')} "
                     f"validation_ok={answer.get('validation_ok')} "
                     f"claims={len(answer.get('claims') or [])} "
                     f"sources={len(answer.get('sources') or [])}")
    return "ANNO| " + " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eval_size", help="run JSON from the size arm")
    ap.add_argument("eval_restructure", help="run JSON from the restructure arm")
    ap.add_argument("--out", default="results/harness50/real_models.md",
                    help="where to write the markdown comparison")
    ap.add_argument("--answer", default=None,
                    help="optional answer-smoke JSON to summarize")
    args = ap.parse_args()

    runs = {"size": load(args.eval_size), "restructure": load(args.eval_restructure)}
    top_k = (runs["size"]["config"].get("retrieval_top_k")
             or runs["restructure"]["config"].get("retrieval_top_k") or 20)
    answer = load(args.answer) if args.answer and Path(args.answer).exists() else None

    md = build_md(runs, answer, top_k)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    sys.stdout.write(md)
    sys.stdout.write(build_anno(runs, answer) + "\n")
    print(f"[real_report] markdown written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
