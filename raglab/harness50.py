#!/usr/bin/env python3
"""Offline BM25 harness: baseline size-chunking vs the restructure strategy.

Runs the two chunking arms over the same 6-document corpus and scores
questions_50.json (17 ar / 17 fr / 16 en, 5 out-of-scope) with the lab's own
judge (evaluate.is_correct_hit + evaluate.compute_metrics).

Why BM25-only: the pinned runtime expects NVIDIA embeddings that are not
reachable from this sandbox (no network to huggingface.co / pytorch.org), so
retrieval here is the lab's lexical path (store.keyword_search, k=20). The
retriever is therefore IDENTICAL for both arms — the only difference is how
the documents were chunked, which is exactly the variable under test.

Usage:  .venv/bin/python harness50.py
Output: results/harness50/{arm_baseline.json, arm_restructure.json,
        comparison.md, validation.md}
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from loader import load_all                                    # noqa: E402
import chunker                                                 # noqa: E402
import store                                                   # noqa: E402
import evaluate                                                # noqa: E402

RESULTS = HERE / "results" / "harness50"
DOCS_DIRS = [HERE.parent / "docs", HERE / "data"]
QUESTIONS = HERE / "questions_50.json"

# Same token budget for both arms: the ONLY difference is the strategy.
BUDGET_TOKENS = 220
OVERLAP_TOKENS = 40
TOP_K = 20
CHROMA_DIM = 8  # placeholder vectors — retrieval is BM25-only


# ---------------------------------------------------------------------------
# Config (standalone: no .env, no provider gate)
# ---------------------------------------------------------------------------

def make_cfg(mode: str, collection: str, chroma_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        CHUNKING_MODE=mode,
        CHUNK_SIZE_TOKENS=BUDGET_TOKENS,
        CHUNK_OVERLAP_TOKENS=OVERLAP_TOKENS,
        SPLIT_ON_HEADINGS_FIRST=True,
        CHUNK_OVERLAP_SENTENCE_AWARE=True,
        RESTRUCTURE_RTL_REPAIR="1",
        CHROMA_COLLECTION_NAME=collection,
        CHROMA_DIR=chroma_dir,
        STORE_BATCH_SIZE=64,
        INDEX_EXCLUDE_BOILERPLATE=False,
        KEYWORD_SEARCH_INCLUDE_METADATA=True,
    )


def placeholder_vector(text: str) -> list[float]:
    """Deterministic 8-dim stand-in. BM25 never looks at vectors; they only
    keep Chroma happy (cosine space needs real floats)."""
    h = abs(hash(text))
    return [((h >> (8 * i)) & 0xFF) / 255.0 for i in range(CHROMA_DIM)]


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def build_arm(name: str, mode: str, docs: list[dict]) -> dict:
    cfg = make_cfg(mode, f"harness50_{name}", RESULTS / f"chroma_{name}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"\n[harness50] ===== arm={name} | mode={mode} | "
          f"budget={BUDGET_TOKENS} overlap={OVERLAP_TOKENS} =====")

    chunks = chunker.chunk_all(docs, cfg)
    print(f"[harness50] {len(chunks)} chunk(s) | "
          f"{sum(c.token_count for c in chunks)} tokens total")

    client = store._client(cfg)
    try:
        client.delete_collection(cfg.CHROMA_COLLECTION_NAME)
        print(f"[harness50] deleted stale collection '{cfg.CHROMA_COLLECTION_NAME}'")
    except Exception:  # noqa: BLE001
        pass
    collection = client.get_or_create_collection(
        name=cfg.CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    fp = store.chunk_fp(cfg)
    batch = 64
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        collection.add(
            ids=[f"{c.source}::chunk_{c.index:04d}" for c in part],
            embeddings=[placeholder_vector(c.text) for c in part],
            documents=[c.text for c in part],
            metadatas=[{
                "document": c.source,
                "source": c.source,
                "language": c.language,
                "heading": c.heading,
                "chunk_index": c.index,
                "origin": c.origin,
                "section_type": c.section_type,
                "token_count": c.token_count,
                "chunk_fp": fp,
            } for c in part],
        )
    print(f"[harness50] collection '{cfg.CHROMA_COLLECTION_NAME}' | "
          f"count={collection.count()} | chunk_fp={fp}")

    cases = evaluate.load_question_set(QUESTIONS)
    per_question = []
    for case in cases:
        q = evaluate.prepare_query_text(case["question"])
        hits = store.keyword_search(collection, q, k=TOP_K,
                                    include_metadata=True, cfg=None)
        is_oos = case["category"] == "out-of-scope"
        correct = None if is_oos else evaluate.find_correct_hit(case, hits)
        any_lang = None if is_oos else evaluate.find_correct_any_lang(case, hits)
        per_question.append({
            "id": case["id"],
            "question": case["question"],
            "normalized_query": q,
            "language": case.get("language"),
            "category": case["category"],
            "expected": {k: v for k, v in case.items()
                         if k.startswith("expected") and v is not None},
            "is_out_of_scope": is_oos,
            "hits": [{
                "rank": h["rank"],
                "id": h["id"],
                "heading": (h.get("metadata") or {}).get("heading"),
                "language": (h.get("metadata") or {}).get("language"),
                "document": (h.get("metadata") or {}).get("document"),
                "metadata": h.get("metadata") or {},
                "keyword_score": h.get("keyword_score"),
                "score": h.get("keyword_score"),
                "text": h["text"],
            } for h in hits],
            "correct_rank": correct["rank"] if correct else None,
            "correct_score": (correct.get("keyword_score")
                              if correct else None),
            "correct_id": correct["id"] if correct else None,
            "correct_any_lang_rank": any_lang["rank"] if any_lang else None,
            "correct_any_lang_id": any_lang["id"] if any_lang else None,
            "hit_at_1": bool(correct and correct["rank"] <= 1),
            "hit_at_3": bool(correct and correct["rank"] <= 3),
            "hit_at_5": bool(correct and correct["rank"] <= 5),
        })

    metrics = evaluate.compute_metrics(per_question)
    run = {
        "arm": name,
        "mode": mode,
        "budget_tokens": BUDGET_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "retrieval": "bm25-only (store.keyword_search, k=%d)" % TOP_K,
        "chunk_fp": fp,
        "metrics": metrics,
        "questions": per_question,
    }
    out = RESULTS / f"arm_{name}.json"
    out.write_text(json.dumps(run, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"[harness50] wrote {out}")
    return run, chunks


# ---------------------------------------------------------------------------
# Substring validation (authoring safety net)
# ---------------------------------------------------------------------------

def validate_substrings(cases: list[dict], docs: list[dict],
                        restructure_chunks: list) -> list[str]:
    """Every expected_substring must exist (a) in the raw document text and
    (b) inside at least one restructure-arm chunk of the expected document.
    (a) can legitimately fail for the scrambled PDFs — that is precisely the
    baseline's weakness — so it is reported, not fatal. (b) is fatal."""
    by_name = {d["name"]: d for d in docs}
    chunk_text_by_doc: dict[str, list[str]] = {}
    for c in restructure_chunks:
        chunk_text_by_doc.setdefault(c.source, []).append(c.text)

    lines = ["# questions_50.json — expected_substring validation",
             "",
             "| id | doc | in raw text | in restructure chunks |",
             "|---|---|---|---|"]
    fatal = []
    for case in cases:
        sub = case.get("expected_substring")
        if not sub:
            continue
        doc = case.get("expected_document", "")
        raw = by_name.get(doc, {}).get("text", "")
        in_raw = evaluate.normalize_for_match(sub) in evaluate.normalize_for_match(raw)
        in_re = any(evaluate.normalize_for_match(sub)
                    in evaluate.normalize_for_match(t)
                    for t in chunk_text_by_doc.get(doc, []))
        flag = "YES" if in_re else "**NO — fix the case**"
        lines.append(f"| {case['id']} | {doc} | {'yes' if in_raw else 'no (baseline misses by design)'} | {flag} |")
        if not in_re:
            fatal.append(case["id"])
    report = "\n".join(lines) + "\n"
    (RESULTS / "validation.md").write_text(report, encoding="utf-8")
    if fatal:
        print(f"[harness50] VALIDATION FAILED — substrings missing from the "
              f"restructure arm for: {', '.join(fatal)}")
    else:
        print(f"[harness50] validation OK — every expected_substring is "
              f"present in the restructure arm")
    return fatal


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def comparison_table(base: dict, restr: dict) -> str:
    b, r = base["metrics"], restr["metrics"]
    L = ["# harness50 — size-220/40 baseline vs restructure (BM25-only, k=20)",
         "",
         "Corpus: 6 documents (4 Arabic docs — 2 PDFs stored in visual word "
         "order with corrupted digits — + 1 Arabic guide + 1 Arabic intro; "
         "2 parallel fictional FR/AR product sheets). Retrieval: BM25 lexical "
         "only, identical for both arms; the only variable is the chunking "
         "strategy. Questions: 50 (17 ar / 17 fr / 16 en, 5 out-of-scope).",
         "",
         "## Overall (45 answerable)",
         "",
         "| arm | hit@1 | hit@3 | hit@5 |",
         "|---|---|---|---|",
         f"| size-220/40 baseline | {pct(b['overall']['hit@1'])} | "
         f"{pct(b['overall']['hit@3'])} | {pct(b['overall']['hit@5'])} |",
         f"| restructure-220/40 | {pct(r['overall']['hit@1'])} | "
         f"{pct(r['overall']['hit@3'])} | {pct(r['overall']['hit@5'])} |",
         "",
         "## By category",
         "",
         "| category | n | base hit@1/3/5 | restructure hit@1/3/5 |",
         "|---|---|---|---|"]
    for cat in sorted(set(b["by_category"]) | set(r["by_category"])):
        bc = b["by_category"].get(cat, {})
        rc = r["by_category"].get(cat, {})
        L.append(f"| {cat} | {bc.get('n', rc.get('n'))} | "
                 f"{pct(bc.get('hit@1'))}/{pct(bc.get('hit@3'))}/{pct(bc.get('hit@5'))} | "
                 f"{pct(rc.get('hit@1'))}/{pct(rc.get('hit@3'))}/{pct(rc.get('hit@5'))} |")
    L += ["", "## By question language",
          "",
          "| lang | n | base hit@1/3/5 | restructure hit@1/3/5 |",
          "|---|---|---|---|"]
    for lang in sorted(set(b["by_language"]) | set(r["by_language"])):
        bc = b["by_language"].get(lang, {})
        rc = r["by_language"].get(lang, {})
        L.append(f"| {lang} | {bc.get('n', rc.get('n'))} | "
                 f"{pct(bc.get('hit@1'))}/{pct(bc.get('hit@3'))}/{pct(bc.get('hit@5'))} | "
                 f"{pct(rc.get('hit@1'))}/{pct(rc.get('hit@3'))}/{pct(rc.get('hit@5'))} |")
    L += ["", "## Per-question outcome",
          "",
          "| id | lang | cat | doc | base rank | restr rank |",
          "|---|---|---|---|---|---|"]
    for qb, qr in zip(base["questions"], restr["questions"]):
        L.append(f"| {qb['id']} | {qb['language']} | {qb['category']} | "
                 f"{(qb['expected'].get('expected_document') or '—')[:34]} | "
                 f"{qb['correct_rank'] or 'miss'} | {qr['correct_rank'] or 'miss'} |")
    oob, oor = b["out_of_scope"], r["out_of_scope"]
    L += ["",
          f"## Out-of-scope (n={oob['n']}): max top-1 BM25 score "
          f"base={oob['max_top1_score']:.1f} vs restructure={oor['max_top1_score']:.1f}"
          if oob["max_top1_score"] is not None and oor["max_top1_score"] is not None else "",
          ""]
    return "\n".join(L).rstrip() + "\n"


# ---------------------------------------------------------------------------
def main() -> int:
    docs = load_all(DOCS_DIRS)
    cases = evaluate.load_question_set(QUESTIONS)
    langs = {c["language"] for c in cases}
    print(f"[harness50] corpus: {len(docs)} docs | questions: {len(cases)} | "
          f"langs: {sorted(langs)}")

    base, _base_chunks = build_arm("baseline", "size", docs)
    restr, restr_chunks = build_arm("restructure", "restructure", docs)

    fatal = validate_substrings(cases, docs, restr_chunks)

    table = comparison_table(base, restr)
    (RESULTS / "comparison.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"[harness50] wrote {RESULTS / 'comparison.md'}")
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
