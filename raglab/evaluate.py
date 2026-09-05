"""evaluate.py — run the question set against the collection and measure retrieval.

Reads questions.json (cases: question, language, category, expected chunk
index OR expected substring; out-of-scope cases have no expected match),
embeds each query, retrieves top-k (vector or hybrid), decides hit/miss, and
prints:

  - per-question table (hit, correct rank, score; every miss flagged),
  - overall hit@1 / hit@3 / hit@5,
  - hit rate by category and by query language,
  - mean score of correct chunk vs mean best-incorrect score (separation),
  - for out-of-scope cases, the maximum similarity returned.

The full run is saved to results/eval_<timestamp>.json together with the
chunking parameters and embedding model so runs can be compared.
"""

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from loader import normalize_arabic, normalize_text
from store import (best_variant_merge, blend_hybrid, collection_languages,
                   keyword_search, query_vector, rrf_merge)

VALID_CATEGORIES = {"verbatim", "paraphrase", "cross-lingual", "out-of-scope"}


# ---------------------------------------------------------------------------
# Question set
# ---------------------------------------------------------------------------


def load_question_set(path: Path) -> list[dict]:
    """Load and validate questions.json; every warning is printed, never silent."""
    if not path.exists():
        raise FileNotFoundError(
            f"questions file not found: {path} (expected next to config.py)"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    print(f"[evaluate] loaded {len(cases)} question(s) from {path.name}")

    for i, case in enumerate(cases):
        problems = []
        if not case.get("id"):
            problems.append("missing id")
        if not case.get("question"):
            problems.append("missing question")
        if case.get("category") not in VALID_CATEGORIES:
            problems.append(f"invalid category {case.get('category')!r}")
        out_of_scope = case.get("category") == "out-of-scope"
        has_expected = (
            case.get("expected_chunk_index") is not None
            or bool(case.get("expected_substring"))
        )
        if out_of_scope and has_expected:
            problems.append("out-of-scope case must have no expected match")
        if not out_of_scope and not has_expected:
            problems.append("non out-of-scope case needs expected_chunk_index or expected_substring")
        if problems:
            print(f"[evaluate] WARNING case #{i} ({case.get('id', '?')}): "
                  + "; ".join(problems))
    return cases


def normalize_for_match(text: str) -> str:
    """Normalization used to match expected substrings against chunk text."""
    return normalize_arabic(normalize_text(text)).casefold()


def prepare_query_text(text: str) -> str:
    """Queries go through the same cleaning as documents (no translation)."""
    return normalize_text(normalize_arabic(text))


# ---------------------------------------------------------------------------
# Hit logic
# ---------------------------------------------------------------------------


def is_correct_hit(case: dict, hit: dict) -> bool:
    """True if this hit matches the case's expectation (language-aware)."""
    meta = hit.get("metadata") or {}
    expected_lang = case.get("expected_lang")
    if expected_lang and meta.get("language") != expected_lang:
        return False

    expected_index = case.get("expected_chunk_index")
    if expected_index is not None:
        return meta.get("chunk_index") == expected_index

    expected_sub = case.get("expected_substring")
    if expected_sub:
        return normalize_for_match(expected_sub) in normalize_for_match(hit["text"])
    return False


def is_correct_any_lang(case: dict, hit: dict) -> bool:
    """Like is_correct_hit but ignoring the expected_lang constraint — used to
    tell apart a retrieval failure from a language-routing observation."""
    relaxed = {k: v for k, v in case.items() if k != "expected_lang"}
    return is_correct_hit(relaxed, hit)


def find_correct_hit(case: dict, hits: list) -> dict | None:
    """First (highest-ranked) hit that satisfies the case expectation."""
    for hit in hits:
        if is_correct_hit(case, hit):
            return hit
    return None


def find_correct_any_lang(case: dict, hits: list) -> dict | None:
    """First hit matching the expectation regardless of language."""
    for hit in hits:
        if is_correct_any_lang(case, hit):
            return hit
    return None


# ---------------------------------------------------------------------------
# Running the evaluation
# ---------------------------------------------------------------------------


def run_evaluation(cfg, embedder, collection, cases: list, mode: str = "vector",
                   top_k: int = 20, translator=None,
                   blend_lambda: float | None = None) -> dict:
    """Embed every question, retrieve, decide hit/miss, build the full run dict.

    mode selects the per-variant retrieval:
      - "vector": dense cosine only;
      - "rrf"    : vector + BM25 fused by reciprocal rank fusion;
      - "blend"  : vector + BM25 fused by weighted score blend
                   (lambda * cosine + (1-lambda) * normalized BM25);
                   blend_lambda overrides cfg.HYBRID_BLEND_LAMBDA (CI sweep).

    When translator is not None, each query is also translated into every
    corpus language (query variants); results are fused per chunk by their
    best score across variants (store.best_variant_merge), so a chunk wins on
    the language in which it matches best.
    """
    if mode not in {"vector", "rrf", "blend"}:
        raise ValueError(f"unknown retrieval mode {mode!r}")
    corpus_langs = collection_languages(collection)
    translation_enabled = bool(
        translator is not None and getattr(cfg, "QUERY_TRANSLATION_ENABLED", False)
    )
    if translation_enabled:
        print(f"[evaluate] query translation ENABLED | model={translator.model} "
              f"| corpus languages={corpus_langs}")
    else:
        print("[evaluate] query translation disabled (original queries only)")
    lambd = (getattr(cfg, "HYBRID_BLEND_LAMBDA", 0.7)
             if blend_lambda is None else blend_lambda)
    print(f"[evaluate] running {len(cases)} question(s) | "
          f"mode={mode} | blend_lambda={lambd:.2f} | "
          f"recording top_k={top_k}")
    score_key = {"vector": "similarity", "rrf": "rrf_score",
                 "blend": "blend_score"}[mode]

    per_question = []
    for case in cases:
        q_text = prepare_query_text(case["question"])
        if translation_enabled:
            variants = translator.build_variants(
                q_text, case.get("language"), corpus_langs)
        else:
            variants = [{
                "label": f"{case.get('language') or '?'}(original)",
                "lang": case.get("language"),
                "translated": False,
                "text": q_text,
            }]

        variant_hit_lists = []
        for variant in variants:
            q_embedding = embedder.embed_query(variant["text"])
            vector_hits = query_vector(collection, q_embedding, k=top_k, cfg=cfg)
            if mode == "vector":
                variant_hit_lists.append(vector_hits)
            elif mode == "rrf":
                kw_hits = keyword_search(collection, variant["text"], k=top_k,
                            include_metadata=getattr(
                                cfg, "KEYWORD_SEARCH_INCLUDE_METADATA",
                                True), cfg=cfg)
                variant_hit_lists.append(
                    rrf_merge(vector_hits, kw_hits,
                              k=cfg.RRF_RANK_CONSTANT)[:top_k])
            else:  # blend
                kw_hits = keyword_search(collection, variant["text"], k=top_k,
                                         include_metadata=getattr(
                                             cfg,
                                             "KEYWORD_SEARCH_INCLUDE_METADATA",
                                             True), cfg=cfg)
                variant_hit_lists.append(
                    blend_hybrid(vector_hits, kw_hits, lambd=lambd)[:top_k])

        fused = best_variant_merge(variant_hit_lists, score_key=score_key,
                                   labels=[v["label"] for v in variants],
                                   tie_break=getattr(
                                       cfg, "FUSION_TIE_BREAK",
                                       "same_lang_margin"))
        hits = fused[:top_k]

        is_oos = case["category"] == "out-of-scope"
        correct_hit = None if is_oos else find_correct_hit(case, hits)
        any_lang_hit = None if is_oos else find_correct_any_lang(case, hits)

        # "score" is the metric that determined the ranking in this mode.
        def score_of(h, _key=score_key):
            return h.get(_key)

        record = {
            "id": case["id"],
            "question": case["question"],
            "normalized_query": q_text,
            "language": case.get("language"),
            "category": case["category"],
            "expected": {k: v for k, v in case.items()
                         if k.startswith("expected") and v is not None},
            "is_out_of_scope": is_oos,
            "query_variants": [{
                "label": v["label"], "lang": v.get("lang"),
                "translated": v["translated"], "text": v["text"],
            } for v in variants],
            "top_variant": hits[0].get("from_variant") if hits else None,
            "hits": [{
                "rank": h["rank"],
                "id": h["id"],
                "heading": (h.get("metadata") or {}).get("heading"),
                "language": (h.get("metadata") or {}).get("language"),
                "document": (h.get("metadata") or {}).get("document"),
                "variant": h.get("from_variant"),
                "variant_ranks": h.get("variant_ranks"),
                "relative_score": h.get("relative_score"),
                "similarity": h.get("similarity"),
                "keyword_score": h.get("keyword_score"),
                "rrf_score": h.get("rrf_score"),
                "blend_score": h.get("blend_score"),
                "kw_norm": h.get("kw_norm"),
                "score": score_of(h),
                "text": h["text"],
            } for h in hits],
            "correct_rank": correct_hit["rank"] if correct_hit else None,
            "correct_score": score_of(correct_hit) if correct_hit else None,
            "correct_id": correct_hit["id"] if correct_hit else None,
            "correct_variant": correct_hit.get("from_variant") if correct_hit else None,
            # Where the answer sits when language is NOT constrained: separates
            # retrieval failure from language-routing behavior on strict cases.
            "correct_any_lang_rank": any_lang_hit["rank"] if any_lang_hit else None,
            "correct_any_lang_id": any_lang_hit["id"] if any_lang_hit else None,
            "hit_at_1": bool(correct_hit and correct_hit["rank"] <= 1),
            "hit_at_3": bool(correct_hit and correct_hit["rank"] <= 3),
            "hit_at_5": bool(correct_hit and correct_hit["rank"] <= 5),
        }
        per_question.append(record)

    metrics = compute_metrics(per_question)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "provider": cfg.EMBEDDING_PROVIDER,
            "embedding_model": cfg.active_embedding_model(),
            "chunk_size_tokens": cfg.CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": cfg.CHUNK_OVERLAP_TOKENS,
            "split_on_headings_first": cfg.SPLIT_ON_HEADINGS_FIRST,
            "retrieval_top_k": top_k,
            "hybrid": mode == "rrf",
            "retrieval_mode": mode,
            "hybrid_blend_lambda": lambd,
            "fusion_tie_break": getattr(cfg, "FUSION_TIE_BREAK",
                                        "same_lang_margin"),
            "query_translation_enabled": translation_enabled,
            "query_translation_model": translator.model if translation_enabled else None,
            # Each variant's scores are normalized by its own best match, then
            # fused — raw cross-variant scores are not comparable (the original
            # query's language always scores higher).
            "query_translation_fusion": "best_variant_relative",
            "query_translation_languages": corpus_langs,
            "rrf_rank_constant": cfg.RRF_RANK_CONSTANT,
            "keyword_include_metadata": getattr(
                cfg, "KEYWORD_SEARCH_INCLUDE_METADATA", True),
        },
        "metrics": metrics,
        "questions": per_question,
    }


def compute_metrics(per_question: list) -> dict:
    """Aggregate hit rates, separation and out-of-scope behaviour."""
    evaluable = [q for q in per_question if not q["is_out_of_scope"]]
    oos = [q for q in per_question if q["is_out_of_scope"]]
    n = len(evaluable)

    def rate(cases, cutoff):
        if not cases:
            return None
        return sum(1 for q in cases if q[f"hit_at_{cutoff}"]) / len(cases)

    overall = {
        "n": n,
        "hit@1": rate(evaluable, 1),
        "hit@3": rate(evaluable, 3),
        "hit@5": rate(evaluable, 5),
    }

    by_category = {}
    for cat in sorted({q["category"] for q in evaluable}):
        subset = [q for q in evaluable if q["category"] == cat]
        by_category[cat] = {
            "n": len(subset),
            "hit@1": rate(subset, 1),
            "hit@3": rate(subset, 3),
            "hit@5": rate(subset, 5),
        }

    by_language = {}
    for lang in sorted({q.get("language") for q in evaluable if q.get("language")}):
        subset = [q for q in evaluable if q.get("language") == lang]
        by_language[lang] = {
            "n": len(subset),
            "hit@1": rate(subset, 1),
            "hit@3": rate(subset, 3),
            "hit@5": rate(subset, 5),
        }

    # Separation: score of the correct chunk vs the best incorrect chunk,
    # computed only for cases where the correct chunk was found (rank <= top_k).
    correct_scores, best_incorrect_scores = [], []
    for q in evaluable:
        if q["correct_rank"] is None:
            continue
        correct_scores.append(q["correct_score"])
        others = [h for h in q["hits"] if h["id"] != q["correct_id"]]
        if others:
            best_incorrect_scores.append(max(h["score"] for h in others))

    separation = {
        "n_correct_retrieved": len(correct_scores),
        "mean_correct_score": (statistics.mean(correct_scores) if correct_scores else None),
        "n_with_best_incorrect": len(best_incorrect_scores),
        "mean_best_incorrect_score": (statistics.mean(best_incorrect_scores) if best_incorrect_scores else None),
        "gap_mean_correct_minus_best_incorrect": (
            (statistics.mean(correct_scores) - statistics.mean(best_incorrect_scores))
            if correct_scores and best_incorrect_scores else None
        ),
    }

    oos_scores = [
        q["hits"][0]["score"] if q["hits"] else None for q in oos
    ]
    oos_scores = [s for s in oos_scores if s is not None]
    out_of_scope = {
        "n": len(oos),
        "max_top1_score": (max(oos_scores) if oos_scores else None),
        "mean_top1_score": (statistics.mean(oos_scores) if oos_scores else None),
        "per_question": [
            {
                "id": q["id"],
                "question": q["question"],
                "max_score": q["hits"][0]["score"] if q["hits"] else None,
                "top_chunk_id": q["hits"][0]["id"] if q["hits"] else None,
                "top_chunk_text": q["hits"][0]["text"] if q["hits"] else None,
            }
            for q in oos
        ],
    }

    return {
        "overall": overall,
        "by_category": by_category,
        "by_language": by_language,
        "separation": separation,
        "out_of_scope": out_of_scope,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_report(run: dict):
    """Print the whole report; every number is labelled, every miss flagged."""
    m = run["metrics"]
    conf = run["config"]
    print("\n" + "=" * 78)
    print(f"EVALUATION REPORT | provider={conf['provider']} model={conf['embedding_model']} "
          f"| hybrid={conf['hybrid']}")
    print(f"chunks: size={conf['chunk_size_tokens']} overlap={conf['chunk_overlap_tokens']} "
          f"split_headings={conf['split_on_headings_first']} | recorded k={conf['retrieval_top_k']}")
    print("=" * 78)

    # -- per-question table ---------------------------------------------------
    print("\n--- PER-QUESTION TABLE ---")
    header = (f"{'id':<11}{'lang':<6}{'category':<14}{'hit@1':<6}{'hit@3':<6}"
              f"{'hit@5':<6}{'status':<7}{'rank':<6}{'score':<10}question")
    print(header)
    print("-" * len(header))
    for q in run["questions"]:
        if q["is_out_of_scope"]:
            status, rank, score = "n/a", "n/a", f"{q['hits'][0]['score']:.4f}" if q["hits"] else "-"
        else:
            hit = q["correct_rank"] is not None
            status = "YES" if hit else "MISS"
            rank = str(q["correct_rank"]) if hit else "-"
            score = f"{q['correct_score']:.4f}" if hit else "-"
        print(f"{q['id']:<11}{str(q['language']):<6}{q['category']:<14}"
              f"{'Y' if q['hit_at_1'] else '-':<6}{'Y' if q['hit_at_3'] else '-':<6}"
              f"{'Y' if q['hit_at_5'] else '-':<6}{status:<7}{rank:<6}{score:<10}"
              f"{q['question'][:60]}")
    print("  (score = cosine similarity in vector mode, RRF fusion score in hybrid mode)")

    # -- flag misses in detail ------------------------------------------------
    misses = [q for q in run["questions"]
              if not q["is_out_of_scope"] and q["correct_rank"] is None]
    if misses:
        print("\n--- MISSES (flagged) ---")
        for q in misses:
            print(f"!! {q['id']} [{q['language']} / {q['category']}] "
                  f"correct chunk NOT in top {conf['retrieval_top_k']} "
                  f"(strict expected_lang)")
            print(f"   question : {q['question']}")
            print(f"   expected : {q['expected']}")
            if q.get("correct_any_lang_rank") is not None:
                print(f"   note     : the answer WAS retrieved (any language) at "
                      f"rank {q['correct_any_lang_rank']} — this is a LANGUAGE-"
                      f"ROUTING observation, not a chunking/retrieval failure.")
            if q["hits"]:
                top = q["hits"][0]
                print(f"   top hit  : rank {top['rank']} (top score "
                      f"{top['score']:.4f}) lang={top['language']} "
                      f"heading={top['heading']!r}")
    else:
        print(f"\n--- MISSES: none (all {len(run['questions'])} non OOS questions hit at "
              f"top {conf['retrieval_top_k']}) ---")

    # -- overall + breakdowns --------------------------------------------------
    o = m["overall"]
    print("\n--- OVERALL HIT RATE (only questions with an expected match) ---")
    print(f"  n={o['n']}   hit@1={o['hit@1']:.3f}   hit@3={o['hit@3']:.3f}   "
          f"hit@5={o['hit@5']:.3f}")

    print("\n--- HIT RATE BY CATEGORY ---")
    for cat, d in sorted(m["by_category"].items()):
        print(f"  {cat:<14} n={d['n']:<3} hit@1={d['hit@1']:.3f} "
              f"hit@3={d['hit@3']:.3f} hit@5={d['hit@5']:.3f}")

    print("\n--- HIT RATE BY QUERY LANGUAGE ---")
    for lang, d in sorted(m["by_language"].items()):
        print(f"  {lang:<14} n={d['n']:<3} hit@1={d['hit@1']:.3f} "
              f"hit@3={d['hit@3']:.3f} hit@5={d['hit@5']:.3f}")

    s = m["separation"]
    print("\n--- SEPARATION (mean ranking score) ---")
    print("  (ranking score = cosine similarity in vector mode, RRF in hybrid mode)")
    print(f"  correct chunk            : {s['mean_correct_score']:.4f} "
          f"(n={s['n_correct_retrieved']})")
    print(f"  best incorrect chunk     : {s['mean_best_incorrect_score']:.4f} "
          f"(n={s['n_with_best_incorrect']})")
    print(f"  gap (correct - incorrect): {s['gap_mean_correct_minus_best_incorrect']:.4f}"
          if s["gap_mean_correct_minus_best_incorrect"] is not None
          else "  gap: not computable (no correct chunk was ever retrieved)")

    os_ = m["out_of_scope"]
    print("\n--- OUT-OF-SCOPE (no expected match; what a refusal threshold must beat) ---")
    print(f"  n={os_['n']}   max top-1 score={os_['max_top1_score']:.4f}   "
          f"mean top-1 score={os_['mean_top1_score']:.4f}")
    for item in os_["per_question"]:
        score = item["max_score"]
        print(f"  {item['id']:<6} max={score:.4f}  "
              f"top chunk: {item['top_chunk_id']} | "
              f"{item['top_chunk_text'][:90]}...")

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("  - hit@1/3/5: how often the correct chunk ranked that high; the")
    print("    distance between hit@1 and hit@5 shows how lucky k=5 is.")
    print("  - separation gap: how far the correct chunk scores above the best")
    print("    wrong chunk; a small or negative gap means rank only, not meaning.")
    print("  - out-of-scope max: the highest score a question with no answer")
    print("    can still reach; a refusal threshold slightly above that value")
    print("    keeps you from hallucinating an answer on unseen questions (lab-only).")
    print("=" * 78)


def save_run(run: dict, results_dir: Path) -> Path:
    """Write the full run to results/eval_<timestamp>.json; returns the path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"eval_{stamp}.json"
    out.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[evaluate] full run saved to {out}")
    return out
