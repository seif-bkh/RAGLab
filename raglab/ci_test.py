"""ci_test.py — end-to-end integration test for GitHub Actions.

Runs the FULL pipeline against the real Gemini embedding API using the
GEMINI_API_KEY provided by the CI secret (written to .env by the workflow):

    inspect (offline) -> sanity check -> ingest --reset -> query ->
    evaluate -> saved results JSON

Every step is a CLI invocation via main.main(), so argparse is exercised too.
Mechanics are asserted (counts, dimensions, metadata, files, metric shape).
Retrieval QUALITY is never asserted: misses are findings, not test failures —
the printed metrics are the report you compare across runs.

CI artifact: every milestone is appended to results/ci_progress.txt and any
crash is written to results/ci_test_error.txt, so a failed run is inspectable
from the uploaded artifact even if the raw log cannot be fetched.
"""

import glob
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone

import config as cfg
import main
from evaluate import load_question_set
from store import get_collection

failures: list[str] = []
CURRENT_STEP = "startup"

cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE = cfg.RESULTS_DIR / "ci_progress.txt"
ERROR_FILE = cfg.RESULTS_DIR / "ci_test_error.txt"
CURRENT_STEP = "startup"


def progress(line: str) -> None:
    print(line, flush=True)
    with open(PROGRESS_FILE, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {line}\n")


def annotate_error(lines: list[str]) -> None:
    """Emit GitHub Actions ::error:: annotations (readable via the check-runs
    API, even when the raw log/artifact hosts are unreachable)."""
    for line in lines[:12]:  # keep annotations compact
        print(f"::error::{line}", flush=True)


def notify(line: str) -> None:
    """Emit a ::notice:: annotation (same API-readability benefit)."""
    print(f"::notice::{line}", flush=True)


def hit3(d: dict) -> str:
    """'hit@1/hit@3/hit@5' as one compact string."""
    return f"{d['hit@1']:.3f}/{d['hit@3']:.3f}/{d['hit@5']:.3f}"


def cats_str(m: dict) -> str:
    """Compact per-category hit rates."""
    return " ".join(
        f"{c}=h1:{m['by_category'][c]['hit@1']:.3f}/"
        f"h3:{m['by_category'][c]['hit@3']:.3f}/"
        f"h5:{m['by_category'][c]['hit@5']:.3f}"
        for c in sorted(m["by_category"]))


def langs_str(m: dict) -> str:
    """Compact per-language hit rates."""
    return " ".join(
        f"{l}=h1:{m['by_language'][l]['hit@1']:.3f}/"
        f"h3:{m['by_language'][l]['hit@3']:.3f}/"
        f"h5:{m['by_language'][l]['hit@5']:.3f}"
        for l in sorted(m["by_language"]))


def xl_detail_str(r: dict) -> str:
    """Per cross-lingual question: rank + which query variant found it."""
    return " | ".join(
        f"{q['id']} {q.get('language')}->{q.get('expected_lang')} "
        f"rank={q.get('correct_rank')} via={q.get('correct_variant') or '-'} "
        f"top={q.get('top_variant') or '-'}"
        for q in r.get("questions", [])
        if q.get("category") == "cross-lingual")


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[ci] {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def get_latest_eval() -> dict | None:
    files = sorted(glob.glob(str(cfg.RESULTS_DIR / "eval_*.json")))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as fh:
        return json.load(fh)


def run_steps() -> None:
    """The full pipeline. Any failure here is caught by run_main()."""
    global CURRENT_STEP
    # --- 1. Offline inspect (no API calls) -----------------------------------
    CURRENT_STEP = "inspect"
    progress("\n[ci] STEP 1/7 — inspect (load + chunk, offline)")
    rc = main.main(["inspect"])
    check("inspect exits 0", rc == 0, f"rc={rc}")

    # --- 2. Startup report + 3-language sanity check (1 small API call) ------
    CURRENT_STEP = "sanity"
    progress("\n[ci] STEP 2/7 — embedding sanity check (provider/model/dimension + "
             "EN/FR/AR pairwise cosine)")
    embedder = main.make_embedder(skip_sanity=False)
    expected_dim = int(getattr(cfg, "GEMINI_OUTPUT_DIMENSIONALITY", 0) or 0) or None
    check("vector dimension matches config",
          expected_dim is None or embedder._dimension == expected_dim,
          f"got={embedder._dimension} config={expected_dim}")
    # One compact notice (GitHub shows at most 10 annotations per step).
    sanity = " | ".join(getattr(embedder, "sanity_lines", ["dimension: ?"]))
    notify(f"sanity: {sanity}")

    # --- 3. Ingest (real embeddings, --reset) ---------------------------------
    CURRENT_STEP = "ingest"
    progress("\n[ci] STEP 3/7 — ingest --reset")
    rc = main.main(["ingest", "--reset", "--skip-sanity-check"])
    check("ingest exits 0", rc == 0, f"rc={rc}")
    check("embedding cache written", cfg.EMBEDDING_CACHE_PATH.exists())

    collection = get_collection(cfg, reset=False)
    count = collection.count()
    check("collection count > 0", count > 0, f"count={count}")

    raw = collection.get(include=["metadatas"])
    metas = raw["metadatas"]
    langs = {m.get("language") for m in metas}
    check("both FR and AR documents stored", {"fr", "ar"} <= langs, str(sorted(langs)))

    required_fields = {"document", "language", "heading", "chunk_index", "source",
                       "origin", "section_type", "ingested_at", "embedding_model",
                       "token_count"}
    check("every record carries required metadata",
          bool(metas) and required_fields <= set(metas[0]),
          str(sorted(metas[0]) if metas else []))
    check("metadata embedding model matches config",
          bool(metas) and metas[0].get("embedding_model") == cfg.EMBEDDING_MODEL,
          metas[0].get("embedding_model") if metas else "none")

    # v2 chunking strategy: boilerplate/front-matter chunks are chunked but
    # not indexed, so they cannot act as retrieval magnets.
    exclude = bool(getattr(cfg, "INDEX_EXCLUDE_BOILERPLATE", False))
    types_seen = {m.get("section_type") for m in metas}
    if exclude:
        check("boilerplate chunks excluded from the index",
              types_seen == {"content"}, f"stored types={sorted(types_seen)}")
    notify(f"chunking: stored={count} chunks, section_types={sorted(types_seen)}, "
           f"exclude_boilerplate={exclude} "
           f"(size={cfg.CHUNK_SIZE_TOKENS} overlap={cfg.CHUNK_OVERLAP_TOKENS} "
           f"sentence_overlap={getattr(cfg, 'CHUNK_OVERLAP_SENTENCE_AWARE', True)})")

    # --- 4. Query: vector-only + hybrid with language filter -------------------
    CURRENT_STEP = "query"
    progress("\n[ci] STEP 4/7 — query (vector + hybrid + lang filter + translation)")

    # 4a. Query translation plumbing (cross-lingual experiment). Two batched
    # calls warm the cache; `evaluate` below then reuses it, so the API cost
    # stays tiny. Quality stays a reported finding; PLUMBING is asserted.
    translator = None
    if getattr(cfg, "QUERY_TRANSLATION_ENABLED", False):
        progress("[ci] query translation ENABLED — plumbing check "
                 "(2 batched calls, evaluation reuses the cache)")
        translator = main.make_translator()
        check("query translator constructed", translator is not None)
        if translator is not None:
            samples = [c for c in load_question_set(cfg.QUESTIONS_FILE)
                       if c.get("category") == "cross-lingual"]
            for target in ("ar", "fr"):
                texts = [c["question"] for c in samples
                         if c.get("language") != target]
                if texts:
                    translator.translate_many(texts, target)
            ok_trans = (translator.api_calls >= 1
                        and translator.failures == 0)
            check("translation API calls succeeded", ok_trans,
                  f"calls={translator.api_calls} failures={translator.failures} "
                  f"cache_hits={translator.cache_hits} "
                  f"active_model={translator.active_model} "
                  f"last_error={translator.last_error or 'none'}")
            if not ok_trans:
                annotate_error([
                    f"translation failed: active_model="
                    f"{translator.active_model} last_error="
                    f"{translator.last_error or 'unknown'}",
                    "fix: check QUERY_TRANSLATION_MODEL / key permissions "
                    "(workflow makes query-translation a hard plumbing check)"])
            sample = next((c for c in samples if c.get("language") == "ar"
                           and c.get("expected_lang") == "fr"), None)
            example = ""
            if sample:
                tr = translator.translate_one(sample["question"], "fr")
                if tr:
                    example = f" | e.g. {sample['id']} ar->fr: {tr!r}"
            active = ("" if translator.active_model == translator.model
                      else f"->{translator.active_model}")
            notify(f"query-translation: enabled=True model="
                   f"{translator.model}{active} "
                   f"api_calls={translator.api_calls} "
                   f"cache_hits={translator.cache_hits} "
                   f"failures={translator.failures}{example}")

    rc = main.main(["query",
                    "Quel est le frais mensuel du Compte Courant Atlas ?",
                    "--k", "5", "--skip-sanity-check"])
    check("vector query exits 0", rc == 0, f"rc={rc}")
    rc = main.main(["query",
                    "Which Atlas account is meant for independent professionals?",
                    "--k", "5", "--hybrid", "--lang", "fr", "--skip-sanity-check"])
    check("hybrid query exits 0", rc == 0, f"rc={rc}")

    # --- 5. Evaluate (vector-only baseline) -------------------------------------
    CURRENT_STEP = "evaluate"
    progress("\n[ci] STEP 5/7 — evaluate (questions.json, vector-only)")
    rc = main.main(["evaluate", "--skip-sanity-check"])
    check("evaluate exits 0", rc == 0, f"rc={rc}")

    v_metrics = v_sep = None
    run = get_latest_eval()
    check("timestamped results JSON written", run is not None,
          f"results_dir={cfg.RESULTS_DIR}")
    if run:
        metrics = run["metrics"]
        check("metrics sections present",
              {"overall", "by_category", "by_language", "separation", "out_of_scope"}
              <= set(metrics))
        check("run config records model + chunking params",
              run["config"].get("embedding_model") == cfg.EMBEDDING_MODEL
              and "chunk_size_tokens" in run["config"])
        check("run config records query translation status",
              "query_translation_enabled" in run["config"])
        check("per-question records include hits",
              bool(run.get("questions"))
              and all("hits" in q and "category" in q for q in run["questions"]))
        check("per-question records include query variants",
              bool(run.get("questions"))
              and all("query_variants" in q for q in run["questions"]))
        # Printable conclusions for the log + compact annotations (GitHub
        # shows at most 10 annotations per step, so vector and hybrid runs
        # share compressed lines).
        v_metrics = metrics
        v_sep = sep = metrics["separation"]
        overall = metrics["overall"]
        oos = metrics["out_of_scope"]
        progress(f"\n[ci] CONCLUSION — overall hit rates:"
                 f"  hit@1={overall['hit@1']:.3f}  hit@3={overall['hit@3']:.3f}"
                 f"  hit@5={overall['hit@5']:.3f}  (n={overall['n']})")
        xl = [q for q in run.get("questions", [])
              if q.get("category") == "cross-lingual"]
        if xl:
            progress("[ci]   cross-lingual detail (translation variants):")
            for q in xl:
                progress(f"[ci]     {q['id']} {q.get('language')}"
                         f"->{q.get('expected_lang')} "
                         f"correct_rank={q.get('correct_rank')} "
                         f"correct_variant={q.get('correct_variant')} "
                         f"top_variant={q.get('top_variant')} "
                         f"variants="
                         f"{[v['label'] for v in q.get('query_variants', [])]}")
        for cat, d in sorted(metrics["by_category"].items()):
            progress(f"[ci]   category {cat:<14} n={d['n']:<3} hit@1={d['hit@1']:.3f} "
                     f"hit@3={d['hit@3']:.3f} hit@5={d['hit@5']:.3f}")
        for lang, d in sorted(metrics["by_language"].items()):
            progress(f"[ci]   language {lang:<14} n={d['n']:<3} hit@1={d['hit@1']:.3f} "
                     f"hit@3={d['hit@3']:.3f} hit@5={d['hit@5']:.3f}")
        progress(f"[ci]   separation  correct={sep['mean_correct_score']:.4f} "
                 f"best_incorrect={sep['mean_best_incorrect_score']:.4f} "
                 f"gap={sep['gap_mean_correct_minus_best_incorrect']:.4f}")
        # One compact line for overall/categories/languages (annotation budget).
        notify(f"evaluation: overall h1/h3/h5={hit3(overall)} (n={overall['n']}) | "
               f"cats: {cats_str(metrics)} | langs: {langs_str(metrics)}")
        notify(f"evaluation: separation correct={sep['mean_correct_score']:.4f} "
               f"best_incorrect={sep['mean_best_incorrect_score']:.4f} "
               f"gap={sep['gap_mean_correct_minus_best_incorrect']:.4f} "
               f"oos_max_top1={oos['max_top1_score']:.4f} "
               f"oos_mean_top1={oos['mean_top1_score']:.4f} (oos_n={oos['n']})")
        xl_detail = xl_detail_str(run)
        sl_detail = " | ".join(
            f"{q['id']} {q.get('language')} rank={q.get('correct_rank')} "
            f"via={q.get('correct_variant') or '-'}"
            for q in run.get("questions", [])
            if q.get("category") in ("verbatim", "paraphrase"))
        # One compact line (GitHub shows at most 10 annotations per step).
        notify(f"evaluation: detail  xl: {xl_detail or '-'}  sl: {sl_detail or '-'}")
        misses = [q for q in run["questions"]
                  if not q["is_out_of_scope"] and q["correct_rank"] is None]
        for q in misses:
            progress(f"[ci]   MISS  {q['id']} [{q['language']}/{q['category']}] "
                     f"not in top {run['config']['retrieval_top_k']} — "
                     f"{q['question'][:70]}")
            # What DID come back instead? Top 3 hits (log-level only).
            top = q["hits"][:3]
            parts = []
            for h in top:
                sid = h.get("id", "?")
                if "::" in sid:
                    sid = sid.split("::", 1)[1]
                score = h.get("score")
                score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "?"
                heading = (h.get("heading") or "").replace("\n", " ")
                snippet = (h.get("text") or "").replace("\n", " ")[:42]
                parts.append(
                    f"#{h['rank']} {sid} sim={score_s} "
                    f"{h.get('language') or '?'} "
                    f"v={h.get('variant') or '?'} h='{heading[:20]}' "
                    f"txt='{snippet}...'"
                )
            progress(f"[ci]     " + " | ".join(parts))

    # --- 5b. Hybrid A/B: same questions, vector + BM25 RRF per variant ---------
    CURRENT_STEP = "hybrid-evaluate"
    progress("\n[ci] STEP 6/7 — hybrid evaluation "
             "(vector + BM25 RRF, translated variants)")
    # eval_*.json names have 1-second precision; the hybrid run could overwrite
    # the vector file, so preserve it under a clearly-labelled copy.
    vector_files = sorted(glob.glob(str(cfg.RESULTS_DIR / "eval_*.json")))
    if vector_files:
        shutil.copy(vector_files[-1], cfg.RESULTS_DIR / "eval_000_vector_ab.json")
    rc = main.main(["evaluate", "--hybrid", "--skip-sanity-check"])
    check("hybrid evaluate exits 0", rc == 0, f"rc={rc}")
    run_h = get_latest_eval()
    check("hybrid results JSON written", run_h is not None)
    if run_h:
        check("hybrid run records hybrid=True",
              run_h["config"].get("hybrid") is True)
        m_h = run_h["metrics"]
        o_h = m_h["overall"]
        s_h = m_h["separation"]
        oos_h = m_h["out_of_scope"]

        def delta(a, b):
            return None if a is None or b is None else a - b

        progress(f"\n[ci] HYBRID — overall hit rates:"
                 f"  hit@1={o_h['hit@1']:.3f}  hit@3={o_h['hit@3']:.3f}"
                 f"  hit@5={o_h['hit@5']:.3f}  (n={o_h['n']})")
        for cat, d in sorted(m_h["by_category"].items()):
            progress(f"[ci]   category {cat:<14} n={d['n']:<3} "
                     f"hit@1={d['hit@1']:.3f} hit@3={d['hit@3']:.3f} "
                     f"hit@5={d['hit@5']:.3f}")
        for lang, d in sorted(m_h["by_language"].items()):
            progress(f"[ci]   language {lang:<14} n={d['n']:<3} "
                     f"hit@1={d['hit@1']:.3f} hit@3={d['hit@3']:.3f} "
                     f"hit@5={d['hit@5']:.3f}")
        progress(f"[ci]   separation  correct={s_h['mean_correct_score']:.4f} "
                 f"best_incorrect={s_h['mean_best_incorrect_score']:.4f} "
                 f"gap={s_h['gap_mean_correct_minus_best_incorrect']:.4f}")
        if v_metrics is not None:
            o_v = v_metrics["overall"]
            s_v = v_sep
            notify(f"evaluation: hybrid h1/h3/h5={hit3(o_h)} "
                   f"(Δ vs vector "
                   f"{delta(o_h['hit@1'], o_v['hit@1']):+.3f}/"
                   f"{delta(o_h['hit@3'], o_v['hit@3']):+.3f}/"
                   f"{delta(o_h['hit@5'], o_v['hit@5']):+.3f}) | "
                   f"cats: {cats_str(m_h)} | langs: {langs_str(m_h)}")
            notify(f"evaluation: hybrid sep gap="
                   f"{s_h['gap_mean_correct_minus_best_incorrect']:.4f} "
                   f"(vector {s_v['gap_mean_correct_minus_best_incorrect']:.4f}; "
                   f"oos_max_top1={oos_h['max_top1_score']:.4f}) | "
                   f"xl detail: {xl_detail_str(run_h) or '-'}")
        else:
            notify(f"evaluation: hybrid h1/h3/h5={hit3(o_h)} "
                   f"(no vector baseline in this run) | "
                   f"cats: {cats_str(m_h)} | langs: {langs_str(m_h)}")

    # --- 7. Answer stub exists (regression guard) ------------------------------
    CURRENT_STEP = "answer-stub"
    progress("\n[ci] STEP 7/7 — answer.py stub")
    import answer  # noqa: E402
    stub = answer.generate_answer("test", [])
    check("answer stub returns placeholder", "[STUB]" in stub)

    # --- 7. Verdict ---------------------------------------------------------------
    progress("\n" + "=" * 78)
    if failures:
        progress(f"[ci] CI PIPELINE FAILED: {len(failures)} assertion(s):")
        for f in failures:
            print("  -", f)
        with open(ERROR_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"[assertions] failed during step {CURRENT_STEP}\n")
            for f in failures:
                fh.write(f"  - {f}\n")
        annotate_error([f"RAGLab CI FAILED: {len(failures)} check(s) at step {CURRENT_STEP}"]
                       + [f"  check failed: {f}" for f in failures]
                       + ["  details: artifact raglab-eval-results / ci_test_error.txt"])
        sys.exit(1)
    notify("RAGLab CI PASSED: full pipeline mechanics + evaluation OK "
           "(metrics above, results JSON in the raglab-eval-results artifact)")
    progress("[ci] CI PIPELINE PASSED — all mechanics verified; metrics above are the report.")
    print("=" * 78)
    sys.exit(0)


def run_main() -> None:
    """Wrapper: any crash in run_steps is captured to the artifact."""
    try:
        run_steps()
    except SystemExit as exc:
        # Missing key/SDK raise SystemExit WITH a message string; capture it so
        # the CI artifact explains the failure. Integer exits (0/1) are ours.
        message = exc.code if isinstance(exc.code, str) else ""
        if message:
            progress(f"[ci] CLEAN EXIT in step {CURRENT_STEP}: {message}")
            with open(ERROR_FILE, "a", encoding="utf-8") as fh:
                fh.write(f"[SystemExit] step={CURRENT_STEP}\n{message}\n")
            annotate_error([f"RAGLab CI exited at step {CURRENT_STEP}: {message[:180]}",
                            "  details: artifact raglab-eval-results / ci_test_error.txt"])
        raise
    except BaseException as exc:  # noqa: BLE001 — CI diagnostics: keep the cause
        progress(f"[ci] CRASH in step {CURRENT_STEP}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        with open(ERROR_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"[crash] step={CURRENT_STEP} type={type(exc).__name__}\n{exc}\n")
            fh.write(traceback.format_exc())
        annotate_error([f"RAGLab CI CRASHED at step {CURRENT_STEP}: "
                        f"{type(exc).__name__}: {str(exc)[:180]}",
                        "  details: artifact raglab-eval-results / ci_test_error.txt"])
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 78)
    print("RAGLab CI integration test — provider:", cfg.EMBEDDING_PROVIDER,
          "| model:", cfg.EMBEDDING_MODEL)
    print("=" * 78)
    run_main()
