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
import os
import re
import shutil
import sys
import traceback
from datetime import datetime, timezone

import config as cfg
import main
from evaluate import (load_question_set, normalize_for_match,
                       prepare_query_text)
from store import get_collection, query_vector

failures: list[str] = []
CURRENT_STEP = "startup"

cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE = cfg.RESULTS_DIR / "ci_progress.txt"
ERROR_FILE = cfg.RESULTS_DIR / "ci_test_error.txt"
CURRENT_STEP = "startup"
# RAGLAB_CI_REAL_ONLY=1 skips the fictional-corpus legs (saves ~40
# embedding API calls/run — the free tier is ~100 RPD and the docs/
# corpus alone needs ~15 ingest + ~32 query embeddings).
REAL_ONLY = os.environ.get("RAGLAB_CI_REAL_ONLY", "0").strip() \
    in {"1", "true", "yes", "on"}

REQUIRED_METADATA_FIELDS = {
    "document", "language", "heading", "chunk_index", "source",
    "origin", "section_type", "ingested_at", "embedding_model",
    "token_count",
}


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


def eval_summary(run: dict) -> str:
    """ONE compact annotation line for one evaluation run.

    GitHub shows at most 10 annotations per workflow step, and the whole
    ci_test.py runs as a single step, so every mode/corpus pair gets exactly
    one line: main hit rates + categories + languages + separation + the
    per-cross-lingual-question detail. Verbose per-category/per-language
    tables stay in the progress log / results artifact.
    """
    m = run["metrics"]
    o = m["overall"]
    s = m["separation"]
    oos = m["out_of_scope"]
    return (
        f"h1/h3/h5={hit3(o)} (n={o['n']}) | cats: {cats_str(m)} | "
        f"langs: {langs_str(m)} | sep_gap="
        f"{s['gap_mean_correct_minus_best_incorrect']:.4f} "
        f"oos_max_top1={oos['max_top1_score']:.4f} "
        f"(oos_n={oos['n']}) | xl: {xl_detail_str(run) or '-'}"
    )


def delta_str(run: dict, baseline: dict) -> str:
    """"(Δ vs baseline −/+/-)" for h1/h3/h5."""
    b = baseline["metrics"]["overall"]
    c = run["metrics"]["overall"]
    return (f"(Δ {c['hit@1'] - b['hit@1']:+.3f}/"
            f"{c['hit@3'] - b['hit@3']:+.3f}/"
            f"{c['hit@5'] - b['hit@5']:+.3f})")


# --- hybrid-blend lambda sweep -----------------------------------------------
SWEEP_LAMBDAS = (0.55, 0.65, 0.75, 0.85, 0.95)
REAL_SWEEP_LAMBDAS = (0.65, 0.75, 0.85)
TIEBREAK_MODES = ("variant_order", "same_lang_margin", "raw")


def qrank(run: dict, qid: str):
    """correct_rank of one question id (None when missing)."""
    for q in run.get("questions", []):
        if q["id"] == qid:
            return q.get("correct_rank")
    return None


def eval_blend_at(lam: float, extra_args: tuple = ()) -> dict | None:
    """Run evaluate --hybrid-blend with a specific lambda.

    Query embeddings come from the persistent on-disk cache, so after the
    first mode every extra lambda costs zero additional embedding API calls
    (only local BM25 + fusion re-run). cfg.HYBRID_BLEND_LAMBDA is patched for
    the duration of the call and restored afterwards.
    """
    global CURRENT_STEP
    old = cfg.HYBRID_BLEND_LAMBDA
    cfg.HYBRID_BLEND_LAMBDA = lam
    try:
        rc = main.main(["evaluate", "--hybrid-blend", *extra_args,
                        "--skip-sanity-check"])
    finally:
        cfg.HYBRID_BLEND_LAMBDA = old
    check(f"blend lambda={lam:.2f} evaluate exits 0", rc == 0, f"rc={rc}")
    run = get_latest_eval()
    check(f"blend lambda={lam:.2f} results JSON written", run is not None)
    if run is None:
        return None
    got = run["config"].get("hybrid_blend_lambda")
    check(f"blend lambda={lam:.2f} recorded in run config",
          got is not None and abs(float(got) - lam) < 1e-9, f"recorded={got}")
    return run


def blend_sweep(extra_args: tuple, label: str, baseline: dict | None,
                rrf_run: dict | None, default_run: dict | None,
                anchor_q: str, lambdas: tuple) -> None:
    """Evaluate several lambdas, report one compact line + the acceptance verdict.

    Acceptance (user's): the blend should recover the vector hit@1 while
    keeping the RRF recall gain on the anchor question (q10 on the fictional
    set). Mechanics are asserted; quality is a reported finding.
    """
    global CURRENT_STEP
    CURRENT_STEP = f"{label}-blend-sweep"
    progress(f"\n[ci] STEP 7b — {label} blend score-lambda sweep "
             f"({', '.join(f'{l:.2f}' for l in lambdas)})")
    rows = []
    for lam in lambdas:
        run_lam = eval_blend_at(lam, extra_args)
        if run_lam is None:
            continue
        shutil.copy(sorted(glob.glob(str(cfg.RESULTS_DIR / "eval_*.json")))[-1],
                    cfg.RESULTS_DIR / f"sweep_{label}_blend_l{lam:.2f}.json")
        o = run_lam["metrics"]["overall"]
        rows.append((lam, run_lam))
        progress(f"[ci]   {label} lambda={lam:.2f}: h1={o['hit@1']:.3f} "
                 f"h3={o['hit@3']:.3f} h5={o['hit@5']:.3f} "
                 f"(n={o['n']}) {anchor_q} rank={qrank(run_lam, anchor_q)}")
    if not rows:
        return
    if default_run is not None:
        rows.append((cfg.HYBRID_BLEND_LAMBDA, default_run))
    joined = " | ".join(
        f"lambda={lam:.2f}:{hit3(r['metrics']['overall'])}"
        f" {anchor_q}={qrank(r, anchor_q)}"
        for lam, r in sorted(rows))
    best_lam, best_run = max(rows, key=lambda t: (
        t[1]["metrics"]["overall"]["hit@1"],
        t[1]["metrics"]["overall"]["hit@3"],
        t[1]["metrics"]["overall"]["hit@5"]))
    b = baseline["metrics"]["overall"] if baseline else None
    bh = best_run["metrics"]["overall"]
    q_b, q_v, q_r = qrank(best_run, anchor_q), (qrank(baseline, anchor_q)
                                                if baseline else None), (qrank(
                                                    rrf_run, anchor_q)
                                                if rrf_run else None)
    recovered = bool(b is not None and bh["hit@1"] >= b["hit@1"])
    # ONE line per sweep (GitHub keeps ~10 annotations per step; the sweep,
    # its acceptance verdict and the spec'd-lambda verdict belong together).
    line = (f"{label} blend sweep ({joined}) || best lambda={best_lam:.2f} "
            f"h1={bh['hit@1']:.3f}/h3={bh['hit@3']:.3f}/h5={bh['hit@5']:.3f}"
            + (f" (vector {b['hit@1']:.3f}/{b['hit@3']:.3f}/{b['hit@5']:.3f})"
               if b else "")
            + f" | {anchor_q} rank={q_b} (vector={q_v}, rrf={q_r}) | "
            f"hit@1 recovered: {'YES' if recovered else 'NO'}")
    d70 = next((r for lam, r in rows if abs(lam - 0.70) < 1e-9), None)
    if d70 is not None and baseline is not None:
        od = d70["metrics"]["overall"]
        rec70 = od["hit@1"] >= b["hit@1"]
        line += (f" || lambda=0.70: "
                 f"h1={od['hit@1']:.3f}/h3={od['hit@3']:.3f}/h5={od['hit@5']:.3f}"
                 f" (vector h1={b['hit@1']:.3f}) "
                 f"hit@1 recovered: {'YES' if rec70 else 'NO'}")
    notify(line)


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


_REAL_QUOTA = re.compile(r"429|RESOURCE_EXHAUSTED|quota", re.I)


def _real_eval(args: list, stage: str) -> int | None:
    """Run one real-docs evaluation, deferring cleanly on the daily 429 quota.

    Every query embedding is cache-saved as it completes, so a rerun (after
    the cache artifact is restored) continues exactly where the quota stopped
    the run — no progress is lost and the pipeline stays green.
    """
    global CURRENT_STEP
    CURRENT_STEP = stage
    try:
        return main.main(args)
    except RuntimeError as exc:
        if _REAL_QUOTA.search(str(exc)):
            progress(f"[ci] real-docs {stage} DEFERRED — daily embedding quota "
                     "exhausted (429 RESOURCE_EXHAUSTED); query cache saved; "
                     "rerun continues from where it stopped.")
            notify(f"real-docs: {stage} DEFERRED (daily embedding quota 429 — "
                   "cache artifact saved; rerun continues from where it stopped)")
            (cfg.RESULTS_DIR / "real_docs_status.json").write_text(
                json.dumps({"status": f"deferred-quota-{stage}",
                            "detail": str(exc)[:300]},
                           ensure_ascii=False, indent=2))
            return None
        raise


def run_steps() -> None:
    """The full pipeline. Any failure here is caught by run_main()."""
    global CURRENT_STEP
    # --- 1. Offline inspect (no API calls) -----------------------------------
    CURRENT_STEP = "inspect"
    progress("\n[ci] STEP 1 — inspect (load + chunk, offline)")
    rc = main.main(["inspect"])
    check("inspect exits 0", rc == 0, f"rc={rc}")

    # --- 1b. Real-docs OFFLINE chunking coverage (no API, deterministic) -----
    # With paragraph-boundary preservation, every expected substring must live
    # inside >= 1 chunk BEFORE any embedding happens. This is the hard gate:
    # if it fails the chunker must be fixed — no retriever can ever return a
    # phrase that no chunk contains. Runs even when the daily quota is gone,
    # so the chunker fix can be verified for free today.
    CURRENT_STEP = "real-chunks-offline"
    progress("\n[ci] STEP 1b — real-docs offline chunking coverage (no API)")
    real_docs_dir = str((cfg.PROJECT_DIR.parent / "docs").resolve())
    from loader import load_all  # noqa: E402
    from chunker import chunk_all  # noqa: E402
    offline_cases = load_question_set(cfg.PROJECT_DIR / "questions_real.json")
    try:
        docs_real = load_all([real_docs_dir])
        chunks_real = chunk_all(docs_real, cfg)
        norm_real = [normalize_for_match(c.text) for c in chunks_real]
        missing = []
        for case in offline_cases:
            nsub = normalize_for_match(case.get("expected_substring") or "")
            if nsub and not any(nsub in t for t in norm_real):
                missing.append(case["id"])
        ncov = len(offline_cases) - len(missing)
        notify(f"real-docs offline chunking: chunks={len(chunks_real)} "
               f"coverage={ncov}/{len(offline_cases)} "
               + ("all covered" if not missing else "miss=" + ",".join(missing)))
        check("real-docs offline: every expected substring in >= 1 chunk",
              not missing,
              ("all covered" if not missing else "missing=" + ",".join(missing)))
    except Exception as exc:  # noqa: BLE001 — report, don't crash the pipeline
        check("real-docs offline chunking runs", False,
              f"{type(exc).__name__}: {exc}")

    # --- 2. Startup report + 3-language sanity check (1 small API call) ------
    CURRENT_STEP = "sanity"
    progress("\n[ci] STEP 2 — embedding sanity check (provider/model/dimension + "
             "EN/FR/AR pairwise cosine)")
    embedder = main.make_embedder(skip_sanity=False)
    hf_provider = (cfg.EMBEDDING_PROVIDER or "").strip().lower() in {
        "huggingface", "hf", "sentence_transformers",
        "sentence-transformers"}
    if hf_provider:
        # Local models report their dimension at load; a known size can be
        # asserted via HF_EMBEDDING_DIM (0 = auto-detect, skip the assert).
        expected_dim = int(getattr(cfg, "HF_EMBEDDING_DIM", 0) or 0) or None
    else:
        expected_dim = int(
            getattr(cfg, "GEMINI_OUTPUT_DIMENSIONALITY", 768) or 0) or None
    check("vector dimension matches config",
          expected_dim is None or embedder._dimension == expected_dim,
          f"got={embedder._dimension} config={expected_dim}")
    # One compact notice (GitHub shows at most 10 annotations per step).
    sanity = " | ".join(getattr(embedder, "sanity_lines", ["dimension: ?"]))
    notify(f"sanity: {sanity}")

    # --- 3. Ingest (real embeddings, --reset) ---------------------------------
    CURRENT_STEP = "ingest"
    if not REAL_ONLY:
        progress("\n[ci] STEP 3 — ingest --reset")
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

        check("every record carries required metadata",
              bool(metas) and REQUIRED_METADATA_FIELDS <= set(metas[0]),
              str(sorted(metas[0]) if metas else []))
        check("metadata embedding model matches config",
              bool(metas) and metas[0].get("embedding_model")
              == cfg.active_embedding_model(),
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
    progress("\n[ci] STEP 4 — query (vector + hybrid + lang filter + translation)")

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
            # With the translation-cache artifact restored, a warm run makes
            # zero API calls (all cache hits) — that is success, not a
            # skipped check: require no failures AND at least one served
            # translation (fresh call or cache hit).
            ok_trans = (translator.failures == 0
                        and (translator.api_calls >= 1
                             or translator.cache_hits >= 1))
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

    if not REAL_ONLY:
        rc = main.main(["query",
                        "Quel est le frais mensuel du Compte Courant Atlas ?",
                        "--k", "5", "--skip-sanity-check"])
        check("vector query exits 0", rc == 0, f"rc={rc}")
        rc = main.main(["query",
                        "Which Atlas account is meant for independent professionals?",
                        "--k", "5", "--hybrid", "--lang", "fr", "--skip-sanity-check"])
        check("hybrid query exits 0", rc == 0, f"rc={rc}")


    if not REAL_ONLY:
        # --- 5. Mode × tie-break A/B: vector | rrf | blend ----------------------
        # The first mode embeds every query variant (~46 API inputs); all the
        # remaining modes AND the lambda sweep reuse the on-disk cache, so the
        # entire comparison costs zero additional embedding API calls.
        CURRENT_STEP = "evaluate"
        progress("\n[ci] STEP 5 — evaluation A/B (vector | rrf | blend) x "
                 f"tie-break {', '.join(TIEBREAK_MODES)} "
                 "(cached embeddings, no extra API calls)")
        ab_rows: list[tuple[str, dict]] = []
        for tie in TIEBREAK_MODES:
            old_tie = cfg.FUSION_TIE_BREAK
            cfg.FUSION_TIE_BREAK = tie
            runs: dict[str, dict] = {}
            for mode_label, mode_args in (("vector", []),
                                          ("rrf", ["--hybrid"]),
                                          ("blend", ["--hybrid-blend"])):
                rc = main.main(["evaluate", *mode_args,
                                "--skip-sanity-check"])
                check(f"{tie}/{mode_label} evaluate exits 0", rc == 0,
                      f"rc={rc}")
                run_m = get_latest_eval()
                check(f"{tie}/{mode_label} results JSON written",
                      run_m is not None)
                if run_m is None:
                    continue
                check(f"{tie}/{mode_label} records tie-break",
                      run_m["config"].get("fusion_tie_break") == tie,
                      f"got={run_m['config'].get('fusion_tie_break')}")
                shutil.copy(
                    sorted(glob.glob(str(cfg.RESULTS_DIR / "eval_*.json")))[-1],
                    cfg.RESULTS_DIR / f"tie_{tie}_{mode_label}.json")
                o = run_m["metrics"]["overall"]
                runs[mode_label] = run_m
                progress(f"[ci]   tie={tie:<18} {mode_label:<7} "
                         f"h1={o['hit@1']:.3f} h3={o['hit@3']:.3f} "
                         f"h5={o['hit@5']:.3f} (n={o['n']})")
            cfg.FUSION_TIE_BREAK = old_tie
            if "blend" in runs and "vector" in runs and "rrf" in runs:
                ab_rows.append((tie, runs))

        if not ab_rows:
            check("at least one tie-break A/B row recorded", False)
        else:
            best_tie, best = max(ab_rows, key=lambda t: (
                t[1]["blend"]["metrics"]["overall"]["hit@1"],
                t[1]["blend"]["metrics"]["overall"]["hit@3"],
                t[1]["blend"]["metrics"]["overall"]["hit@5"]))
            cfg.FUSION_TIE_BREAK = best_tie
            notify("fictional tie-break A/B (h1/h3/h5, q10): " + " | ".join(
                f"{tie} v={hit3(r['vector']['metrics']['overall'])}"
                f"@q10={qrank(r['vector'], 'q10')} "
                f"r={hit3(r['rrf']['metrics']['overall'])}"
                f"@q10={qrank(r['rrf'], 'q10')} "
                f"b={hit3(r['blend']['metrics']['overall'])}"
                f"@q10={qrank(r['blend'], 'q10')}"
                for tie, r in ab_rows)
                + f" || best={best_tie} (blend h1="
                + f"{best['blend']['metrics']['overall']['hit@1']:.3f}, "
                  f"vector="
                + f"{best['vector']['metrics']['overall']['hit@1']:.3f}, "
                  f"rrf="
                + f"{best['rrf']['metrics']['overall']['hit@1']:.3f})")
            # Lambda sweep under the winning tie-break + acceptance verdict.
            blend_sweep((), "fictional", best["vector"], best["rrf"],
                        best["blend"], "q10", SWEEP_LAMBDAS)

    # --- 6b. REAL documents (docs/): ingest + vector/hybrid evaluation --------
    # The user's real corpus: BCT circular 2019-08, internal Islamic-banking
    # guide, law 2016-48 (PDFs, Arabic) and two DOCX guides (Arabic). Their own
    # question set lives in questions_real.json.
    CURRENT_STEP = "real-docs-ingest"
    real_docs_dir = str((cfg.PROJECT_DIR.parent / "docs").resolve())
    progress(f"\n[ci] STEP 8 — ingest --reset --data-dir {real_docs_dir}")
    real_ready = False
    # Bind the run variables up-front: later legs (jina A/B) compare against
    # them, and the Gemini leg may defer (quota) before they are assigned.
    run_real = run_real_h = run_real_b = None
    try:
        rc = main.main(["ingest", "--reset", "--data-dir", real_docs_dir,
                        "--skip-sanity-check"])
        check("real-docs ingest exits 0", rc == 0, f"rc={rc}")
        collection_real = get_collection(cfg, reset=False)
        count_real = collection_real.count()
        check("real-docs collection count > 0", count_real > 0,
              f"count={count_real}")
        metas_real = (collection_real.get(include=["metadatas"])["metadatas"] or [])
        langs_real = {m.get("language") for m in metas_real}
        check("real-docs collection is Arabic-only", langs_real == {"ar"},
              str(sorted(langs_real)))
        check("real-docs metadata complete",
              bool(metas_real) and REQUIRED_METADATA_FIELDS <= set(metas_real[0]))
        real_ready = bool(rc == 0 and count_real > 0 and langs_real == {"ar"})
    except RuntimeError as exc:
        # Free-tier daily embedding quota (reset midnight Pacific): the batches
        # embedded so far were saved into the embedding cache artifact by the
        # workflow, so a rerun after the reset continues from where it stopped.
        # A pure quota condition DEFERS instead of failing the pipeline.
        if _REAL_QUOTA.search(str(exc)):
            progress("[ci] real-docs ingest DEFERRED — daily embedding quota "
                     "exhausted (429 RESOURCE_EXHAUSTED). Batch cache saved; "
                     "rerun after the reset continues the ingest from there.")
            notify("real-docs: DEFERRED (daily embedding quota 429 — cache "
                   "artifact saved; rerun after reset continues the ingest)")
            (cfg.RESULTS_DIR / "real_docs_status.json").write_text(
                json.dumps({"status": "deferred-quota",
                            "detail": str(exc)[:500]},
                           ensure_ascii=False, indent=2))
        else:
            raise

    if real_ready:
        # --- 6b.0 Diagnostics: coverage + GLOBAL reachability ----------------
        # Answers may sit beyond top_k (then "correct chunk NOT in top 20" is a
        # ranking problem) or in no chunk at all (then it is a
        # chunking/extraction problem). This scans the WHOLE collection with
        # k=count and reports the correct chunk's rank under every query
        # variant; translations of the worst questions are printed too.
        CURRENT_STEP = "real-diagnostics"
        progress("\n[ci] STEP 8b — real-docs diagnostics (coverage + reachability)")
        real_cases = load_question_set(cfg.PROJECT_DIR / "questions_real.json")
        all_records = collection_real.get(include=["documents", "metadatas"])
        ids_all = all_records["ids"] or []
        texts_all = all_records["documents"] or []
        metas_all = all_records["metadatas"] or []
        norm_all = [normalize_for_match(t) for t in texts_all]
        diag_translator = (main.make_translator(quiet=True)
                           if getattr(cfg, "QUERY_TRANSLATION_ENABLED", False)
                           else None)
        diag_embedder = main.make_embedder(skip_sanity=True)
        corpus_langs = ["ar"]
        problem_rows = []
        unreachable = []
        covered_ok = True
        for case in real_cases:
            sub = case.get("expected_substring")
            if not sub or case.get("category") == "out-of-scope":
                continue
            nsub = normalize_for_match(sub)
            holders = [i for i, t in enumerate(norm_all) if nsub in t]
            if not holders:
                covered_ok = False
                unreachable.append(case["id"])
                # Where does the phrase split? Print which chunks hold its
                # prefix vs suffix (25 chars each) so the boundary is visible.
                pre, suf = nsub[:25], nsub[-25:]
                pre_h = [ids_all[i] for i, t in enumerate(norm_all)
                         if pre in t]
                suf_h = [ids_all[i] for i, t in enumerate(norm_all)
                         if suf in t]
                progress(f"[ci]   diag {case['id']}: UNREACHABLE — expected "
                         "substring not inside ANY chunk (chunking/"
                         "extraction issue, not ranking) | prefix in: "
                         f"{pre_h[:3]} | suffix in: {suf_h[:3]}")
                continue
            correct_id = ids_all[holders[0]]
            m = metas_all[holders[0]] or {}
            q_text = prepare_query_text(case["question"])
            if diag_translator is not None and diag_translator.available:
                variants = diag_translator.build_variants(
                    q_text, case.get("language"), corpus_langs)
            else:
                variants = [{"label": f"{case.get('language')}(original)",
                             "text": q_text}]
            ranks = {}
            for v in variants:
                emb = diag_embedder.embed_query(v["text"])
                hits = query_vector(collection_real, emb, k=count_real)
                ranks[v["label"]] = next((h["rank"] for h in hits
                                          if h["id"] == correct_id), None)
            best = min((r for r in ranks.values() if r), default=None)
            progress(f"[ci]   diag {case['id']}: correct="
                     f"{m.get('source')}::{m.get('chunk_index')} | ranks="
                     + ", ".join(f"{l}={r}" for l, r in ranks.items()))
            if best is None or best > 5:
                problem_rows.append((case["id"], best, variants))
        check("real-docs: every expected substring present in >= 1 chunk",
              covered_ok)
        # ONE aggregated line: per-case notices get dropped by the GitHub
        # 10-annotations-per-step cap, which is exactly what hid the
        # UNREACHABLE id in the previous run.
        parts = []
        if unreachable:
            parts.append("UNREACHABLE (substring not in any chunk): "
                         + ", ".join(unreachable))
            # For each unreachable id, the chunks holding prefix vs suffix
            # (visible in progress; ids summary here — the phrase is split by
            # a chunk boundary exactly where prefix/suffix separate).
            for case in real_cases:
                if case["id"] not in unreachable:
                    continue
                nsub = normalize_for_match(case.get("expected_substring") or "")
                pre, suf = nsub[:25], nsub[-25:]
                pre_h = [ids_all[i] for i, t in enumerate(norm_all) if pre in t]
                suf_h = [ids_all[i] for i, t in enumerate(norm_all) if suf in t]
                parts.append(f"{case['id']} prefix→{pre_h[:2]} suffix→{suf_h[:2]}")
        if problem_rows:
            parts.append("beyond top-5 globally: " + " | ".join(
                f"{qid}: best={best}"
                + (" [" + "; ".join(
                    f"{v['label'][:12]}={v['text'][:55]!r}"
                    for v in variants) + "]" if best is None or best > 10
                   else "")
                for qid, best, variants in problem_rows[:3]))
        if parts:
            notify("real-diagnostics: " + " || ".join(parts))
        else:
            notify("real-diagnostics: all questions covered and reachable "
                   "in top-5 globally (ranking-only misses)")

        # 6b.1 vector evaluation on the REAL question set
        CURRENT_STEP = "real-evaluate"
        progress("\n[ci] STEP 9 — evaluate --questions questions_real.json "
                 "(vector-only)")
        rc = _real_eval(["evaluate", "--questions", "questions_real.json",
                         "--skip-sanity-check"], "real-evaluate")
        if rc is None:
            return
        check("real-docs vector evaluate exits 0", rc == 0, f"rc={rc}")
        run_real = get_latest_eval()
        check("real-docs vector results JSON written", run_real is not None)
        if run_real:
            shutil.copy(
                sorted(glob.glob(str(cfg.RESULTS_DIR / "eval_*.json")))[-1],
                cfg.RESULTS_DIR / "eval_001_real_vector_ab.json")
            m_real = run_real["metrics"]
            o_real = m_real["overall"]
            progress(f"\n[ci] REAL-DOCS (vector) — overall hit rates:"
                     f"  hit@1={o_real['hit@1']:.3f}  hit@3={o_real['hit@3']:.3f}"
                     f"  hit@5={o_real['hit@5']:.3f}  (n={o_real['n']})")
            for cat, d in sorted(m_real["by_category"].items()):
                progress(f"[ci]   category {cat:<14} n={d['n']:<3} "
                         f"hit@1={d['hit@1']:.3f} hit@3={d['hit@3']:.3f} "
                         f"hit@5={d['hit@5']:.3f}")
            for lang, d in sorted(m_real["by_language"].items()):
                progress(f"[ci]   language {lang:<14} n={d['n']:<3} "
                         f"hit@1={d['hit@1']:.3f} hit@3={d['hit@3']:.3f} "
                         f"hit@5={d['hit@5']:.3f}")
            notify(f"real-docs: vector | stored={count_real} chunks "
                   f"langs={sorted(langs_real)} | {eval_summary(run_real)}")

        # 6b.2 hybrid evaluation on the REAL question set
        CURRENT_STEP = "real-hybrid-evaluate"
        progress("\n[ci] STEP 10 — evaluate --hybrid --questions questions_real.json")
        rc = _real_eval(["evaluate", "--hybrid", "--questions",
                         "questions_real.json", "--skip-sanity-check"],
                        "real-hybrid-evaluate")
        if rc is None:
            return
        check("real-docs hybrid evaluate exits 0", rc == 0, f"rc={rc}")
        run_real_h = get_latest_eval()
        check("real-docs hybrid results JSON written", run_real_h is not None)
        if run_real_h:
            shutil.copy(
                sorted(glob.glob(str(cfg.RESULTS_DIR / "eval_*.json")))[-1],
                cfg.RESULTS_DIR / "eval_002_real_hybrid_ab.json")
            m_rh = run_real_h["metrics"]
            o_rh = m_rh["overall"]
            progress(f"\n[ci] REAL-DOCS (hybrid) — overall hit rates:"
                     f"  hit@1={o_rh['hit@1']:.3f}  hit@3={o_rh['hit@3']:.3f}"
                     f"  hit@5={o_rh['hit@5']:.3f}  (n={o_rh['n']})")
            d = delta_str(run_real_h, run_real) if run_real is not None else ""
            notify(f"real-docs: rrf {d} | {eval_summary(run_real_h)}")

        # 6b.3 blend evaluation on the REAL question set
        CURRENT_STEP = "real-blend-evaluate"
        progress("\n[ci] STEP 11 — evaluate --hybrid-blend --questions questions_real.json")
        rc = _real_eval(["evaluate", "--hybrid-blend", "--questions",
                         "questions_real.json", "--skip-sanity-check"],
                        "real-blend-evaluate")
        if rc is None:
            return
        check("real-docs blend evaluate exits 0", rc == 0, f"rc={rc}")
        run_real_b = get_latest_eval()
        check("real-docs blend results JSON written", run_real_b is not None)
        if run_real_b:
            check("real-docs blend run records retrieval_mode=blend",
                  run_real_b["config"].get("retrieval_mode") == "blend")
            m_rb = run_real_b["metrics"]
            progress(f"\n[ci] REAL-DOCS (blend) — overall hit rates:"
                     f"  hit@1={m_rb['overall']['hit@1']:.3f}  "
                     f"hit@3={m_rb['overall']['hit@3']:.3f}  "
                     f"hit@5={m_rb['overall']['hit@5']:.3f}  "
                     f"(n={m_rb['overall']['n']})")
            # Preserve the default-lambda run, then sweep lambda (cache reuse);
            # the merged sweep notice carries the blend result + verdict.
            # (No standalone blend notice: GitHub keeps ~10 per step.)
            files = sorted(glob.glob(str(cfg.RESULTS_DIR / "eval_*.json")))
            if files:
                shutil.copy(files[-1], cfg.RESULTS_DIR
                            / "eval_005_real_blend_ab.json")
            blend_sweep(("--questions", "questions_real.json"), "real",
                        run_real, run_real_h, run_real_b, "rq10",
                        REAL_SWEEP_LAMBDAS)

    # --- 6b.4 REAL chunk-size A/B (220 vs 340 tokens) -------------------------
    # rq13's expected substring was NOT inside any 220-token chunk (phrase
    # split by a boundary). Bigger chunks should keep the sentence whole;
    # query embeddings are cached, so this costs only the new doc
    # embeddings. Summary is merged into the final PASS line (annotation cap).
    CURRENT_STEP = "real-chunksize"
    progress("\n[ci] STEP 11b — real-docs chunk-size A/B (220 vs 340 tokens)")
    chunksize_ab = {}
    # run only when the real leg actually completed
    if real_ready:
        old_size = cfg.CHUNK_SIZE_TOKENS
        old_overlap = cfg.CHUNK_OVERLAP_TOKENS
        cfg.CHUNK_SIZE_TOKENS = 340
        cfg.CHUNK_OVERLAP_TOKENS = 60
        try:
            rc = main.main(["ingest", "--reset", "--data-dir",
                            real_docs_dir, "--skip-sanity-check"])
            check("chunk-size 340 ingest exits 0", rc == 0, f"rc={rc}")
            col340 = get_collection(cfg, reset=False)
            cnt340 = col340.count()
            progress(f"[ci]   chunk-size 340: {cnt340} chunks")
            # coverage at 340 (local scan, no API): how many of the 14
            # expected substrings live inside >= 1 chunk now?
            cov340 = {"covered": 0, "total": 0, "miss": []}
            if cnt340 > 0:
                all340 = col340.get(include=["documents"])["documents"] or []
                norm340 = [normalize_for_match(t) for t in all340]
                for case in real_cases:
                    nsub = normalize_for_match(
                        case.get("expected_substring") or "")
                    if not nsub:
                        continue
                    cov340["total"] += 1
                    if any(nsub in t for t in norm340):
                        cov340["covered"] += 1
                    else:
                        cov340["miss"].append(case["id"])
            progress(f"[ci]   chunk-size 340 coverage: "
                     f"{cov340['covered']}/{cov340['total']} "
                     + (f"miss={','.join(cov340['miss'])}"
                        if cov340["miss"] else "all covered"))
            if cnt340 > 0:
                for mode, margs, key in (("vector", [], "vector"),
                                         ("rrf", ["--hybrid"], "rrf"),
                                         ("blend0.7",
                                          ["--hybrid-blend"], "blend07")):
                    rc = _real_eval(["evaluate", *margs, "--questions",
                                     "questions_real.json",
                                     "--skip-sanity-check"],
                                    f"real-chunk340-{mode}")
                    if rc is None:
                        progress(f"[ci]   chunk340 {mode}: deferred")
                        break
                    runx = get_latest_eval()
                    if runx is not None:
                        o = runx["metrics"]["overall"]
                        chunksize_ab[key] = {
                            "h1": o["hit@1"], "h3": o["hit@3"],
                            "h5": o["hit@5"], "n": o["n"],
                            "chunks": cnt340,
                        }
                        progress(f"[ci]   chunk340 {mode}: "
                                 f"h1={o['hit@1']:.3f} h3={o['hit@3']:.3f} "
                                 f"h5={o['hit@5']:.3f}")
        except RuntimeError as exc:
            # Daily 429: defer cleanly (like the rest of the real leg) instead
            # of crashing; the cache artifact saves what was embedded.
            if _REAL_QUOTA.search(str(exc)):
                progress("[ci] chunk-size 340 ingest DEFERRED — daily "
                         "embedding quota (429); A/B skipped, rerun after "
                         "the midnight-Pacific reset continues from cache.")
                notify("real chunk-size A/B: DEFERRED (429 quota — cache saved; "
                       "rerun after reset)")
            else:
                raise
        finally:
            cfg.CHUNK_SIZE_TOKENS = old_size
            cfg.CHUNK_OVERLAP_TOKENS = old_overlap

    # --- 6c. REAL docs with Jina embeddings (provider A/B) --------------------
    # The user added a Jina API key. jina-embeddings-v5-omni-small is
    # multilingual (100+ langs incl. Arabic); the SAME 836 chunks are re-
    # embedded with retrieval.passage/query tasks. Runs even when the Gemini
    # daily quota is exhausted (separate key/quota); 429 defers cleanly.
    # Provider-scoped cache => the Gemini resumable cache is never clobbered.
    CURRENT_STEP = "real-jina"
    progress("\n[ci] STEP 11c — real-docs Jina A/B "
             "(jina-embeddings-v5-omni-small)")
    jina_summary = ""
    if not os.environ.get("JINA_API_KEY", "").strip():
        progress("[ci]   JINA_API_KEY not set — jina A/B skipped")
    else:
        old_provider = cfg.EMBEDDING_PROVIDER
        old_model = cfg.EMBEDDING_MODEL
        cfg.EMBEDDING_PROVIDER = "jina"
        cfg.EMBEDDING_MODEL = getattr(cfg, "JINA_EMBEDDING_MODEL",
                                      "jina-embeddings-v5-omni-small")
        try:
            jemb = main.make_embedder(skip_sanity=False)
            jdims = jemb._dimension
            jsan = " | ".join(getattr(jemb, "sanity_lines", ["dimension: ?"]))
            notify(f"real-docs jina: embedder dims={jdims} | "
                   f"api={getattr(jemb, 'last_response_preview', None)} | {jsan}")
            rc = main.main(["ingest", "--reset", "--data-dir",
                            real_docs_dir, "--skip-sanity-check"])
            check("jina ingest exits 0", rc == 0, f"rc={rc}")
            jcol = get_collection(cfg, reset=False)
            jcnt = jcol.count()
            progress(f"[ci]   jina ingest: {jcnt} chunks, dims={jdims}")
            jina_res = {}
            for mode, margs, key in (("vector", [], "vector"),
                                     ("rrf", ["--hybrid"], "rrf"),
                                     ("blend0.75", ["--hybrid-blend"],
                                      "blend")):
                rc = _real_eval(["evaluate", *margs, "--questions",
                                 "questions_real.json",
                                 "--skip-sanity-check"],
                                f"real-jina-{mode}")
                if rc is None:
                    progress(f"[ci]   jina {mode}: deferred")
                    break
                runj = get_latest_eval()
                if runj is not None:
                    o = runj["metrics"]["overall"]
                    jina_res[key] = (o["hit@1"], o["hit@3"], o["hit@5"])
                    progress(f"[ci]   jina {mode}: h1={o['hit@1']:.3f} "
                             f"h3={o['hit@3']:.3f} h5={o['hit@5']:.3f}")
            if jina_res:
                jline = " | ".join(
                    f"{k}={v[0]:.3f}/{v[1]:.3f}/{v[2]:.3f}"
                    for k, v in jina_res.items())
                gline = ("gemini vector " + hit3(run_real["metrics"]["overall"])
                         if run_real else "gemini vector (deferred)")
                jina_summary = (f" || real-docs jina A/B (h1/h3/h5): "
                                f"chunks={jcnt} {jline} (vs {gline})")
                notify(f"real-docs jina: chunks={jcnt} | {jline}"
                       + (f" | vs gemini vector {hit3(run_real['metrics']['overall'])}"
                          if run_real else ""))
        except RuntimeError as exc:
            if _REAL_QUOTA.search(str(exc)):
                progress("[ci] jina A/B DEFERRED — API quota/rate limit (429); "
                         "cache saved, rerun continues.")
                notify("real-docs jina A/B: DEFERRED (429 — cache saved; "
                       "rerun continues from cache)")
            else:
                raise
        finally:
            cfg.EMBEDDING_PROVIDER = old_provider
            cfg.EMBEDDING_MODEL = old_model

    # --- 7. Answer stub exists (regression guard) ------------------------------
    CURRENT_STEP = "answer-stub"
    progress("\n[ci] STEP 12 — answer.py stub")
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
    # final PASS line merges the chunk-size A/B so it survives the note cap
    if chunksize_ab:
        v220 = hit3(run_real["metrics"]["overall"]) if run_real else "-"
        r220 = hit3(run_real_h["metrics"]["overall"]) if run_real_h else "-"
        b220 = (hit3(run_real_b["metrics"]["overall"])
                if run_real_b else "-")
        summary = (" || real chunk-size A/B (h1/h3/h5): 220t "
                   f"vector={v220} rrf={r220} blend(.7)={b220}")
        summary += " || 340t " + " ".join(
            f"{k}={d['h1']:.3f}/{d['h3']:.3f}/{d['h5']:.3f}"
            for k, d in sorted(chunksize_ab.items()))
        summary += (" (chunks="
                    + ",".join(str(d["chunks"]) for d in chunksize_ab.values())
                    + f", rq13-reachable={cov340['covered'] == cov340['total']})")
    else:
        summary = ""
    notify("RAGLab CI PASSED: full pipeline mechanics + evaluation OK "
           "(metrics above, results JSON in the raglab-eval-results artifact)"
           + summary + jina_summary)
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
          "| model:", cfg.active_embedding_model())
    print("=" * 78)
    run_main()
