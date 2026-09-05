"""main.py — CLI entry point for the RAGLab mini laboratory.

Subcommands (run from this project folder, inside the virtualenv):

    python main.py inspect                    load + chunk only, print every chunk
    python main.py ingest [--reset]           chunk -> embed -> store in ChromaDB
    python main.py query "question" [-k 5] [--lang fr] [--hybrid]
    python main.py evaluate [--hybrid] [--top-k N]

There is intentionally no answer generation: evaluate.py / answer.py keep a
clearly marked stub. Run `python main.py <subcommand> --help` for details.
"""

import argparse
import statistics
import sys
from pathlib import Path

import config as cfg
from chunker import chunk_all
from embedder import build_embedder
from evaluate import (load_question_set, prepare_query_text, print_report,
                      run_evaluation, save_run)
from loader import load_all
from store import (best_variant_merge, blend_hybrid, collection_languages,
                   get_collection, keyword_search, query_vector, rrf_merge,
                   store_chunks)
from translate import QueryTranslator, detect_language


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def banner(title: str):
    print("=" * 78)
    print(title)
    print("=" * 78)


def make_embedder(skip_sanity: bool = False):
    """Build the embedding provider and, unless asked otherwise, print the
    provider/model/dimension startup report + 3-language sanity check."""
    embedder = build_embedder(cfg)
    if not skip_sanity:
        embedder.startup_report()
    else:
        print(f"[embedder] provider={embedder.provider_name} model={embedder.model} "
              f"(sanity check skipped)")
    return embedder


def make_translator(quiet: bool = False):
    """Build the query translator when enabled, else None (original queries).

    Never raises: an unavailable/failing translator simply means the lab runs
    without query translation (and records it in the evaluation run config).
    """
    if not bool(getattr(cfg, "QUERY_TRANSLATION_ENABLED", False)):
        if not quiet:
            print("[translate] QUERY_TRANSLATION_ENABLED=False — original queries only")
        return None
    try:
        return QueryTranslator(cfg)
    except SystemExit:
        raise  # missing SDK: let the actionable message propagate
    except Exception as exc:  # noqa: BLE001 — degrade, never block retrieval
        print(f"[translate] WARNING: disabled ({exc})")
        return None


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def cmd_inspect(args) -> int:
    banner("INSPECT — load and chunk only, NO embedding, NO API calls")
    data_dirs = args.data_dir if args.data_dir else [cfg.DATA_DIR]
    docs = load_all(data_dirs)
    if not docs:
        print("[inspect] nothing to inspect; add files to", cfg.DATA_DIR)
        return 1

    print(f"\n[inspect] parameters: chunk_size={cfg.CHUNK_SIZE_TOKENS} tokens | "
          f"overlap={cfg.CHUNK_OVERLAP_TOKENS} tokens | "
          f"split_on_headings_first={cfg.SPLIT_ON_HEADINGS_FIRST} | "
          f"sentence_aware_overlap={cfg.CHUNK_OVERLAP_SENTENCE_AWARE}")
    print(f"[inspect] query translation: enabled="
          f"{cfg.QUERY_TRANSLATION_ENABLED} | model="
          f"{cfg.QUERY_TRANSLATION_MODEL} | fusion=best_variant_max_score | "
          f"cache={cfg.QUERY_TRANSLATION_CACHE_PATH.name}")
    chunks = chunk_all(docs, cfg)
    print(f"\n[inspect] {len(chunks)} chunk(s) across {len(docs)} document(s)\n")

    for chunk in chunks:
        print("-" * 78)
        print(f"chunk #{chunk.index:03d} | source={chunk.source} | "
              f"language={chunk.language} | tokens={chunk.token_count} | "
              f"section={chunk.section_type}")
        print(f"heading: {chunk.heading or '(none)'}")
        if chunk.notes:
            print(f"notes  : {', '.join(chunk.notes)}")
        print("text:")
        print(chunk.text)

    # Token distribution summary.
    counts = [c.token_count for c in chunks]
    if counts:
        print("\n" + "=" * 78)
        print(f"[inspect] SUMMARY | chunks={len(counts)} | "
              f"tokens total={sum(counts)}")
        print(f"[inspect] token distribution: min={min(counts)} "
              f"median={statistics.median(counts):.0f} "
              f"mean={statistics.mean(counts):.1f} max={max(counts)}")
        print("[inspect] token histogram (by chunk):")
        step = max(1, cfg.CHUNK_SIZE_TOKENS // 5)
        for low in range(0, max(counts) + 1, step):
            high = low + step
            n = sum(1 for c in counts if low <= c < high)
            if n:
                bar = "#" * n
                print(f"  {low:>4}-{high:<4}: {n:>2} {bar}")
        print("=" * 78)
    return 0


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


def cmd_ingest(args) -> int:
    banner("INGEST — load, chunk, embed, store")
    data_dirs = args.data_dir if args.data_dir else [cfg.DATA_DIR]
    docs = load_all(data_dirs)
    chunks = chunk_all(docs, cfg)
    if not chunks:
        print("[ingest] no chunks produced; nothing to do")
        return 1

    embedder = make_embedder(skip_sanity=args.skip_sanity_check)

    print(f"\n[ingest] embedding {len(chunks)} chunk(s) "
          f"(batch size {cfg.EMBEDDING_BATCH_SIZE})...")
    embeddings = embedder.embed_texts([c.text for c in chunks])
    assert len(embeddings) == len(chunks)
    print(f"[ingest] embedding done | cache hits={embedder.cache_hits} | "
          f"API calls={embedder.api_calls} | dimension={len(embeddings[0])}")

    collection = get_collection(cfg, reset=args.reset)
    if collection.count() > 0 and not args.reset:
        print(f"[ingest] WARNING: collection already holds {collection.count()} record(s); "
              "chunk ids are deterministic, so re-adding the same chunks does not create "
              "duplicates, but metadata from an older ingest (e.g. timestamps) may remain. "
              "Use --reset for a guaranteed clean rebuild.")

    pairs = list(zip(chunks, embeddings))
    store_chunks(collection, pairs, cfg)

    print(f"\n[ingest] final collection count = {collection.count()}")
    print(f"[ingest] model used            = {embedder.model}")
    print(f"[ingest] cache file            = {cfg.EMBEDDING_CACHE_PATH.name}")
    print("[ingest] next: python main.py query \"...\" or python main.py evaluate")
    return 0


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


def cmd_query(args) -> int:
    banner("QUERY — embed the question, retrieve from ChromaDB")
    question = " ".join(args.question).strip()
    if not question:
        print("[query] no question given")
        return 1

    q_text = prepare_query_text(question)
    print(f"[query] raw question : {question}")
    if q_text != question:
        print(f"[query] cleaned text : {q_text}")

    embedder = make_embedder(skip_sanity=args.skip_sanity_check)
    collection = get_collection(cfg, reset=False)
    if collection.count() == 0:
        print("[query] ERROR: collection is empty. Run: python main.py ingest --reset")
        return 1

    k = args.k
    lang = args.lang
    translator = None if args.no_translation else make_translator()
    mode = "blend" if args.hybrid_blend else ("rrf" if args.hybrid else "vector")
    print(f"[query] k={k} | lang_filter={lang or 'none'} | mode={mode} | "
          f"blend_lambda={cfg.HYBRID_BLEND_LAMBDA:.2f} | "
          f"translation={'on' if translator else 'off'}")

    # Original query + translations into each corpus language (best-score
    # fusion across variants), so cross-lingual questions see both language
    # halves of the corpus.
    qlang = args.query_lang or detect_language(q_text)
    print(f"[query] detected query language: {qlang}")
    if translator:
        variants = translator.build_variants(q_text, qlang,
                                             collection_languages(collection))
        for v in variants:
            tag = "original" if not v["translated"] else "translated"
            print(f"[query] variant [{v['label']}] ({tag}): {v['text']}")
    else:
        variants = [{
            "label": f"{qlang}(original)", "lang": qlang,
            "translated": False, "text": q_text,
        }]

    variant_hit_lists = []
    for variant in variants:
        q_embedding = embedder.embed_query(variant["text"])
        print(f"[query] embedded variant {variant['label']} "
              f"dimension={len(q_embedding)}")
        v_hits = query_vector(collection, q_embedding, k=k, lang=lang,
                        cfg=cfg)
        if mode == "rrf":
            kw_hits = keyword_search(collection, variant["text"], k=k,
                        include_metadata=getattr(
                            cfg, "KEYWORD_SEARCH_INCLUDE_METADATA",
                            True))
            v_hits = rrf_merge(v_hits, kw_hits, k=cfg.RRF_RANK_CONSTANT)[:k]
        elif mode == "blend":
            kw_hits = keyword_search(collection, variant["text"], k=k, cfg=cfg)
            v_hits = blend_hybrid(v_hits, kw_hits,
                                  lambd=cfg.HYBRID_BLEND_LAMBDA)[:k]
        variant_hit_lists.append(v_hits)

    if len(variant_hit_lists) > 1:
        score_key = {"vector": "similarity", "rrf": "rrf_score",
                     "blend": "blend_score"}[mode]
        hits = best_variant_merge(
            variant_hit_lists, score_key=score_key,
            labels=[v["label"] for v in variants],
            tie_break=getattr(cfg, "FUSION_TIE_BREAK",
                              "same_lang_margin"))[:k]
        print(f"[query] fused {len(hits)} hit(s) from {len(variants)} "
              f"variant(s) (best-score fusion, tie_break="
              f"{cfg.FUSION_TIE_BREAK})")
    else:
        hits = variant_hit_lists[0]
        print(f"[query] retrieved {len(hits)} hit(s)")

    for hit in hits:
        meta = hit.get("metadata") or {}
        print("-" * 78)
        print(f"rank       : {hit['rank']}")
        if hit.get("from_variant") is not None:
            print(f"best variant: {hit['from_variant']}")
            ranks = hit.get("variant_ranks") or {}
            if ranks:
                print(f"variant ranks: {ranks}")
        if hit.get("similarity") is not None:
            print(f"similarity : {hit['similarity']:+.4f}  (cosine, 1 - distance)")
        if hit.get("relative_score") is not None:
            print(f"relative   : {hit['relative_score']:.4f}  "
                  f"(score / best score of its variant)")
        if hit.get("keyword_score") is not None:
            print(f"BM25 score : {hit['keyword_score']:.4f}")
        if hit.get("kw_norm") is not None:
            print(f"BM25 norm  : {hit['kw_norm']:.4f}  (score / max of variant)")
        if hit.get("blend_score") is not None:
            print(f"blend score: {hit['blend_score']:.4f}  "
                  f"(lambda*cosine + (1-lambda)*BM25_norm)")
        if hit.get("rrf_score") is not None:
            print(f"RRF score  : {hit['rrf_score']:.4f}  (1/(k+rank) fusion)")
        print(f"language   : {meta.get('language')}")
        print(f"heading    : {meta.get('heading') or '(none)'}")
        print(f"source     : {meta.get('source')} | chunk {meta.get('chunk_index')} | "
              f"ingested {meta.get('ingested_at')} | model {meta.get('embedding_model')}")
        print(f"id         : {hit['id']}")
        print("text:")
        print(hit["text"])

    if not hits:
        print(f"[query] no hits returned (lang filter {lang!r} may exclude "
              "everything)")
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def cmd_evaluate(args) -> int:
    banner("EVALUATE — run questions.json against the collection")
    qpath = (Path(args.questions).resolve() if args.questions
             else cfg.QUESTIONS_FILE)
    cases = load_question_set(qpath)
    if not cases:
        print("[evaluate] no questions in", cfg.QUESTIONS_FILE)
        return 1

    collection = get_collection(cfg, reset=False)
    if collection.count() == 0:
        print("[evaluate] ERROR: collection is empty. Run: python main.py ingest --reset")
        return 1

    embedder = make_embedder(skip_sanity=args.skip_sanity_check)
    translator = None if args.no_translation else make_translator()
    mode = "blend" if args.hybrid_blend else ("rrf" if args.hybrid else "vector")
    run = run_evaluation(cfg, embedder, collection, cases,
                         mode=mode, top_k=args.top_k,
                         translator=translator)
    print_report(run)
    save_run(run, cfg.RESULTS_DIR)
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="RAGLab — local, transparent multilingual RAG retrieval lab.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="load+chunk only, print every chunk")
    p_inspect.add_argument("--data-dir", action="append", default=None,
                           metavar="PATH",
                           help="extra directory to load (repeatable; default: data/)")
    p_inspect.set_defaults(func=cmd_inspect)

    p_ingest = sub.add_parser("ingest", help="chunk, embed and store into ChromaDB")
    p_ingest.add_argument("--data-dir", action="append", default=None,
                          metavar="PATH",
                          help="extra directory to load (repeatable; default: data/)")
    p_ingest.add_argument("--reset", action="store_true", default=False,
                          help="delete the collection first (fresh start)")
    p_ingest.add_argument("--skip-sanity-check", action="store_true", default=False,
                          dest="skip_sanity_check",
                          help="skip the default 3-language sanity check")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="embed a question and retrieve top-k")
    p_query.add_argument("question", nargs="+", help="the question text")
    p_query.add_argument("-k", "--k", type=int, default=cfg.RETRIEVAL_TOP_K,
                         help=f"number of results (default {cfg.RETRIEVAL_TOP_K})")
    p_query.add_argument("--lang", choices=["en", "fr", "ar"], default=None,
                         help="filter stored chunks by language")
    p_query.add_argument("--query-lang", choices=["en", "fr", "ar"], default=None,
                         dest="query_lang",
                         help="language of the question (default: auto-detected)")
    qmode = p_query.add_mutually_exclusive_group()
    qmode.add_argument("--hybrid", action="store_true", default=False,
                       help="merge BM25 with vector search by RRF")
    qmode.add_argument("--hybrid-blend", action="store_true", default=False,
                       dest="hybrid_blend",
                       help="score-blend BM25 with vector search "
                            f"(lambda={cfg.HYBRID_BLEND_LAMBDA:.2f})")
    p_query.add_argument("--no-translation", action="store_true", default=False,
                         dest="no_translation",
                         help="disable query translation (original query only)")
    p_query.add_argument("--skip-sanity-check", action="store_true", default=False,
                         dest="skip_sanity_check",
                         help="skip the default 3-language sanity check")
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser("evaluate", help="run questions.json and report metrics")
    emode = p_eval.add_mutually_exclusive_group()
    emode.add_argument("--hybrid", action="store_true", default=False,
                       help="use vector + BM25 RRF fusion for evaluation")
    emode.add_argument("--hybrid-blend", action="store_true", default=False,
                       dest="hybrid_blend",
                       help="use vector + BM25 score-blend fusion for evaluation "
                            f"(lambda={cfg.HYBRID_BLEND_LAMBDA:.2f})")
    p_eval.add_argument("--top-k", type=int, default=cfg.EVAL_TOP_K,
                        help=f"hits recorded per question (default {cfg.EVAL_TOP_K})")
    p_eval.add_argument("--questions", default=None, metavar="PATH",
                        help="question set JSON (default: questions.json)")
    p_eval.add_argument("--no-translation", action="store_true", default=False,
                        dest="no_translation",
                        help="disable query translation (baseline comparison)")
    p_eval.add_argument("--skip-sanity-check", action="store_true", default=False,
                        dest="skip_sanity_check",
                        help="skip the default 3-language sanity check")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n[main] interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
