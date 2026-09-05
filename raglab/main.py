"""main.py — CLI entry point for the RAGLab mini laboratory.

Subcommands (run from this project folder, inside the virtualenv):

    python main.py inspect                    load + chunk only, print every chunk
    python main.py ingest [--reset]           chunk -> embed -> store in ChromaDB
    python main.py query "question" [-k 5] [--lang fr] [--hybrid]
    python main.py evaluate [--hybrid] [--top-k N]

Answer generation is optional: `answer` uses cited source excerpts and safe
refusal; ordinary query/evaluate commands still measure retrieval only.
"""

import argparse
import json
import statistics
from datetime import datetime, timezone
from types import SimpleNamespace
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
from retrieval import retrieve, expand_neighbors
from artifacts import write_json


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
          f"{cfg.NVIDIA_TRANSLATION_MODEL if cfg.QUERY_TRANSLATION_PROVIDER == 'nvidia' else cfg.QUERY_TRANSLATION_MODEL} | fusion=best_variant_max_score | "
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

    if not args.reset:
        existing = get_collection(cfg, reset=False)
        if existing.count():
            raise RuntimeError("Collection is nonempty. Use python main.py ingest --reset to replace its snapshot safely.")

    embedder = make_embedder(skip_sanity=args.skip_sanity_check)

    print(f"\n[ingest] embedding {len(chunks)} chunk(s) "
          f"(batch size {embedder.batch_size})...")
    embeddings = embedder.embed_texts([c.text for c in chunks])
    assert len(embeddings) == len(chunks)
    print(f"[ingest] embedding done | cache hits={embedder.cache_hits} | "
          f"API calls={embedder.api_calls} | dimension={len(embeddings[0])}")

    collection = get_collection(cfg, reset=args.reset)

    pairs = list(zip(chunks, embeddings))
    store_chunks(collection, pairs, cfg)

    print(f"\n[ingest] final collection count = {collection.count()}")
    print(f"[ingest] model used            = {embedder.model}")
    print(f"[ingest] cache file            = {embedder.cache.path.name}")
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

    qlang = args.query_lang or detect_language(q_text)
    hits, variants = retrieve(cfg, embedder, collection, q_text, language=qlang,
                              translator=translator, mode=mode, top_k=k,
                              lang_filter=lang, variant_strategy=args.variant_strategy)
    for variant in variants:
        print(f"[query] variant [{variant['label']}]: {variant['text']}")
    if translator:
        print("[query] " + translator.summary())

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


def cmd_answer(args):
    from answer import AnswerGenerator, needs_private_or_live_data
    local = SimpleNamespace(**{key: getattr(cfg, key) for key in dir(cfg) if key.isupper()})
    local.ANSWER_MODEL = args.model
    generator = AnswerGenerator(local)
    question = " ".join(args.question).strip()
    if not question:
        raise ValueError("Question must not be empty")
    language = args.query_lang or detect_language(question)
    hits, variants = [], []
    if not needs_private_or_live_data(question):
        collection = get_collection(cfg)
        if not collection.count():
            raise ValueError("Collection is empty; run python main.py ingest --reset --data-dir ../docs")
        embedder = make_embedder(skip_sanity=True)
        translator = None if args.no_translation else make_translator()
        hits, variants = retrieve(cfg, embedder, collection, prepare_query_text(question),
                                  language=language, translator=translator, top_k=args.k,
                                  variant_strategy=args.variant_strategy)
        hits = expand_neighbors(collection, hits, args.neighbor_radius)
    result = generator.answer(question, hits, language)
    result.update(question=question, query_variants=variants)
    print(result["answer"])
    for source in result["sources"]:
        used = [e["quote"] for c in result["claims"] for e in c["evidence"] if e["source_id"] == source["source_id"]]
        if used or args.show_context:
            print(f"\n[{source['source_id']}] {source['document']} — {source['chunk_id']}")
            print(source["text"] if args.show_context else "\n".join(used))
    path = Path(args.output) if args.output else cfg.RESULTS_DIR / ("answer_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + ".json")
    write_json(path, result)
    print(f"\n[answer] status={result['status']} reason={result['reason']} | {path}")
    return 0 if result["validation_ok"] else 2


def cmd_benchmark(args):
    from nvidia_benchmark import run
    result = run(stage=args.stage, quality=not args.skip_translation_references)
    return 0 if result["status"] == "completed" else 2


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
    p_query.add_argument("--variant-strategy", choices=["original", "best", "translated"],
                         default=cfg.QUERY_VARIANT_STRATEGY)
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

    from nvidia_api import ANSWER_MODELS
    p_answer = sub.add_parser("answer", help="generate a document-grounded, cited answer or refusal")
    p_answer.add_argument("question", nargs="+")
    p_answer.add_argument("--model", choices=ANSWER_MODELS, default=cfg.ANSWER_MODEL)
    p_answer.add_argument("--query-lang", choices=["en", "fr", "ar"], default=None)
    p_answer.add_argument("-k", type=int, default=cfg.ANSWER_TOP_K)
    p_answer.add_argument("--no-translation", action="store_true")
    p_answer.add_argument("--variant-strategy", choices=["original", "best", "translated"], default=cfg.QUERY_VARIANT_STRATEGY)
    p_answer.add_argument("--neighbor-radius", type=int, choices=[0, 1, 2], default=cfg.ANSWER_NEIGHBOR_RADIUS)
    p_answer.add_argument("--show-context", action="store_true")
    p_answer.add_argument("--output", help="where to save the complete answer/evidence JSON")
    p_answer.set_defaults(func=cmd_answer)
    p_bench = sub.add_parser("benchmark", help="exact-model NVIDIA comparison (uses API quota)")
    p_bench.add_argument("--stage", choices=["retrieval", "all"], default="retrieval")
    p_bench.add_argument("--skip-translation-references", action="store_true")
    p_bench.set_defaults(func=cmd_benchmark)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, RuntimeError) as exc:
        if argv is not None:
            raise  # programmatic runners retain their quota/error handling
        print(f"[main] ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[main] interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
