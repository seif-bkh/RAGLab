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

import config as cfg
from chunker import chunk_all
from embedder import build_embedder
from evaluate import (load_question_set, prepare_query_text, print_report,
                      run_evaluation, save_run)
from loader import load_all
from store import (get_collection, keyword_search, query_vector, rrf_merge,
                   store_chunks)


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


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def cmd_inspect(args) -> int:
    banner("INSPECT — load and chunk only, NO embedding, NO API calls")
    docs = load_all(cfg.DATA_DIR)
    if not docs:
        print("[inspect] nothing to inspect; add files to", cfg.DATA_DIR)
        return 1

    print(f"\n[inspect] parameters: chunk_size={cfg.CHUNK_SIZE_TOKENS} tokens | "
          f"overlap={cfg.CHUNK_OVERLAP_TOKENS} tokens | "
          f"split_on_headings_first={cfg.SPLIT_ON_HEADINGS_FIRST}")
    chunks = chunk_all(docs, cfg)
    print(f"\n[inspect] {len(chunks)} chunk(s) across {len(docs)} document(s)\n")

    for chunk in chunks:
        print("-" * 78)
        print(f"chunk #{chunk.index:03d} | source={chunk.source} | "
              f"language={chunk.language} | tokens={chunk.token_count}")
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
    docs = load_all(cfg.DATA_DIR)
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
    print(f"[query] k={k} | lang_filter={lang or 'none'} | hybrid={args.hybrid}")

    q_embedding = embedder.embed_query(q_text)
    print(f"[query] question embedding dimension={len(q_embedding)}")

    vector_hits = query_vector(collection, q_embedding, k=k, lang=lang)
    if args.hybrid:
        kw_hits = keyword_search(collection, q_text, k=k)
        hits = rrf_merge(vector_hits, kw_hits, k=cfg.RRF_RANK_CONSTANT)[:k]
        print(f"[query] vector hits={len(vector_hits)} | keyword hits={len(kw_hits)} | "
              f"fused hits={len(hits)}")
    else:
        hits = vector_hits
        print(f"[query] retrieved {len(hits)} hit(s)")

    for hit in hits:
        meta = hit.get("metadata") or {}
        print("-" * 78)
        print(f"rank       : {hit['rank']}")
        if hit.get("similarity") is not None:
            print(f"similarity : {hit['similarity']:+.4f}  (cosine, 1 - distance)")
        if hit.get("keyword_score") is not None:
            print(f"BM25 score : {hit['keyword_score']:.4f}")
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
    cases = load_question_set(cfg.QUESTIONS_FILE)
    if not cases:
        print("[evaluate] no questions in", cfg.QUESTIONS_FILE)
        return 1

    collection = get_collection(cfg, reset=False)
    if collection.count() == 0:
        print("[evaluate] ERROR: collection is empty. Run: python main.py ingest --reset")
        return 1

    embedder = make_embedder(skip_sanity=args.skip_sanity_check)
    run = run_evaluation(cfg, embedder, collection, cases,
                         hybrid=args.hybrid, top_k=args.top_k)
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
    p_inspect.set_defaults(func=cmd_inspect)

    p_ingest = sub.add_parser("ingest", help="chunk, embed and store into ChromaDB")
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
    p_query.add_argument("--hybrid", action="store_true", default=False,
                         help="merge BM25 keyword search with vector search by RRF")
    p_query.add_argument("--skip-sanity-check", action="store_true", default=False,
                         dest="skip_sanity_check",
                         help="skip the default 3-language sanity check")
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser("evaluate", help="run questions.json and report metrics")
    p_eval.add_argument("--hybrid", action="store_true", default=False,
                        help="use vector + BM25 RRF fusion for evaluation")
    p_eval.add_argument("--top-k", type=int, default=cfg.EVAL_TOP_K,
                        help=f"hits recorded per question (default {cfg.EVAL_TOP_K})")
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
