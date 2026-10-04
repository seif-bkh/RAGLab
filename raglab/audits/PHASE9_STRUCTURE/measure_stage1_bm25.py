#!/usr/bin/env python3
"""Run a local, retrieval-only diagnostic over the Phase 9 Stage 1 cases.

The script uses the adopted restructure chunking path and the production
store.keyword_search BM25 implementation. Chroma is ephemeral and receives
fixed placeholder vectors solely because a collection requires embeddings;
no vector search, embedding provider, chat model, or external API is called.

The case labels remain provisional. This run measures retrieval of the cited
source spans; it does not measure customer-facing answer behavior.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RAGLAB = REPO / "raglab"
sys.path.insert(0, str(RAGLAB))

import chromadb  # noqa: E402
import chunker  # noqa: E402
import evaluate  # noqa: E402
import harness50  # noqa: E402
import store  # noqa: E402
from loader import load_all  # noqa: E402

CASES_PATH = HERE / "stage1_cases_draft.json"
DOCS_DIRS = [REPO / "docs", RAGLAB / "data"]
TOP_K = 20
CHROMA_DIM = 8
RESULTS_DIR = RAGLAB / "results" / "phase9_stage1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def placeholder_vector(_text: str) -> list[float]:
    """Stable dummy vector; only the lexical path is queried in this run."""
    return [1.0] + [0.0] * (CHROMA_DIM - 1)


def source_name_for_evidence(evidence_file: str, docs: list[dict]) -> str:
    """Map an adopted *_corrected.md citation to its original docs/ filename."""
    name = Path(evidence_file).name
    suffix = "_corrected.md"
    if not name.endswith(suffix):
        raise ValueError(f"Expected an adopted corrected codex path, got {evidence_file!r}")
    expected_stem = name[:-len(suffix)]
    matches = [doc["name"] for doc in docs
               if Path(doc["name"]).stem == expected_stem]
    if len(matches) != 1:
        raise ValueError(
            f"Could not map {evidence_file!r} to exactly one loaded source; "
            f"matches={matches!r}")
    return matches[0]


def make_collection(chunks: list, cfg):
    """Create a transient collection for the production lexical search code."""
    try:
        settings = chromadb.config.Settings(anonymized_telemetry=False)
        client = chromadb.EphemeralClient(settings=settings)
    except Exception:  # noqa: BLE001 — tolerate Chroma Settings API variation
        client = chromadb.EphemeralClient()

    collection = client.create_collection(
        name="phase9_stage1_bm25",
        metadata={"hnsw:space": "cosine", "measurement": "bm25-only"},
    )
    fingerprint = store.chunk_fp(cfg)
    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        part = chunks[start:start + batch_size]
        collection.add(
            ids=[f"{chunk.source}::chunk_{chunk.index:04d}" for chunk in part],
            embeddings=[placeholder_vector(chunk.text) for chunk in part],
            documents=[chunk.text for chunk in part],
            metadatas=[{
                "document": chunk.source,
                "source": chunk.source,
                "language": chunk.language,
                "heading": chunk.heading,
                "chunk_index": chunk.index,
                "origin": chunk.origin,
                "section_type": chunk.section_type,
                "token_count": chunk.token_count,
                "chunk_fp": fingerprint,
            } for chunk in part],
        )
    if collection.count() != len(chunks):
        raise RuntimeError(
            f"Transient Chroma collection count {collection.count()} "
            f"does not match chunk count {len(chunks)}")
    return client, collection, fingerprint


def first_rank_for_quote(hits: list[dict], quote: str, source: str) -> int | None:
    wanted = evaluate.normalize_for_match(quote)
    for hit in hits:
        metadata = hit.get("metadata") or {}
        actual_source = metadata.get("document") or metadata.get("source")
        if actual_source == source and wanted in evaluate.normalize_for_match(hit["text"]):
            return hit["rank"]
    return None


def corpus_occurrences(chunks: list, quote: str, source: str) -> list[int]:
    wanted = evaluate.normalize_for_match(quote)
    return [i for i, chunk in enumerate(chunks)
            if chunk.source == source
            and wanted in evaluate.normalize_for_match(chunk.text)]


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1,
                       math.ceil(percentile_value * len(ordered)) - 1))
    return ordered[index]


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if payload.get("status") != "draft_pending_owner_review":
        raise SystemExit("Refusing to run: case manifest is no longer the reviewed draft")

    docs = load_all(DOCS_DIRS)
    if not docs:
        raise SystemExit("No source documents were loaded")

    cfg = harness50.make_cfg(
        "restructure", "phase9_stage1_bm25", RESULTS_DIR / "unused_chroma_path")
    chunks = chunker.chunk_all(docs, cfg)
    if not chunks:
        raise SystemExit("Chunker produced no chunks")
    tokenizer = chunker.tokenizer_identity()
    client, collection, fingerprint = make_collection(chunks, cfg)

    # Make sure every target quote exists in its intended adopted-codex chunk
    # before computing ranked retrieval. This is a corpus-integrity check,
    # not retrieval credit.
    evidence_case_ids = []
    evidence_spans_total = 0
    for case in cases:
        refs = case.get("expected_evidence") or []
        if not refs:
            continue
        evidence_case_ids.append(case["id"])
        evidence_spans_total += len(refs)
        for index, ref in enumerate(refs):
            source = source_name_for_evidence(ref["file"], docs)
            if not corpus_occurrences(chunks, ref["quote"], source):
                raise SystemExit(
                    f"Evidence integrity failure: {case['id']} span {index} is "
                    f"not present in any {source!r} restructure chunk")

    per_case = []
    retrieval_latencies_ms: list[float] = []
    for case in cases:
        query = evaluate.prepare_query_text(case["question"])
        start = time.perf_counter()
        hits = store.keyword_search(
            collection,
            query,
            k=TOP_K,
            include_metadata=True,
            cfg=None,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        retrieval_latencies_ms.append(elapsed_ms)

        span_results = []
        for index, ref in enumerate(case.get("expected_evidence") or []):
            source = source_name_for_evidence(ref["file"], docs)
            span_results.append({
                "span_index": index,
                "evidence_file": ref["file"],
                "expected_source_document": source,
                "rank": first_rank_for_quote(hits, ref["quote"], source),
                "quote": ref["quote"],
            })
        found_ranks = [entry["rank"] for entry in span_results
                       if entry["rank"] is not None]
        all_spans_found = bool(span_results) and len(found_ranks) == len(span_results)
        top1 = hits[0] if hits else None
        top1_meta = (top1.get("metadata") or {}) if top1 else {}
        per_case.append({
            "id": case["id"],
            "question": case["question"],
            "language": case.get("language"),
            "expected_behavior": case.get("expected_behavior"),
            "label_status": case.get("review_status"),
            "retrieval_scoring": "evidence-span" if span_results else "diagnostic-only-no-gold-span",
            "normalized_query": query,
            "evidence_spans": span_results,
            "first_evidence_rank": min(found_ranks) if found_ranks else None,
            "complete_evidence_rank": max(found_ranks) if all_spans_found else None,
            "top1_diagnostic": ({
                "source_document": top1_meta.get("document") or top1_meta.get("source"),
                "heading": top1_meta.get("heading"),
                "keyword_score": top1.get("keyword_score"),
            } if top1 else None),
            "retrieval_latency_ms": round(elapsed_ms, 3),
            "top_hits": [{
                "rank": hit["rank"],
                "source_document": (hit.get("metadata") or {}).get("document")
                                   or (hit.get("metadata") or {}).get("source"),
                "heading": (hit.get("metadata") or {}).get("heading"),
                "keyword_score": hit.get("keyword_score"),
                "text_preview": hit["text"][:320],
            } for hit in hits[:5]],
        })

    evidence_cases = [row for row in per_case if row["evidence_spans"]]
    metrics = {}
    for k in (1, 3, 5):
        hit_spans = sum(
            1 for row in evidence_cases for span in row["evidence_spans"]
            if span["rank"] is not None and span["rank"] <= k)
        complete_cases = sum(
            1 for row in evidence_cases
            if row["complete_evidence_rank"] is not None
            and row["complete_evidence_rank"] <= k)
        any_evidence_cases = sum(
            1 for row in evidence_cases
            if row["first_evidence_rank"] is not None
            and row["first_evidence_rank"] <= k)
        metrics[f"evidence_span_recall@{k}"] = {
            "hits": hit_spans,
            "total": evidence_spans_total,
            "rate": hit_spans / evidence_spans_total if evidence_spans_total else None,
        }
        metrics[f"any_evidence_case_hit@{k}"] = {
            "hits": any_evidence_cases,
            "total": len(evidence_cases),
            "rate": any_evidence_cases / len(evidence_cases) if evidence_cases else None,
        }
        metrics[f"complete_evidence_case_hit@{k}"] = {
            "hits": complete_cases,
            "total": len(evidence_cases),
            "rate": complete_cases / len(evidence_cases) if evidence_cases else None,
        }

    latency = {
        "n_queries": len(retrieval_latencies_ms),
        "mean_ms": statistics.mean(retrieval_latencies_ms) if retrieval_latencies_ms else None,
        "median_ms": statistics.median(retrieval_latencies_ms) if retrieval_latencies_ms else None,
        "p95_ms": percentile(retrieval_latencies_ms, 0.95),
        "note": "BM25 search only in this sandbox; excludes document loading, chunking, answer generation, and production serving overhead.",
    }
    run = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "experiment": "phase9_stage1_local_bm25_retrieval_diagnostic",
        "manifest_path": str(CASES_PATH.relative_to(REPO)),
        "manifest_sha256": sha256_file(CASES_PATH),
        "manifest_status": payload.get("status"),
        "case_labels_are_final": False,
        "corpus_documents": [doc["name"] for doc in docs],
        "corpus_document_count": len(docs),
        "chunking": {
            "mode": "restructure",
            "chunk_size_tokens": cfg.CHUNK_SIZE_TOKENS,
            "overlap_tokens": cfg.CHUNK_OVERLAP_TOKENS,
            "split_on_headings_first": cfg.SPLIT_ON_HEADINGS_FIRST,
            "sentence_aware_overlap": cfg.CHUNK_OVERLAP_SENTENCE_AWARE,
            "rtl_repair": cfg.RESTRUCTURE_RTL_REPAIR,
            "tokenizer": tokenizer,
            "chunk_fingerprint": fingerprint,
            "chunk_count": len(chunks),
            "chunk_tokens_sum": sum(chunk.token_count for chunk in chunks),
        },
        "retrieval": {
            "implementation": "raglab.store.keyword_search / BM25Index",
            "top_k": TOP_K,
            "include_metadata": True,
            "chroma_mode": "ephemeral; fixed placeholder embeddings are never queried",
            "vector_search_used": False,
            "embedding_api_calls": 0,
            "generation_api_calls": 0,
            "external_model_calls": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "chromadb": getattr(chromadb, "__version__", "unknown"),
        },
        "evidence_integrity": {
            "answerable_cases": len(evidence_cases),
            "evidence_spans": evidence_spans_total,
            "all_spans_found_in_intended_restructure_source_before_ranking": True,
        },
        "metrics": metrics,
        "retrieval_latency": latency,
        "generation_or_behavior_evaluated": False,
        "non_evidence_cases_are_unscored": True,
        "cases": per_case,
    }
    out_json = RESULTS_DIR / "stage1_bm25_run.json"
    out_json.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    client.clear_system_cache() if hasattr(client, "clear_system_cache") else None
    print(f"[phase9-stage1] wrote {out_json.relative_to(REPO)}")
    print(f"[phase9-stage1] cases={len(cases)} | evidence_cases={len(evidence_cases)} "
          f"| spans={evidence_spans_total} | chunks={len(chunks)} | tokenizer={tokenizer}")
    for k in (1, 3, 5):
        row = metrics[f"complete_evidence_case_hit@{k}"]
        spans = metrics[f"evidence_span_recall@{k}"]
        print(f"[phase9-stage1] complete evidence case@{k}: {row['hits']}/{row['total']} "
              f"({row['rate']:.1%}) | span recall: {spans['hits']}/{spans['total']} "
              f"({spans['rate']:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
