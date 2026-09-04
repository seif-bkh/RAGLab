"""ci_test.py — end-to-end integration test for GitHub Actions.

Runs the FULL pipeline against the real Gemini embedding API using the
GEMINI_API_KEY provided by the CI secret (written to .env by the workflow):

    inspect (offline) -> sanity check -> ingest --reset -> query ->
    evaluate -> saved results JSON

Every step is a CLI invocation via main.main(), so argparse is exercised too.
Mechanics are asserted (counts, dimensions, metadata, files, metric shape).
Retrieval QUALITY is never asserted: misses are findings, not test failures —
the printed metrics are the report you compare across runs.

Not part of the lab workflow itself; CI-only convenience.
"""

import glob
import json
import sys

import config as cfg
import main
from store import get_collection

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[ci] {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)


def get_latest_eval() -> dict | None:
    files = sorted(glob.glob(str(cfg.RESULTS_DIR / "eval_*.json")))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as fh:
        return json.load(fh)


print("=" * 78)
print("RAGLab CI integration test — provider:", cfg.EMBEDDING_PROVIDER,
      "| model:", cfg.EMBEDDING_MODEL)
print("=" * 78)

# --- 1. Offline inspect (no API calls) --------------------------------------
print("\n[ci] STEP 1/6 — inspect (load + chunk, offline)")
rc = main.main(["inspect"])
check("inspect exits 0", rc == 0, f"rc={rc}")

# --- 2. Startup report + 3-language sanity check (1 small API call) ---------
print("\n[ci] STEP 2/6 — embedding sanity check (provider/model/dimension + "
      "EN/FR/AR pairwise cosine)")
embedder = main.make_embedder(skip_sanity=False)
expected_dim = int(getattr(cfg, "GEMINI_OUTPUT_DIMENSIONALITY", 0) or 0) or None
check("vector dimension matches config",
      expected_dim is None or embedder._dimension == expected_dim,
      f"got={embedder._dimension} config={expected_dim}")

# --- 3. Ingest (real embeddings, --reset) ------------------------------------
print("\n[ci] STEP 3/6 — ingest --reset")
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
                   "origin", "ingested_at", "embedding_model", "token_count"}
check("every record carries required metadata",
      bool(metas) and required_fields <= set(metas[0]), str(sorted(metas[0]) if metas else []))
check("metadata embedding model matches config",
      bool(metas) and metas[0].get("embedding_model") == cfg.EMBEDDING_MODEL,
      metas[0].get("embedding_model") if metas else "none")

# --- 4. Query: vector-only + hybrid with language filter ---------------------
print("\n[ci] STEP 4/6 — query (vector + hybrid + lang filter)")
rc = main.main(["query",
                "Quel est le frais mensuel du Compte Courant Atlas ?",
                "--k", "5", "--skip-sanity-check"])
check("vector query exits 0", rc == 0, f"rc={rc}")
rc = main.main(["query",
                "Which Atlas account is meant for independent professionals?",
                "--k", "5", "--hybrid", "--lang", "fr", "--skip-sanity-check"])
check("hybrid query exits 0", rc == 0, f"rc={rc}")

# --- 5. Evaluate -------------------------------------------------------------
print("\n[ci] STEP 5/6 — evaluate (questions.json)")
rc = main.main(["evaluate", "--skip-sanity-check"])
check("evaluate exits 0", rc == 0, f"rc={rc}")

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
    check("per-question records include hits",
          bool(run.get("questions"))
          and all("hits" in q and "category" in q for q in run["questions"]))
    # Printable conclusions for the log.
    overall = metrics["overall"]
    sep = metrics["separation"]
    oos = metrics["out_of_scope"]
    print("\n[ci] CONCLUSION — overall hit rates:"
          f"  hit@1={overall['hit@1']:.3f}  hit@3={overall['hit@3']:.3f}"
          f"  hit@5={overall['hit@5']:.3f}  (n={overall['n']})")
    for cat, d in sorted(metrics["by_category"].items()):
        print(f"[ci]   category {cat:<14} n={d['n']:<3} hit@1={d['hit@1']:.3f} "
              f"hit@3={d['hit@3']:.3f} hit@5={d['hit@5']:.3f}")
    for lang, d in sorted(metrics["by_language"].items()):
        print(f"[ci]   language {lang:<14} n={d['n']:<3} hit@1={d['hit@1']:.3f} "
              f"hit@3={d['hit@3']:.3f} hit@5={d['hit@5']:.3f}")
    print(f"[ci]   separation  correct={sep['mean_correct_score']:.4f} "
          f"best_incorrect={sep['mean_best_incorrect_score']:.4f} "
          f"gap={sep['gap_mean_correct_minus_best_incorrect']:.4f}")
    print(f"[ci]   out-of-scope  max_top1={oos['max_top1_score']:.4f} "
          f"mean_top1={oos['mean_top1_score']:.4f} (n={oos['n']})")
    misses = [q for q in run["questions"]
              if not q["is_out_of_scope"] and q["correct_rank"] is None]
    for q in misses:
        print(f"[ci]   MISS  {q['id']} [{q['language']}/{q['category']}] "
              f"not in top {run['config']['retrieval_top_k']} — "
              f"{q['question'][:70]}")

# --- 6. Answer stub exists (regression guard) --------------------------------
print("\n[ci] STEP 6/6 — answer.py stub")
import answer  # noqa: E402
stub = answer.generate_answer("test", [])
check("answer stub returns placeholder", "[STUB]" in stub)

print("\n" + "=" * 78)
if failures:
    print(f"[ci] CI PIPELINE FAILED: {len(failures)} assertion(s):")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("[ci] CI PIPELINE PASSED — all mechanics verified; metrics above are the report.")
print("=" * 78)
sys.exit(0)
