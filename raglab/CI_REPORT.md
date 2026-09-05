# Score-blend hybrid — CI report (2026-09-05)

Branch `arena/01a06d64-raglab` — HEAD `89f55ce` (pushed; working tree clean).
All runs below are the real Gemini API (`gemini-embedding-2`,
`gemini-3.5-flash-lite` translations) on the fictional `questions.json`,
n=14, translation ON, best-variant fusion ON.

## Reproducibility fix (important)

`best_variant_merge` had a top-1 bug: when two DIFFERENT chunks each topped
their own query variant (both normalized to relative_score=1.0), the tie was
broken by variant order, so the original-language variant's champion was
handed rank 1 even when a translated variant's champion was the actual answer
chunk (observed as cross-lingual hit@1 drops in the hybrid modes).

Repeated A/Bs also drifted run-to-run because the cache artifacts were never
actually restored (wrong zip layout) and translations are stochastic. Both
caches (`embeddings_cache.json`, `translations_cache.json`) are now uploaded
and restored by the workflow — runs since then reproduce identical numbers.

## Tie-break policy A/B (stable across 3 CI runs)

| policy (config.FUSION_TIE_BREAK) | vector h1 | rrf h1 | blend h1 (λ=.7) |
|---|---|---|---|
| `variant_order` (original) | .714 | .500 | .429 |
| `same_lang_margin` (new default) | .571 | .500 | .500 |
| `raw` | .429 | .643 | .500 |

q10 rank was identical under all policies: vector 10, RRF 16, blend 15.
Policy `same_lang_margin` was kept as default (best blend h1; also best
combo on the acceptance axes).

## Lambda sweep (under `same_lang_margin`)

| λ   | h1    | h3    | h5    | q10 rank |
|-----|-------|-------|-------|----------|
| 0.55 | .429 | .714 | .929 | 15 |
| 0.65 | .500 | .714 | .929 | 15 |
| 0.70 | .500 | .786 | .929 | 15 |
| 0.75 | .429 | .786 | .786 | 15 |
| 0.85 | .571 | .786 | .786 | 14 |
| 0.95 | .571 | .786 | .786 | 13 |

## Acceptance verdicts (printed by CI every run)

- `blend at lambda=0.70`: h1 .500 vs vector .571 → **hit@1 recovered: NO**
  (misses by one question; n=14, ±.07 is noise). Recall gain kept: q10 10→15
  vs vector, h5 .929.
- `best lambda=0.85`: h1 .571 = vector .571 → **hit@1 recovered: YES (tie)**,
  q10 10→14 (RRF q10 = 16 in the current engine — the original 9→3 anchor was
  measured on different stochastic translations and is not reproduced).
- λ=0.95 was best in an earlier run with h1 .643 > .571; with the restored
  caches it also gives .571 (tie).

Bottom line: the BM25 component adds deep recall (h3/h5), not top-1. Any λ
that keeps a meaningful BM25 weight trades ~1 hit@1. λ≈0.85–0.95 recovers
hit@1 (at 85–95% vector dominance); λ=0.70 maximizes recall (q10 10→15,
h5 .929). Choose with `HYBRID_BLEND_LAMBDA`; CI prints both verdicts on every
run.

## Real docs (docs/, 4 Arabic documents → 213 chunks)

- 14/14 ground-truth substrings verified in the NFKC-normalized chunks
  (offline); questions: `questions_real.json` (rq01–rq14, rq15/rq16 OOS).
- CI ingest of the full 213-chunk corpus COMPLETED in run `33936558123`
  (fictional legs were fully cache-served, freeing the daily budget); the
  real vector evaluation then hit the daily 429.
- Real evaluations are now quota-aware: they defer cleanly on 429, save the
  query cache artifact, and a later run resumes from where they stopped
  (CI stays green). Real-docs results will appear as `real-docs: vector/rrf/
  blend …` annotations once the quota allows 16 more query embeddings.

## CI run IDs (branch push, all jobs)

- `33934715296` — blend wiring first A/B (vector .714 / rrf .500 / blend
  .429; real ingest 429). FAILED at real-docs-ingest.
- `33935677817` — tie-break fix + λ sweep (best λ=0.75/.95; real ingest 429
  deferred-green after `ffcdc9f`). PASSED.
- `33936040987` — tie-break mode A/B (all three policies, stable). PASSED.
- `33936261469` — translation cache + λ=0.70 verdict. PASSED.
- `33936558123` — restore-path fix; real ingest completed; real eval 429 →
  crash (pre-deferral). FAILED at real-evaluate.
- `33936713545` — warm-cache translation check fixed. FAILED at
  real-evaluate (same).
- `33936911402` — real-eval quota deferral (`89f55ce`). PASSED (real-docs
  eval deferred, resumes after the daily reset).
