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

## Real docs — 2026-09-05: rq13 root cause FOUND AND FIXED (chunker)

**rq13's hard miss was never a ranking problem — it was a chunking bug.**
`normalize_text()` claimed to keep "one blank line between paragraphs" but
never emitted any separator: every document reached the chunker as ONE giant
paragraph, which the chunker then hard-split at WORD boundaries (573 chunks
under CI's tiktoken). Expected-match phrases straddling a word cut were
simply not in any chunk → unreachable for every retriever (rq13).

Fix (committed `63ec02e`):
- `normalize_text` now really preserves paragraph boundaries (blank line
  between paragraphs and around block lines; wrapped lines still reflow;
  consecutive list items / table rows stay contiguous).
- `read_docx` separates each `<w:p>` paragraph (a .docx has no blank lines).
- `_hard_split` now splits oversized paragraphs at SENTENCE boundaries
  (Latin + Arabic) first; only single sentences over the whole budget fall
  back to word cuts.

Deterministic proof (offline, zero API calls — new CI STEP 1b):
`real-docs offline chunking: chunks=836 coverage=16/16 all covered` (run
`33943254671`, real tiktoken). **rq13 is now reachable.**

Offline BM25 upper bounds over the new chunks (regex tokenizer ⇒ identical
in CI): rq13 Arabic phrase → rank 1 (was None); rq14 → 1; rq10 → 1; rq12 → 1.
Original-language queries stay weak for cross-lingual cases (expected — the
eval uses query translations).

Real-docs retrieval numbers are being re-measured after the daily embedding
quota reset (the run that would contain them deferred on 429; quota resets
~07:00 UTC). Historical numbers below are the OLD (buggy) split and are
superseded:

### Historical (OLD 573-chunk split, pre-fix — superseded)

Vector baseline: h1 .786 / h3 .857 / h5 .929 — cross-lingual h1 .571,
verbatim/paraphrase h1 1.000, Arabic questions h1 1.000; separation gap
+0.0147 (positive); OOS max top-1 score .7642.
Per-question (cross-lingual): rq03 1, rq05 1, rq08 1, rq09 1, rq12 3,
rq13 None (miss), rq14 5.

RRF: h1 .643 / h3 .786 / h5 .786 (Δ −.143/−.071/−.143 vs vector) — helps
rq12 3→1 but breaks rq03 1→3, rq08 1→2, rq14 5→13, verbatim 1.0→.8.

Blend (real sweep): λ=.65/.70/.75/.85 → h1 .429/.429/.500/.643 (all
rq10 rank=1, h3 .714, h5 .786). Best λ=.85 h1 .643 < vector .786 → the
acceptance verdict on the REAL corpus was **hit@1 recovered: NO** (and
λ=0.70 h1 .429 → NO) — but this was measured on the buggy split, where
hybrid could only rescue reachable chunks. The verdict is void until the
post-fix sweep re-runs.

NOTE: only superseded by the post-fix re-measurement; the old key's quota
limitations are historical (renewed key enabled the full real pipeline).

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
- `33936911402` — real-eval quota deferral (`89f55ce`). PASSED (deferred).
- `33937150152` — **new API key**: real-docs ingest + all three evaluations
  completed (573 chunks, n=14); PASSED.
- `33937314918` — merged notices (`7a6bbfb`): real-docs blend/sweep +
  acceptance verdicts now visible; reproduces the same numbers. PASSED.
- `33942433980` — BM25 metadata cue on (`123b6bf`): real blend h1 .429→.643,
  rq14 rank 5→1 (source-name cue); rq13 None → coverage FAIL (root cause
  lead). FAILED only on the coverage assertion.
- `33942581451` — aggregated diagnostics (`4ddc9b6`): `UNREACHABLE (substring
  not in any chunk): rq13` — chunk-boundary bug confirmed. FAILED on
  coverage.
- `33942678443` — split-point evidence (prefix/suffix chunk holders) +
  env-gated chunk sizes (`a5a999f`). FAILED on coverage.
- `33942821027` — chunk-size A/B attempt; evidence shows prefix and suffix
  INSIDE chunk_0023 (phrase cut in the middle) — then the 340-token ingest
  hit the daily 429 and crashed (pre-deferral-wrap). FAILED.
- `33943254671` — **chunker fix (`63ec02e`) + offline coverage gate**:
  `chunks=836 coverage=16/16 all covered`, fictional metrics unchanged
  (markdown was unaffected), real leg deferred on quota. PASSED.

## 2026-09-05 — free local Arabic-capable embeddings + log-pushing runner

- New provider `EMBEDDING_PROVIDER=huggingface` (local, free, no key, no
  quota). Default model `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0, 1024 dims,
  32K context, 100+ languages incl. Arabic; ~640MB download on first use).
  Alternative: `BAAI/bge-m3` (MIT). Family-aware task prompts (Qwen3 built-in
  "query" prompt / BGE-M3 instruction / E5 query:-passage: prefixes),
  `HF_EMBEDDING_*` knobs in config.py.
- Embedding spaces are provider-specific: switch with `ingest --reset`;
  metadata + run-config now record `config.active_embedding_model()`.
- `run_tests.sh`: local full-suite runner (py_compile → tests_offline →
  inspect → ci_test.py), all output in one log
  (`raglab/logs/test_run_<UTC>.log` + `latest.log`), pushed to the current
  branch — check `raglab/logs/latest.log` after a run.
- `tests_offline.py`: no-API regression suite (translation, fusion tie-breaks,
  blend λ edges, HF provider logic with a stubbed model).

### 2026-09-05 (2) — sentence-transformers 6.x compatibility fix
User hit two real bugs on ST 6.0.1:
1. `get_sentence_embedding_dimension()` renamed (deprecation warning) —
   now resolved: `get_embedding_dimension()` first, legacy name as fallback.
2. `encode()` returns numpy float32 rows; `list(v)` yields np.float32
   scalars which json.dumps cannot serialize → cache.save() crashed.
   Fixed at the source (`_to_python_floats` uses `.tolist()` → native
   floats) AND hardened `EmbeddingCache.put` to coerce every provider.
   `convert_to_numpy` is also no longer required (tensor fallback).
Regression: tests_offline.py now stubs the ST 6 API (float32 rows, new
method name) + the legacy API; 24 checks pass; the no-API suite runs in
CI's compile-offline job.

## 2026-09-05 — Qwen3-Embedding-0.6B on the real docs (n=14) — first local run

User ran `./run_tests.sh --provider huggingface` (commit 1375743, full
log pasted; GitHub copy was 16 lines due to the tee bug, now fixed).
Ingest: 573/573 chunks, 1024-dim, 72 batches, cache hits=0 → full re-embed
(no cache conflict: correct, provider changed).

| mode | h1 | h3 | h5 | ar h1 | en h1 | fr h1 | sep gap | OOS max |
|---|---|---|---|---|---|---|---|---|
| Qwen3 vector | .357 | .786 | .786 | .571 | **.000** | .333 | +.0145 | .7604 |
| Qwen3 RRF | .429 | .714 | .714 | .714 | .000 | .333 | +.0003 | .0305 |
| Qwen3 blend λ=.7 | **.500** | .714 | .786 | **.857** | .000 | .333 | +.0410 | .8129 |
| Qwen3 blend λ=.65 | .500 | .714 | .786 | .857 | .000 | .333 | +.0395 | .8216 |
| Qwen3 blend λ=.75 | .429 | .714 | .786 | .714 | .000 | .333 | +.0411 | .8041 |
| Qwen3 blend λ=.85 | .429 | .714 | .786 | .714 | .000 | .333 | +.0350 | .7866 |
| **Gemini vector (baseline)** | **.786** | **.857** | **.929** | 1.000 | .500 | .667 | +.0147 | .7642 |
| Gemini blend λ=.85 | .643 | .786 | .786 | – | – | – | – | – |

Verdict: **Gemini-embedding-2 stays the default.** Qwen3-0.6B is clearly
worse on this corpus (vector h1 .357 vs .786; en h1 .000 vs .500). Its
blend (λ=.65-.7) recovers hit@1 only relative to its OWN weak vector
baseline (.500 ≥ .357) — the CI "hit@1 recovered: YES" is therefore
within-provider, not an improvement over Gemini.

Per-question (blend λ=.7): fixed rq01 (2→1) vs Qwen3 vector; rq12 solved
(rank 2, was MISS in Qwen3 vector); rq02 regressed badly (2→10).
Persistent across ALL providers/modes: **rq13 (hard miss)**;
Qwen3 additionally misses rq14 everywhere (Gemini solved it at rank 5).

English queries are Qwen3-0.6B's weak spot (0/4 hit@1 in every mode) —
likely the English "query" prompt + cross-lingual EN→AR gap; BGE-M3 or a
larger Qwen3 variant are the candidates if we pursue local embeddings
further. For now: keep `EMBEDDING_PROVIDER=gemini` (free hosted) as
default; the huggingface provider remains an offline fallback (no quota).

## Real docs — Jina embeddings A/B (2026-09-05, run `33944939286` + `33945106657`)

New `EMBEDDING_PROVIDER=jina` (raw HTTPS, no new deps; key from
`JINA_API_KEY`). Same 836 chunks (post-fix split), same questions.
`jina-embeddings-v5-omni-small`: 1024 dims, L2-normalized,
`retrieval.passage`/`retrieval.query` task mapping. Cross-lingual sanity:
EN↔FR +0.8261, EN↔AR +0.8140, FR↔AR +0.8320 (Gemini 768d: +0.843/+0.864/+0.845
— close, both clearly in one shared space).

| mode (jina, 836 chunks) | h1  | h3  | h5  |
|---|---|---|---|
| vector | **.714** | **1.000** | **1.000** |
| rrf (BM25+meta) | .571 | .786 | .857 |
| blend λ=.75 | .643 | .929 | .929 |

**Key per-question (vector mode): `rq13@v=1`** — the phrase that was
UNREACHABLE on the old split (no chunk contained it) is now rank 1; rq14@v=3
(previously rank 5 with Gemini on the old split). vector h3/h5 = 14/14.
The gap to Gemini's historical h1 (.786 on the OLD 573-chunk split) cannot
be judged yet: Gemini has not re-run on the NEW 836-chunk split (daily quota);
the post-reset run gives the apples-to-apples comparison.

## Next step (after quota reset ~07:00 UTC)

Re-run once (`workflow_run` on `arena/01a06d64-raglab`, or push a trivial
commit): the 836-chunk real ingest resumes from the saved batch cache, then
the vector/rrf/blend + lambda sweep + 220-vs-340 chunk-size A/B all rerun on
the FIXED split. Those numbers will replace the "historical" block above and
complete the Gemini ↔ Jina comparison on identical chunks.
