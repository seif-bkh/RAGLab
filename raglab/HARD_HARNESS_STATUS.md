# Large-harness progress — 6 September 2026

**No frozen dataset and no answer-comparison scores exist yet.** Nothing in this file
is a result; it is the verified state of the pipeline that produces them.

## Verified retained work

- **469 accepted base families / 1,407 language-specific question–reference pairs** of
  the 900 base families the plan enumerates (plus 100 adversarial families derived
  later). Shards 0–3 hold 100 each, shard 4 holds 56, shard 5 holds 8, shard 6 holds 5,
  shards 7–8 are empty. 431 families remain, and each accepted family was re-validated
  against the extracted source text — nothing was counted from a manifest alone.
- **Every one of those 469 families is category `supported`.** The negatives — 200
  out-of-scope families (`hh0651`–`hh0850`) and 50 insufficient-evidence families
  (`hh0851`–`hh0900`) — have not been reached at all, and they are where the answer-only
  agent's refusal behaviour is actually tested. Authoring therefore queues the
  least-progressed shards first (`author_shard_order`), which puts 6–8 ahead of 0–3.
- **0 of 469 accepted families were audited by a different provider than the one that
  drafted them.** They carry a second pass by the same model, which is a useful check but
  is not independent validation. That count is recorded in
  `benchmarks/hard_harness_accepted/manifest.json`, copied into the frozen dataset
  manifest as `audit_independence`, and printed in the report caveats; it is not hidden
  and not presented as provider-independent.
- All four source documents and all 36 PDF pages are accounted for; 230 reference units
  are available. Source-page review is model-assisted, not a human banking/legal
  certification.

## Recovery no longer depends on Actions storage

`benchmarks/hard_harness_accepted/` is the committed per-shard snapshot of accepted
families (469 rows, ~2.3 MB, bookkeeping hashes stripped, `served_model` kept).
`author_shard` recovers from it before it opens a model client, so a run cannot lose
minutes of work to an evicted cache or an expired artifact; a rejected or unverifiable
row is never trusted back in, and recovery issues zero model calls (asserted by test).

## Provider reality, measured on this repo

- The reference audit runs on Google's free tier. It advertises 20 requests/minute for
  `gemini-3.5-flash`, but measured behaviour is far thinner: **four concurrent shards
  were rejected with `RESOURCE_EXHAUSTED … limit: 20` at roughly 4 requests/minute in
  aggregate, while a single worker kept working.** The ceiling is shared per project, so
  fanning out buys nothing; `author_parallelism` is 2 and a shared per-process
  `rate_limit_event_budget` (12 events) stops a job from grinding a hard limit.
- Per the user's decision, providers are mixed: **xKiro Qwen `qwen/qwen3.8-max:free`
  drafts, Google `gemini-3.5-flash` audits/repairs/grades**. Because the auditor is the
  scarce side, authoring is now two-pass (`authoring.audit_mode: drafts_only`): each run
  drafts everything still missing on the fast provider, then promotes as many pending
  drafts as the audit allowance permits. A drafted family waiting for an audit is neither
  lost nor re-drafted, and it is never counted as accepted until an audit approves it.
- A quota event pauses and asks the user; nothing switches model, provider, project or
  billing silently.

## LLM-free retrieval judgement (measured on CI, 6 September)

Verified numbers from run 34005544576, measured on the pinned candidate corpus
(`pinned_corpus_agreement.match: true`, tokenizer `cl100k_base`, 836 chunks, 4 documents,
1,407 questions from 469 accepted families, embedding model
`nvidia/nemotron-3-embed-1b` at 2048 dimensions, 0 embedding API calls because the cache
was restored, and no generative call of any kind):

| | whole span in one top-5 chunk | ≥80% of the span | semantic-only queries (145) | abstain AUC |
|---|---:|---:|---:|---:|
| lexical (BM25) | 2.7% | 9.5% | 0.0% | 0.558 |
| nvidia embeddings | 10.0% | 37.3% | **62.8%** | 0.679 |

- The embedding arm retrieves Arabic evidence for French and English questions at
  essentially the same rate as for Arabic ones (9.8% / 9.8% / 10.2% answer-ready), while the
  lexical arm collapses to 0.2% cross-lingually. On queries sharing no content word with
  their evidence, embeddings find the span 62.8% of the time and lexical finds it never.
  That is the cross-lingual claim this lab cares about, and it needed no answer model.
- **The chunker, not the retriever, is the current ceiling:** only 5 of 230 audited source
  units (2.2%) fit inside a single 220-token chunk, and 217 are split or absent. Handing
  the answer prompt its joined top-5 lifts full-span availability only from 10.0% to
  **12.2%** (lexical 2.7% → 3.2%), because a 450-character span needs *adjacent* chunks,
  not merely five retrieved ones. That 12.2% is the honest ceiling on an answer model given
  this corpus, and no answer-side measurement can exceed it for the strict requirement of
  quoting the span; 37.3% of queries at least get most of the evidence.
- A similarity threshold is not yet a good abstention rule on this corpus: the best
  threshold that wrongly rejects at most 5% of answerable queries catches only 24% of
  queries whose supporting document was deleted (AUC 0.68).
- The two rankers agree on the top chunk for only 13.7% of queries, and the embedding arm
  alone found evidence the lexical arm missed on 61 queries versus 8 the other way.

`--chunk-tokens` / `--chunk-overlap` grade a chunking change against the same labels with
zero model calls (locally, 420-token chunks moved lexical answer-readiness 17.4% → 21.8%).
Run 34006036092 reports the pinned 220-token corpus beside 420 and 640, so a chunking
decision can be made on measurements rather than intuition; the pinned setting is untouched
by that job and a re-pin stays a separate, explicit change.

## LLM-free retrieval judgement (available now)

`python3 hard_harness_main.py judge --arms lexical,vector` scores retrieval without any
generative model: the label is exact source-span containment inside a retrieved chunk,
taken from the accepted snapshot families, so no answer key is read and no model judges
anything. It reports answer-readiness at top-k, recall@1/3/k, MRR, nDCG, the strict
ceiling an answer model is bounded by, a separate slice for queries sharing no content
word with their evidence, `partial_only_rate` as a chunking diagnostic, an abstention AUC
built by deleting the supporting document from the searchable set, a threshold chosen with
a bounded false-rejection rate, a family-clustered bootstrap interval, and the agreement
between the lexical and embedding arms (the anti-circularity check). `.github/workflows/
retrieval-judge.yml` runs it on every chunking, corpus or snapshot change.

The lexical arm measured locally on 469 families / 1,407 questions: answer-ready 17.4%
overall, 46.3% for Arabic and ~3% for French/English, with 455 questions sharing no
content word with their evidence. Those numbers are what a cross-lingual embedding has to
improve on, and the report states plainly that this says nothing about whether answers
would be correct, faithful or properly refused.

## Scaled version first

Per the user's decision, the first pass is a **scaled dataset version: 475 paired
scenarios per language (1,425 records)** at the same 65/20/10/5 proportions as the
1,000-per-language target, frozen separately, scored, and then extended. Counts come
from the plan everywhere: `dataset_scale` derives `counts_per_language`,
`questions_per_language` and the answer-shard count, `select_accepted` takes the lowest
accepted family ID per category (so a later run extends the same version), and the
compile step refuses to freeze a version it cannot fill rather than padding with
duplicates or placeholders. `full_target_questions_per_language: 1000` stays in the plan.

## To reproduce / continue

```bash
cd raglab
python3 hard_harness_main.py collect --sha <head_sha> --destination /tmp/collectN   # via the Checks API
python3 hard_harness_main.py snapshot          # accepted shard output -> committed snapshot
python3 hard_harness_main.py author --shard N  # drafts, then audits what the allowance allows
python3 -m unittest test_hard_harness test_nvidia_pipeline   # 120 tests
```

`gh run download` and `gh run view --log` fail from this environment (Azure blob `EOF`);
`collect` reads the check runs each job publishes instead.

## Scaled retry: 100 families per language, no model in the loop

The judge now takes `--limit-per-language N` (CI default 100, env
`HARD_HARNESS_JUDGE_LIMIT_PER_LANGUAGE`, `0` for the whole pool). One family carries one
question in each of Arabic, English and French, so 100 families is 100 questions per language,
300 queries in total.

The sample is chosen deterministically, with no random seed:

1. families whose gold quote does not exist verbatim in the runtime corpus are ranked last, and
   with 242 of 469 findable the 100 picked are all findable — a miss in this run is a ranking
   miss, not the transcription defect;
2. within that pool the pick round-robins over `(subtype, question_style)`, so the 100 are not
   100 formally-worded definitions: 15-20 per subtype, 19-30 per question style.

`manifest['sample']` records the pool, how many were findable, and the per-bucket counts;
`REPORT.md` states the scope and warns that full-pool numbers are not comparable to sampled ones.

The label-match test was then decoupled from chunking (findability is judged against the
document as `loader.load_all` returns it, not against joined chunk text, whose repeated overlap
words made long spans look absent), so the three chunk sizes below now sample from one identical
question set; the first sweep had 179/222/287 findable per chunking, which is why the 220 row of
an earlier collection is not directly comparable.

CI numbers for this retry (`cl100k_base`, run `2cff1bc`, top_k=5, 300 queries, every label
findable; whole = full gold span inside one retrieved chunk, joined = present in the assembled
top-5, sem = the 179-189 queries sharing no content word with their evidence):

| chunks | arm | whole | joined | no evidence | sem recall | abstain AUC | recall@1 | MRR | ar / fr / en whole |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 220 (pinned) | lexical | 5.7% | 6.3% | 29.0% | 0.0% | 0.566 | 2.7% | 0.046 | 17 / 0 / 0 |
| 220 (pinned) | vector | 23.3% | 26.7% | 29.0% | 52.2% | 0.725 | 10.0% | 0.161 | 24 / 23 / 23 |
| 420 | lexical | 26.7% | 26.7% | 1.0% | 0.0% | 0.601 | 14.7% | 0.211 | 72 / 6 / 2 |
| 420 | **vector** | **87.3%** | 88.0% | 1.0% | **91.1%** | **0.840** | **70.7%** | 0.779 | 88 / 87 / 87 |
| 640 | lexical | 32.3% | 32.3% | 0.0% | 0.0% | 0.592 | 21.7% | 0.281 | 85 / 7 / 5 |
| 640 | **vector** | **95.7%** | 95.7% | 0.0% | 94.7% | 0.809 | 79.3% | 0.871 | 96 / 95 / 96 |

Against the full pool the same arm measured 10.0% / 42.4% / 56.9% whole-span at 220 / 420 / 640,
so roughly half of what looked like retrieval failure was the label defect and the rest was
chunk size. The embedding arm is at cross-lingual parity on the clean sample (87-88% in all three
languages at 420) where BM25 is 72% Arabic and 2-6% elsewhere, because the corpus is Arabic.

Measured locally (lexical arm, fallback token estimator — CI with `cl100k_base` is authoritative,
which is why the same sample is judged in CI at 220/420/640 chunks): answer-ready 32.3% at
pinned chunking, 35.0% at 420, 35.7% at 640, versus 17.4% overall on the unsampled full pool —
the gap is the label defect removed, not a better ranker. Arabic questions in that sample reach
0.94 recall@5 with the evidence at median rank 1, because the corpus is Arabic and BM25 shares
words with Arabic queries only; the English and French variants of the same families are the
embedding arm's job, and that is the number worth reading out of CI.

### Corrected sweep (findability judged on the document)

Judge run `34024960494` failed on a `dict`/attribute bug in that change and was fixed in the
following commit; the re-run of the same matrix is the authoritative 100-per-language number.
The label pool is now chunk-independent — **313 of 469 accepted families (66.7%)** have gold text
that exists in `docs/` as the runtime loader reads it, at every chunk size, so the three rows
below score one identical 100x3 question set instead of 179 / 222 / 287 families. Locally
(fallback estimator) the balanced sample reads 24.7% / 31.3% / 32.7% whole-span at 220 / 420 / 640
with the lexical arm; CI with `cl100k_base` and both arms decides the chunking question.
