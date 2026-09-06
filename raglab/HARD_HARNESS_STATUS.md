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
