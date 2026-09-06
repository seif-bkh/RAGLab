# Large-harness progress — 6 September 2026

**The final 3,000-question dataset and answer comparison are not complete yet.**

## Verified retained work

- The latest collected checkpoint, run **34001048769**, contains **468 audited
  base families / 1,404 language-specific question/reference pairs** (of the 900
  base families plus 100 derived adversarial families that the plan requires).
  Shards 0–3 hold 100 each, shard 4 holds 56, shard 5 holds 8, shard 6 holds 4;
  shards 7–8 are still empty. 432 base families remain.
- All four source documents and all 36 PDF pages are accounted for; 230 reference
  units are available. Source-page review is assistant/model-assisted, not a
  human banking/legal expert certification. Runtime extraction is unchanged.
- Reference authoring stopped on a Google HTTP 429, not on a Qwen/xKiro event.
  The free tier advertises 20 requests per minute for `gemini-3.5-flash`, yet the
  pause arrived while the shard was issuing roughly 4 requests per minute, so the
  usable project budget is thinner than the documented figure. Every completed
  source/author response stays checkpointed and is reused, never regenerated.

## What changed to resume it

- `hard_harness/google_client.py` now distinguishes a *rate* limit from a *quota*
  limit. A 429 that states a bounded wait ("Please retry in 14.15s") is paced: the
  request sleeps for the advertised time, the process-wide interval shared by the
  author and auditor clients doubles, and the request is retried up to
  `quota_retry_attempts` times. An unlabelled 429, a wait longer than
  `max_retry_delay_seconds`, or a repeated refusal still raises, which pauses the
  fleet and asks the user. No model, project or billing state is changed silently.
- A lone transport/5xx failure no longer stops a shard. Only a streak of
  `transport_failure_streak` consecutive failures does, so one 503 "high demand"
  blip cannot discard minutes of work while an outage still halts the run.
- Every shard step gets `HARNESS_DEADLINE_MINUTES` from the workflow (45 for
  authoring, 32 for answering, 105 for grading). A shard that reaches its deadline
  publishes `status: partial_deadline`, keeps all accepted families, and resumes in
  the next run instead of being killed at the job timeout — and a deadline stop is
  not written as a `paused` signal, so it never gates the other shards.
- `author_parallelism` now comes from the plan (4). Each worker is latency-bound at
  roughly 1.5–2 calls per minute, so four workers stay near 6–8 requests per minute
  in aggregate, and the adaptive floor absorbs the surplus if Google disagrees.

## Resume

The unfinished reference author/auditor profile uses **gemini-3.5-flash**, whose
official model pricing lists a free tier, with the plan's `pacing` block governing
intervals and quota waits. The user confirmed that the Google credential is on a
free-tier project. Existing accepted Qwen and Gemini 3.1 Flash-Lite records keep
their provenance; they are not regenerated.

Candidate answering remains **xKiro `qwen/qwen3.8-max:free`** with native
`nvidia/nemotron-3-embed-1b` retrieval. After authoring, the compiler must repair
or flag exact duplicates and ambiguous negatives before freezing the separate
question and answer-key files. Only then may the 3,000 candidate answers and the
comparison run.

Progress is recoverable without Actions artifact access
(`.blob.core.windows.net` is unreachable from this sandbox):

```bash
cd raglab
python3 hard_harness_main.py collect --sha <run head sha> --destination /tmp/collect
```
