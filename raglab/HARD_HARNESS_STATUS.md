# Large-harness progress — 5 September 2026

**Not a completed 3,000-question result.** The agreed target is 1,000 paired
scenarios × Arabic/French/English, with separate anticipated answers and source
references. The full run is authorized, with checkpoints and a prompt before
switching on quota exhaustion.

## Current state

- All four source documents were extracted for inspection. Original PDF page
  images confirmed serious text-layer digit/order corruption, so references are
  not built blindly from the candidate's noisy extraction.
- The fresh `XKIRO_API_KEY_JINKO` credential successfully performed source
  transcription/audit calls. No key value was exported.
- Two source-audit passes exposed inconsistent/overly typographic judgments and
  malformed output. Their completed responses were checkpointed rather than
  discarded. Seven original pages were visually compared by the Arena assistant;
  corrections are hash-bound in `benchmarks/hard_source_reviews.json`. This is
  **not** a human banking/legal expert review.
- A third audit run, **33983351756**, finished with failure status after saving
  its cache, report and artifacts. Its final source findings have **not yet been
  retrieved**: GitHub access returned **HTTP 401**. Do not infer a model quota
  problem from this GitHub authentication error.
- Public question files, answer-key files and 3,000 predictions are **not yet
  released**. Source/reference integrity gates must be resolved first.

## Implemented locally

- Original-page references separate from unchanged runtime corpus.
- Nine resumable 100-family authoring shards, multilingual reference audits,
  full-corpus negative-case audit, duplicate/count/ID checks and separate frozen
  question/reference artifacts.
- Native Nemotron retrieval that does not load answer keys.
- Thirty balanced 100-question prediction shards, with per-case and raw-response
  checkpoints. Invalid completed outputs remain measured failures, not retries
  until success.
- Deterministic refusal/error checks plus calibrated, blinded semantic grading
  that accepts faithful paraphrases. Per-language/provider/category coverage and
  paired-family uncertainty reporting.
- Explicit credential/provider provenance. Google fallback adapter prepared but
  inactive; enabling it requires a versioned profile and confirmation of a
  free-tier project. No paid fallback or account/billing changes are automatic.

## Immediate next step

Reconnect GitHub in Arena (no credentials in chat), retrieve the third audit's
checkpoint, resolve any remaining source issues, push the remaining harness
implementation, then advance the versioned plan through authoring and evaluation.
Completed work is retained locally and in Actions checkpoints; it is not necessary
to restart the source/model calls that were already completed.

See `benchmarks/HARD_HARNESS_PLAN.md` and `hard_harness/README.md` for the protocol.
