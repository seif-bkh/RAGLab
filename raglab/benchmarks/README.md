# Frozen NVIDIA evaluation protocol

## Data

- `retrieval_dev.json`: the previous session's 16 real-document questions
  (14 answerable, 2 out of scope), augmented with explicit source-file constraints
  and approximate answer-concept rubrics. Used for selection/tuning.
- `retrieval_holdout.json`: authored **before the first NVIDIA retrieval run**.
  Six new fact groups, each in Arabic/French/English (18 answerable questions),
  plus 9 out-of-scope/security questions. Validation only. Language variants are
  correlated: this is **six independent held-out facts, not 18**.
- `translations_v2.json`: 18 authored references covering all six directed language
  pairs, legal identifiers, numbers, negation, products and banking terminology.
  These are not professionally certified translations. The original
  `translations.json` remains frozen: an audit of iteration 02 found that its
  `t1_ar_en`/`t1_ar_fr` demanded literal BCT even though the Arabic source says
  only “the central bank.” Version 2 removes those two invalid literal-copy
  demands and requires the central-bank entity instead. **Source inputs and
  reference strings are unchanged.** `reports/translation_constraint_audit.json`
  regrades the same outputs without new model calls. Other literal-copy failures
  remain failures, even when an expansion/transliteration is semantically plausible.
  No retrieval labels or answer rubrics were changed.
- `run_plan.json`: explicit live-run trigger and iteration identifier. Changing
  ordinary source files does not automatically consume NVIDIA quota.

All source evidence substrings are verified against the actual normalized
four-document corpus, and again against its chunks. Expected substrings, answer
rubrics, source labels and answerability labels are **never** supplied to models.
The original `questions_real.json` remains unchanged for historical comparisons.
Its old measurements did not require an explicit source-file match, so its hit
rates are not identical to this stricter benchmark's definition.

## Procedure

1. Require real `cl100k_base` tokenization. Freeze the 220-token / 40-overlap
   chunks, corpus hashes, model ID, native 2048 dimensions and dependency versions.
2. Embed the corpus once with `nvidia/nemotron-3-embed-1b`; use it for every arm.
3. Compare original-query vector search with each exact translator under
   `basic-v1` and `banking-v2`. Use correct model-specific API/prompt contracts.
4. Choose translator/prompt on development data, then test three predeclared
   local retrieval alternatives: translated-only vector, RRF, and blend lambda .85.
   Selection first requires preservation of BCT's institution identity (added
   from the endpoint probe, not holdout labels), then orders hit@5, hit@3,
   hit@1, MRR@10, with simpler original-only on ties. This guard is recorded
   as protocol `nvidia-v2-entity-guard-riva-pivot`; the first retrieval-only run predates it.
   Riva's supported English-centric pairs are composed for FR↔AR; the two hops
   are disclosed, not misrepresented as direct translation.
5. Freeze the selection **before** evaluating holdout. For comparison, also report
   each other model's development-selected profile on holdout. Do not select a
   new winner from holdout results.
6. Measure independent authored translation-reference similarity (chrF++) and
   deterministic invariants; inspect the actual outputs, especially negation and
   institution names. A plausible-looking translation can still be wrong.
7. With `--stage all`, compare the exact Kimi/DeepSeek answerers on development,
   using `grounded-v1` and `grounded-v2` with adjacent-chunk context. This second
   arm changes both prompt and context; it is a **combined strategy comparison**,
   not evidence that either change alone caused an improvement. Select on
   development refusal rate, answer-rubric rate, and validation rate. Only then
   evaluate the chosen answerer on holdout. `--answer-profiles grounded-v1`
   explicitly limits the comparison to the basic profile for **both** answerers.
   It does not imply that expanded-context profiles were completed. Iteration 03
   uses this bounded scope to finish the core comparison after trial-endpoint
   rate limits interrupted iteration 02. Successful exact-request caches are
   reused; new answer calls are serial, paced at least 30 seconds apart, with
   two bounded attempts and a 60-second default 429 backoff (Retry-After honored).
   This was protocol `nvidia-v3-source-valid-references-serial-answers`.
   **Audit:** that run's answer client actually retained a 30-second retry cap
   because the larger configured cap was not forwarded. The corrected wiring
   and runtime (rather than intended) parameter reporting are now covered by a
   regression test. `nvidia-v4-cache-merge-and-retry-cap` also fixes stale
   translator instances overwriting each other's persisted cache entries; it
   has not yet been live-tested. Old measurements are not rewritten.

An unavailable translator/model is **incomplete**, not a baseline score labeled
with that model's name. No automatic model fallback is allowed. Bounded retries,
request counts, prompt/model identities and cached/live status are recorded.
A provider outage opens a circuit instead of spending the full timeout on every
question. Validation failures are separate from transport failures.

## Interpretation

The thresholds in `nvidia_benchmark.py` were declared before retrieval results.
They are deliberately stricter than "the job exited zero": hit@1 ≥ .85, held-out
hit@3 ≥ .95 and hit@5 = 1; answer rubric ≥ .90, citation validation/refusal = 1.
**Passing this small suite is not production certification.** The answer rubric
is approximate string matching, not a blinded human or entailment evaluation.
Quote membership establishes provenance only, not correctness of interpretation.
Repeated measurement of the same holdout is a regression check, not fresh
independent validation. Do not tune prompts or rewrite labels to make it pass.

Broader production claims require larger independently authored bank/legal
questions, Arabic/French subject-matter review, account-data isolation, adversarial
prompt-injection tests, data-governance approval and production endpoint load tests.
