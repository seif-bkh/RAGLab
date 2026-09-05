# Paired 3,000-question harness

**Target, not a claim of completed data:** 1,000 scenarios in Arabic, French and
English (3,000 records). Each language has 650 supported, 200 out-of-scope,
100 adversarial and 50 insufficient-information cases. Actual completion is
recorded in phase manifests; missing cases/scores are never fabricated.

The user authorized continuation through all phases, checkpointing, and a
possible Google fallback after a prompt on quota exhaustion. The first profile
is xKiro Qwen with **XKIRO_API_KEY_JINKO**. The ordinary Qwen/Nemotron application
configuration is not changed by harness-only provider switches.

## Phases

1. `sources`: render the original PDF pages, transcribe and audit them, retain
   logical DOCX text, and keep the agent's existing runtime corpus separate.
   Visually reviewed source overrides are hash-bound and explicitly labeled as
   assistant reviews, not banking/legal expert certification. Unresolved pages
   block release; PDF digit/order corruption is not turned into an answer key.
2. `author --shard N`: nine resumable shards each draft/audit 100 paired base
   families. Definitions, conditions, exceptions, boundary reasoning and false
   premises vary. Every source-backed key has contiguous original-source quotes.
3. `compile-dataset`: audit negative cases against the corpus, add 100 explicitly
   grouped adversarial variants, check duplicates/counts/ID bijections, then freeze
   separate `public/questions.{ar,fr,en}.jsonl` and
   `references/answer_key.{ar,fr,en}.jsonl` files with hashes.
4. `retrieve`: native Nemotron embeds the unchanged runtime corpus and public
   questions. This phase does not download/open answer keys or use oracle context.
5. `predict --shard N`: 30 balanced 100-question shards run normal retrieved-context
   Qwen answers. Each shard has roughly equal language counts. Completed model
   outputs—including invalid ones—are checkpointed, not selectively retried to
   make scores improve. Quota/transport failures remain attempts, not answers.
6. `grade`: the grader alone receives the frozen reference artifact. Deterministic
   checks separate local refusals, provider failures, invalid output and released
   injection markers. A calibrated semantic judge accepts faithful paraphrases,
   checks grounding/required facts/language, and flags questionable references.
   Per-case judgments, per-language/provider/category coverage and paired-family
   uncertainty are saved. Same-model authors/judges are not independent validation.

Workflow: **Large multilingual answer harness**, controlled by
`../benchmarks/hard_harness_plan.json`. Phases are `sources`, `author`, `evaluate`.
The latter includes retrieval, all candidate shards and grading. Source/author
review gates must pass before the plan advances. `sources_run` and `dataset_run`
identify the exact earlier artifacts; ordinary code changes do not spend API quota.

## Persistence and boundaries

- Atomic request caches under `.cache/hard_harness/` scope provider, model, role,
  exact input and output limit. Credentials are **not** in cache keys/content.
  Credential aliases and original response provenance are retained on replay.
- Candidate results are checkpointed per case. A malformed but completed response
  is a measured failure; it is not rerun until it happens to become valid.
- Shard-specific Actions cache keys avoid one worker replacing another worker's
  checkpoint. At most two shards run concurrently. A shared pause artifact prevents queued
  shards from making model calls after a quota event; in-flight shards preserve
  their own checkpoints rather than being forcibly cancelled.
- Every completed/paused job saves artifacts and neutral, byte-bounded Checks.
  `hard_harness_main.py collect --sha SHA` verifies and reconstructs those files
  when redirected artifact downloads are unavailable in the sandbox.
- Public question and private answer-key artifacts are physically separate. Answer
  workers download only public inputs/retrieval; the model never gets expected
  answers, labels or reference rubrics.
- Large vector caches stay out of Git. Dataset/reference files and compact reports
  can be committed after validation. Artifacts/cache retention is finite, so retain
  final question/key files independently for long-term reproducibility.
- Google support is harness-only and inactive by default. Activation requires a
  versioned provider/model/key profile and confirmation that the project uses the
  free tier. No billing settings are changed, no credentials are sent to another
  provider, and mixed results are never called a Qwen-only score.

## Commands

```bash
pip install -r requirements-harness.txt
python hard_harness_main.py sources
python hard_harness_main.py author --shard 0
python hard_harness_main.py compile-dataset
python hard_harness_main.py retrieve
python hard_harness_main.py predict --shard 0
python hard_harness_main.py grade
```

The commands require the preceding phase's validated files in
`results/hard_harness/`. They are orchestration-free Python; provider calls and
intermediate artifacts are inspectable. The workflow supplies secrets from the
selected alias without exposing their values.

This harness strengthens evidence for an **answer-only** agent. It does not prove
that 3,000 language variants are independent facts, that model-authored keys are
expert ground truth, or that a free gateway meets a production SLA.
