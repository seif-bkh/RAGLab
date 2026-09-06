# Proposed 3,000-question answer-agent harness

Status: **sources and the authoring protocol are implemented and running; 469 of
the 900 base families are accepted; the 3,000 cases and every score derived from
them are still absent.** This extends the selected Qwen/Nemotron pipeline. Per the
user's decision the *harness construction* side is now provider-mixed (see
"Provider mix"), because one free tier cannot carry ~1,000 audit requests; the
answer model under test stays `qwen/qwen3.8-max:free` and the embeddings stay
`nvidia/nemotron-3-embed-1b`.

## Target

1,000 Arabic + 1,000 French + 1,000 English questions, with separate, frozen
answer keys. Recommended design: 1,000 scenario families rendered in all three
languages, so language comparisons use equivalent facts and difficulty. That is
3,000 question records, **not 3,000 independent facts**.

Proposed allocation **per language**:

| Category | Count | What it tests |
|---|---:|---|
| Supported document QA | 650 | Definitions, procedures, conditions, exceptions, numerical boundaries, multi-evidence answers and correction of false premises |
| Out of scope | 200 | 100 plausible but unsupported banking details, 50 unrelated questions, 50 private/live-data requests |
| Adversarial | 100 | Legitimate questions with query/source injection, role spoofing, false authority and requests to override source rules |
| Ambiguous / insufficient information | 50 | Missing variables, unspecified products/parties/dates and unsupported assumptions; abstain or request clarification |
| **Total** | **1,000** | Identical category totals for each language |

### Scaled version first

Per the user's decision the first frozen dataset is a **scaled version**:
`questions_per_language` and `counts_per_language` come from the plan, so the
initial pass holds 475 paired scenarios per language (1,425 records) at the same
65/20/10/5 proportions, gets scored, and is then extended to the 1,000-per-language
target recorded as `full_target_questions_per_language`. Authoring still enumerates
the identical 900 base specs, so no cache key changes between versions; the compiler
selects the lowest accepted family ID per category and **refuses to freeze a version
it cannot fill** rather than padding or duplicating.

Only a minority of negative cases should be easy local private-account guards.
Those guard passes must be reported separately from genuine model refusals.
Question families, reused facts and attack-template families remain identifiable;
near-duplicate paraphrases cannot be presented as independent coverage.

## Documents inspected

The existing runtime extraction produced:

| Document | Pages / type | Normalized characters | Runtime chunks |
|---|---|---:|---:|
| Circulaire_BCT_2019-08.pdf | 5 PDF pages | 6,891 | 37 |
| Guide_Interne_Operations_Bancaires_Islamiques.docx | DOCX | 33,210 | 191 |
| Loi_2016-48.pdf | 31 PDF pages | 113,484 | 495 |
| Madkhal_Sayrafa_Islamiya.docx | DOCX | 23,177 | 113 |

The chunk manifest is unchanged:
`807785db25fb798499e67f12f59589f6db33792e88cb00cf104752853046e60e`.

**Reference-quality blocker found during inspection:** the PDF text layer is
not a reliable numerical answer key. On the rendered first BCT page, the date is
14 October 2019 and the circular is 08/2019. The runtime extraction instead
contains `9112` and `80`. Other references/article numbers are corrupted, and
Law 2016-48 has column/word-order problems. Changing extraction library did not
solve this. The original page images were inspected to confirm the mismatch.

Therefore:

- Build PDF references against rendered original pages, with transcription audit
  and explicit page anchors—not uncritically against the noisy runtime text.
- Use logical DOCX text/table evidence where trustworthy.
- Preserve the current runtime corpus during the primary test. If the agent fails
  because its extraction is wrong, classify that as an ingestion/source-quality
  failure; do not silently rewrite the corpus or gold answer to hide it.
- Uncertain source readings are flagged/excluded or become explicit insufficient-
  information cases, never confident fabricated legal facts.

Interesting hard boundaries already found in the guide include the 3-year used-
car condition; strict “does not reach 30%” versus “does not exceed 5%” wording;
lessor/lessee maintenance and insurance responsibilities; late penalties not being
bank revenue; early-settlement concessions; document-signing and possession
conditions; and investment profit/loss allocation. These warrant scenario tests,
not just repeated definition questions.

## Files to produce

- `questions.ar.jsonl`, `questions.fr.jsonl`, `questions.en.jsonl`: IDs, language,
  question text and any deliberately untrusted test input. No answer labels.
- `answer_key.ar.jsonl`, `answer_key.fr.jsonl`, `answer_key.en.jsonl`: same IDs,
  expected behavior, anticipated answer, required facts, forbidden claims,
  acceptable variants, numeric conditions and evidence/page references.
- `source_manifest.json`, `reference_audit.jsonl`, `dataset_manifest.json`:
  document hashes, source quality, provenance, exact counts, duplicate checks,
  author/auditor model roles, and frozen file hashes.
- `predictions.*.jsonl`: actual answers, citations, retrieved hits/scores, timings,
  provider errors and model/request provenance.
- `judgments.*.jsonl` and `REPORT.md`: per-case comparisons, failures and aggregate
  scores, with links back to evidence.

The answering process receives only public question/test-input files and normal
retrieved context. It never loads or transmits the answer-key files. Reference
answers must be generated/audited/frozen **before** candidate answers are produced.

## Evaluation

1. Validate source anchors, exact per-language/category counts, ID bijection,
   duplicates, cross-language equivalence and reference quality.
2. Run a stratified pilot before spending the full request budget; freeze the
   schema and thresholds rather than fixing labels in response to model failures.
3. Test the normal question → Nemotron retrieval → Qwen answer path. Do not feed
   oracle evidence to the answerer. Record retrieval/source coverage separately.
4. Check answer structure and citations deterministically. Compare meaning and
   required facts, not exact answer-string identity; allow faithful paraphrases.
5. Separate correct answers, partial answers, unsupported claims/hallucinations,
   wrong-language answers, correct refusals, over-refusals, invalid output and
   provider failures. Untested cases are never scored as passes or zeros.
6. Report by language, document, difficulty, question/fact family and attack type;
   count local guard refusals separately. Use family-clustered uncertainty, not
   confidence intervals that assume all translations are independent.

With the selected-model constraint, Qwen can help author/audit/judge, but that is
**not an independent judge**. Record the roles, blind the judge to model/aggregate
scores, audit disputed and sampled reference cases, and retain an expert-review
flag. Automated self-consistency cannot certify legal correctness.

## Execution and resource policy

The complete exercise needs more than 3,000 model requests once authoring,
reference audits and semantic comparison are included, plus native query
embeddings. It can take hours and may exceed a free daily token quota.

Use resumable, bounded shards; recheck zero pricing, never fall back to paid SKUs,
and stop/defer on quota or persistent provider failures. Preserve completed
records and request counts. Keep large vector caches out of Git; retain the
question/reference files and compact measurements for inspection.

The user confirmed paired scenarios and automatic continuation through all
3,000 after validation. `XKIRO_API_KEY_JINKO` is the first harness credential.
On quota exhaustion, preserve checkpoints and ask before switching. A hard
3,000-case suite will substantially strengthen evidence for an answer-only agent,
but it is still an evaluation—not a production guarantee by itself.

## Provider mix (authoring side, user-authorized)

- `author_llm` — xKiro `qwen/qwen3.8-max:free` drafts each paired family. This is
  the same model the harness measures, so drafting alone is never validation.
- `llm` — Google `gemini-3.5-flash` audits drafts, audits negatives, repairs
  references and grades answers: a provider other than the candidate's.
- Roles resolve through `role_profile(plan, role)`; a profile outside the two
  approved free tiers fails the plan check before any request. Every phase reads
  its counts and models from the plan, never from literals.
- Measured ceiling: four concurrent authoring shards were rejected with
  `RESOURCE_EXHAUSTED … limit: 20` at ~4 requests/minute in aggregate while a
  single worker kept working, because the quota is shared per project. Auditing is
  therefore capped at two workers with a 12-event wait budget, and authoring is
  two-pass (`authoring.audit_mode: drafts_only`): each run drafts everything still
  missing on the fast provider, then promotes as many pending drafts as the audit
  allowance permits. A drafted family waits in a pending file instead of being
  re-drafted or counted as accepted.
- Mixed provenance is labelled, not smoothed over: the frozen manifest carries the
  per-model `provider_mix` and `audit_independence` counts, and today's 469 accepted
  families are all `same_model_audited: 469, cross_provider_audited: 0`, which the
  report prints as a caveat. No number from a mixed run may be described as
  Qwen-only.
