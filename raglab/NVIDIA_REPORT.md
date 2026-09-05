# NVIDIA RAG evaluation — 5 September 2026

## Bottom line

**The three exact translators have now been measured on the real corpus. The
end-to-end evaluation is not complete, and this is not production-ready.**

- Embeddings: **`nvidia/nemotron-3-embed-1b`, native 2048 dimensions**.
- Development-selected retrieval: **`moonshotai/kimi-k3` / `banking-v2`, vector
  search with original + translated queries (`best`)**. Translation does not
  improve every question and was worse than original-only on held-out top-1.
- Best **completed development** answer profile:
  **`deepseek-ai/deepseek-v4-pro-0813` / `grounded-v1`**, top 5, no neighbor expansion.
  This is the provisional CLI answer default, not a validated production choice.
- Kimi answer evaluation, expanded-context profiles, held-out answers and source
  injection testing were interrupted by HTTP 429 responses. Their missing
  measurements must not be presented as passed tests or as model-quality scores.

A third, serial **basic-profile** comparison is prepared locally. Its push failed
with a GitHub authentication error; an independent `gh api` request also returned
HTTP 401. **Reconnect GitHub in Arena to launch that run. No credentials should
be sent in chat.** No iteration-03 API run has been launched.

## What was actually executed

| Experiment | Evidence | Outcome |
|---|---|---|
| Exact-model endpoint probes | [Probe records](reports/nvidia_model_probes.json) | Nemotron, Kimi and Riva responded; DeepSeek initially timed out |
| Iteration 01: retrieval | [Run 33969545639](https://github.com/seif-bkh/RAGLab/actions/runs/33969545639), [measurements](reports/nvidia_iteration_01.json) | Original/Kimi retrieval measured; DeepSeek timeout, invalid direct Riva pair routing and a reference-test rate limit prevented completion |
| Iteration 02: all stages | [Run 33970405295](https://github.com/seif-bkh/RAGLab/actions/runs/33970405295), [measurements](reports/nvidia_iteration_02.json) | All translators and their selected-prompt reference suites completed; answer/security testing incomplete |
| Report recovery | [Run 33971261399](https://github.com/seif-bkh/RAGLab/actions/runs/33971261399) | Successful; no model calls. The original compact summary was 60,204 bytes, exceeding the publisher's 60,000-byte guard |
| Iteration 03 | [Versioned plan](benchmarks/run_plan.json) | Prepared, **not launched**; GitHub reconnection required |

Iteration 02 ran commit `be5f4448b09cd8905db6eaaf718d215bf4036e11`.
Recovery published neutral measurement Checks, not passing quality certifications.
The saved JSON retains actual translations, rankings, generated claims and evidence
quotes. Unused answer-context bodies were omitted from Checks; full sources and
outputs are in the original workflow artifact.

## Fixed corpus and evaluation boundaries

- All **four real Arabic PDF/DOCX documents in `docs/`**, not just fictional bank
  product sheets.
- **836 chunks**, actual `cl100k_base` tokenizer, 220-token chunks / 40 overlap.
- One native Nemotron embedding space, local persistent ChromaDB/cosine, shared
  by all translation arms. No silent model substitution or reduced dimensions.
- Chunk-manifest SHA-256:
  `807785db25fb798499e67f12f59589f6db33792e88cb00cf104752853046e60e`.
- All expected evidence was reachable. **Zero heading metadata chunks** were
  detected; PDF extraction artifacts and missing structural metadata remain issues.
- Development: 14 answerable + 2 refusal questions. Held out: **six independent
  facts in three languages**, giving 18 answerable variants, plus 9 refusal/security
  variants. These are not 18 independent facts or a representative banking workload.
- Retrieval requires the expected evidence **and source file**. Selection uses
  development data only; held-out labels and answer rubrics were not rewritten.
  Repeated retrieval on this same holdout is a regression check, not fresh validation.

## Retrieval results

Each translator below uses its development-selected prompt. Percentages are
source-constrained hit rates. All arms use the same Nemotron vectors/corpus.

| Split | Translator / prompt | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---|---:|---:|---:|---:|
| Development | Original query only | 71.4% | 92.9% | 100% | .8238 |
| Development | Kimi / banking-v2 | **85.7%** | **100%** | **100%** | **.9167** |
| Development | DeepSeek / basic-v1 | 78.6% | 100% | 100% | .8810 |
| Development | Riva / banking-v2 | 71.4% | 92.9% | 100% | .8274 |
| Holdout | Original query only | **94.4%** | 100% | 100% | **.9722** |
| Holdout | Kimi / banking-v2 | 88.9% | 100% | 100% | .9444 |
| Holdout | DeepSeek / basic-v1 | 88.9% | 100% | 100% | .9444 |
| Holdout | Riva / banking-v2 | 88.9% | 100% | 100% | .9444 |

Both DeepSeek prompts tied on these development metrics. Riva's banking prompt
preserved the development BCT entity where its basic prompt did not. Kimi's
basic prompt reached 78.6% top-1, versus banking-v2's 85.7%.

The predeclared alternatives did not beat Kimi/vector/best on development:
translated-only vector matched its top-1 but had lower MRR; RRF fell to 50.0%
top-1, and the .85 dense/keyword blend to 64.3%. They were not promoted.

**Interpretation:** Kimi is the development-selected configuration, not a
universal winner. Original-only Nemotron is a strong, lower-latency baseline:
`query --no-translation` avoids the translation round trip. In particular, the
French held-out blank-document question dropped from rank 1 to 2 with translation.
No configuration was selected by optimizing that held-out result.

### Riva routing correction

Riva's supported English-centric pair tags are now used correctly. French↔Arabic
is explicitly composed through English using **the same Riva model**. Both hops
are validated; route and intermediate English text are cached and inspectable.
The first run's rejected wrong-script outputs were an adapter failure, not a valid
measurement of Riva's French↔Arabic translation quality.

## Translation-reference results and fixture audit

All three selected-prompt suites completed **18 examples across six directions**.
chrF++ measures similarity to authored references, not semantic correctness.

| Translator / prompt | chrF++ | Original strict constraints | Corrected strict constraints |
|---|---:|---:|---:|
| Kimi / banking-v2 | **79.38** | 16/18 | **18/18** |
| DeepSeek / basic-v1 | 74.85 | 12/18 | 14/18 |
| Riva / banking-v2 | 67.21 | 13/18 | 15/18 |

**The correction is a disclosed fixture fix, not new model output.** Two Arabic
sources said only “the central bank,” but the original fixture required literal
“BCT” in their English/French translations. All three models correctly retained
“central bank” without inventing that acronym. `translations_v2.json` removes
those two impossible literal-copy demands and checks the central-bank entity
instead. Inputs, authored reference strings, outputs and chrF++ are unchanged.
The [audit record](reports/translation_constraint_audit.json) regrades the saved
outputs with **zero new API calls**; the original fixture/results remain intact.

Remaining failures include literal Atlas/TND/BCT expansion or transliteration,
which violates this experiment's identifier-preservation policy but is not always
a semantic error. There is also a real omission: Riva's French→Arabic reference
translation dropped the BCT institution entirely. DeepSeek added “card” to one
Atlas fee sentence although the source did not specify a card. Expert review is
still needed; a strict-constraint pass is not translation certification.

## Grounded-answer results

These results use the frozen Kimi-selected retrieval profile.

| Answer profile | Measured outcome |
|---|---|
| Kimi / grounded-v1 | 8/16 development cases attempted: 5 valid, rubric-passing answers; 3 HTTP 429 failures; remaining cases untested |
| DeepSeek / grounded-v1 | **16/16 development cases completed**: 12/14 strict answer-rubric passes; 2/2 correct system refusals; 13/13 rendered answers had valid citations/quotes |
| DeepSeek / grounded-v2 + neighbors | 4/16 attempted: 1 valid answer, 3 HTTP 429 failures; not a completed quality comparison |
| Kimi / grounded-v2 + neighbors | Not reached after its provider circuit opened |
| Selected DeepSeek held-out answers | First 3 attempts returned HTTP 429; **no usable held-out answer-quality result** |
| Three untrusted-source injection tests | Provider errors; **not evaluated**, not evidence that attacks succeeded or were resisted |

DeepSeek's two development refusals include one model refusal on an unsupported
question and one local pre-API guard for private/live account information.
Mean successful uncached client-call durations were 31.46 s for DeepSeek/v1
(n=15) and 45.29 s for Kimi/v1 (n=5). These include client pacing/retries; they are
small observations, **not comparable production SLAs or a load test**.

Manual inspection of the two DeepSeek rubric failures matters:

- `rq05`: the answer “cost plus a pre-determined profit margin” is supported by
  its quoted BCT circular. It fails the benchmark's internal-guide source
  constraint and a narrow wording group. This is not evidence of a fabricated
  profit-margin answer. The frozen score remains 12/14; this note is not a
  replacement blinded evaluation.
- `rq08`: DeepSeek refused the French Salam question despite relevant internal-
  guide evidence being retrieved. This is a genuine false refusal to investigate.

Citation membership proves that quoted text occurred in the supplied source;
it does **not** prove that a claim follows from it. The substring rubric is also
only a proxy for factual/legal correctness.

## Implemented and verified locally

The repository now has exact model-specific HTTP/SSE calls, strict task/dimension/
vector validation, tokenizer-aware index fingerprints, resumable model-scoped
caches, explicit Riva pivot provenance, shared CLI/evaluation retrieval, and
optional JSON claims with verbatim evidence/citation validation. Invalid outputs
fail closed; provider failures are distinct from “not in the documents.” No
orchestration framework, UI, account access or cloud vector database was added.

**106 offline checks pass** (59 legacy + 47 NVIDIA/pipeline/report tests), together
with compilation and `pip check`. The latest completed remote CI also passed
([run 33971261523](https://github.com/seif-bkh/RAGLab/actions/runs/33971261523));
newer local changes await authenticated push. Offline/CI success is not a live
model-quality or security certificate.

## Next controlled run and production blockers

Iteration 03 changes operational pacing and corrects the reference fixture;
it does not change source inputs, retrieval/answer labels or generation prompts.
It compares **grounded-v1 for both answerers**, reuses exact successful responses,
and retries missing answers serially: one worker, minimum 30 s request spacing,
180 s request timeout, two attempts, default 60 s HTTP-429 backoff with bounded
Retry-After handling. Expanded-context results remain explicitly incomplete.

After GitHub is reconnected, push the current session branch to launch the
versioned plan. For an authorized local environment, the equivalent command is:

```bash
cd raglab
python main.py benchmark --stage all --answer-profiles grounded-v1
```

**Production remains blocked by:**

1. Incomplete held-out generation, Kimi answer comparison and adversarial-context
   evaluation; no validated end-to-end winner yet.
2. Trial endpoint timeouts/rate limits and tens-of-seconds observed answer calls;
   no approved hosting, capacity, cost or SLA validation.
3. Tiny correlated evaluation sets, approximate rubrics, and no independent
   Arabic/French banking/legal review.
4. PDF extraction artifacts and missing structural metadata; robust document
   versioning, broader ingestion validation and larger source-poisoning tests.
5. Deployment-specific data governance and access controls before exposing any
   service or confidential banking data (outside this local CLI lab's scope).

`production_ready` remains **false**. Passing the prepared run would still not,
by itself, certify a banking product.
