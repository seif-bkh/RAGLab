# Free gateway RAG results — 5 September 2026

> Historical model comparison. The supported runtime is now pinned to xKiro
> Qwen 3.8 Max Free and native Nemotron embeddings; retired providers are removed.
> See [current configuration](README.md) and [production readiness](READINESS.md).

## Recommendation

**Best tested free option: `qwen/qwen3.8-max:free` through xKiro.**

Use it with the existing **`nvidia/nemotron-3-embed-1b` native 2048-dimensional
embeddings**, original-query vector retrieval, top 5 chunks, and `grounded-v1`
answers. Query translation is disabled in this evaluated profile, avoiding the
NVIDIA chat rate limits encountered earlier.

The Qwen profile passed the declared development, held-out answer and three
source-injection gates. **This is a promising, measured lab configuration—not
production certification.** Gateway model identities are not independently
verified, the evaluation is small, and free availability/prices can change.

**Historical KiosAPI tests were blocked by key routing**, not graded as poor
model quality. That provider is now removed from the supported runtime; it is
not a dependency or a remaining action for the selected pipeline.

## Fresh, full-development comparison

All four xKiro candidates received the same 16 development cases: 14 answerable
questions and two refusal cases. These were **fresh client calls**, without
reading or writing successful-answer caches. Prompts, context and labels were
unchanged. All candidate SKUs passed live zero-price checks before inference.

| xKiro request SKU | Strict answer rubric | Validation, all cases | System refusals | Mean client-call time |
|---|---:|---:|---:|---:|
| **`qwen/qwen3.8-max:free`** | **13/14 — 92.9%** | **16/16** | **2/2** | **4.68 s** |
| `minimax/minimax-m3:free` | 11/14 — 78.6% | 15/16 | 2/2 | 6.07 s |
| `mistralai/mistral-small-2603` | 6/14 — 42.9% | 11/16 | 2/2 | 8.27 s |
| `deepseek/deepseek-v4-flash` | 7/14 — 50.0% | 13/16 | 2/2 | 7.46 s |

These are **source-constrained concept-rubric scores**, not independently judged
semantic accuracy. Invalid quotes/JSON fail closed instead of being delivered
as answers. The time column includes failed/invalid-output client calls and
pacing/retries, excludes local guards and cached responses, and is not a full
live-query latency measurement or an SLA. Each model had 15 measured client calls;
the private-account refusal required no inference.

Notable findings:

- **Qwen:** no provider errors, invalid outputs or false refusals in development.
  Its one rubric miss was the French Salam answer: it used the valid conjugation
  **“achète”**, whereas the narrow rubric required “achat” or “livraison.” Its
  cited source matched. The frozen score remains **13/14**, not manually promoted
  to 14/14.
- **MiniMax M3:** one rejected quote, one insufficient-evidence refusal despite
  available evidence, and “credit cards” instead of the expected deferred-debit
  terminology. The latter needs banking-language review.
- **Mistral Small:** several quote/JSON validation failures and language errors,
  including Spanish for an English question and Arabic for another English one.
- **DeepSeek Flash:** an incomplete stream, quotation failures, a false refusal
  and some narrow wording-rubric misses. Its 50% strict score is not a claim
  that every failed rubric represented a factually wrong answer.

Qwen was selected from **development results before its held-out calls**.

## Qwen held-out and security results

| Check | Result |
|---|---:|
| Answer rubric | **18/18** |
| Answerable-question false refusals | **0/18** |
| Citation/quote validation | **18/18 rendered answers** |
| Validation across all held-out cases | **27/27** |
| Private/live-data refusal checks | **9/9** |
| Untrusted-source injection fixtures | **3/3** |
| Mean held-out answer client-call time | **5.25 s**, 18 calls |

All 18 answerable held-out cases made new client calls; none was replayed from
an answer cache. **The nine held-out refusals were local capability guards,
not nine demonstrated LLM refusals.** Development also included one genuinely
unsupported question that the model refused.

The three synthetic source-injection tests covered Arabic, French and English.
Qwen answered the legitimate questions with citations and did not emit the
injected override marker. Three tests are not a security proof.

The held-out answer set consists of **six facts expressed in three languages**,
not 18 independent facts. These fixtures were already used in earlier experiments;
this is not a new independent validation dataset. Quote validation checks source
membership after normalization; it does **not** establish semantic entailment,
legal correctness or faithful interpretation.

## KiosAPI: tested, but no model-quality result

Live public pricing advertised these exact request SKUs exclusively in its
zero-multiplier `Free` group:

- `moonshotai/kimi-k3`
- `Qwen/Qwen3.8-27B`
- `nvidia/nemotron-3-super-120b-a12b`
- `deepseek-v4-flash`

Each was attempted on two development questions, with bounded retries. All
returned HTTP 503 errors rather than usable model answers. The dominant error was:

> No available channel for model … under group default (distributor)

There were also provider-reported CPU-overload errors. The earlier authenticated
`/v1/models` response was empty, consistent with the current routing problem.
Neither observation proves that the advertised models are intrinsically bad or
unavailable to a correctly configured token.

**Current status:** this integration and its secret references have been removed.
No token/group change is needed for the selected Qwen/Nemotron pipeline. The
following comparison status describes the historical run, not a current dependency.

The overall workflow deliberately remains **incomplete** because KiosAPI is not
fully measured and the DeepSeek arm had a provider failure. That is distinct from
the selected Qwen profile passing its quality gates.

## What “free” means here

- **xKiro:** the live model API had to report `access_tier=free`, zero USD input
  and output prices, and zero separately priced cache tokens where specified.
  Paid Kimi K3 and the dated DeepSeek 0813 SKU were excluded from xKiro testing.
- **KiosAPI:** exact SKUs had to be exclusive to `Free`, with group multiplier
  zero and no custom billing expression. `model_price=0` alone is insufficient
  for token-priced models; mixed Free/paid SKUs were rejected.
- Pricing was rechecked between stages and after five minutes between new calls.
  No paid sibling, stripped `:free` suffix or client-side model fallback was used.
- The final xKiro run used **81 logical inference calls / 83 HTTP attempts**,
  excluding pricing reads. It made **zero embedding or translation API calls**.
  These are provider-advertised zero-priced SKUs, not an independent billing audit.

The services are gateways. A matching response model label does not verify the
actual upstream engine; xKiro documents internal routing while reporting the
requested model name. Do not confuse these results with direct NVIDIA-hosted
measurements, model-vendor benchmarks or a deployment SLA.

## Reproducibility and artifacts

The answer experiment used the immutable original-query retrieval artifact from
NVIDIA run **33970405295**: four real Arabic PDF/DOCX files, 836 chunks, actual
`cl100k_base` tokenization, chunk size 220 / overlap 40, and native Nemotron
2048-dimensional embeddings. Commit, question hashes, chunk manifest and each
retrieved hit's source/text were verified before inference. No gold answer or
rubric was sent to a model.

| Experiment | Run | Saved measurements |
|---|---|---|
| Initial pricing preflight | [33976033405](https://github.com/seif-bkh/RAGLab/actions/runs/33976033405) | [free_iteration_01.json](reports/free_iteration_01.json): pricing HTTP 403; **zero inference calls** |
| Screen + MiniMax follow-through | [33976171738](https://github.com/seif-bkh/RAGLab/actions/runs/33976171738) | [free_iteration_02.json](reports/free_iteration_02.json) |
| Fresh full xKiro comparison | [33977206216](https://github.com/seif-bkh/RAGLab/actions/runs/33977206216) | **[free_iteration_03.json](reports/free_iteration_03.json)** |

The first pricing failure was corrected using the same scoped authentication
and explicit client identification as the successful catalog probe; the free-
price gate was not bypassed. MiniMax initially won the small screen, but its
weaker full-development result motivated the fresh four-model comparison.
Earlier outcomes remain preserved; no labels or prior results were rewritten.

JSON records retain actual answers, evidence quotes, errors, pricing snapshots,
model/request IDs and provenance. Full unused source bodies remain in workflow
artifacts, not the compact Checks records. Workflow artifacts have limited
retention; a later replay may require a newly versioned frozen retrieval artifact.
See [methodology](benchmarks/FREE_MODELS.md) and [the execution plan](benchmarks/free_provider_plan.json).

## Use the tested profile

The CLI now defaults to the selected Qwen/Nemotron pair. Other provider/model
choices are rejected; query translation is disabled. Configure `NVIDIA_API_KEY` and `XKIRO_API_KEY` in the local
environment or `.env`, then:

```bash
cd raglab
python main.py ingest --reset --data-dir ../docs
python main.py answer "Quelle est la définition du financement Salam selon le guide interne ?" \
  --query-lang fr
```

This retains native Nemotron embeddings and original-query retrieval. The answer
factory checks current free-price eligibility before billable embedding calls;
private/live-account questions are refused locally before any provider or index
access. Each provider receives only its own credential at its documented HTTPS
endpoint; redirects are disabled for gateway requests.

The generator and scoring pipeline were exercised live in the experiments.
**128 offline checks pass** (59 legacy + 69 pipeline/gateway/report/CLI tests),
with compilation and `pip check` clean. The new CLI provider wiring and no-network
refusal paths are covered by regressions, and the Qwen CLI private-account refusal
was exercised locally without a provider call. No additional live CLI inference
was claimed.

**Before production:** independently authored multilingual banking/legal tests,
expert review, stronger extraction/metadata, approved data hosting/retention,
access controls, adversarial testing and load/SLA validation are still required.
Use only approved nonconfidential material with trial/free gateways.
