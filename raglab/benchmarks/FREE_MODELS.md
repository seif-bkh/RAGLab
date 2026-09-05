# Free-model shortlist and safeguards

> Historical experiments; not the supported runtime. Current configuration uses
> only xKiro Qwen 3.8 Max Free and native Nemotron embeddings. See the main guide
> and readiness assessment. Old measurements/labels remain unchanged.


Research date: **2026-09-05**. This is an explicitly broader, gateway-reported
answer experiment authorized after the exact NVIDIA comparison. The embeddings
remain the frozen native `nvidia/nemotron-3-embed-1b` results.

## Sources of truth

- [xKiro live model API contract](https://docs.xkiro.com/api/list-models/):
  `access_tier` and current USD input/output/cache prices come from
  `GET https://api.xkiro.com/v1/models`.
- [xKiro public model/price table](https://xkiro.com/models): useful for shortlisting,
  but rechecked against the API at execution. Individual promotional pages may
  disappear while catalog entries remain; do not infer availability from a page title.
- [xKiro reasoning controls](https://docs.xkiro.com/guides/reasoning/): controls
  differ by model. Use the lowest supported explicit effort from live metadata;
  do not assume a universal thinking flag works.
- [KiosAPI public pricing](https://kiosapi.com/api/pricing): request SKU names,
  token/call pricing mode, enabled groups and group multipliers. Its `Free` group
  was advertised with multiplier zero and a warning that request stability is
  not guaranteed.
- [KiosAPI model-list contract](https://kiosapi.mintlify.app/api-reference/models/list-models.md):
  the authenticated list is scoped to a key's access. An empty list is an access
  diagnostic, not proof that documented inference SKUs do not exist.

**Not used as a free-price guarantee:** third-party social posts, open-weight
licenses, a “start free” button, free signup credits, or an old blanket promotion.
xKiro's current table lists Kimi K3 and the dated DeepSeek V4 Pro 0813 SKU as paid;
they were not included in this free xKiro shortlist. Undated/free SKUs are not
relabeled as those exact NVIDIA models.

## Predeclared candidates

These are candidates to *test*, not claims of best quality. The mix covers
larger general models and smaller/Flash alternatives for a quality/latency check.

| Provider | Request SKU | Reason to include |
|---|---|---|
| xKiro | `qwen/qwen3.8-max:free` | Larger Qwen candidate |
| xKiro | `minimax/minimax-m3:free` | Independent general-model family |
| xKiro | `mistralai/mistral-small-2603` | Smaller Mistral alternative, including French questions |
| xKiro | `deepseek/deepseek-v4-flash` | Flash alternative; explicitly not the paid dated SKU |
| KiosAPI | `moonshotai/kimi-k3` | Advertised exclusively under Free, despite xKiro's paid Kimi SKU |
| KiosAPI | `Qwen/Qwen3.8-27B` | Smaller Qwen SKU; distinct from paid lowercase aliases |
| KiosAPI | `nvidia/nemotron-3-super-120b-a12b` | Answer-model candidate, not the embedding model |
| KiosAPI | `deepseek-v4-flash` | Another Free-only answer candidate |

The executable shortlist and screen are in `free_provider_plan.json`. Every SKU
can be excluded if its live pricing changes. KiosAPI entries must be **exclusive
to Free**, have a zero group multiplier and no custom billing expression.
`model_price=0` alone is insufficient for token billing; mixed Free/paid SKUs
are deliberately excluded rather than assuming which group a key will select.

## Experiment

1. Validate the original-query retrieval artifact from NVIDIA run 33970405295:
   native 2048 dimensions, corpus/chunk manifest, unchanged question hashes and
   exact hit text/source correspondence. No new embedding or translation calls.
2. Screen eligible models on seven fixed development cases: five answerable
   variants and two refusal cases (one of those is a local private-data guard).
3. Evaluate each provider's screen winner on all 16 development cases.
4. Freeze the overall winner before all 27 held-out cases and three synthetic
   source-injection tests. Selection uses refusal, strict rubric, validation,
   then observed client-call latency. There is no holdout tuning.
5. Keep every status, error, answer claim, citation and quoted evidence inspectable.
   At most 100 logical inference calls, at most two attempts each; no client-side
   paid/model fallback. Pricing is rechecked between stages and after five minutes
   between new calls. Keys never enter reports or caches.

## Development iteration after the first measured run

The first screen favored MiniMax M3 (5/5 answerable screen cases), but its full
sweep passed only 10/14 development rubrics, including two rejected quotations.
Its 17/18 held-out rubric result does not erase those development defects. The
next plan therefore compares **all four xKiro candidates on fresh full development
calls**, without reading or writing successful-answer caches. This avoids turning
selective retries of invalid outputs into apparently improved model quality.
Prompts, source context and labels stay unchanged; holdout is still not a
selection input. An explicit development rubric ≥.90 / validation=1 / refusal=1
gate is now recorded. Failed/invalid-output call durations are retained too;
previous run latency averages excluded invalid-output responses.

KiosAPI is deferred: actual inference returned `model_not_found` under group
`default (distributor)` (plus occasional CPU-overload errors). A token permitted
to use the zero-multiplier **Free** routing group is needed before further tests.
No model-quality score is assigned to these transport/routing failures.

A gateway can route internally; a matching response label does not independently
verify the upstream model. Prices/availability may change, “free” does not mean an
SLA, and this procedure is not an independent billing audit or a banking/legal
quality certification. The same small-suite and proxy-rubric limitations as the
NVIDIA evaluation apply. Context is original-only, so answer scores must not be
presented as a pure model swap against the earlier translated-query experiment.
