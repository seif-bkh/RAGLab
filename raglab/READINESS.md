# Production-readiness assessment

**Decision: not production-ready for a banking service.**

The selected stack is now **NVIDIA `nvidia/nemotron-3-embed-1b` (native 2048)
+ original-query Chroma retrieval + xKiro `qwen/qwen3.8-max:free` answers**.
Other providers/models and separate chat translation are no longer supported
runtime choices. Historical reports remain evidence, not active dependencies.

Removing an unused provider reduces complexity; it does not establish factual,
security, privacy or availability guarantees. The current implementation is
suitable for a **supervised pilot with approved nonconfidential documents**,
not unsupervised customer-facing banking advice or automated financial decisions.

## What the evidence supports

The post-cleanup [selected-pipeline regression](reports/selected_pipeline_01.json),
[Actions run 33979521271](https://github.com/seif-bkh/RAGLab/actions/runs/33979521271),
completed successfully with all declared regression gates true:

- 13/14 development answer-rubric passes; the remaining French case used “achète”
  where the narrow rubric expected “achat”/“livraison.” The original score is retained.
- 18/18 held-out answer-rubric passes and successful citation/quote validation.
- 3/3 synthetic Arabic/French/English source-injection fixtures passed.
- 9/9 private/live-data refusals, **all performed by local guards**, not nine
  independent demonstrations of model refusal quality.
- 36 new Qwen client calls, no replayed answer results, no provider errors.
  Nemotron retrieval used the verified frozen snapshot; no new embedding calls.
- Approximately 5.26 s development / 4.91 s held-out answer client-call means.
  These exclude live embedding/retrieval time and are not production percentiles.
- **133 offline checks pass** (59 historical regressions + 74 current pipeline/
  contract checks); compilation, dependency checks and remote CI passed without
  the retired Google generation SDK.

The simpler default CLI was also exercised on a private-account question:
it selected xKiro Qwen automatically and refused locally with zero inference.

Implemented protections include exact model selection, no fallback, live
zero-price checks, scoped credentials, bounded retries, stale-index/vector
validation, token-bounded context, normalized source-quote membership and
fail-closed invalid responses. Tests cover these contracts and default routing.

The selected-pipeline regression passed, but `production_ready` remains **false**.
Completing this suite and proving production readiness are different claims.

## Must resolve before customer-facing production

### 1. Approved and dependable model serving

xKiro is a routed gateway. A matching model label is not independently verified
upstream identity/version. The free SKU can change availability/pricing; fail-
closed checks avoid silent paid/model fallback but can make the service unavailable.

**Exit evidence:** approved provider/data-processing terms, retention/residency
review, an acceptable model/version guarantee, measured capacity, incident/support
arrangements, and an availability/latency commitment suitable for the intended use.
Do not infer these from a free catalog entry or a handful of successful requests.

### 2. Independent banking/legal quality evaluation

The held-out answers cover **six independent facts in three languages**, not 18
independent facts. Labels are authored examples and substring proxies. Quote
membership after normalization does not prove that a claim follows from a quote.
The same holdout has been reused across iterations.

**Exit evidence:** a larger independently authored Arabic/French/English question
set, qualified banking/legal review, explicit critical-error criteria (amounts,
dates, negation, obligations and entity identity), unsupported-question tests,
and evaluation on new documents not used during development.

### 3. Deployment security and document access boundaries

Three known injection fixtures and capability regexes are useful regressions,
not a security assessment. This CLI has no multi-user authentication, document
ACLs/tenant isolation or account/transaction integration. Those features were
not requested and have not been added implicitly.

**Exit evidence if exposed as a service:** access control and authorized-document
retrieval, adversarial/source-poisoning testing, secret/data-leak testing,
auditability and a reviewed policy for human escalation. Keep financial actions
outside the model unless separately designed and validated.

### 4. Corpus and operational reliability

The four-document evaluation corpus has PDF extraction artifacts and **zero
recognized heading-metadata chunks**. Scanned PDFs, tables and general ingestion
lifecycle behavior are not comprehensively validated. Cache locks coordinate
instances inside one process only; concurrent writer processes are unsupported.

**Exit evidence:** extraction/OCR and source-location validation, version/update/
delete procedures, representative load and fault-injection tests, p95/p99 latency
and error-rate targets, backups/restore drills, operational metrics and an
owner for incidents. Local Chroma persistence alone is not high availability.

## Pilot boundary

Use read-only document QA with human review, approved nonconfidential inputs,
visible evidence/citations, explicit refusal/error handling and bounded traffic.
Do not present benchmark percentages as probabilities of correctness, claim
verified upstream weights, or describe this as a production banking product.

See [the current guide](README.md) and [historical comparison details](FREE_MODELS_REPORT.md).
