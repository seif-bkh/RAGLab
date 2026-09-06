# Large-harness progress — 6 September 2026

**The final 3,000-question dataset and answer comparison are not complete yet.**

## Verified retained work

- The latest collected checkpoint, run **33997136207**, contains **460 audited
  base families / 1,380 language-specific question/reference pairs**.
- Shards 0–3 each contain 100 accepted families. Shard 4 contains 48, shard 5
  retains 8, and shard 6 retains 4. The remaining base families are not yet complete.
- Reference authoring stopped on repeated Google HTTP 503 high demand, not a
  confirmed quota error. All completed source/author responses remain checkpointed.
- All four source documents and all 36 PDF pages are accounted for; 230 reference
  units are available. Source-page review is assistant/model-assisted, not a
  human banking/legal expert certification. Runtime extraction is unchanged.

## Resume

The unfinished reference author/auditor profile now explicitly requests
**gemini-3.5-flash**, whose official model pricing lists a free tier. The user
confirmed that the Google credential is on a free-tier project. No billing
settings are changed, and no model switch is hidden in an existing score.

Existing accepted Qwen and Gemini 3.1 Flash-Lite records keep their provenance;
they are not regenerated. Candidate answering remains **xKiro Qwen / JINKO**
with native Nemotron retrieval. After authoring, the compiler must repair/flag
exact duplicates and ambiguous negatives before freezing separate question and
answer-key files. Only then may the 3,000 candidate answers and comparison run.
