# Large-harness progress — 5 September 2026

**The 3,000-question dataset and final comparison are not complete yet.**

## Recovered checkpoint

- GitHub access is working again. The saved authoring run **33987445824** was
  retrieved and file fingerprints verified.
- **315 audited base scenario families (945 language-specific question/reference
  pairs)** are retained: shard 0 has 98, shard 1 has 93, shard 2 has 70, shard 3
  has 54. These are drafts, not yet the released 1,000-per-language dataset.
- Twelve families require repair (mostly malformed JSON, one primary-evidence
  mismatch). Shards 2/3 paused on provider capacity/server errors; the other five
  shards deferred without model calls. No daily-quota exhaustion was established.
- The source reference checkpoint remains ready: 230 evidence units, 36 original
  PDF pages accounted for, and 21 assistant-visual reviews with hash-bound fixes.
  This is not legal-expert certification. Runtime extraction remains unchanged.

## Resume safeguards

- Accepted family caches are reused. A previous-run artifact fallback restores
  audited references if an Actions cache was evicted; source/spec checks still apply.
- Reference author/audit requests now explicitly request a JSON object. Their
  request-cache identity includes this mode, so malformed old responses are not
  mistaken for repaired outputs. Already accepted references are not regenerated.
- The answering agent's ordinary payload and completed-response caches are unchanged.
  Candidate outputs, including invalid completed outputs, are retained as results.
- **155 offline checks pass** (59 historical plus 96 pipeline/harness checks).
- The harness still uses **XKIRO_API_KEY_JINKO**. Google remains inactive; any
  switch is explicit, provenance-preserving and requires free-tier-project confirmation.

## Next

Resume unfinished reference families/shards, validate the full paired dataset,
freeze separate question and answer-key files, then execute and compare all
3,000 cases. A shared pause signal stops newly queued model work on provider
or credential failures. No missing questions, references, or scores are fabricated.
