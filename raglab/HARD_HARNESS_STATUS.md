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
- **157 offline checks pass** (59 historical plus 98 pipeline/harness checks).
- The user confirmed a **Google free-tier project** for unfinished reference
  authoring/auditing. The reference profile now requests Gemini 3.1 Flash-Lite
  using an available approved Google secret alias; candidate answers remain
  targeted at xKiro Qwen with JINKO. Saved Qwen references retain their provenance.

## Latest retry

Run **33993104191** retried the first unfinished reference with the same JINKO
profile, but received xKiro's “temporarily at capacity” stream error after bounded
attempts. Newly queued shards deferred without inference. This is a provider
capacity failure, not confirmed daily-quota exhaustion.

All 315 audited families remain in earlier artifacts and per-family caches.
Checkpoint export now includes **all** validated cached families before retrying
an earlier gap, so a pause cannot make later cached IDs disappear from an artifact.

Google reference fallback is now explicitly authorized and configured. It is
not presented as Qwen-only authoring and does not change the ordinary answering
application. Google requests are globally paced across author/auditor clients,
with one active reference shard to respect shared free-tier limits.

## Next

Resume unfinished reference families/shards with the confirmed reference provider, validate the full paired dataset,
freeze separate question and answer-key files, then execute and compare all
3,000 cases. A shared pause signal stops newly queued model work on provider
or credential failures. No missing questions, references, or scores are fabricated.
