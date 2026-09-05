# Large-harness progress — 5 September 2026

**The 3,000-question dataset and final scores are not complete yet.**

- GitHub access was restored and the saved third source audit was retrieved.
- All 36 original PDF pages are accounted for. The Arena assistant visually
  reviewed 21 pages total, including the 14 pages flagged in the final audit;
  corrections are file-hash-bound in `benchmarks/hard_source_reviews.json`.
- Reference-source preparation now passes with 230 usable evidence units across
  all four documents. It was rebuilt from saved/visually reviewed text with
  **zero new model calls**. This is not legal-expert certification.
- The runtime extraction/chunk manifest remains unchanged; ingestion errors are
  not silently fixed or converted into the answer key.
- The base authoring plan has 900 families (650 supported, 200 out of scope,
  50 ambiguous). A further 100 adversarial variants make 1,000 families, paired
  across Arabic/French/English. Question/reference files remain separate.
- Authoring, frozen-key validation, sharded prediction, semantic grading and
  quota-resume code are implemented. The fresh JINKO credential remains selected;
  Google fallback is available only through an explicit, confirmed plan switch.

Next: publish the reviewed source checkpoint, run the nine author/audit shards,
freeze all three question files and separate answer keys, then run/compare all
3,000 cases. Completed responses and attempts are retained in checkpoints.
