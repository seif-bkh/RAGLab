# Running the hard harness on your own machine

Everything here is the same code CI runs; the only difference is which credentials you have to
supply yourself. Two rules make the difference safe to ignore: the dataset version is pinned in
`raglab/benchmarks/hard_harness_plan.json` (chunking, sample size, per-role provider/model), and no
phase ever switches provider or model on its own — a spent quota pauses and says so.

## 1. Set up

```bash
git clone <this repo> && cd RAGLab
cp raglab/.env.example raglab/.env      # paste the 3 keys: XKIRO_API_KEY, EXPERIENTIAL_API_KEY, NVIDIA_API_KEY
./raglab/local.sh setup                 # .venv at the repo root + requirements-harness.txt
./raglab/local.sh check                 # doctor: free, and it exits non-zero if a key is short
```

That leaves you in `raglab/` with everything the phases need. The wrapper is optional - each
subcommand is one `hard_harness_main.py` call, listed in the table below, and `./local.sh help` prints
the same. If you would rather not use it:

```bash
python3 -m venv .venv && source .venv/bin/activate    # from the repo root
pip install -r raglab/requirements-harness.txt
cd raglab && cp .env.example .env && python hard_harness_main.py doctor
```

`doctor` makes no completion call. It lists models per key (which is free), prints which profile each
role resolves to, reports the tokenizer and corpus state, and exits non-zero if the current plan
phase is missing a credential — so it is safe to run before anything that costs money, and useful
in a script. Use `--no-probe` for an offline readout and `--json` for machines.

Required: `NVIDIA_API_KEY` (embeddings), `XKIRO_API_KEY` (answers), `EXPERIENTIAL_API_KEY`
(the judge — billed per token, no free tier). Optional: `XKIRO_API_KEY_JINKO` for the author role,
`GOOGLE_API_KEY`/`GEMINI_API_KEY` for reference-side auditing.

`doctor` loads `raglab/.env` through `config.py`, the same import every phase performs, and prints
what that load did: how many assignments the file has, which names it put in the environment, and
whether `python-dotenv` is installed at all. A key can be present in the file and still invisible to a
run - no `dotenv`, an empty assignment, a name spelled differently, or an `export` in your shell that
`.env` will not overwrite - so each of those reads differently instead of all saying "missing". (They
used to all say missing: the first version of this command never imported `config`, so a filled `.env`
looked empty. That is now pinned by a test.)

If `doctor` says the tokenizer is an estimator fallback, **stop before measuring anything**: chunk
boundaries move, and your recall numbers will not mean the same as CI's. Warm `TIKTOKEN_CACHE_DIR`
with `cl100k_base` first.

## 2. Start from what is already frozen, not from paid phases

`raglab/results/hard_harness/` is git-ignored, so a fresh clone is empty. The published checkpoints
carry the whole state — sources audit, 476 accepted families, the frozen 100×3 dataset, 300 answers
and 300 grading rows. Pull them instead of re-running anything:

```bash
./raglab/local.sh restore                 # asks gh for the last harness run's sha, then collects it
./raglab/local.sh report                  # prints the 0.780 table from those rows
```

`restore` takes an optional sha if you want a specific run (`./local.sh restore 0d2589f…`);
`python hard_harness_main.py collect --sha <sha> --destination results/hard_harness` is the same call.

Collected checkpoints are verified by fingerprint on the way in, so a truncated publish fails loudly
rather than scoring partial data. Grading replays its cached judgments and makes no new judge
requests; that is how the 0.780 figure can be re-derived locally for free.

## 3. Phases, and what each one costs

| command | provider traffic |
|---|---|
| `judge --arms lexical,vector [--chunk-tokens N]` | embeddings only, no completion. Retrieval quality and abstain separability |
| `retrieve` | embeddings for the corpus and the 300 queries |
| `predict --shard 0..2` | **100 answer calls per shard** on Qwen; never reads the answer key |
| `grade` | ~6 comparisons per judge call on GLM; every call is cached by request hash |
| `compile-dataset` | no answer traffic; it freezes the public questions and the sealed key files |
| `sources`, `author --shard N` | vision + drafting/auditing traffic; both are resumable and shard-cached |

Order for a fresh extension of the dataset is `sources → author → compile-dataset → retrieve →
predict → grade`. Batch size in `grade` (six per request) is a cache-key constraint, not a
tunable: changing it invalidates every judgment already paid for.

## 4. What a local number is allowed to claim

- Every published row records `provider` and `model`; answering is xKiro Qwen, judging is
  Experiential Labs GLM-5.3-Flash, and neither is reported as the other's result.
- `status: paused` means quota, not performance. Nothing is imputed for unanswered cases, and a
  phase that only partially graded reports `partial`, never a score.
- Rows where the judge could not answer in usable form become `judge_unusable` with `correct=None`
  and are excluded from every denominator, with `complete_with_judge_gaps` as the status.
- The frozen sample is all-supported: refusal correctness on out-of-scope and insufficient-evidence
  questions, and injection resistance, are **not** measured yet. Do not present 0.780 as a general
  capability score.

## 5. Tests

```bash
cd raglab && python -m unittest test_hard_harness test_nvidia_pipeline   # 145 offline tests
python tests_offline.py                                                  # 59 pipeline checks
```

Both run without any credential: they exercise the client contract with fakes, so a passing suite
proves the plumbing, not the providers.
