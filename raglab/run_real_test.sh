#!/usr/bin/env bash
# Real-model test: both chunking arms with the pinned NVIDIA embeddings,
# plus one cited xKiro/Qwen answer. This is the version of harness50 that
# spends API calls, so it is only runnable where the endpoints are
# reachable (not from the restricted sandbox) with keys in raglab/.env.
#
# Usage:  ./run_real_test.sh
# Reads:  NVIDIA_API_KEY and XKIRO_API_KEY from raglab/.env
# Writes: results/evaluate_*.json per arm + one answers JSON, prints the
#         metrics table for each arm (compare across the two runs).
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="../.venv/bin/python"
[ -x "$PY" ] || PY="python3"

if [ ! -f .env ]; then
  echo "ERROR: raglab/.env not found. Copy .env.example to .env and fill in"
  echo "       NVIDIA_API_KEY and XKIRO_API_KEY, then re-run."
  exit 2
fi

for arm in size restructure; do
  echo
  echo "############################################################"
  echo "# ARM=$arm — ingest (NVIDIA nemotron embeddings) + evaluate"
  echo "############################################################"
  CHUNKING_MODE="$arm" "$PY" main.py ingest --reset --skip-sanity-check \
      --data-dir ../docs --data-dir data || exit 1
  CHUNKING_MODE="$arm" "$PY" main.py evaluate --skip-sanity-check \
      --no-translation --top-k 20 --questions questions_50.json || exit 1
done

echo
echo "############################################################"
echo "# Answer smoke test — xKiro/Qwen cited answer (restructure arm)"
echo "############################################################"
"$PY" main.py answer "ما هي المرابحة؟" --query-lang ar \
    --output results/harness50/real_answer_murabaha.json || exit 1

echo
echo "Done. Compare the two evaluate_*.json runs in results/ "
"(metrics.overall + metrics.by_category), and the answer JSON for the"
"citation contract."
