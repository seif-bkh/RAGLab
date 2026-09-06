#!/usr/bin/env bash
# One-liners for testing this lab on your own machine. Each subcommand is a thin wrapper around
# hard_harness_main.py (or chat.py), so CI and local runs execute the same code. Only `answers`,
# `grade`, `index`, `judge`, `sources` and `author` touch a provider; `check`, `probe`, `restore`,
# `report`, `chat --check` and `test` are free. Keys live in raglab/.env, which is git-ignored, and
# no subcommand here prints one - values are masked to their first eight characters.
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"      # resolved before cd, so $0 stays valid
cd "$(dirname "$SELF")"                                      # -> raglab/
# README.md creates raglab/.venv, this script creates one at the repo root; the first found wins.
ROOT_VENV="../.venv"
VENV=""
for candidate in "$ROOT_VENV" ".venv"; do
  [ -f "$candidate/bin/activate" ] && VENV="$candidate" && break
done
run() {
  [ -n "$VENV" ] || { echo "no venv yet - run: ./local.sh setup"; exit 1; }
  . "$VENV/bin/activate"
  exec python3 "$@"
}
case "${1:-help}" in
  setup)                                              # fresh clone: deps + a .env to fill in
    [ -n "$VENV" ] || VENV="$ROOT_VENV"
    python3 -m venv "$VENV"
    . "$VENV/bin/activate"
    python3 -m pip install -q -r requirements-harness.txt
    if [ -f .env ]; then echo "raglab/.env exists - leaving it alone"; else
      cp .env.example .env; echo "created raglab/.env - paste your keys, then: ./local.sh check"; fi
    echo "virtualenv: $VENV"
    ;;
  check)   run hard_harness_main.py doctor ;;
  probe)   run hard_harness_main.py doctor --json ;;
  restore)                                            # no API traffic: pull the published checkpoints
    sha="${2:-$(gh run list --workflow hard-harness.yml --limit 1 --json headSha -q '.[0].headSha' 2>/dev/null || true)}"
    [ -n "$sha" ] || { echo "could not ask GitHub for the last harness run sha - pass one: ./local.sh restore <sha>"; exit 1; }
    run hard_harness_main.py collect --sha "$sha" --destination results/hard_harness
    ;;
  report)  run hard_harness_main.py report ;;
  judge)   shift; run hard_harness_main.py judge --arms lexical,vector "$@" ;;  # embeddings only
  index)   run hard_harness_main.py retrieve ;;                               # corpus + query embeddings
  answers) run hard_harness_main.py predict --shard "${2:-0}" ;;              # 100 Qwen answer calls
  grade)   run hard_harness_main.py grade ;;                                   # judge tokens, cached per request
  chat)    shift; run chat.py "$@" ;;                                          # the Q&A REPL (NVIDIA Nemotron)
  test)    run -m unittest test_hard_harness test_nvidia_pipeline ;;
  help|*)
    awk '/^#!/ {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$SELF"     # the header comment block
    echo
    echo "subcommands: setup check probe restore report judge index answers grade chat test"
    echo "run order for a fresh clone:  setup -> check -> restore -> report"
    echo "ask questions:                ./raglab/chat.sh   (or ./local.sh chat --ingest first)"
    ;;
esac
