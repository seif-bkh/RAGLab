#!/usr/bin/env bash
# One-liners for testing the hard harness on your own machine. Each subcommand is a thin wrapper
# around hard_harness_main.py, so CI and local runs execute the same code. Only `answers`, `grade`,
# `index`, `judge`, `sources` and `author` touch a provider; `check`, `restore`, `report` and `test`
# are free. Keys live in raglab/.env (git-ignored) and no subcommand here prints one.
set -euo pipefail
SELF="$0"                                            # $0 goes stale once we cd
SELF="$(cd "$(dirname "$SELF")" && pwd)/$(basename "$SELF")"
cd "$(dirname "$SELF")"                                    # -> raglab/
VENV="../.venv"                                       # repo root, already git-ignored
run() {
  [ -f "$VENV/bin/activate" ] || { echo "no venv yet - run: ./local.sh setup"; exit 1; }
  . "$VENV/bin/activate"
  exec python3 hard_harness_main.py "$@"
}
case "${1:-help}" in
  setup)                                              # fresh clone: deps + a .env to fill in
    python3 -m venv "$VENV"
    . "$VENV/bin/activate"
    python3 -m pip install -q -r requirements-harness.txt
    if [ -f .env ]; then echo "raglab/.env exists - leave it alone"; else
      cp .env.example .env; echo "created raglab/.env - paste your three keys, then: ./local.sh check"; fi
    ;;
  check)   run doctor ;;
  probe)   run doctor --json ;;
  restore)                                            # no API traffic: pull the published checkpoints
    sha="${2:-$(gh run list --workflow hard-harness.yml --limit 1 --json headSha -q '.[0].headSha' 2>/dev/null || true)}"
    [ -n "$sha" ] || { echo "could not ask GitHub for the last harness run sha - pass one: ./local.sh restore <sha>"; exit 1; }
    run collect --sha "$sha" --destination results/hard_harness
    ;;
  report)  run report ;;
  judge)   shift; run judge --arms lexical,vector "$@" ;;    # embeddings only, no completion
  index)   run retrieve ;;                            # embedding calls for corpus + queries
  answers) run predict --shard "${2:-0}" ;;            # 100 Qwen answer calls
  grade)   run grade ;;                               # judge tokens; every request is cached
  test)    [ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
           exec python3 -m unittest test_hard_harness test_nvidia_pipeline ;;
  help|*)
    awk '/^#!/ {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$SELF"   # the header comment block
    echo
    echo "subcommands: setup check probe restore report judge index answers grade test"
    echo "run order for a fresh clone:  setup -> check -> restore -> report"
    ;;
esac
