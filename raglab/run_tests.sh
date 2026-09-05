#!/usr/bin/env bash
# Supported stack checks. No automatic Git writes or legacy model calls.
# Usage: ./run_tests.sh --offline | ./run_tests.sh --benchmark
# Live regression needs the frozen retrieval artifact described in README.md.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="../.venv/bin/python"
[ -x "$PY" ] || PY="python3"
mode=offline
while [ $# -gt 0 ]; do
    case "$1" in
        --offline) mode=offline ;;
        --benchmark) mode=benchmark ;;
        --no-push) : ;;
        --legacy|--provider|--stage|--real-only) echo 'Historical model experiments are retired. Use --offline or --benchmark.'; exit 2 ;;
        --push) echo 'This runner never commits or pushes files.'; exit 2 ;;
        -h|--help) sed -n '1,4p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 2 ;;
    esac
    shift
done
mkdir -p logs
LOG="logs/test_run_$(date -u +%Y%m%d_%H%M%S).log"
run_all() {
    echo "=== RAGLab checks $(date -u +%FT%TZ) ==="
    echo "python=$($PY --version) mode=$mode"
    "$PY" -m compileall -q . || return 1
    "$PY" tests_offline.py || return 1
    "$PY" -m unittest -v test_nvidia_pipeline test_hard_harness || return 1
    "$PY" main.py inspect || return 1
    "$PY" -m pip check || return 1
    if [ "$mode" = benchmark ]; then
        echo '### Selected Qwen/Nemotron regression (explicit API calls)'
        "$PY" main.py benchmark
    else
        echo '### Offline only: no model calls'
    fi
}
run_all 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
printf '\nEXIT=%s\n' "$status" | tee -a "$LOG"
cp "$LOG" logs/latest.log
printf 'Complete log: %s\n' "$LOG"
exit "$status"
