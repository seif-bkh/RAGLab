#!/usr/bin/env bash
# Run the lab's checks with a complete, local log. No automatic Git writes.
# Usage: ./run_tests.sh --offline | ./run_tests.sh --stage all
#        ./run_tests.sh --provider gemini --legacy
# --no-push and --real-only remain accepted for old command-line compatibility.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="../.venv/bin/python"
[ -x "$PY" ] || PY="python3"
mode=benchmark
stage=retrieval
while [ $# -gt 0 ]; do
    case "$1" in
        --offline) mode=offline ;;
        --legacy) mode=legacy ;;
        --stage) stage="${2:?--stage needs retrieval or all}"; shift ;;
        --provider) export EMBEDDING_PROVIDER="${2:?--provider needs a provider}"; shift ;;
        --real-only) export RAGLAB_CI_REAL_ONLY=1 ;;
        --no-push) : ;;
        --push) echo 'Logs are now local/Actions artifacts; commit only a reviewed summary explicitly.'; exit 2 ;;
        -h|--help) sed -n '1,5p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 2 ;;
    esac
    shift
done
mkdir -p logs
LOG="logs/test_run_$(date -u +%Y%m%d_%H%M%S).log"
run_all() {
    echo "=== RAGLab checks $(date -u +%FT%TZ) ==="
    echo "python=$($PY --version) mode=$mode stage=$stage"
    echo '### Compile'
    "$PY" -m compileall -q . || return 1
    echo '### Offline regressions'
    "$PY" tests_offline.py || return 1
    "$PY" -m unittest -v test_nvidia_pipeline || return 1
    echo '### Inspect fictional documents (zero model API calls)'
    "$PY" main.py inspect || return 1
    "$PY" -m pip check || return 1
    if [ "$mode" = benchmark ]; then
        echo '### Exact-model NVIDIA benchmark (live API quota)'
        "$PY" nvidia_benchmark.py --stage "$stage"
    elif [ "$mode" = legacy ]; then
        echo '### Historical provider experiments (explicit opt-in)'
        "$PY" ci_test.py
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
