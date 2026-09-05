#!/usr/bin/env bash
# run_tests.sh — run the whole RAGLab test suite locally, capture EVERYTHING
# into ONE log file, and push that log to GitHub so it can be checked
# afterwards (the file lands in the repo: raglab/logs/<timestamp>.log and a
# copy at raglab/logs/latest.log on the current branch).
#
# Usage:
#   ./run_tests.sh                          # full suite (needs .env provider key)
#   ./run_tests.sh --provider huggingface   # full suite with the LOCAL model
#                                           #   (free, no key, no quota)
#   ./run_tests.sh --real-only              # skip fictional legs (save quota)
#   ./run_tests.sh --offline                # compile + unit + inspect, zero API
#   ./run_tests.sh --no-push                # keep the log local only
#
# The tests' exit code is preserved, and the log is pushed even when tests
# fail (the summary line says EXIT=<code>). The API key is NEVER written to
# the log (only "present/absent").
set -uo pipefail

cd "$(dirname "$0")"
REPO="$(cd .. && pwd)"
BRANCH="$(cd "$REPO" && git branch --show-current 2>/dev/null || echo UNKNOWN)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/test_run_${STAMP}.log"
LATEST="$LOG_DIR/latest.log"

mode="full"
provider="${EMBEDDING_PROVIDER:-}"
real_only=0
push=1

while [ $# -gt 0 ]; do
    case "$1" in
        --offline) mode="offline" ;;
        --real-only) real_only=1 ;;
        --no-push) push=0 ;;
        --provider)
            provider="${2:?--provider needs a value (gemini|huggingface|openai|cohere|voyage)}"
            shift ;;
        -h|--help) sed -n '1,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see header comments)"; exit 2 ;;
    esac
    shift
done

[ -n "$provider" ] && export EMBEDDING_PROVIDER="$provider"

PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

# --- header (values only; the key itself is never logged) -------------------
{
    echo "=== RAGLab test run $(date -u +%FT%TZ) ==="
    echo "repo     : $REPO"
    echo "branch   : $BRANCH"
    echo "commit   : $(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || echo '?')"
    echo "python   : $("$PY" --version 2>&1)"
    echo "provider : ${EMBEDDING_PROVIDER:-gemini}"
    echo "model    : $("$PY" -c 'import config; print(config.active_embedding_model())' 2>/dev/null || echo '?')"
    if [ -f .env ] && grep -q '^[A-Z_]*API_KEY=..*' .env; then
        echo "api_key  : present (value not logged)"
    else
        echo "api_key  : absent (offline/offline-API steps only)" 
    fi
    echo "mode     : $mode   real_only=$real_only   push=$push"
    echo "log      : $LOG"
    echo "-------------------------------------------------------------------"
    echo
} | tee "$LOG"

# --- the suite --------------------------------------------------------------
run_all() {
    echo "### [1/4] py_compile — all modules"
    "$PY" -m py_compile config.py loader.py chunker.py embedder.py store.py \
        evaluate.py answer.py main.py ci_test.py translate.py tests_offline.py \
        || return 1

    echo "### [2/4] unit tests (offline, no API)"
    "$PY" tests_offline.py || return 1

    echo "### [3/4] inspect — load + chunk the corpus (offline)"
    "$PY" main.py inspect || return 1

    echo "### [4/4] ci_test.py — full pipeline integration harness"
    [ "$mode" = "offline" ] && { echo "(--offline: skipping ci_test.py)"; return 0; }
    [ "$real_only" = 1 ] && export RAGLAB_CI_REAL_ONLY=1
    "$PY" ci_test.py
}

set +e
run_all
status=$?
{
    echo
    echo "-------------------------------------------------------------------"
    echo "EXIT=$status"
    echo "=== end of RAGLab test run ==="
} | tee -a "$LOG"
set -e

# --- push the log to GitHub -------------------------------------------------
cp "$LOG" "$LATEST"
if [ "$push" = 1 ]; then
    echo "### pushing $LOG to origin/$BRANCH (tests continued: EXIT=$status)"
    (
        cd "$REPO"
        git add "raglab/logs/$(basename "$LOG")" raglab/logs/latest.log 2>/dev/null
        git -c user.name="RAGLab test runner" -c user.email="raglab@local" \
            commit -q -m "test log: $(date -u +%FT%TZ) mode=$mode provider=${EMBEDDING_PROVIDER:-gemini} exit=$status" \
            && echo "committed log: $(git rev-parse --short HEAD)" \
            || echo "nothing to commit (log unchanged?)"
        git push origin "$BRANCH" \
            && echo "pushed to origin/$BRANCH" \
            || echo "PUSH FAILED — the log is committed locally; check git auth"
    )
else
    echo "(--no-push: log kept locally at $LOG)"
fi

exit "$status"
