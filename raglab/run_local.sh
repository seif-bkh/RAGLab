#!/usr/bin/env bash
# raglab/run_local.sh — ONE command to run RAGLab on your own machine.
#
# The service and the front are two processes (the front is a pure REST client:
# see local_front.py's docstring), which is correct but means two terminals and
# a "did the service come up yet?" wait. This wrapper does that dance: it starts
# the service, waits until /health answers, runs the front action you asked for,
# and stops the service again (unless you tell it not to).
#
#   ./raglab/run_local.sh                    start the service, open the front menu, stop on exit
#   ./raglab/run_local.sh --status           doctor report (profile, index state, keys), no prompts
#   ./raglab/run_local.sh --ingest           build the index for this profile (first run), then stop
#   ./raglab/run_local.sh --ingest --reset   rebuild it (chunking changed / index is stale)
#   ./raglab/run_local.sh --ask "ما هي المرابحة؟"
#   ./raglab/run_local.sh --search "What is Murabaha?" -k 5
#   ./raglab/run_local.sh --interactive      straight into the chat REPL (the `front chat>` prompt)
#   ./raglab/run_local.sh --smoke            state-aware endpoint suite (keyless/empty/stale/ready)
#   ./raglab/run_local.sh --no-start         use a service that is ALREADY running on the port
#   ./raglab/run_local.sh --keep             leave the service running when the front exits
#   ./raglab/run_local.sh --port 8100        another port (default 8000, or RAGLAB_PORT)
#   ./raglab/run_local.sh --host 0.0.0.0     bind address for the service (default 127.0.0.1)
#   ./raglab/run_local.sh --keys             set/replace the keys in raglab/.env (hidden prompt,
#                                            nothing in shell history) — no service is started
#   ./raglab/run_local.sh --keys NVIDIA_API_KEY        just one key
#
# Your shell does not have to be in the repo. This finds the clone wherever it
# lives under ~ (guarding the empty result: a bare `cd ""` would silently stay
# put) and then runs the command of your choice:
#   R="$(git rev-parse --show-toplevel 2>/dev/null || find ~ -maxdepth 5 -type d -name .git -ipath '*raglab*' -printf '%h\n' 2>/dev/null | head -1)";
#   [ -n "$R" ] && cd "$R" && ./raglab/run_local.sh --keys
#
# RAGLAB_PYTHON=/path/to/python overrides the interpreter (a venv somewhere
# else, or a CI run where deps live in the system python).
#
# Anything this script does not recognize is passed to local_front.py unchanged,
# so every flag documented by `python local_front.py --help` works here.
# Service logs: raglab/logs/service_<port>.log (tail -f to watch it).
#
# Setup, once (see AGENTS.md §9):
#   python3 -m venv raglab/.venv
#   raglab/.venv/bin/pip install -r raglab/requirements-benchmark.txt -r raglab/requirements-service.txt
#   cp raglab/.env.example raglab/.env     # then paste NVIDIA_API_KEY / XKIRO_API_KEY / GOOGLE_API_KEY
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"       # -> .../raglab
PORT="${RAGLAB_PORT:-8000}"
HOST="127.0.0.1"
START=1
KEEP=0
KEYS=0
FRONT_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="${2:?--port needs a number}"; shift 2 ;;
        --host) HOST="${2:?--host needs an address}"; shift 2 ;;
        --no-start) START=0; shift ;;
        --keep) KEEP=1; shift ;;
        --keys) KEYS=1; shift ;;
        -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
        *) FRONT_ARGS+=("$1"); shift ;;
    esac
done

# The venv README.md/local.sh create first (raglab/.venv), then the root one.
VENV=""
for candidate in "$HERE/.venv" "$HERE/../.venv"; do
    [ -x "$candidate/bin/python" ] && VENV="$candidate" && break
done
PY="${RAGLAB_PYTHON:-}"
[ -z "$PY" ] && [ -n "$VENV" ] && PY="$VENV/bin/python"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    echo "[run] no virtualenv found. Set it up once (from the repo root):"
    echo "[run]   python3 -m venv raglab/.venv"
    echo "[run]   raglab/.venv/bin/pip install -r raglab/requirements-benchmark.txt -r raglab/requirements-service.txt"
    echo "[run] (or point RAGLAB_PYTHON at an existing interpreter)"
    exit 2
fi
# Key setup needs the interpreter, not the service — and must not start one.
if [ "$KEYS" = "1" ]; then
    "$PY" "$HERE/set_keys.py" "${FRONT_ARGS[@]}"
    exit $?
fi

BASE_URL="http://127.0.0.1:${PORT}"

health_ok() {
    "$PY" - "$BASE_URL" <<'PYCHECK' 2>/dev/null
import json, sys, urllib.error, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1] + "/health", timeout=3) as resp:
        sys.exit(0 if json.loads(resp.read().decode()).get("status") == "ok" else 1)
except Exception:
    sys.exit(1)
PYCHECK
}

STARTED_PID=""
cleanup() {
    if [ -n "$STARTED_PID" ] && [ "$KEEP" != "1" ]; then
        kill "$STARTED_PID" 2>/dev/null || true
        wait "$STARTED_PID" 2>/dev/null || true
        echo "[run] service stopped (pid $STARTED_PID)"
    elif [ -n "$STARTED_PID" ]; then
        echo "[run] service left running (pid $STARTED_PID) on $BASE_URL — stop it with: kill $STARTED_PID"
    fi
}
trap cleanup EXIT

if health_ok; then
    echo "[run] a RAGLab service already answers on $BASE_URL — using it (nothing will be stopped)"
elif [ "$START" = "1" ]; then
    mkdir -p "$HERE/logs"
    LOG="$HERE/logs/service_${PORT}.log"
    echo "[run] starting the service on $BASE_URL (log: $LOG)"
    ( cd "$HERE" && exec "$PY" -m uvicorn service:app --host "$HOST" --port "$PORT" ) \
        > "$LOG" 2>&1 &
    STARTED_PID=$!
    for _ in $(seq 1 60); do
        health_ok && break
        if ! kill -0 "$STARTED_PID" 2>/dev/null; then
            echo "[run] the service exited during startup — last lines of $LOG:"
            tail -20 "$LOG"
            exit 2
        fi
        sleep 0.5
    done
    if ! health_ok; then
        echo "[run] the service did not answer /health within 30s — last lines of $LOG:"
        tail -20 "$LOG"
        exit 2
    fi
    echo "[run] service is up"
else
    echo "[run] --no-start given but nothing answers on $BASE_URL."
    echo "[run] start it yourself:  cd $HERE && $PY -m uvicorn service:app --host 0.0.0.0 --port $PORT"
    exit 2
fi

# No front flag: open the console menu (same 13 menus as app.py, over REST).
"$PY" "$HERE/local_front.py" --base-url "$BASE_URL" "${FRONT_ARGS[@]}"
