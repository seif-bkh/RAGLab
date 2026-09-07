#!/usr/bin/env bash
# raglab/chat.sh — ask questions about the four documents and get a cited answer back.
#
#   ./raglab/chat.sh                      readiness report, then the chat
#   ./raglab/chat.sh "ما رأس مال ..."     ask once and exit
#   ./raglab/chat.sh --ingest             first run: embed the corpus (NVIDIA calls, then cached)
#   ./raglab/chat.sh --ingest --reset     rebuild it: chunking changed, or the index is stale
#   ./raglab/chat.sh --show-context -k 8  "quel est le délai ..."
#   ./raglab/chat.sh --check              configuration + index state, no completion call
#
# Answering uses NVIDIA_API_KEY with nvidia/nemotron-3.5-lightning-30b-a3b, a free endpoint on
# build.nvidia.com. That is deliberately NOT the benchmark's answerer: the frozen harness pins xKiro
# Qwen and refuses any other model, so no score in this repo belongs to this one. Retrieval, the
# verbatim-citation check and the private/live-question refusal are the lab's own, unchanged. It reads its
# own collection (raglab_chat) at the plan-pinned 640-token chunks — 82% whole-document recall measured,
# against 11% at the app's 220 default — and a refusal prints the excerpts the model was handed, so
# 'not in the corpus' can be told apart from 'retrieval missed it'.
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)"
cd "$SELF"
# Two locations are supported because README.md creates raglab/.venv and local.sh creates the one at
# the repo root; the first one found wins.
VENV=""
for candidate in "$SELF/../.venv" "$SELF/.venv"; do
  [ -f "$candidate/bin/activate" ] && VENV="$candidate" && break
done
if [ -z "$VENV" ]; then
  echo "no virtualenv found in raglab/.venv or .venv - run: ./raglab/local.sh setup"
  exit 1
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"

case "${1:-}" in
  -h|--help) exec python3 chat.py --help ;;
  --check)   exec python3 chat.py --check ;;
esac

if [ "$#" -eq 0 ]; then
  # Report first, then chat: the report is what tells you an empty index or a missing key is coming,
  # instead of the first question failing three calls in.
  python3 chat.py --check || { echo; echo "fix the lines above, then re-run ./raglab/chat.sh"; exit 1; }
  echo
  echo "asking the corpus. Ctrl-D or :quit to leave, :help for commands."
  echo
  exec python3 chat.py
fi
exec python3 chat.py "$@"
