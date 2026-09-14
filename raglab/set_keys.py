#!/usr/bin/env python3
"""Set provider keys in raglab/.env — hidden prompt, nothing in shell history.

Run it through the one-command wrapper (works from anywhere, see AGENTS.md §9):

    ./raglab/run_local.sh --keys
    ./raglab/run_local.sh --keys NVIDIA_API_KEY      # just one of the three

or directly from this directory:  .venv/bin/python set_keys.py

Why this exists: `export NVIDIA_API_KEY=nvapi-…` leaves the secret in the shell
history and in `ps` output, and a visible prompt puts it on the screen. getpass
reads without echo, and profiles.write_env_assignment rewrites the single
matching line so the rest of .env (and its comments) survives untouched.

Bad pastes are refused instead of stored: a stray trailing space, a chat
client's smart quotes, a placeholder like `your-key-here`. That matters because
a key with whitespace in it fails later with a confusing 401 from the provider.
Nothing is ever echoed back — the confirmation prints profiles.masked(), the
same first-8-characters form the service and the front use.
"""

from __future__ import annotations

import getpass
import sys

from profiles import ENV_PATH, looks_like_placeholder, masked, write_env_assignment

KEY_NAMES = ("NVIDIA_API_KEY", "XKIRO_API_KEY", "GOOGLE_API_KEY")

# Characters a chat client or an editor likes to slip into a paste. A real key
# is ASCII with no whitespace, so any of these means "copy it again, plain".
_QUOTES = "\"'`“”‘’«»"
_BOM = "\ufeff\u200b\u00a0"


def paste_problem(value: str) -> str:
    """Why this pasted value is not usable as a key ('' = looks fine)."""
    if not value:
        return "empty"
    if looks_like_placeholder(value):
        return "looks like a placeholder, not a key"
    if any(ch.isspace() for ch in value):
        return "contains whitespace — a stray space or newline came with the paste"
    if any(ch in _QUOTES for ch in value):
        return "contains quote characters — paste the value without surrounding quotes"
    if any(ch in _BOM for ch in value):
        return "contains invisible characters (BOM/zero-width space)"
    if not value.isascii():
        return "contains non-ASCII characters — a chat client may have rewritten the dash"
    return ""


def store(name: str, value: str, writer=None, echo=print) -> bool:
    """Write one key through the .env writer. Returns True if it was stored."""
    value = (value or "").strip()
    problem = paste_problem(value)
    if problem:
        echo(f"[keys] {name}: NOT written — {problem}")
        return False
    (writer or write_env_assignment)(name, value)
    echo(f"[keys] {name}: written to {ENV_PATH} (starts with {masked(value)})")
    return True


def main(argv=None) -> int:
    names = list(argv if argv is not None else sys.argv[1:]) or list(KEY_NAMES)
    echo = print
    echo(f"[keys] writing to {ENV_PATH} — input is hidden, Enter skips a key, Ctrl-D stops")
    written, skipped = [], []
    for name in names:
        try:
            value = getpass.getpass(f"{name}: ")
        except (EOFError, KeyboardInterrupt):
            echo()
            skipped.extend(n for n in names if n not in written)   # this one included
            break
        (written if store(name, value, echo=echo) else skipped).append(name)
    echo("[keys] " + (f"written: {', '.join(written)}" if written else "nothing written")
         + (f" · skipped: {', '.join(skipped)}" if skipped else ""))
    echo("[keys] check what the service sees with:  ./raglab/run_local.sh --status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
