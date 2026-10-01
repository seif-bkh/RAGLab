#!/usr/bin/env python3
"""Request audit log — Phase 6, item 3 (RAGLAB_ROADMAP.md).

A per-request audit trail for the answer/search endpoints: trace id,
UTC timestamp, the PII-scrubbed question, status, evidence state (when the
sufficiency fields are on), model, citation counts and latency. Entries are
JSONL — one line per request, newest last.

Governance:
- RETENTION (max entries kept) and the entry schema are DECLARED REVIEW
  DATA; the default keeps the log bounded on disk without a cron.
- PII: the caller scrubs BEFORE recording (service.scrub_pii) — this module
  never sees raw identifiers by contract, and a test enforces the marker.
- Storage lives under the service's RESULTS_DIR (gitignored by design), so
  the trail never lands in the repository.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

# DECLARED (REVIEW DATA): how many entries the on-disk trail keeps.
RETENTION_MAX_ENTRIES = 500

ENTRY_FIELDS = ("trace_id", "ts", "endpoint", "question", "language",
                "status", "reason", "evidence_status", "model", "claims",
                "sources", "retrieved", "validation_ok", "seconds", "error")


def new_trace_id() -> str:
    """Short, sortable-enough trace identifier for one request."""
    return uuid.uuid4().hex[:12]


def record(path: Path, entry: dict) -> dict:
    """Append one entry (JSONL) and enforce retention. Returns the entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: entry.get(k) for k in ENTRY_FIELDS if k in entry}
    row.setdefault("trace_id", new_trace_id())
    row.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    _trim(path)
    return row


def _trim(path: Path) -> None:
    """Keep at most RETENTION_MAX_ENTRIES entries (drop the oldest)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) <= RETENTION_MAX_ENTRIES:
        return
    kept = lines[-RETENTION_MAX_ENTRIES:]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
    tmp.replace(path)


def recent(path: Path, limit: int = 20) -> list[dict]:
    """The newest `limit` entries, newest first."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a torn last line never breaks the reader
        if len(out) >= limit:
            break
    return out
