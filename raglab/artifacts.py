"""Atomic JSON artifacts; cache locks coordinate instances within one process.

Separate writer processes still need external coordination and are unsupported.
"""
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path

_CACHE_LOCKS = {}
_CACHE_LOCKS_GUARD = threading.Lock()


def cache_lock(path):
    key = str(Path(path).resolve())
    with _CACHE_LOCKS_GUARD:
        return _CACHE_LOCKS.setdefault(key, threading.RLock())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize before touching the old file; NaN/inf are never valid artifacts.
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
