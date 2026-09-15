"""The service's owned document store — the target of the gateway feed.

The gateway owns a document store and pushes documents to the active agent;
this module is RAGLab's receiving side. Pushed documents are stored as plain
files in one directory (RAGLAB_DOCUMENTS_DIR) with a `.meta.json` sidecar
each, under a namespaced filename (`pushed-<id><ext>`) so they can never
collide with the repo's own corpus files. The directory is appended to the
profile's data_dirs by the service, so every existing path (ingest, inspect,
chunks/search) sees pushed documents without special-casing.

Versioning is content-based: pushing the same id with identical bytes is a
no-op (idempotent); different bytes bumps the version and leaves the doc
"stale" (old chunks still serve) until the next ingest rebuilds the index.
Deletion removes the file AND purges the document's chunks from every local
collection, so the index stays truthful without a full rebuild.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

# ids are filename components in the store: one safe charset, no traversal
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
# every stored file is namespaced so it cannot collide with repo corpus files
PREFIX = "pushed-"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def valid_id(raw) -> str:
    """A safe document id (also used as the filename stem) or raise."""
    if not isinstance(raw, str):
        raise ValueError("document id must be a string")
    doc_id = raw.strip()
    if not ID_RE.match(doc_id):
        raise ValueError(
            "document id must match [A-Za-z0-9][A-Za-z0-9._-]{0,79} "
            "(no paths, no spaces)")
    if doc_id in {".", ".."}:
        raise ValueError("document id cannot be '.' or '..'")
    return doc_id


def extension_of(filename: str) -> str:
    """The lowercased extension (with dot) of a filename."""
    return Path(filename).suffix.lower()


class DocumentStore:
    """Files + sidecars in one directory; thread-safe; atomic writes."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()   # reentrant: save() calls find() under the lock

    # -- paths ----------------------------------------------------------------

    @staticmethod
    def stored_name(doc_id: str, ext: str) -> str:
        return f"{PREFIX}{doc_id}{ext}"

    def _file(self, stored: str) -> Path:
        return self.root / stored

    def _sidecar(self, stored: str) -> Path:
        return self.root / (stored + ".meta.json")

    # -- reads ----------------------------------------------------------------

    def _read_record(self, sidecar: Path) -> dict | None:
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or "id" not in data:
            return None
        return data

    def find(self, doc_id: str) -> dict | None:
        """The record for `doc_id`, or None (also matches by stored stem)."""
        with self.lock:
            for sidecar in self.root.glob(PREFIX + "*.meta.json"):
                record = self._read_record(sidecar)
                if record and record.get("id") == doc_id:
                    return record
        return None

    def list(self) -> list[dict]:
        with self.lock:
            records = [self._read_record(sidecar)
                       for sidecar in sorted(self.root.glob(PREFIX + "*.meta.json"))]
        return [record for record in records if record]

    # -- writes ---------------------------------------------------------------

    def save(self, doc_id: str, filename: str, content: bytes) -> tuple[dict, str]:
        """Store `content` as document `doc_id`.

        Returns (record, action) with action in {"created", "replaced",
        "unchanged"} — identical bytes are a no-op; different bytes bump the
        version (the old index chunks keep serving until the next ingest,
        which the status computation surfaces as "stale").
        """
        sha = hashlib.sha256(content).hexdigest()
        ext = extension_of(filename)
        stored = self.stored_name(doc_id, ext)
        with self.lock:
            existing = self.find(doc_id)
            if existing:
                if existing.get("sha256") == sha and existing.get("stored_as") == stored:
                    return dict(existing), "unchanged"
                # a different extension means the old file must go too
                old = self._file(existing.get("stored_as", ""))
                if old.is_file() and existing.get("stored_as") != stored:
                    old.unlink(missing_ok=True)
                    self._sidecar(existing["stored_as"]).unlink(missing_ok=True)
                record = {**existing, "filename": filename, "stored_as": stored,
                          "bytes": len(content), "sha256": sha,
                          "version": int(existing.get("version", 1)) + 1,
                          "updated_at": _now()}
                action = "replaced"
            else:
                record = {"id": doc_id, "filename": filename, "stored_as": stored,
                          "bytes": len(content), "sha256": sha, "version": 1,
                          "received_at": _now(), "updated_at": _now()}
                action = "created"
            tmp = self._file(stored + ".tmp")
            tmp.write_bytes(content)
            tmp.replace(self._file(stored))
            self._sidecar(stored).write_text(
                json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
            return dict(record), action

    def delete(self, doc_id: str) -> dict | None:
        """Remove the document and its sidecar; returns the record deleted."""
        with self.lock:
            record = self.find(doc_id)
            if not record:
                return None
            stored = record.get("stored_as", "")
            self._file(stored).unlink(missing_ok=True)
            self._sidecar(stored).unlink(missing_ok=True)
            return record

    # -- facts for the API layer ------------------------------------------------

    @staticmethod
    def public_row(record: dict, *, status: str, chunks: int) -> dict:
        """The API row: storage facts + index status; the full hash is internal."""
        return {"id": record["id"], "filename": record["filename"],
                "stored_as": record["stored_as"], "bytes": record["bytes"],
                "sha256_16": record["sha256"][:16], "version": record["version"],
                "received_at": record["received_at"],
                "updated_at": record["updated_at"],
                "status": status, "chunks_in_index": chunks}
