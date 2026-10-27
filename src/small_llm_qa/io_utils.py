"""Atomic artifacts and a resumable, strictly validated per-sample journal."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(path):
    path = Path(path)
    return canonical_hash({str(p.relative_to(path)).replace("\\", "/"): file_hash(p)
                           for p in sorted(path.rglob("*")) if p.is_file()})


def write_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if immutable and path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Immutable artifact differs: {path}; use a new run directory")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def read_journal(path, expected_ids):
    path = Path(path)
    if not path.exists():
        return {}
    contents = path.read_bytes()
    lines = contents.splitlines(keepends=True)
    found = {}
    valid_bytes = 0
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i == len(lines) - 1 and not line.endswith(b"\n"):
                path.with_suffix(".incomplete-tail").write_bytes(line)
                path.write_bytes(contents[:valid_bytes])
                break
            raise ValueError(f"Corrupt completed journal record at line {i+1}: {path}")
        sid = row["Sample_id"]
        if sid not in expected_ids or sid in found:
            raise ValueError(f"Duplicate or unexpected journal ID: {sid}")
        found[sid] = row
        valid_bytes += len(line)
    # A fully written JSON object without newline is valid; terminate it before append.
    if found and path.read_bytes() and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as f:
            f.write(b"\n")
    return found


def append_journal(path, record):
    with Path(path).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
