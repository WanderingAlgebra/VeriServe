"""Atomic artifacts and ordinary Git commands shared by the experiments."""
from __future__ import annotations
import csv
import hashlib
import io
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from veriserve_research import ROOT
from .legacy import atomic_json as atomic_json, read_json as read_json, stable_hash as stable_hash, append_jsonl


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_npz(path, **arrays):
    b = io.BytesIO()
    np.savez_compressed(b, **arrays)
    atomic_bytes(path, b.getvalue())


def safe_error(error):
    # Remote failures may echo a credential-bearing URL; never persist that URL.
    return re.sub(r"(?:https?|ssh)://\S+", "[redacted URL]", str(error))


def git(*args, cwd=ROOT, env=None):
    result = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True,
                            timeout=600)
    if result.returncode:
        raise RuntimeError(safe_error(result.stderr))
    return result.stdout.strip()


def event(run, kind, **values):
    append_jsonl(Path(run) / "events.jsonl", {"time": now(), "event": kind, **values})


def write_csv(path, rows, fields):
    b = io.StringIO(newline="")
    writer = csv.DictWriter(b, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, b.getvalue().encode())
