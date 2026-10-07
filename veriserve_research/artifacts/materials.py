"""Material relocation and corruption recovery for execution artifacts."""
from __future__ import annotations
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from veriserve_research import ROOT
from .io import read_json, atomic_bytes, digest, event, now
HERE = ROOT / "stages/stage3"


def material_key(path, *, root=None):
    """Store repository paths independently of the machine's checkout location."""
    root = Path(root) if root is not None else ROOT
    path = Path(path).resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        # Synthetic self-checks also use material directories outside the checkout.
        return str(path)


def material_path(key, *, root=None):
    """Relocate old stage material addresses without rewriting signed records."""
    root = Path(root) if root is not None else ROOT
    path = Path(key)
    if ".." in path.parts:
        raise ValueError("Material paths cannot traverse parent directories")
    if not path.is_absolute():
        return root / path
    for i, part in enumerate(path.parts[:-1]):
        if part == "stages" and path.parts[i + 1] in ("stage2", "stage3"):
            return root.joinpath(*path.parts[i:])
    return path


def validate_materials(record, *, root=None):
    for field in ("source_material_hashes", "material_hashes", "record_hashes", "probe_hashes"):
        for key, expected in record.get(field, {}).items():
            path = material_path(key, root=root)
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f"Missing or changed source material: {key}")


def archive_diagnostic(run, name):
    """Keep prior hardware/resource diagnostics before a later session replaces them."""
    if name not in ("runtime_environment.json", "capacity_check.json"):
        raise ValueError("Only session diagnostic files can be archived here")
    run = Path(run)
    path = run / name
    if not path.exists():
        return None
    archive = run / "session_checks" / f"{time.time_ns()}-{name}"
    atomic_bytes(archive, path.read_bytes())
    event(run, "SESSION_DIAGNOSTIC_ARCHIVED", original=name,
          archive=str(archive.relative_to(run)), sha256=digest(archive))
    return archive


def safe_read_json(path, default=None):
    """Preserve corrupt bytes; restore a committed copy, or stop without regeneration."""
    path = Path(path)
    try:
        return read_json(path, default)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        preserved = path.with_name(f"{path.name}.corrupt-{now().replace(':', '')}")
        shutil.copyfile(path, preserved)
        run = next((p for p in path.parents if (p / "manifest.json").exists()
                    and p.parent == HERE / "runs"), None)
        commit = None
        if run is not None and path.name != "backup.json":
            try:
                status = read_json(run / "backup.json", {})
                commit = status.get("last_successful_commit") or (
                    status.get("commit") if status.get("status") == "PUSHED" else None)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        if commit and re.fullmatch(r"[0-9a-f]{40}", commit):
            relative = str(path.relative_to(ROOT))
            result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT,
                                    capture_output=True, timeout=30)
            if result.returncode == 0:
                try:
                    value = json.loads(result.stdout)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
                else:
                    atomic_bytes(path, result.stdout)
                    event(run, "CORRUPT_FILE_RESTORED", path=relative, commit=commit,
                          preserved=str(preserved.relative_to(ROOT)))
                    return value
        raise RuntimeError(f"Corrupt JSON retained at {preserved}; no valid committed copy. "
                           "Stop and recover the original record before continuing.") from exc


import numpy as np


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def group(value, boundaries):
    if not finite(value) or not boundaries:
        return None
    return "low" if value <= boundaries[0] else "mid" if value <= boundaries[1] else "high"


def exclusion(arm):
    if not arm or arm.get("status") != "COMPLETE":
        return "INFRASTRUCTURE" if arm and arm.get("status", "").startswith("INFRA") else "PENDING"
    reason = (arm.get("grading") or {}).get("exclusion") or ""
    if reason.startswith("GOLD_"):
        return "GOLD_DATA" if reason == "GOLD_UNPARSEABLE" else "SCORING_ERROR"
    if reason in {"PREDICTION_TIMEOUT", "VERIFY_TIMEOUT", "VERIFY_ERROR"}:
        return "SCORING_ERROR"
    return None if arm.get("Y") in (0, 1) else "SCORING_UNAVAILABLE"


def backend(arm):
    info = (arm or {}).get("backend") or {}
    return json.dumps(info, ensure_ascii=False, sort_keys=True) if info else "UNKNOWN_BACKEND"
