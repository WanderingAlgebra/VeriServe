"""Read and validate saved trajectories/features; no collection or fitting."""
from __future__ import annotations
import hashlib
import json
import time
from collections import Counter
import numpy as np
from veriserve_research import ROOT
from ..artifacts.io import atomic_json, atomic_bytes, read_json, stable_hash, digest, event, git, safe_error


def record_path(run, split, uid):
    return run / "records" / split / (hashlib.sha256(uid.encode()).hexdigest()[:20] + ".json")


def valid_features(path, record):
    files = record.get("feature_files", [])
    if not files:
        return False
    try:
        for entry in files:
            p = path.parent / entry["name"]
            if digest(p) != entry["sha256"]:
                return False
            with np.load(p, allow_pickle=False) as arrays:
                if not np.isfinite(arrays["hidden"]).all():
                    return False
        return True
    except (OSError, ValueError, EOFError):
        return False


def read_record(path, run, *, recover=True):
    if not path.exists():
        return None
    try:
        value = read_json(path)
        if (not isinstance(value, dict) or not {"prompt_ids", "generated_ids", "token_sha256", "stage"} <= value.keys()
                or any(not isinstance(value[key], list) or not value[key]
                       or any(type(token) is not int or token < 0 for token in value[key])
                       for key in ("prompt_ids", "generated_ids"))):
            raise ValueError("Invalid original token record structure")
    except (ValueError, OSError) as exc:
        if not recover:
            raise ValueError(f"Invalid original token JSON {path}; analysis leaves source files unchanged") from exc
        # Never replace original tokens or good features because a JSON became unreadable.
        event(run, "CORRUPT_JSON", path=str(path.relative_to(run)), error=safe_error(exc))
        atomic_bytes(path.with_suffix(f".corrupt-{time.time_ns()}"), path.read_bytes())
        try:
            restored = json.loads(git("show", f"HEAD:{path.relative_to(ROOT)}"))
            if stable_hash({"prompt_ids": restored["prompt_ids"], "generated_ids": restored["generated_ids"]}) != restored["token_sha256"]:
                raise ValueError("Committed token checksum mismatch")
        except (RuntimeError, ValueError, KeyError):
            raise ValueError(f"Unreadable original token JSON {path}; preserved files, recover from a valid backup") from exc
        atomic_json(path, restored)
        event(run, "JSON_RECOVERED_FROM_GIT", path=str(path.relative_to(run)))
        value = restored
    if value:
        if stable_hash({"prompt_ids": value["prompt_ids"], "generated_ids": value["generated_ids"]}) != value["token_sha256"]:
            raise ValueError(f"Token checksum mismatch in {path}; preserve and inspect manually")
    return value


def load_examples(run, manifest, split):
    examples = []
    counts = Counter({key: 0 for key in ("generated", "completed", "correct", "wrong", "truncated",
                                       "unscorable", "no_steps", "format_error_trajectories", "features_missing_or_corrupt")})
    counts["planned"] = len(manifest["splits"][split])
    for row in manifest["splits"][split]:
        path = record_path(run, split, row["unique_id"])
        record = read_record(path, run, recover=False)
        if not record:
            counts["not_generated"] += 1
            continue
        counts["generated"] += 1
        counts["completed"] += record["completion"] == "EOS"
        counts["truncated"] += record["completion"] in {"LENGTH_TRUNCATED", "CONTEXT_LIMIT"}
        if record.get("label") is None:
            counts["unscorable"] += 1
            counts[record.get("exclusion") or "not_scored"] += 1
        else:
            counts["correct" if record["label"] == 0 else "wrong"] += 1
        counts["no_steps"] += not record.get("boundaries", {}).get("steps", [])
        counts["format_error_trajectories"] += bool(record.get("boundaries", {}).get("format_errors"))
        if not valid_features(path, record):
            counts["features_missing_or_corrupt"] += 1
            continue
        if record.get("label") is None:
            continue
        parts, layers = [], []
        for entry in record["feature_files"]:
            with np.load(path.parent / entry["name"], allow_pickle=False) as data:
                parts.append(data["hidden"])
                layers.extend(data["layers"].tolist())
        hidden = np.concatenate(parts, axis=0)
        if layers != list(range(1, len(layers) + 1)):
            raise ValueError("Feature layer indices are not contiguous hidden_states[1..L]")
        steps = record["boundaries"]["steps"]
        expected_positions = [len(record["prompt_ids"]) + s["end_token_index"] for s in steps]
        if record["boundaries"]["end"]:
            expected_positions.append(len(record["prompt_ids"]) + record["boundaries"]["end"]["end_token_index"])
        with np.load(path.parent / record["feature_files"][0]["name"], allow_pickle=False) as data:
            if data["positions"].tolist() != expected_positions:
                raise ValueError("Saved feature positions differ from token metadata")
        if hidden.shape[1] != len(expected_positions):
            raise ValueError("Feature/position count differs")
        examples.append({"id": row["unique_id"], "y": record["label"], "steps": steps,
                         "hidden": hidden, "has_end": record["boundaries"]["end"] is not None,
                         "record": record})
    counts["analyzable_end"] = sum(x["has_end"] for x in examples)
    counts["analyzable_intermediate"] = sum(bool(x["steps"]) for x in examples)
    return examples, dict(counts)
