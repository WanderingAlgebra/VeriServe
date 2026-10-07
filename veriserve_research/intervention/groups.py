"""Freeze dev risk groups before test; this code participates in execution identity."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
from ..artifacts.io import read_json, atomic_json, stable_hash
from ..artifacts.materials import finite, group, exclusion, backend
from ..artifacts.provenance import source_hashes
ARMS = ("NOW", "DELAY_2")
PHASES = ("pilot", "dev", "test")


def load_rows(run, manifest):
    snapshots, arms = {}, {}
    for path in sorted((run / "snapshots").rglob("*.json")):
        record = read_json(path)
        if record and record.get("unique_id"):
            key = record.get("phase"), record["unique_id"]
            if key in snapshots:
                raise ValueError(f"Duplicate authoritative snapshot: {key}")
            snapshots[key] = record
    for path in sorted((run / "arms").rglob("*.json")):
        record = read_json(path)
        if record and record.get("unique_id") and record.get("arm") in ARMS:
            key = record.get("phase"), record["unique_id"], record["arm"]
            if key in arms:
                raise ValueError(f"Duplicate authoritative arm: {key}")
            arms[key] = record
    rows = []
    for phase in PHASES:
        for item in manifest["splits"].get(phase, []):
            uid = item["unique_id"]
            snapshot = snapshots.get((phase, uid), {})
            pair = {a: arms.get((phase, uid, a)) for a in ARMS}
            eligible = snapshot.get("status") == "ELIGIBLE"
            exclusions = {a: exclusion(pair[a]) for a in ARMS}
            complete = eligible and all(pair[a] and pair[a].get("status") == "COMPLETE" for a in ARMS)
            times = {a: ((pair[a] or {}).get("timing") or {}).get("T_postfork_wall") for a in ARMS}
            same_backend = complete and backend(pair["NOW"]) != "UNKNOWN_BACKEND" and backend(pair["NOW"]) == backend(pair["DELAY_2"])
            valid_time = same_backend and all(finite(times[a]) and times[a] >= 0 for a in ARMS)
            rows.append({"unique_id": uid, "phase": phase, "snapshot": snapshot, "arms": pair,
                         "eligible": eligible, "complete": complete, "exclusions": exclusions,
                         "quality_valid": eligible and all(exclusions[a] is None for a in ARMS),
                         "time_valid": bool(valid_time),
                         "backend": backend(pair["NOW"]) if same_backend else None})
    return rows


def freeze_groups(cfg, run, manifest=None):
    """Freeze dev quantiles using only eligible snapshot scores, never action outcomes."""
    run = Path(run)
    manifest = manifest or read_json(run / "manifest.json")
    dev = [r for r in load_rows(run, manifest) if r["phase"] == "dev"]
    eligible = [r for r in dev if r["eligible"]]
    scores = [{"unique_id": r["unique_id"], "q_B": r["snapshot"].get("q_B"), "q_C": r["snapshot"].get("q_C")} for r in eligible]
    score_hash = stable_hash(scores)
    previous = read_json(run / "dev_groups.json")
    if previous and previous.get("status") == "FROZEN":
        if previous["score_hash"] != score_hash:
            raise ValueError("Frozen dev scores changed; use a new run_id")
        return previous
    reason = None
    if not dev or any(r["snapshot"].get("status") not in ("ELIGIBLE", "NO_ELIGIBLE_ANCHOR") or (r["eligible"] and not r["complete"]) for r in dev):
        reason = "DEV_NOT_COMPLETE"
    elif len(eligible) < 3:
        reason = "FEWER_THAN_THREE_ELIGIBLE_DEV_ANCHORS"
    boundaries = {}
    for method in ("B", "C"):
        values = [s[f"q_{method}"] for s in scores]
        if not values or not all(finite(v) and 0 <= v <= 1 for v in values):
            reason = reason or f"INVALID_OR_MISSING_DEV_Q_{method}"
            boundaries[method] = None
        else:
            thresholds = np.quantile(values, [1 / 3, 2 / 3]).tolist()
            boundaries[method] = thresholds
            if thresholds[0] >= thresholds[1]:
                reason = reason or f"NON_DISTINCT_DEV_BOUNDARIES_{method}"
    result = {"status": "INSUFFICIENT_DEV" if reason else "FROZEN", "reason": reason,
              "source": "eligible_dev_snapshot_scores_only", "quantile_method": "numpy linear",
              "rule": "q <= first: low; q <= second: mid; otherwise: high",
              "planned_dev": len(dev), "eligible_dev": len(eligible), "score_hash": score_hash,
              "boundaries": boundaries, "config_hash": manifest.get("config_hash"),
              "source_hashes": source_hashes(),
              "dev_group_counts": {m: dict(Counter(group(s[f"q_{m}"], boundaries.get(m)) for s in scores)) for m in ("B", "C")}}
    atomic_json(run / "dev_groups.json", result)
    if reason:
        from ..analysis.fixed_step import analyze
        analyze(cfg, run, manifest)
        raise ValueError(f"Cannot freeze risk groups: {reason}")
    return result
