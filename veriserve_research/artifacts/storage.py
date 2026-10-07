"""Fixed-step manifests, diagnostics and frozen-backup gates."""
from __future__ import annotations
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from veriserve_research import ROOT
from .io import atomic_json, read_json, stable_hash, atomic_bytes, digest, git, now, event
from .provenance import SHA, environment, source_hashes, run_identity, manifest_version, require_current
from .materials import safe_read_json, archive_diagnostic
from .backup import backup_fixed_step as backup
HERE = ROOT / "stages/stage3"
STAGE2 = ROOT / "stages/stage2/runs/20261004-36fe9009f1f5"
INITIAL_VERIFIER_RESOLUTION = {
    "repo": "Qwen/Qwen2.5-Math-PRM-7B", "sha": "0610740060112df12585d00a1c5f4624d2f59051",
    "method": "hub_model_info", "stage1_recorded_revision_found": False,
    "revisioned_local_cache_found": False, "persisted_before_download": True,
}


def workingtree():
    # Artifact writes are excluded so recording the manifest does not dirty its own provenance.
    paths = [*source_hashes(), "stages/stage3/config.json", "stages/stage3/README.md",
             ".gitignore", ".gitattributes"]
    status = git("status", "--porcelain=v1", "--untracked-files=all", "--", *paths)
    return {"head_commit": git("rev-parse", "HEAD"), "source_clean": not bool(status),
            "source_status": status.splitlines(),
            "scope": "source files, config, documentation and LFS metadata; excludes run artifacts"}


def verifier_provenance(cfg, config_path=None):
    if cfg.get("verifier_revision_source"):
        return cfg["verifier_revision_source"]
    if (cfg["verifier"]["id"] == INITIAL_VERIFIER_RESOLUTION["repo"]
            and cfg["revisions"]["verifier"] == INITIAL_VERIFIER_RESOLUTION["sha"]):
        result = dict(INITIAL_VERIFIER_RESOLUTION)
        if config_path is not None:
            result["config_file_mtime_utc"] = datetime.fromtimestamp(
                Path(config_path).stat().st_mtime, timezone.utc).isoformat()
            result["timestamp_basis"] = "observed persisted config mtime; exact Hub response timestamp was not recorded"
        return result
    return {"method": "configured_exact_revision", "repo": cfg["verifier"]["id"],
            "sha": cfg["revisions"]["verifier"], "original_resolution_provenance": "unavailable"}


def save_json(path, value):
    atomic_json(Path(path), value)


def resolve_verifier(repo):
    """Use recorded stage 1 assets before a local revisioned cache or Hub resolution."""
    candidates = [ROOT / "stages/stage1/.cache/assets.json", ROOT / ".cache/assets.json",
                  Path("/root/autodl-tmp/VeriServe/stages/stage1/.cache/assets.json"),
                  Path("/root/autodl-tmp/VeriServe/.cache/assets.json")]
    candidates += sorted((ROOT / "stages/stage1").glob("**/manifest.json"))
    candidates += sorted((ROOT / "stages/stage1").glob("**/assets.json"))

    def revisions(value):
        found = set()
        if isinstance(value, dict):
            if value.get("id") == repo and SHA.fullmatch(str(value.get("revision", ""))):
                found.add(value["revision"])
            if isinstance(value.get("model_revisions"), dict):
                rev = value["model_revisions"].get("verifier")
                if SHA.fullmatch(str(rev)) and repo in json.dumps(value):
                    found.add(rev)
            for child in value.values():
                found.update(revisions(child))
        elif isinstance(value, list):
            for child in value:
                found.update(revisions(child))
        return found

    found = set()
    sources = []
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        try:
            values = revisions(read_json(path))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if values:
            found.update(values)
            sources.append(str(path))
    if len(found) > 1:
        raise RuntimeError(f"Conflicting recorded stage 1 PRM revisions in {sources}")
    if found:
        return found.pop(), {"method": "stage1_recorded_assets", "paths": sources}
    from huggingface_hub.constants import HF_HUB_CACHE
    cache = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    snapshots = sorted(p.name for p in (cache / "snapshots").glob("*")
                       if SHA.fullmatch(p.name))
    if len(snapshots) == 1:
        return snapshots[0], {"method": "local_revisioned_cache", "path": str(cache)}
    from huggingface_hub import HfApi
    sha = HfApi().model_info(repo, timeout=30).sha
    if not SHA.fullmatch(str(sha)):
        raise RuntimeError("Hub did not return an exact verifier SHA")
    return sha, {"method": "hub_model_info", "resolved_at": now()}


def probe_hashes(stage2):
    result = {}
    for name, dimensions in (("B", 3584), ("C", 2)):
        metadata = safe_read_json(stage2 / f"probe_{name}.json")
        if metadata.get("method") != name or metadata.get("layer") != (19 if name == "B" else None):
            raise ValueError(f"Unexpected fixed probe {name} metadata")
        with np.load(stage2 / f"probe_{name}.npz", allow_pickle=False) as weights:
            if set(weights.files) != {"mean", "scale", "w", "b"}:
                raise ValueError(f"Unexpected fixed probe {name} arrays")
            for key in ("mean", "scale", "w"):
                if weights[key].shape != (dimensions,) or not np.isfinite(weights[key]).all():
                    raise ValueError(f"Invalid probe {name} {key}")
            if (weights["scale"] <= 0).any() or float(weights["b"]) != metadata["b"]:
                raise ValueError(f"Probe {name} metadata/weights mismatch")
        for ext in ("json", "npz"):
            path = stage2 / f"probe_{name}.{ext}"
            result[str(path.relative_to(ROOT))] = digest(path)
    return result


def prepare(config_path, resume=True):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = safe_read_json(config_path)
    if not isinstance(cfg, dict):
        raise ValueError("Stage 3 config must be a JSON object")
    stage2 = ROOT / cfg["stage2_run"]
    original = safe_read_json(stage2 / "manifest.json")
    baseline = original["config"]
    for name in ("model", "prompt", "dataset", "dataset_split", "attention", "dtype",
                 "batch_size", "do_sample"):
        if cfg[name] != baseline[name]:
            raise ValueError(f"Stage 3 must keep original stage 2 {name}")
    revision = cfg.setdefault("revisions", {})
    for name in ("model", "tokenizer", "dataset"):
        if revision.get(name) != baseline["revisions"][name]:
            raise ValueError(f"Stage 3 must keep fixed stage 2 {name} SHA")
    if not revision.get("verifier"):
        revision["verifier"], provenance = resolve_verifier(cfg["verifier"]["id"])
        revision["verifier_tokenizer"] = revision["verifier"]
        cfg["verifier_revision_source"] = provenance
        atomic_json(config_path, cfg)  # Exact SHA is durable before downloads or run_id.
    if not revision.get("verifier_tokenizer"):
        revision["verifier_tokenizer"] = revision["verifier"]
        atomic_json(config_path, cfg)
    for name, sha in revision.items():
        if not SHA.fullmatch(str(sha)):
            raise ValueError(f"{name} revision must be an exact 40-character SHA")
    if revision["verifier_tokenizer"] != revision["verifier"]:
        raise ValueError("PRM model and tokenizer must use the same fixed SHA")
    unused = original["unused_ids"]
    if len(unused) != 190 or len(set(unused)) != 190:
        raise ValueError("Original stage 2 unused_ids must contain exactly 190 unique IDs")
    used = {row["unique_id"] for rows in original["splits"].values() for row in rows}
    if used & set(unused):
        raise ValueError("Stage 3 IDs intersect original stage 2 splits")
    expected_sizes = {"pilot": 20, "dev": 40, "test": 130}
    if cfg["split_sizes"] != expected_sizes:
        raise ValueError("Stage 3 splits must remain 20/40/130")
    config_hash = stable_hash(cfg)
    probes = probe_hashes(stage2)
    inputs = {"manifest": digest(stage2 / "manifest.json"), "probes": probes}
    run = HERE / "runs" / run_identity(cfg, "fixed-step", inputs)
    run.mkdir(parents=True, exist_ok=True)
    manifest = safe_read_json(run / "manifest.json")
    if manifest:
        require_current(manifest, "fixed-step")
        if manifest["config_hash"] != config_hash or manifest["probe_hashes"] != probes:
            raise ValueError("Existing manifest/config/probe mismatch; preserve and use a new run")
        if [r["unique_id"] for phase in expected_sizes for r in manifest["splits"][phase]] != unused:
            raise ValueError("Existing ordered split IDs changed")
        if (manifest["stage2_manifest_sha256"] != digest(stage2 / "manifest.json")
                or any(manifest["split_hashes"][phase] != stable_hash(manifest["splits"][phase])
                       for phase in expected_sizes)):
            raise ValueError("Original manifest or restored question fields changed")
        frozen = safe_read_json(run / "protocol_frozen.json", {})
        if frozen and frozen.get("source_hashes") != source_hashes():
            raise ValueError("Frozen source files changed; preserve old records and use a new run")
        if not frozen and not any((run / "snapshots").glob("**/*.json")):
            updated = {**manifest, "prepare_base_commit": manifest.get("prepare_base_commit", manifest["code_commit"]),
                       "source_hashes": source_hashes(), "environment": environment(),
                       "code_commit": git("rev-parse", "HEAD"), "workingtree": workingtree(),
                       "verifier_revision_source": verifier_provenance(cfg, config_path)}
            if updated != manifest:
                atomic_json(run / "manifest.json", updated)
                event(run, "PREPARE_PROVENANCE_REFRESHED", code_commit=updated["code_commit"],
                      prepare_base_commit=updated["prepare_base_commit"], source_hashes=updated["source_hashes"])
                manifest = updated
        return cfg, run, manifest
    from datasets import load_dataset
    rows = list(load_dataset(cfg["dataset"], split=cfg["dataset_split"],
                             revision=revision["dataset"]))
    by_id = {row["unique_id"]: row for row in rows}
    if len(by_id) != len(rows) or not set(unused) <= by_id.keys():
        raise ValueError("Fixed dataset has duplicate or missing unique_id values")
    splits, offset = {}, 0
    for phase, size in expected_sizes.items():
        splits[phase] = [by_id[uid] for uid in unused[offset:offset + size]]
        offset += size
    manifest = {**manifest_version("fixed-step"),
        "run_id": run.name, "created": now(), "config": cfg, "config_hash": config_hash,
        "revisions": revision, "splits": splits, "ordered_ids": unused,
        "split_hashes": {phase: stable_hash(values) for phase, values in splits.items()},
        "stage2_manifest_sha256": digest(stage2 / "manifest.json"),
        "probe_hashes": probes, "source_hashes": source_hashes(),
        "code_commit": git("rev-parse", "HEAD"), "prepare_base_commit": git("rev-parse", "HEAD"),
        "workingtree": workingtree(), "environment": environment(),
        "verifier_revision_source": verifier_provenance(cfg, config_path),
        "feature_extraction": cfg["feature_extraction"],
        "probe_interpretation": "risk of incorrect unintervened complete trajectory",
        "feedback_adaptation": "checkpoint_step=0: No reasoning steps have been accepted. Restart from Step 1.",
        "grading": original["grading"],
        "timing": "T_postfork_wall is follow-up processing time from the common snapshot; offline feature/probe and fork restoration are separate",
    }
    atomic_json(run / "manifest.json", manifest)
    event(run, "PREPARED", sizes=expected_sizes, source_hashes=manifest["source_hashes"])
    return cfg, run, manifest


def completed_problems(run):
    run = Path(run)
    counts = {phase: 0 for phase in ("pilot", "dev", "test")}
    for path in (run / "snapshots").glob("**/*.json"):
        snapshot = safe_read_json(path)
        phase = snapshot.get("phase")
        if phase not in counts:
            continue
        if snapshot.get("status") == "NO_ELIGIBLE_ANCHOR":
            counts[phase] += 1
            continue
        key = stable_hash(snapshot["unique_id"])[:20]
        arms = [safe_read_json(run / "arms" / f"{key}.{arm}.json", {})
                for arm in ("NOW", "DELAY_2")]
        if all(a.get("completed") or a.get("status") == "COMPLETED" for a in arms):
            counts[phase] += 1
    return counts


def ensure_frozen_backup(run, frozen):
    """A frozen file alone never proves it was pushed; persist pending state before retry."""
    run = Path(run)
    frozen_path, groups_path = run / "protocol_frozen.json", run / "dev_groups.json"
    if digest(groups_path) != frozen["dev_groups_sha256"]:
        raise RuntimeError("Frozen dev groups hash changed; preserve files and recover the original groups")
    status = safe_read_json(run / "backup.json", {})
    if (status.get("status") == "PUSHED"
            and status.get("protocol_frozen_sha256") == digest(frozen_path)
            and status.get("dev_groups_sha256") == frozen["dev_groups_sha256"]
            and status.get("source_hashes") == frozen["source_hashes"]
            and status.get("stage3_source_commit") == status.get("commit")):
        return status
    last = status.get("last_successful_commit") or (
        status.get("commit") if status.get("status") == "PUSHED" else None)
    atomic_json(run / "backup.json", {**status, "status": "PENDING", "time": now(),
                "reason": "frozen protocol must be pushed before test",
                "last_successful_commit": last})
    event(run, "FROZEN_BACKUP_PENDING", last_successful_commit=last,
          protocol_frozen_sha256=digest(frozen_path))
    return backup(run, "freeze dev groups and protocol before independent test")


def self_check():
    """One CPU-only check for storage trust boundaries, callable from the main self-check."""
    with tempfile.TemporaryDirectory(prefix="stage3-storage-check-") as tmp:
        path = Path(tmp) / "record.json"
        save_json(path, {"token_ids": [1, 2, 3]})
        assert safe_read_json(path)["token_ids"] == [1, 2, 3]
        path.write_text("{broken", encoding="utf-8")
        try:
            safe_read_json(path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Corrupt records must not become new generation requests")
        assert path.read_text() == "{broken"
        assert len(list(path.parent.glob("record.json.corrupt-*"))) == 1
        run = path.parent / "run"
        diagnostic = run / "runtime_environment.json"
        original_bytes = b'{ "gpu": "earlier GPU", "session": 1 }\n'
        atomic_bytes(diagnostic, original_bytes)
        archive = archive_diagnostic(run, diagnostic.name)
        assert diagnostic.read_bytes() == archive.read_bytes() == original_bytes
        save_json(diagnostic, {"gpu": "later GPU", "session": 2})
        assert archive.read_bytes() == original_bytes
        save_json(run / "dev_groups.json", {"B": [1, 2], "C": [1, 2]})
        frozen = {"dev_groups_sha256": digest(run / "dev_groups.json"), "source_hashes": {"code.py": "hash"}}
        save_json(run / "protocol_frozen.json", frozen)
        pushed = {"status": "PUSHED", "commit": "a" * 40, "stage3_source_commit": "a" * 40,
                  "source_hashes": frozen["source_hashes"], "dev_groups_sha256": frozen["dev_groups_sha256"],
                  "protocol_frozen_sha256": digest(run / "protocol_frozen.json")}
        save_json(run / "backup.json", pushed)
        assert ensure_frozen_backup(run, frozen) == pushed
        save_json(run / "backup.json", {"status": "PUSHED", "commit": "a" * 40})
        original_backup = globals()["backup"]
        def fake_backup(folder, reason):
            pending = safe_read_json(folder / "backup.json")
            assert pending["status"] == "PENDING" and pending["last_successful_commit"] == "a" * 40
            return {"status": "SELF_CHECK_RETRY_OBSERVED"}
        globals()["backup"] = fake_backup
        try:
            assert ensure_frozen_backup(run, frozen)["status"] == "SELF_CHECK_RETRY_OBSERVED"
        finally:
            globals()["backup"] = original_backup
    assert probe_hashes(STAGE2)
    original = read_json(STAGE2 / "manifest.json")
    ids = original["unused_ids"]
    assert len(ids) == len(set(ids)) == 190
    assert not set(ids) & {r["unique_id"] for rows in original["splits"].values() for r in rows}
    return {"status": "PASSED", "checks": ["atomic_roundtrip", "corruption_preserved",
            "fixed_probe_metadata_weights", "190_original_unused_ids_disjoint", "frozen_backup_interrupt_retry",
            "session_diagnostics_archived_byte_exact"]}
